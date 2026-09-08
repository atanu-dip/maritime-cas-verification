"""
COLREGS classification, role assignment and compliance scoring.

Encounter classification follows the sector geometry of the Rules rather than
a heading-difference heuristic:

  Rule 13 (Overtaking)  a vessel coming up from more than 22.5 deg abaft the
                        beam of another -- i.e. bearing *of the overtaking
                        vessel from the overtaken vessel* outside +/-112.5 deg
                        -- is overtaking, and must keep clear regardless of
                        any later change of bearing.
  Rule 14 (Head-on)     nearly reciprocal courses, each vessel near the other's
                        bow; both alter to starboard.
  Rule 15 (Crossing)    the vessel which has the other on her own starboard
                        side keeps out of the way, and shall avoid crossing
                        ahead of the other vessel.
  Rule 18               overrides the above by vessel category: a power-driven
                        vessel keeps out of the way of sailing / fishing /
                        restricted-in-manoeuvrability / not-under-command
                        vessels.

Rule 13 is tested first, because an overtaking situation stays an overtaking
situation even if the bearing later opens into a crossing sector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (SECTOR_BOW, SECTOR_HEAD_ON, SECTOR_OVERTAKING, cpa,
                       relative_bearing, wrap_pi)
from .vessels import VESSEL_STATUS, ShipState

HEAD_ON = "head_on"
CROSSING = "crossing"
OVERTAKING = "overtaking"
OVERTAKEN = "overtaken"
NO_RISK = "no_risk"

GIVE_WAY = "give_way"
STAND_ON = "stand_on"
BOTH_GIVE_WAY = "both_give_way"


def classify(own: ShipState, tgt: ShipState) -> str:
    """
    Classify the encounter from own-ship's point of view.
    Returns one of: head_on, crossing, overtaking, overtaken.
    """
    b_tgt_from_own = relative_bearing(own.position, own.heading, tgt.position)
    b_own_from_tgt = relative_bearing(tgt.position, tgt.heading, own.position)
    dpsi = abs(float(wrap_pi(tgt.heading - own.heading)))

    # Rule 13 first: is one vessel coming up on the other from abaft the beam?
    if dpsi < SECTOR_OVERTAKING:
        # roughly co-directional
        if abs(b_own_from_tgt) > SECTOR_BOW and own.speed_kn > tgt.speed_kn:
            return OVERTAKING          # own-ship is overtaking the target
        if abs(b_tgt_from_own) > SECTOR_BOW and tgt.speed_kn > own.speed_kn:
            return OVERTAKEN           # target is overtaking own-ship

    # Rule 14: nearly reciprocal, each near the other's bow
    if abs(abs(dpsi) - np.pi) < SECTOR_HEAD_ON and \
            abs(b_tgt_from_own) < np.radians(15.0) and \
            abs(b_own_from_tgt) < np.radians(15.0):
        return HEAD_ON

    return CROSSING


def assign_role(own: ShipState, tgt: ShipState, encounter: str | None = None) -> str:
    """
    Determine own-ship's obligation. Rule 18 (vessel category) takes precedence
    over the steering rules for the encounter geometry.
    """
    encounter = encounter or classify(own, tgt)

    own_priv = VESSEL_STATUS.get(own.spec.status, 0)
    tgt_priv = VESSEL_STATUS.get(tgt.spec.status, 0)
    if own_priv != tgt_priv:
        # Rule 18: the less privileged vessel keeps out of the way.
        return GIVE_WAY if own_priv < tgt_priv else STAND_ON

    if encounter == HEAD_ON:
        return BOTH_GIVE_WAY            # Rule 14: both alter to starboard
    if encounter == OVERTAKING:
        return GIVE_WAY                 # Rule 13: overtaking vessel keeps clear
    if encounter == OVERTAKEN:
        return STAND_ON
    if encounter == CROSSING:
        b = relative_bearing(own.position, own.heading, tgt.position)
        # Rule 15: other on my starboard side -> I give way
        return GIVE_WAY if 0.0 < b <= SECTOR_BOW else STAND_ON
    return STAND_ON


def risk_of_collision(own: ShipState, tgt: ShipState, dcpa_limit: float,
                      tcpa_limit: float) -> bool:
    """Rule 7: risk exists if predicted CPA is close and approaching."""
    t, d = cpa(own.position, own.velocity, tgt.position, tgt.velocity)
    return bool(d < dcpa_limit and 0.0 < t < tcpa_limit)


# Compliance scoring

@dataclass
class ComplianceReport:
    encounter: str
    role: str
    starboard_alteration: float = np.nan   # 1 = complied, 0 = violated, nan = n/a
    substantial_action: float = np.nan     # Rule 8(b)
    early_action: float = np.nan           # Rule 8(a) / 16
    no_cross_ahead: float = np.nan         # Rule 15 second clause
    stand_on_held: float = np.nan          # Rule 17(a)
    overall: float = np.nan

    def as_dict(self):
        return {
            "encounter": self.encounter, "role": self.role,
            "starboard_alteration": self.starboard_alteration,
            "substantial_action": self.substantial_action,
            "early_action": self.early_action,
            "no_cross_ahead": self.no_cross_ahead,
            "stand_on_held": self.stand_on_held,
            "compliance": self.overall,
        }


def score_compliance(result: dict, substantial_deg: float = 30.0,
                     early_tcpa_s: float = 600.0,
                     stand_on_tolerance_deg: float = 10.0) -> ComplianceReport:
    """
    Score a completed run against the steering rules.

    `result` is the dict returned by runner.run_scenario and must contain
    heading history, the initial encounter classification and role, the
    per-step TCPA history, and the index of first substantive manoeuvre.
    """
    enc = result["encounter"]
    role = result["role"]
    rep = ComplianceReport(encounter=enc, role=role)

    hdg = np.asarray(result["own_heading"])
    h0 = hdg[0]
    net = np.degrees(wrap_pi(hdg - h0))
    max_stbd = float(np.max(net))
    max_port = float(-np.min(net))
    total_alteration = max(max_stbd, max_port)

    gives_way = role in (GIVE_WAY, BOTH_GIVE_WAY)

    if gives_way:
        # A give-way vessel that never alters has not "kept out of the way" at
        # all (Rule 16); scoring that as compliant because no port turn was
        # made would reward inaction.
        if total_alteration < 5.0:
            rep.starboard_alteration = 0.0
        else:
            rep.starboard_alteration = 1.0 if max_stbd >= max_port else 0.0
        # ...large enough to be readily apparent to another vessel (Rule 8(b))
        rep.substantial_action = 1.0 if total_alteration >= substantial_deg else 0.0
        # ...and taken in ample time (Rule 8(a) / Rule 16)
        idx = result.get("first_manoeuvre_idx")
        if idx is None:
            rep.early_action = 0.0
        else:
            tcpa_at_action = result["tcpa_hist"][idx]
            rep.early_action = 1.0 if tcpa_at_action >= early_tcpa_s else 0.0

        if enc == CROSSING:
            rep.no_cross_ahead = 1.0 if not result.get("crossed_ahead", False) else 0.0
    else:
        # Rule 17(a)(i): the stand-on vessel keeps her course and speed.
        # Late action to avoid collision (17(b)) is permitted, so only penalise
        # alteration made while the situation was still developing.
        early_idx = result.get("early_window_idx", len(hdg))
        early_net = np.abs(net[:max(early_idx, 1)])
        rep.stand_on_held = 1.0 if float(np.max(early_net, initial=0.0)) \
            <= stand_on_tolerance_deg else 0.0

    scores = [v for v in (rep.starboard_alteration, rep.substantial_action,
                          rep.early_action, rep.no_cross_ahead,
                          rep.stand_on_held) if not np.isnan(v)]
    rep.overall = float(np.mean(scores)) if scores else np.nan
    return rep


__all__ = ["classify", "assign_role", "risk_of_collision", "score_compliance",
           "ComplianceReport", "HEAD_ON", "CROSSING", "OVERTAKING", "OVERTAKEN",
           "GIVE_WAY", "STAND_ON", "BOTH_GIVE_WAY"]
