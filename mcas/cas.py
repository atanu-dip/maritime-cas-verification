"""
Collision Avoidance System strategies.

Four are implemented so that the comparison means something:

  NoAction         -- the baseline. Establishes what the scenario space
                      actually contains: without it a "97% safe" figure is
                      uninterpretable, because a benign scenario set scores
                      well with no avoidance at all.

  RuleBasedCOLREGS -- classify the encounter, assign the give-way/stand-on
                      role, and make a single substantial alteration to
                      starboard in ample time (Rules 8, 13-17). Resumes the
                      original course once finally past and clear (Rule 8(d)).

  VelocityObstacle -- searches admissible course/speed changes for the least
                      deviation that clears the required DCPA, restricted to
                      starboard alterations while give-way so that the
                      geometric solution stays rule-compliant.

  MPC              -- receding-horizon optimisation over course and speed with
                      explicit terms for safety, COLREGS compliance, and path
                      efficiency. The weights expose the trade-off directly.

All strategies share one interface and one state machine
(CRUISING -> AVOIDING -> RESUMING) so that path-efficiency metrics are
comparable across them.
"""
from __future__ import annotations

import numpy as np

from . import colregs as cr
from .geometry import (KNOTS_TO_MS, cpa, range_rate, relative_bearing, wrap_pi)
from .vessels import ShipState

CRUISING, AVOIDING, RESUMING = "cruising", "avoiding", "resuming"


class CASBase:
    """Common interface and manoeuvre bookkeeping."""
    name = "base"

    def __init__(self, dcpa_limit: float = 926.0, tcpa_limit: float = 1200.0):
        self.dcpa_limit = dcpa_limit
        self.tcpa_limit = tcpa_limit
        self.reset()

    def reset(self, own: ShipState | None = None):
        self.phase = CRUISING
        self.base_heading = None if own is None else own.heading
        self.base_speed = None if own is None else own.speed_kn
        self.first_manoeuvre_idx = None
        self.role = None
        self.encounter = None

    #  helpers 
    def _risk(self, own, tgt):
        t, d = cpa(own.position, own.velocity, tgt.position, tgt.velocity)
        return (d < self.dcpa_limit and 0.0 < t < self.tcpa_limit), t, d

    def _past_and_clear(self, own, tgt):
        """Rule 8(d): finally past and clear -- range opening and outside limit."""
        rng = float(np.linalg.norm(tgt.position - own.position))
        opening = range_rate(own.position, own.velocity,
                             tgt.position, tgt.velocity) > 0
        b = abs(float(relative_bearing(own.position, own.heading, tgt.position)))
        abaft_the_beam = b > np.radians(112.5)
        # Seamanlike "finally past and clear" (Rule 8(d)): the other vessel is
        # drawing abaft the beam and the range is opening. The range test alone
        # is not enough -- two vessels on nearly parallel courses can open only
        # very slowly and never satisfy it, leaving the CAS locked in the
        # avoidance phase and shadowing the target indefinitely.
        return opening and (abaft_the_beam or rng > 3.0 * self.dcpa_limit)

    def _note_manoeuvre(self, step):
        if self.first_manoeuvre_idx is None:
            self.first_manoeuvre_idx = step

    def decide(self, own, tgt, t, step):
        raise NotImplementedError


class NoAction(CASBase):
    """Baseline: hold course and speed regardless of risk."""
    name = "No action (baseline)"

    def decide(self, own, tgt, t, step):
        return self.base_heading, self.base_speed


class RuleBasedCOLREGS(CASBase):
    """Rule-compliant single substantial alteration to starboard."""
    name = "Rule-based COLREGS"

    def __init__(self, dcpa_limit=926.0, tcpa_limit=1200.0,
                 alteration_deg=35.0, act_tcpa_s=900.0,
                 stand_on_fallback_tcpa_s=300.0):
        self.alteration = np.radians(alteration_deg)
        self.act_tcpa = act_tcpa_s
        self.fallback_tcpa = stand_on_fallback_tcpa_s
        super().__init__(dcpa_limit, tcpa_limit)

    def decide(self, own, tgt, t, step):
        self.encounter = cr.classify(own, tgt)
        self.role = cr.assign_role(own, tgt, self.encounter)
        risk, tcpa, dcpa = self._risk(own, tgt)

        if self.phase == AVOIDING:
            if self._past_and_clear(own, tgt):
                self.phase = RESUMING
            else:
                return self._evasive_heading(own, tgt), self.base_speed

        if self.phase == RESUMING:
            return self.base_heading, self.base_speed

        gives_way = self.role in (cr.GIVE_WAY, cr.BOTH_GIVE_WAY)
        if risk and gives_way and tcpa < self.act_tcpa:
            # Rule 8(a)/16: act in ample time, and make it substantial (8(b))
            self.phase = AVOIDING
            self._note_manoeuvre(step)
            return self._evasive_heading(own, tgt), self.base_speed

        if risk and not gives_way and tcpa < self.fallback_tcpa and dcpa < 0.5 * self.dcpa_limit:
            # Rule 17(b): the stand-on vessel must act when collision cannot be
            # avoided by the give-way vessel's action alone.
            self.phase = AVOIDING
            self._note_manoeuvre(step)
            return self._evasive_heading(own, tgt), self.base_speed

        return self.base_heading, self.base_speed

    def _evasive_heading(self, own, tgt):
        """
        Smallest rule-compliant alteration that actually clears.

        A fixed 35 deg turn is substantial (Rule 8(b)) but not necessarily
        effective: if it happens to leave own-ship on a course near-parallel to
        the target, the range never opens. So the alteration is searched from
        the minimum substantial value upwards for the first heading that
        achieves the required DCPA, while staying on the side the Rules
        require.
        """
        b = relative_bearing(own.position, own.heading, tgt.position)
        # Overtaking a vessel fine on the starboard bow: passing down her port
        # side is the shorter and safer alteration.
        sign = -1.0 if (self.encounter == cr.OVERTAKING and b > 0) else 1.0
        target_dcpa = 1.10 * self.dcpa_limit
        speed_ms = own.speed_kn * KNOTS_TO_MS

        fallback = float(wrap_pi(self.base_heading + sign * self.alteration))
        for deg in range(int(np.degrees(self.alteration)), 91, 5):
            h = float(wrap_pi(self.base_heading + sign * np.radians(deg)))
            vel = speed_ms * np.array([np.sin(h), np.cos(h)])
            _, d = cpa(own.position, vel, tgt.position, tgt.velocity)
            if d >= target_dcpa:
                return h
        return fallback


class VelocityObstacle(CASBase):
    """Least-deviation admissible course/speed change clearing the DCPA limit."""
    name = "Velocity obstacle"

    def __init__(self, dcpa_limit=926.0, tcpa_limit=1200.0,
                 max_course_change_deg=60.0, course_step_deg=5.0,
                 speed_options=(1.0, 0.8, 0.6), margin=1.15):
        self.max_course = np.radians(max_course_change_deg)
        self.course_step = np.radians(course_step_deg)
        self.speed_options = speed_options
        self.margin = margin
        super().__init__(dcpa_limit, tcpa_limit)

    def decide(self, own, tgt, t, step):
        self.encounter = cr.classify(own, tgt)
        self.role = cr.assign_role(own, tgt, self.encounter)
        risk, tcpa, dcpa = self._risk(own, tgt)

        if self.phase == AVOIDING and self._past_and_clear(own, tgt):
            self.phase = RESUMING
        if self.phase == RESUMING:
            return self.base_heading, self.base_speed

        if not risk and self.phase == CRUISING:
            return self.base_heading, self.base_speed

        gives_way = self.role in (cr.GIVE_WAY, cr.BOTH_GIVE_WAY)
        target = self.dcpa_limit * self.margin

        # Candidate alterations, ordered by increasing deviation. While
        # give-way, only starboard alterations are admissible (Rule 8(b)).
        n = int(self.max_course / self.course_step)
        deltas = [0.0]
        for k in range(1, n + 1):
            deltas.append(k * self.course_step)
            if not gives_way:
                deltas.append(-k * self.course_step)

        best = None
        for sf in self.speed_options:
            for d in deltas:
                h = float(wrap_pi(self.base_heading + d))
                v = self.base_speed * sf
                vel = v * KNOTS_TO_MS * np.array([np.sin(h), np.cos(h)])
                _, d_cpa = cpa(own.position, vel, tgt.position, tgt.velocity)
                if d_cpa >= target:
                    cost = abs(d) + 2.0 * (1.0 - sf)
                    if best is None or cost < best[0]:
                        best = (cost, h, v)
            if best is not None:
                break

        if best is None:                       # no admissible solution: hardest turn
            h = float(wrap_pi(self.base_heading + self.max_course))
            v = self.base_speed * self.speed_options[-1]
        else:
            _, h, v = best

        if abs(float(wrap_pi(h - self.base_heading))) > np.radians(2.0) or \
                v < self.base_speed - 0.1:
            self.phase = AVOIDING
            self._note_manoeuvre(step)
        return h, v


class MPC(CASBase):
    """
    Receding-horizon course/speed selection.

    Cost = w_safety * domain intrusion + w_colregs * rule penalty
           + w_effort * deviation from the planned track.
    Optimised over a discrete grid, re-solved every `replan_every` steps.
    """
    name = "MPC"

    def __init__(self, dcpa_limit=926.0, tcpa_limit=1200.0,
                 horizon_s=1200.0, dt_pred=30.0, replan_every=6,
                 course_options_deg=(-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60),
                 speed_factors=(1.0, 0.85, 0.7),
                 w_safety=30.0, w_colregs=5.0, w_effort=1.0):
        self.horizon = horizon_s
        self.dt_pred = dt_pred
        self.replan_every = replan_every
        self.course_options = [np.radians(c) for c in course_options_deg]
        self.speed_factors = speed_factors
        self.w_safety, self.w_colregs, self.w_effort = w_safety, w_colregs, w_effort
        self._cached = None
        super().__init__(dcpa_limit, tcpa_limit)

    def reset(self, own=None):
        super().reset(own)
        self._cached = None

    def decide(self, own, tgt, t, step):
        self.encounter = cr.classify(own, tgt)
        self.role = cr.assign_role(own, tgt, self.encounter)
        risk, tcpa, dcpa = self._risk(own, tgt)

        if self.phase == AVOIDING and self._past_and_clear(own, tgt):
            self.phase = RESUMING
        if self.phase == RESUMING:
            return self.base_heading, self.base_speed
        if not risk and self.phase == CRUISING:
            return self.base_heading, self.base_speed

        if self._cached is not None and step % self.replan_every != 0:
            return self._cached

        gives_way = self.role in (cr.GIVE_WAY, cr.BOTH_GIVE_WAY)
        best, best_cost = None, np.inf
        for sf in self.speed_factors:
            for dc in self.course_options:
                h = float(wrap_pi(self.base_heading + dc))
                v = self.base_speed * sf
                cost = self._rollout_cost(own, tgt, h, v, dc, sf, gives_way)
                if cost < best_cost:
                    best_cost, best = cost, (h, v)

        self._cached = best
        if abs(float(wrap_pi(best[0] - self.base_heading))) > np.radians(2.0) or \
                best[1] < self.base_speed - 0.1:
            self.phase = AVOIDING
            self._note_manoeuvre(step)
        return best

    def _rollout_cost(self, own, tgt, h, v, dc, sf, gives_way):
        # First-order heading lag: a ship does not acquire a new course
        # instantly, and a rollout that assumes it does over-credits small
        # alterations. tau is a coarse fit to the Nomoto response.
        tau = max(2.5 * own.spec.T(own.speed_kn), 30.0)
        tvel = tgt.velocity
        p, q = own.position.copy(), tgt.position.copy()
        psi = own.heading
        n = int(self.horizon / self.dt_pred)

        worst = np.inf
        for _ in range(n):
            psi = psi + wrap_pi(h - psi) * min(self.dt_pred / tau, 1.0)
            vel = v * KNOTS_TO_MS * np.array([np.sin(psi), np.cos(psi)])
            p = p + vel * self.dt_pred
            q = q + tvel * self.dt_pred
            worst = min(worst, float(np.linalg.norm(q - p)))

        # Hinge rather than a quadratic: a squared penalty makes a near-miss
        # cheap enough that the optimiser trades it against a small course
        # change, which is exactly the behaviour a CAS must not have.
        margin = 1.1 * self.dcpa_limit
        safety = 0.0 if worst >= margin else \
            1.0 + 3.0 * (margin - worst) / margin
        effort = abs(dc) / np.radians(60.0) + 1.5 * (1.0 - sf)

        colregs_pen = 0.0
        if gives_way and dc < 0:
            colregs_pen += 1.0                       # alteration to port
        if gives_way and 0 < abs(dc) < np.radians(20.0):
            colregs_pen += 0.5                       # not readily apparent
        if not gives_way and abs(dc) > np.radians(10.0):
            colregs_pen += 0.8                       # stand-on vessel manoeuvring

        return (self.w_safety * safety + self.w_colregs * colregs_pen
                + self.w_effort * effort)


CAS_REGISTRY = {
    "no_action": NoAction,
    "rule_based": RuleBasedCOLREGS,
    "velocity_obstacle": VelocityObstacle,
    "mpc": MPC,
}

__all__ = ["CASBase", "NoAction", "RuleBasedCOLREGS", "VelocityObstacle", "MPC",
           "CAS_REGISTRY", "CRUISING", "AVOIDING", "RESUMING"]
