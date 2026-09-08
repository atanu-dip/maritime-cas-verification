"""
Simulation runner.

Own-ship is guided by a CAS; the target vessel is a non-cooperative stand-on
test target holding course and speed. That is deliberately the harder case: it
removes any assumption that the other vessel will solve the problem, which is
the situation an autonomous CAS has to be certified against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import colregs as cr
from .cas import CAS_REGISTRY, CASBase
from .geometry import FujiiDomain, NM_TO_M, cpa, relative_bearing, wrap_pi
from .scenarios import EncounterSpec, solve_initial_geometry
from .vessels import NomotoModel, ShipState

DEFAULT_SAFETY_M = 0.5 * NM_TO_M          # 926 m


def run_scenario(own0: ShipState, tgt0: ShipState, cas: CASBase,
                 dt: float = 5.0, horizon_s: float = 3600.0,
                 safety_m: float = DEFAULT_SAFETY_M,
                 dynamics=None, spec: EncounterSpec | None = None) -> dict:
    """Simulate one encounter and return full histories plus summary metrics."""
    dynamics = dynamics or NomotoModel()
    own, tgt = own0.copy(), tgt0.copy()
    n = int(horizon_s / dt)

    cas.reset(own)
    encounter0 = cr.classify(own, tgt)
    role0 = cr.assign_role(own, tgt, encounter0)

    traj_own = np.zeros((n, 2))
    traj_tgt = np.zeros((n, 2))
    own_heading = np.zeros(n)
    tgt_heading = np.zeros(n)
    own_speed = np.zeros(n)
    rudder = np.zeros(n)
    yaw_rate = np.zeros(n)
    dist = np.zeros(n)
    dcpa_hist = np.zeros(n)
    tcpa_hist = np.zeros(n)
    intrusion = np.zeros(n)
    bearing = np.zeros(n)

    domain = FujiiDomain(own.spec.length_m)
    base_heading, base_speed = own.heading, own.speed_kn
    crossed_ahead = False
    early_window_idx = n

    for i in range(n):
        traj_own[i] = own.position
        traj_tgt[i] = tgt.position
        own_heading[i] = own.heading
        tgt_heading[i] = tgt.heading
        own_speed[i] = own.speed_kn
        rudder[i] = own.rudder
        yaw_rate[i] = own.yaw_rate
        d = float(np.linalg.norm(tgt.position - own.position))
        dist[i] = d
        t_c, d_c = cpa(own.position, own.velocity, tgt.position, tgt.velocity)
        tcpa_hist[i] = t_c
        dcpa_hist[i] = d_c
        intrusion[i] = domain.intrusion(tgt.position, tgt.heading, own.position)
        bearing[i] = relative_bearing(own.position, own.heading, tgt.position)

        # Window in which a stand-on vessel is still expected to hold course.
        # Freeze at the first moment TCPA falls below the late-action
        # threshold; tracking the last moment it was above lets a manoeuvre
        # that opens TCPA back up re-extend the window retrospectively.
        if early_window_idx == n and t_c <= 300.0 and i > 0:
            early_window_idx = i

        # did own-ship pass ahead of the target's bow at close range?
        if d < 3.0 * safety_m:
            b_own_from_tgt = relative_bearing(tgt.position, tgt.heading, own.position)
            if abs(b_own_from_tgt) < np.radians(30.0):
                crossed_ahead = True

        h_cmd, v_cmd = cas.decide(own, tgt, i * dt, i)
        dynamics.step(own, h_cmd, v_cmd, dt)

        tv = tgt.velocity
        tgt.x += tv[0] * dt
        tgt.y += tv[1] * dt

    min_idx = int(np.argmin(dist))
    min_dist = float(dist[min_idx])
    min_intrusion = float(np.min(intrusion))

    # Path efficiency. Distance travelled is a poor measure here: a vessel
    # loses speed in a turn, so an avoiding ship can travel *less* than one
    # holding course. What actually costs the operator is lost progress along
    # the intended track, so that is what is reported.
    step_len = np.linalg.norm(np.diff(traj_own, axis=0), axis=1)
    path_len = float(step_len.sum())
    straight_len = float(base_speed * 0.5144444 * (n - 1) * dt)
    track_unit = np.array([np.sin(base_heading), np.cos(base_heading)])
    advance = float((traj_own[-1] - traj_own[0]) @ track_unit)
    progress_loss_pct = 100.0 * (straight_len - advance) / max(straight_len, 1.0)
    excess_path_pct = 100.0 * (path_len - straight_len) / max(straight_len, 1.0)
    cross_track = float(np.max(np.abs(traj_own[:, 0] - traj_own[0, 0])))
    heading_changes = int(np.sum(np.abs(np.degrees(
        wrap_pi(np.diff(own_heading)))) > 0.25))

    result = {
        "traj_own": traj_own, "traj_tgt": traj_tgt,
        "own_heading": own_heading, "tgt_heading": tgt_heading,
        "own_speed": own_speed, "rudder": rudder, "yaw_rate": yaw_rate,
        "dist": dist, "dcpa_hist": dcpa_hist, "tcpa_hist": tcpa_hist,
        "intrusion": intrusion, "bearing": bearing,
        "dt": dt, "n": n, "safety_m": safety_m,
        "min_dist": min_dist, "min_dist_idx": min_idx,
        "min_dist_t": min_idx * dt,
        "min_intrusion": min_intrusion,
        "safe": bool(min_dist >= safety_m),
        "domain_respected": bool(min_intrusion >= 1.0),
        "collision": bool(min_dist < 0.5 * (own.spec.length_m + tgt.spec.length_m)),
        "encounter": encounter0, "role": role0,
        "cas_name": cas.name,
        "first_manoeuvre_idx": cas.first_manoeuvre_idx,
        "early_window_idx": early_window_idx,
        "crossed_ahead": crossed_ahead,
        "excess_path_pct": excess_path_pct,
        "progress_loss_pct": progress_loss_pct,
        "cross_track_m": cross_track,
        "heading_changes": heading_changes,
        "total_alteration_deg": float(np.max(np.abs(np.degrees(
            wrap_pi(own_heading - base_heading))))),
        "spec": spec,
        "domain": domain,
    }
    result["compliance"] = cr.score_compliance(result)
    return result


def run_batch(scenario_set, cas_keys=("no_action", "rule_based",
                                      "velocity_obstacle", "mpc"),
              dt: float = 5.0, horizon_s: float = 3600.0,
              safety_m: float = DEFAULT_SAFETY_M, dynamics=None,
              progress=None):
    """
    Run every (scenario, CAS) pair.

    `scenario_set` is a dict mapping encounter name -> list of
    (spec, own_state, tgt_state, diagnostics) as produced by a sampler.
    Returns (dataframe_of_summaries, dict_of_full_results).
    """
    rows, store = [], {}
    total = sum(len(v) for v in scenario_set.values()) * len(cas_keys)
    done = 0
    for enc, items in scenario_set.items():
        for j, (spec, own0, tgt0, diag) in enumerate(items):
            for key in cas_keys:
                cas = CAS_REGISTRY[key](dcpa_limit=safety_m)
                res = run_scenario(own0, tgt0, cas, dt=dt, horizon_s=horizon_s,
                                   safety_m=safety_m, dynamics=dynamics, spec=spec)
                sid = f"{enc}_{j:04d}"
                store[(sid, key)] = res
                row = {
                    "scenario_id": sid, "encounter_requested": enc,
                    "encounter_observed": res["encounter"], "role": res["role"],
                    "cas": key, "cas_name": res["cas_name"],
                    "dcpa_target_m": spec.dcpa_m, "tcpa_target_s": spec.tcpa_s,
                    "own_speed_kn": spec.own_speed_kn,
                    "tgt_speed_kn": spec.tgt_speed_kn,
                    "own_vessel": spec.own_vessel, "tgt_vessel": spec.tgt_vessel,
                    "tgt_status": spec.tgt_status, "visibility": spec.visibility,
                    "initial_range_m": diag["initial_range_m"],
                    "relative_bearing_deg": diag["relative_bearing_deg"],
                    "tgt_heading_deg": spec.tgt_heading_deg,
                    "min_dist_m": res["min_dist"],
                    "min_intrusion": res["min_intrusion"],
                    "safe": res["safe"],
                    "domain_respected": res["domain_respected"],
                    "collision": res["collision"],
                    "excess_path_pct": res["excess_path_pct"],
                    "progress_loss_pct": res["progress_loss_pct"],
                    "cross_track_m": res["cross_track_m"],
                    "heading_changes": res["heading_changes"],
                    "total_alteration_deg": res["total_alteration_deg"],
                    "crossed_ahead": res["crossed_ahead"],
                }
                row.update(res["compliance"].as_dict())
                rows.append(row)
                done += 1
                if progress and done % max(total // 20, 1) == 0:
                    progress(done, total)
    return pd.DataFrame(rows), store


__all__ = ["run_scenario", "run_batch", "DEFAULT_SAFETY_M"]
