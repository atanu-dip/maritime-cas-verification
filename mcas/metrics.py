"""
Aggregate metrics: encounter-space coverage, interval estimates, and the
safety / compliance / efficiency trade-off.

Coverage is *measured*, not asserted. A claim such as "95% of COLREGS
situations covered" is only meaningful if the situation space has been defined
and the occupancy of its cells counted, so that is what this module does: it
discretises the encounter space, records which cells the generated scenarios
reach, and reports the fraction of *reachable* cells hit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Encounter space discretisation

BEARING_EDGES = np.arange(-180.0, 180.1, 22.5)        # 16 bins
REL_HEADING_EDGES = np.arange(-180.0, 180.1, 30.0)    # 12 bins
SPEED_RATIO_EDGES = np.array([0.0, 0.7, 1.0, 1.4, np.inf])
DCPA_EDGES = np.array([0.0, 250.0, 500.0, 926.0, 1500.0])
STATUSES = ["power_driven", "fishing", "restricted_manoeuvrability",
            "not_under_command"]
VISIBILITIES = ["good", "moderate", "restricted"]


def _wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def annotate_space(df: pd.DataFrame) -> pd.DataFrame:
    """Add discrete encounter-space cell indices to a results frame."""
    out = df.copy()
    out["rel_heading_deg"] = _wrap180(out["tgt_heading_deg"])
    out["speed_ratio"] = out["tgt_speed_kn"] / out["own_speed_kn"]
    out["bin_bearing"] = np.digitize(_wrap180(out["relative_bearing_deg"]),
                                     BEARING_EDGES) - 1
    out["bin_rel_heading"] = np.digitize(out["rel_heading_deg"],
                                         REL_HEADING_EDGES) - 1
    out["bin_speed_ratio"] = np.digitize(out["speed_ratio"], SPEED_RATIO_EDGES) - 1
    out["bin_dcpa"] = np.digitize(out["dcpa_target_m"], DCPA_EDGES) - 1
    return out


def coverage_report(df: pd.DataFrame, reference: pd.DataFrame | None = None):
    """
    Measure coverage of the encounter space.

    `reference` is a large sample used to establish which cells are physically
    reachable; without it every cell of the full cross-product is treated as
    reachable, which understates coverage because many combinations (e.g. a
    reciprocal heading on a quarter bearing) cannot occur.
    """
    d = annotate_space(df).drop_duplicates("scenario_id")
    ref = annotate_space(reference).drop_duplicates("scenario_id") \
        if reference is not None else d

    def cells(frame, keys):
        return set(map(tuple, frame[keys].to_numpy()))

    report = {}
    marginals = {
        "relative_bearing": (["bin_bearing"], len(BEARING_EDGES) - 1),
        "relative_heading": (["bin_rel_heading"], len(REL_HEADING_EDGES) - 1),
        "speed_ratio": (["bin_speed_ratio"], len(SPEED_RATIO_EDGES) - 1),
        "dcpa_band": (["bin_dcpa"], len(DCPA_EDGES) - 1),
    }
    for name, (keys, total) in marginals.items():
        hit = len(cells(d, keys))
        report[name] = {"hit": hit, "total": total, "pct": 100.0 * hit / total}

    for name, col, vocab in (("vessel_status", "tgt_status", STATUSES),
                             ("visibility", "visibility", VISIBILITIES)):
        hit = d[col].nunique()
        report[name] = {"hit": hit, "total": len(vocab),
                        "pct": 100.0 * hit / len(vocab)}

    joint_keys = ["bin_bearing", "bin_rel_heading"]
    reachable = cells(ref, joint_keys)
    hit_joint = cells(d, joint_keys) & reachable
    report["bearing_x_heading"] = {
        "hit": len(hit_joint), "total": len(reachable),
        "pct": 100.0 * len(hit_joint) / max(len(reachable), 1)}

    role_keys = ["encounter_observed", "role"]
    reachable_roles = cells(ref, role_keys)
    hit_roles = cells(d, role_keys) & reachable_roles
    report["encounter_x_role"] = {
        "hit": len(hit_roles), "total": len(reachable_roles),
        "pct": 100.0 * len(hit_roles) / max(len(reachable_roles), 1)}
    return report


def coverage_grid(df: pd.DataFrame):
    """2-D occupancy counts over (relative bearing x relative heading)."""
    d = annotate_space(df).drop_duplicates("scenario_id")
    nb, nh = len(BEARING_EDGES) - 1, len(REL_HEADING_EDGES) - 1
    grid = np.zeros((nh, nb), dtype=int)
    for b, h in zip(d["bin_bearing"], d["bin_rel_heading"]):
        if 0 <= b < nb and 0 <= h < nh:
            grid[h, b] += 1
    return grid


# Interval estimates

def wilson_interval(successes: int, n: int, z: float = 1.96):
    """
    Wilson score interval for a binomial proportion.

    Preferred to the normal approximation here because safety rates sit near
    0 or 1, where the naive interval can extend outside [0, 1].
    """
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = successes / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def rate_table(df: pd.DataFrame, by=("cas_name",), column="safe"):
    """Rate with Wilson bounds for each group."""
    rows = []
    for key, g in df.groupby(list(by)):
        k = int(g[column].sum())
        n = len(g)
        p, lo, hi = wilson_interval(k, n)
        entry = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        entry.update({"n": n, "successes": k, "rate": p, "lo": lo, "hi": hi})
        rows.append(entry)
    return pd.DataFrame(rows)



# Trade-off analysis

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table: safety, compliance and efficiency per strategy."""
    g = df.groupby("cas_name")
    out = pd.DataFrame({
        "n": g.size(),
        "safety_rate": g["safe"].mean(),
        "domain_respected": g["domain_respected"].mean(),
        "collision_rate": g["collision"].mean(),
        "mean_min_dist_m": g["min_dist_m"].mean(),
        "worst_min_dist_m": g["min_dist_m"].min(),
        "p05_min_dist_m": g["min_dist_m"].quantile(0.05),
        "compliance": g["compliance"].mean(),
        "progress_loss_pct": g["progress_loss_pct"].mean(),
        "mean_alteration_deg": g["total_alteration_deg"].mean(),
    })
    ci = rate_table(df, by=("cas_name",), column="safe").set_index("cas_name")
    out["safety_lo"] = ci["lo"]
    out["safety_hi"] = ci["hi"]
    return out.sort_values("safety_rate", ascending=False)


def pareto_front(points: np.ndarray, maximise=(True, True)):
    """Indices of the non-dominated points."""
    pts = np.asarray(points, float).copy()
    for j, mx in enumerate(maximise):
        if not mx:
            pts[:, j] = -pts[:, j]
    keep = []
    for i, p in enumerate(pts):
        if not np.any(np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)):
            keep.append(i)
    return keep


def compliance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Per-rule compliance rates by strategy (NaN where the rule does not apply)."""
    rules = ["starboard_alteration", "substantial_action", "early_action",
             "no_cross_ahead", "stand_on_held"]
    return df.groupby("cas_name")[rules].mean()


__all__ = ["annotate_space", "coverage_report", "coverage_grid",
           "wilson_interval", "rate_table", "summary_table", "pareto_front",
           "compliance_matrix", "BEARING_EDGES", "REL_HEADING_EDGES"]
