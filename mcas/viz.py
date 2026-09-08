"""
Visualisation.

Three families of figure:

  Per-encounter   trajectory plots, relative-motion (radar plotting) diagrams,
                  PPI displays, CPA time series, control histories, animations
  Validation      turning-circle manoeuvring check against IMO criteria
  Aggregate       coverage maps, safety rates with intervals, distance
                  distributions, per-rule compliance, safety/efficiency Pareto,
                  automatic failure galleries
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.lines import Line2D

from .geometry import NM_TO_M, wrap_pi
from .metrics import (BEARING_EDGES, REL_HEADING_EDGES, coverage_grid,
                      pareto_front, rate_table)

OWN_C = "#1b4f8f"
TGT_C = "#c0392b"
SAFE_C = "#2e7d52"
WARN_C = "#e08b18"
GRID_C = "#cfd6dd"

CAS_COLORS = {
    "No action (baseline)": "#8a8f96",
    "Rule-based COLREGS": "#1b4f8f",
    "Velocity obstacle": "#2e7d52",
    "MPC": "#8e44ad",
}


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150,
        "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
        "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
    })



# Primitives

def ship_polygon(pos, heading, length, beam, scale=1.0):
    """Plan-view hull outline in world coordinates."""
    L, B = length * scale / 2.0, beam * scale / 2.0
    pts = np.array([[0.0, L], [B, 0.35 * L], [B, -L], [-B, -L], [-B, 0.35 * L]])
    c, s = np.cos(heading), np.sin(heading)
    x = pos[0] + pts[:, 0] * c + pts[:, 1] * s
    y = pos[1] - pts[:, 0] * s + pts[:, 1] * c
    return np.column_stack([x, y])


def _draw_ship(ax, pos, heading, spec, color, scale, label=None, alpha=0.95):
    poly = ship_polygon(pos, heading, spec.length_m, spec.beam_m, scale)
    ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="black",
                         linewidth=0.6, alpha=alpha, zorder=5, label=label))


def _nm_axis(ax):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v / NM_TO_M:.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("East (nm)")
    ax.set_ylabel("North (nm)")



# 1. Encounter trajectory

def plot_encounter(res, title=None, ship_scale=18.0, show_domain=True,
                   mark_every_s=300.0, ax=None):
    """Top-down track plot with hull outlines, ship domain and CPA marker."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.0, 7.0))
    to, tt = res["traj_own"], res["traj_tgt"]
    dt = res["dt"]

    ax.plot(to[:, 0], to[:, 1], color=OWN_C, lw=1.8, label="Own ship (CAS)")
    ax.plot(tt[:, 0], tt[:, 1], color=TGT_C, lw=1.8, ls="--", label="Target (stand-on)")

    step = max(int(mark_every_s / dt), 1)
    ax.scatter(to[::step, 0], to[::step, 1], s=9, color=OWN_C, zorder=4)
    ax.scatter(tt[::step, 0], tt[::step, 1], s=9, color=TGT_C, zorder=4)

    i = res["min_dist_idx"]
    ax.plot([to[i, 0], tt[i, 0]], [to[i, 1], tt[i, 1]], color="black",
            lw=1.1, ls=":", zorder=6)
    mid = (to[i] + tt[i]) / 2.0
    ok = res["min_dist"] >= res["safety_m"]
    ax.annotate(f"CPA {res['min_dist']:.0f} m\n@ {res['min_dist_t'] / 60:.0f} min",
                xy=mid, xytext=(12, 12), textcoords="offset points",
                fontsize=8.5, color=SAFE_C if ok else TGT_C,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=SAFE_C if ok else TGT_C, lw=0.8))

    ax.add_patch(Circle(tt[i], res["safety_m"], fill=False, ls="--",
                        ec=WARN_C, lw=1.0, zorder=2))
    if show_domain and res.get("domain") is not None:
        b = res["domain"].boundary(tt[i], res["tgt_heading"][i])
        ax.plot(b[:, 0], b[:, 1], color=TGT_C, lw=0.9, alpha=0.55, zorder=2)

    for k, (traj, hdg, c) in enumerate(((to, res["own_heading"], OWN_C),
                                        (tt, res["tgt_heading"], TGT_C))):
        spec = res["spec"] if False else None
        _draw_ship(ax, traj[0], hdg[0],
                   _spec_for(res, k), c, ship_scale, alpha=0.55)
        _draw_ship(ax, traj[i], hdg[i], _spec_for(res, k), c, ship_scale)

    ax.set_aspect("equal")
    _nm_axis(ax)
    ax.set_title(title or f"{res['encounter'].replace('_', '-').title()} "
                          f"({res['role'].replace('_', ' ')}) — {res['cas_name']}")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=WARN_C, ls="--", label="Safety limit"),
                Line2D([], [], color=TGT_C, alpha=0.55, label="Fujii domain")]
    ax.legend(handles=handles, loc="best", fontsize=8.5)
    if created:
        plt.tight_layout()
    return ax.figure


def _spec_for(res, which):
    """Vessel spec for own (0) / target (1); falls back to a default hull."""
    from .vessels import VESSEL_LIBRARY
    sp = res.get("spec")
    if sp is None:
        return VESSEL_LIBRARY["container_feeder"]
    return VESSEL_LIBRARY[sp.own_vessel if which == 0 else sp.tgt_vessel]


# 2. Strategy comparison grid

def plot_strategy_comparison(results_by_cas, title="", ship_scale=18.0):
    """Same encounter under every strategy, on a shared scale."""
    n = len(results_by_cas)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.9), squeeze=False)
    xs, ys = [], []
    for r in results_by_cas.values():
        xs += [r["traj_own"][:, 0], r["traj_tgt"][:, 0]]
        ys += [r["traj_own"][:, 1], r["traj_tgt"][:, 1]]
    xlim = (min(a.min() for a in xs), max(a.max() for a in xs))
    ylim = (min(a.min() for a in ys), max(a.max() for a in ys))
    pad = 0.08 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    for ax, (name, r) in zip(axes[0], results_by_cas.items()):
        plot_encounter(r, title=name, ship_scale=ship_scale, ax=ax)
        ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
        ax.set_ylim(ylim[0] - pad, ylim[1] + pad)
        ax.get_legend().remove()
        ok = r["min_dist"] >= r["safety_m"]
        ax.set_title(f"{name}\nCPA {r['min_dist']:.0f} m · "
                     f"{'SAFE' if ok else 'UNSAFE'} · "
                     f"compliance {r['compliance'].overall:.2f}",
                     color=SAFE_C if ok else TGT_C, fontsize=10)
    fig.suptitle(title, fontsize=12.5, y=1.02)
    plt.tight_layout()
    return fig


# 3. Relative motion diagram (radar plotting sheet)

def plot_relative_motion(res, ax=None, title=None):
    """
    Target track in own-ship's relative frame — the classic radar plotting
    presentation, in which a constant-bearing closing track appears as a
    straight line through the origin.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6.2, 6.2))
    rel = res["traj_tgt"] - res["traj_own"]
    h = res["own_heading"]
    fwd = np.column_stack([np.sin(h), np.cos(h)])
    stbd = np.column_stack([np.cos(h), -np.sin(h)])
    x = np.sum(rel * stbd, axis=1)
    y = np.sum(rel * fwd, axis=1)

    ax.plot(x, y, color=TGT_C, lw=1.8)
    ax.scatter([x[0]], [y[0]], s=45, color=TGT_C, zorder=5, label="Target start")
    i = res["min_dist_idx"]
    ax.scatter([x[i]], [y[i]], s=70, marker="X", color="black", zorder=6,
               label=f"CPA {res['min_dist']:.0f} m")
    ax.scatter([0], [0], s=90, marker="^", color=OWN_C, zorder=6, label="Own ship")

    for r in (res["safety_m"], 2 * res["safety_m"], 3 * res["safety_m"]):
        ax.add_patch(Circle((0, 0), r, fill=False, ec=GRID_C, lw=0.8, zorder=1))
    ax.add_patch(Circle((0, 0), res["safety_m"], fill=False, ec=WARN_C,
                        ls="--", lw=1.2, zorder=2))
    lim = max(np.abs(np.concatenate([x, y]))) * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID_C, lw=0.8)
    ax.axvline(0, color=GRID_C, lw=0.8)
    ax.set_xlabel("Starboard of own ship (m)")
    ax.set_ylabel("Ahead of own ship (m)")
    ax.set_title(title or "Relative motion (own-ship frame)")
    ax.legend(fontsize=8.5, loc="upper left")
    if created:
        plt.tight_layout()
    return ax.figure

# 4. PPI radar display

def plot_ppi(res, ax=None, max_range_nm=6.0):
    """Bearing/range presentation as seen on own-ship's radar."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6.2, 6.2),
                             subplot_kw={"projection": "polar"})
    rel = res["traj_tgt"] - res["traj_own"]
    rng = np.linalg.norm(rel, axis=1)
    brg = np.arctan2(rel[:, 0], rel[:, 1]) - res["own_heading"]
    brg = wrap_pi(brg)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    sc = ax.scatter(brg, rng, c=np.arange(len(rng)) * res["dt"] / 60.0,
                    cmap="viridis", s=7)
    i = res["min_dist_idx"]
    ax.scatter([brg[i]], [rng[i]], s=90, marker="X", color=TGT_C, zorder=6)
    ax.set_rmax(min(max_range_nm * NM_TO_M, rng.max() * 1.05))
    ax.set_rlabel_position(135)
    ax.set_thetagrids(range(0, 360, 30))
    ax.set_title("Radar PPI — target bearing and range", pad=16)
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.10, shrink=0.78)
    cb.set_label("Elapsed time (min)")
    if created:
        plt.tight_layout()
    return ax.figure


# 5. CPA / domain time series and control history

def plot_cpa_timeseries(results_by_cas, safety_m, title=""):
    """Predicted DCPA, actual range and TCPA against time, per strategy."""
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True)
    for name, r in results_by_cas.items():
        t = np.arange(r["n"]) * r["dt"] / 60.0
        c = CAS_COLORS.get(name, None)
        axes[0].plot(t, r["dcpa_hist"], lw=1.6, color=c, label=name)
        axes[1].plot(t, r["dist"], lw=1.6, color=c)
        axes[2].plot(t, r["tcpa_hist"] / 60.0, lw=1.6, color=c)
    axes[0].axhline(safety_m, color=WARN_C, ls="--", lw=1.1)
    axes[1].axhline(safety_m, color=WARN_C, ls="--", lw=1.1)
    axes[0].set_ylabel("Predicted DCPA (m)")
    axes[1].set_ylabel("Actual range (m)")
    axes[2].set_ylabel("TCPA (min)")
    axes[2].set_xlabel("Time (min)")
    axes[0].set_title(title or "Collision-risk evolution")
    axes[0].legend(fontsize=8.5, ncol=2)
    plt.tight_layout()
    return fig


def plot_control_history(res, title=""):
    """Rudder, heading, yaw rate and speed — the manoeuvre as actually flown."""
    t = np.arange(res["n"]) * res["dt"] / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.6), sharex=True)
    axes[0].plot(t, np.degrees(res["rudder"]), color=OWN_C, lw=1.5)
    axes[0].set_ylabel("Rudder (deg)")
    axes[1].plot(t, np.degrees(wrap_pi(res["own_heading"] - res["own_heading"][0])),
                 color=OWN_C, lw=1.5)
    axes[1].axhline(30, color=WARN_C, ls=":", lw=1.0)
    axes[1].axhline(-30, color=WARN_C, ls=":", lw=1.0)
    axes[1].set_ylabel("Course change (deg)")
    axes[2].plot(t, np.degrees(res["yaw_rate"]) * 60.0, color=OWN_C, lw=1.5)
    axes[2].set_ylabel("Yaw rate (deg/min)")
    axes[3].plot(t, res["own_speed"], color=OWN_C, lw=1.5)
    axes[3].set_ylabel("Speed (kn)")
    axes[3].set_xlabel("Time (min)")
    idx = res.get("first_manoeuvre_idx")
    if idx is not None:
        for a in axes:
            a.axvline(idx * res["dt"] / 60.0, color=SAFE_C, ls="--", lw=1.0)
    axes[0].set_title(title or f"Control history — {res['cas_name']}")
    plt.tight_layout()
    return fig


def plot_domain_intrusion(results_by_cas, title=""):
    """Normalised Fujii-domain radius; below 1.0 is an intrusion."""
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for name, r in results_by_cas.items():
        t = np.arange(r["n"]) * r["dt"] / 60.0
        ax.plot(t, r["intrusion"], lw=1.6, color=CAS_COLORS.get(name), label=name)
    ax.axhline(1.0, color=TGT_C, ls="--", lw=1.2)
    ax.set_ylim(0, 4)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Normalised domain radius")
    ax.set_title(title or "Ship-domain intrusion (< 1.0 = violation)")
    ax.legend(fontsize=8.5, ncol=2)
    plt.tight_layout()
    return fig



# 6. Manoeuvring validation

def plot_turning_circles(circles: dict, title="Turning-circle validation"):
    """Non-dimensional turning circles against the IMO manoeuvring standard."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    ax = axes[0]
    for name, c in circles.items():
        tr = c["track"] / c["L"]
        ax.plot(tr[:, 0], tr[:, 1], lw=1.6, label=f"{name} (L={c['L']:.0f} m)")
    ax.set_aspect("equal")
    ax.set_xlabel("Transfer (ship lengths)")
    ax.set_ylabel("Advance (ship lengths)")
    ax.set_title("35° rudder turning circles, scaled by length")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    names = list(circles)
    adv = [circles[n]["advance_L"] for n in names]
    td = [circles[n]["tactical_diameter_L"] for n in names]
    xp = np.arange(len(names))
    ax.bar(xp - 0.2, adv, 0.38, label="Advance", color=OWN_C)
    ax.bar(xp + 0.2, td, 0.38, label="Tactical diameter", color=SAFE_C)
    ax.axhline(4.5, color=WARN_C, ls="--", lw=1.2)
    ax.axhline(5.0, color=TGT_C, ls="--", lw=1.2)
    ax.text(len(names) - 0.4, 4.55, "IMO advance ≤ 4.5 L", fontsize=8,
            color=WARN_C, ha="right")
    ax.text(len(names) - 0.4, 5.05, "IMO tactical diameter ≤ 5 L", fontsize=8,
            color=TGT_C, ha="right")
    ax.set_xticks(xp)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8.5)
    ax.set_ylabel("Ship lengths")
    ax.set_title("Compliance with IMO Res. MSC.137(76)")
    ax.legend(fontsize=8.5)
    fig.suptitle(title, fontsize=12.5)
    plt.tight_layout()
    return fig


# 7. Coverage

def plot_coverage_polar(df, title="Encounter-space coverage"):
    """Where in bearing/range space the generated scenarios actually land."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6),
                             subplot_kw={"projection": "polar"})
    d = df.drop_duplicates("scenario_id")
    for ax, (col, cmap, lab) in zip(
            axes, [("dcpa_target_m", "viridis", "Target DCPA (m)"),
                   ("initial_range_m", "plasma", "Initial range (m)")]):
        th = np.radians(d["relative_bearing_deg"].to_numpy())
        r = d["initial_range_m"].to_numpy()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        sc = ax.scatter(th, r, c=d[col], cmap=cmap, s=16, alpha=0.85)
        ax.set_thetagrids(range(0, 360, 30))
        ax.set_title(lab, pad=14, fontsize=10)
        fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.75)
    fig.suptitle(f"{title} — initial relative bearing and range", fontsize=12.5)
    plt.tight_layout()
    return fig


def plot_coverage_grid(df, title="Coverage of the encounter space"):
    """Occupancy of the (relative bearing × relative heading) cells."""
    grid = coverage_grid(df)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto", origin="lower")
    ax.set_xticks(range(len(BEARING_EDGES) - 1))
    ax.set_xticklabels([f"{int(BEARING_EDGES[i])}" for i in
                        range(len(BEARING_EDGES) - 1)], fontsize=7.5, rotation=45)
    ax.set_yticks(range(len(REL_HEADING_EDGES) - 1))
    ax.set_yticklabels([f"{int(REL_HEADING_EDGES[i])}" for i in
                        range(len(REL_HEADING_EDGES) - 1)], fontsize=7.5)
    ax.set_xlabel("Relative bearing of target (deg)")
    ax.set_ylabel("Relative heading of target (deg)")
    filled = int((grid > 0).sum())
    ax.set_title(f"{title} — {filled} of {grid.size} cells occupied "
                 f"({100 * filled / grid.size:.0f}%)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j]:
                ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                        fontsize=6.5,
                        color="white" if grid[i, j] > grid.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Scenarios in cell")
    ax.grid(False)
    plt.tight_layout()
    return fig



# 8. Aggregate results

def plot_safety_rates(df, by="encounter_requested", title=""):
    """Safety rate per strategy with Wilson 95% intervals."""
    tab = rate_table(df, by=(by, "cas_name"), column="safe")
    groups = list(dict.fromkeys(tab[by]))
    cas = list(dict.fromkeys(tab["cas_name"]))
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    w = 0.8 / len(cas)
    for k, c in enumerate(cas):
        sub = tab[tab["cas_name"] == c].set_index(by).reindex(groups)
        xp = np.arange(len(groups)) + (k - (len(cas) - 1) / 2) * w
        # Clamp: at a 100% rate the Wilson upper bound lands exactly on 1.0 and
        # floating point can make (hi - rate) a tiny negative number, which
        # errorbar rejects.
        err = np.clip(np.vstack([sub["rate"] - sub["lo"],
                                 sub["hi"] - sub["rate"]]), 0.0, None)
        err = np.nan_to_num(err, nan=0.0)
        ax.bar(xp, sub["rate"], w * 0.92, label=c, color=CAS_COLORS.get(c))
        ax.errorbar(xp, sub["rate"], yerr=err, fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=2.5)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g.replace("_", "-").title() for g in groups])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Safety rate")
    ax.set_title(title or "Safety rate by encounter type (Wilson 95% intervals)")
    ax.legend(fontsize=8.5, ncol=2)
    plt.tight_layout()
    return fig


def plot_distance_distributions(df, safety_m, title=""):
    """Distribution of achieved minimum separation per strategy."""
    cas = list(dict.fromkeys(df["cas_name"]))
    data = [df.loc[df["cas_name"] == c, "min_dist_m"].to_numpy() for c in cas]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for pc, c in zip(parts["bodies"], cas):
        pc.set_facecolor(CAS_COLORS.get(c, OWN_C))
        pc.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.9)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.random.normal(i, 0.045, len(d)), d, s=5, alpha=0.28,
                   color="black", zorder=3)
    ax.axhline(safety_m, color=WARN_C, ls="--", lw=1.3,
               label=f"Safety limit ({safety_m:.0f} m)")
    ax.set_xticks(range(1, len(cas) + 1))
    ax.set_xticklabels([c.replace(" ", "\n") for c in cas], fontsize=8.5)
    ax.set_ylabel("Minimum separation (m)")
    ax.set_yscale("log")
    ax.set_title(title or "Achieved minimum separation")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    return fig


def plot_compliance_heatmap(matrix, title="COLREGS compliance by rule"):
    """Per-rule compliance rates; blank cells are rules that never applied."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    m = matrix.to_numpy(dtype=float)
    im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([c.replace("_", "\n") for c in matrix.columns], fontsize=8.5)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="black" if 0.25 < m[i, j] < 0.85 else "white")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Compliance rate", shrink=0.85)
    plt.tight_layout()
    return fig


def plot_pareto(summary, title="Safety, compliance and operational cost"):
    """Safety against compliance, sized by the progress lost to manoeuvring."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    idx = list(summary.index)

    ax = axes[0]
    xs = summary["compliance"].to_numpy()
    ys = summary["safety_rate"].to_numpy()
    sizes = 120 + 900 * summary["progress_loss_pct"].to_numpy() / \
        max(summary["progress_loss_pct"].max(), 1e-6)
    for i, name in enumerate(idx):
        ax.scatter(xs[i], ys[i], s=sizes[i], color=CAS_COLORS.get(name, OWN_C),
                   alpha=0.85, edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(name, (xs[i], ys[i]), xytext=(0, -22),
                    textcoords="offset points", ha="center", fontsize=8.5)
    front = pareto_front(np.column_stack([xs, ys]), maximise=(True, True))
    order = np.argsort(xs[front])
    ax.plot(xs[np.array(front)[order]], ys[np.array(front)[order]],
            color="black", ls="--", lw=1.0, alpha=0.6, zorder=2,
            label="Pareto front")
    ax.set_xlabel("COLREGS compliance")
    ax.set_ylabel("Safety rate")
    ax.set_title("Marker area ∝ progress lost to manoeuvring")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    xp = np.arange(len(idx))
    ax.bar(xp, summary["progress_loss_pct"], 0.55,
           color=[CAS_COLORS.get(n, OWN_C) for n in idx])
    ax.set_xticks(xp)
    ax.set_xticklabels([n.replace(" ", "\n") for n in idx], fontsize=8.5)
    ax.set_ylabel("Along-track progress lost (%)")
    ax.set_title("Operational cost of avoidance")
    fig.suptitle(title, fontsize=12.5)
    plt.tight_layout()
    return fig


def plot_failure_gallery(df, store, n=6, cas=None, title=""):
    """The worst outcomes, which are the scenarios worth inspecting."""
    d = df if cas is None else df[df["cas"] == cas]
    worst = d.nsmallest(n, "min_dist_m")
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14.4, 4.7 * rows), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), worst.iterrows()):
        r = store[(row["scenario_id"], row["cas"])]
        plot_encounter(r, ship_scale=16.0, ax=ax,
                       title=f"{row['scenario_id']} · {row['cas_name']}\n"
                             f"CPA {row['min_dist_m']:.0f} m · "
                             f"{row['role'].replace('_', ' ')}")
        ax.get_legend().remove()
    for ax in axes.ravel()[len(worst):]:
        ax.axis("off")
    fig.suptitle(title or f"Worst {n} outcomes", fontsize=13, y=1.005)
    plt.tight_layout()
    return fig


def plot_dcpa_sweep(df, safety_m, title=""):
    """Achieved separation against commanded DCPA — the difficulty sweep."""
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for c in dict.fromkeys(df["cas_name"]):
        sub = df[df["cas_name"] == c]
        ax.scatter(sub["dcpa_target_m"], sub["min_dist_m"], s=11, alpha=0.5,
                   color=CAS_COLORS.get(c), label=c)
        bins = np.linspace(0, sub["dcpa_target_m"].max(), 9)
        idx = np.digitize(sub["dcpa_target_m"], bins) - 1
        med = [np.median(sub["min_dist_m"][idx == k]) if (idx == k).any() else np.nan
               for k in range(len(bins) - 1)]
        ax.plot(0.5 * (bins[:-1] + bins[1:]), med, lw=2.0,
                color=CAS_COLORS.get(c))
    ax.axhline(safety_m, color=WARN_C, ls="--", lw=1.2, label="Safety limit")
    ax.plot([0, df["dcpa_target_m"].max()], [0, df["dcpa_target_m"].max()],
            color="black", ls=":", lw=1.0, label="No avoidance (y = x)")
    ax.set_xlabel("Commanded DCPA of the scenario (m)")
    ax.set_ylabel("Achieved minimum separation (m)")
    ax.set_title(title or "Difficulty sweep: separation recovered by each strategy")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    return fig



# 9. Animation

def animate_encounter(res, stride=4, ship_scale=22.0, title=""):
    """Animated plan view; returns a matplotlib FuncAnimation."""
    import matplotlib.animation as animation
    to, tt = res["traj_own"], res["traj_tgt"]
    frames = range(0, res["n"], stride)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    allx = np.concatenate([to[:, 0], tt[:, 0]])
    ally = np.concatenate([to[:, 1], tt[:, 1]])
    pad = 0.10 * max(allx.ptp(), ally.ptp())
    ax.set_xlim(allx.min() - pad, allx.max() + pad)
    ax.set_ylim(ally.min() - pad, ally.max() + pad)
    ax.set_aspect("equal")
    _nm_axis(ax)
    ax.set_title(title or f"{res['encounter'].replace('_', '-').title()} — "
                          f"{res['cas_name']}")

    own_line, = ax.plot([], [], color=OWN_C, lw=1.6)
    tgt_line, = ax.plot([], [], color=TGT_C, lw=1.6, ls="--")
    ring = Circle((0, 0), res["safety_m"], fill=False, ec=WARN_C, ls="--", lw=1.0)
    ax.add_patch(ring)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID_C))
    hulls = []

    def update(i):
        for h in hulls:
            h.remove()
        hulls.clear()
        own_line.set_data(to[:i + 1, 0], to[:i + 1, 1])
        tgt_line.set_data(tt[:i + 1, 0], tt[:i + 1, 1])
        ring.center = (tt[i, 0], tt[i, 1])
        for k, (traj, hdg, c) in enumerate(((to, res["own_heading"], OWN_C),
                                            (tt, res["tgt_heading"], TGT_C))):
            poly = Polygon(ship_polygon(traj[i], hdg[i],
                                        _spec_for(res, k).length_m,
                                        _spec_for(res, k).beam_m, ship_scale),
                           closed=True, facecolor=c, edgecolor="black",
                           linewidth=0.6, zorder=5)
            ax.add_patch(poly)
            hulls.append(poly)
        d = res["dist"][i]
        txt.set_text(f"t = {i * res['dt'] / 60:5.1f} min\n"
                     f"range = {d:7.0f} m\n"
                     f"DCPA  = {res['dcpa_hist'][i]:7.0f} m\n"
                     f"rudder = {np.degrees(res['rudder'][i]):5.1f}°")
        return own_line, tgt_line, ring, txt

    return animation.FuncAnimation(fig, update, frames=frames,
                                   interval=60, blit=False)


__all__ = [n for n in dir() if n.startswith("plot_") or n in
           ("animate_encounter", "use_style", "ship_polygon", "CAS_COLORS")]
