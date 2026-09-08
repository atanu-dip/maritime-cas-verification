"""
Synthetic sea-scene rendering, and monocular range estimation.

Rendering exists for one reason: the closed loop. Own-ship's avoidance
manoeuvre changes what her camera sees, and no recorded video can show the
view that a manoeuvre which never happened would have produced. Real imagery
is therefore used to *characterise* the detector (see `errormodel`), and
rendering is used to close the loop.

Two independent monocular range cues are implemented, because they fail in
different ways and that difference is itself a result:

  Horizon-based   R = h_cam * fx / (v_waterline - v_horizon)
                  Needs only the camera height, which is known exactly. Precise
                  near, and degrades quadratically with range as the waterline
                  approaches the horizon. Fails completely beyond the geometric
                  horizon, where the waterline is not in the image at all.

  Size-based      R = fx * H_ship / height_px
                  Needs a prior on the vessel's dimensions, so it carries a
                  roughly constant *relative* error set by how wrong that prior
                  is. Survives past the horizon, but is biased for hull-down
                  targets whose silhouette is truncated.

Both have sigma_R proportional to R^2, which is the fundamental reason a single
camera is a poor long-range ranging instrument and why the tracker in
`tracking` matters so much.
"""
from __future__ import annotations

import numpy as np

from .camera import CameraSpec, project_vessel, visibility_range_m


# Rendering

SKY_TOP = np.array([104, 152, 198], float)
SKY_HORIZON = np.array([196, 214, 232], float)
SEA_HORIZON = np.array([104, 126, 142], float)
SEA_NEAR = np.array([46, 66, 84], float)
HULL = np.array([48, 52, 58], float)
SUPERSTRUCTURE = np.array([206, 208, 210], float)


def render_scene(cam: CameraSpec, targets, visibility: str = "good",
                 sea_state: float = 0.35, seed: int | None = None,
                 sun_glare: bool = False):
    """
    Render own-ship's forward view.

    `targets` is a list of dicts from `camera.project_vessel`, each optionally
    carrying a "superstructure_frac" key. Returns a uint8 HxWx3 image.
    """
    rng = np.random.default_rng(seed)
    H, W = cam.height, cam.width
    v_hor = cam.horizon_row

    rows = np.arange(H, dtype=float)[:, None]
    img = np.zeros((H, W, 3), float)

    # sky: vertical gradient toward the horizon 
    sky_t = np.clip(rows / max(v_hor, 1.0), 0.0, 1.0)
    sky = SKY_TOP[None, None, :] * (1 - sky_t[..., None]) \
        + SKY_HORIZON[None, None, :] * sky_t[..., None]

    # sea: gradient from horizon colour down to near-field 
    sea_t = np.clip((rows - v_hor) / max(H - v_hor, 1.0), 0.0, 1.0)
    sea = SEA_HORIZON[None, None, :] * (1 - sea_t[..., None]) \
        + SEA_NEAR[None, None, :] * sea_t[..., None]

    mask_sky = (rows < v_hor)
    img = np.where(mask_sky[..., None], sky, sea)
    img = np.repeat(img, W, axis=1) if img.shape[1] == 1 else img

    # wave texture: coarser and higher-contrast in the near field 
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    below = (yy > v_hor)
    depth = np.clip((yy - v_hor) / max(H - v_hor, 1.0), 0.0, 1.0)
    waves = (np.sin(xx / (6.0 + 30.0 * depth) + 4.0 * depth)
             * np.sin(yy / (2.0 + 12.0 * depth)))
    waves = waves * (6.0 + 22.0 * depth) * sea_state
    waves = waves + rng.normal(0.0, 2.5 * sea_state, size=(H, W))
    img += (waves * below)[..., None]

    # targets 
    for t in targets:
        if t is None:
            continue
        _draw_vessel(img, cam, t, rng)

    # atmospheric haze: contrast falls with range 
    vis_m = visibility_range_m(visibility)
    img = _apply_haze(img, cam, vis_m)

    if sun_glare:
        img = _apply_glare(img, cam, rng)

    img += rng.normal(0.0, 2.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_vessel(img, cam: CameraSpec, t, rng):
    """Hull block plus a lighter superstructure, alpha-blended by range haze."""
    H, W = img.shape[:2]
    x0, y_top, x1, y_bot = t["box"]
    u0, u1 = int(np.floor(min(x0, x1))), int(np.ceil(max(x0, x1)))
    v0, v1 = int(np.floor(y_top)), int(np.ceil(y_bot))
    u0, u1 = max(u0, 0), min(u1, W)
    v0, v1 = max(v0, 0), min(v1, H)
    if u1 <= u0 or v1 <= v0:
        return

    total_h = max(v1 - v0, 1)
    sup_frac = float(t.get("superstructure_frac", 0.45))
    v_split = v0 + int((1.0 - sup_frac) * total_h) if total_h > 2 else v1

    img[v_split:v1, u0:u1] = HULL
    if v_split > v0:
        # superstructure sits inboard of the hull
        inset = max(int(0.22 * (u1 - u0)), 0)
        img[v0:v_split, u0 + inset:u1 - inset] = SUPERSTRUCTURE

    # a thin wake smear just below the waterline sells the scale
    if v1 < H - 1 and (u1 - u0) > 3:
        wake_h = max(int(0.25 * total_h), 1)
        w0, w1 = v1, min(v1 + wake_h, H)
        img[w0:w1, u0:u1] += 26.0


def _apply_haze(img, cam: CameraSpec, vis_m: float):
    """
    Blend toward the horizon colour with an extinction that grows with range.
    Range is inferred per-row from the waterline geometry, so haze correctly
    thickens toward the horizon rather than uniformly down the image.
    """
    H, W = img.shape[:2]
    rows = np.arange(H, dtype=float)
    R = cam.row_to_range(rows)
    R = np.where(np.isfinite(R), R, cam.horizon_range_m)
    R = np.clip(R, 0.0, 4.0 * cam.horizon_range_m)
    # Koschmieder: contrast falls as exp(-3.912 R / V)
    trans = np.exp(-3.912 * R / max(vis_m, 1.0))
    trans = np.clip(trans, 0.0, 1.0)[:, None, None]
    haze_colour = SKY_HORIZON[None, None, :]
    return img * trans + haze_colour * (1.0 - trans)


def _apply_glare(img, cam: CameraSpec, rng):
    H, W = img.shape[:2]
    cx = rng.uniform(0.25 * W, 0.75 * W)
    cy = cam.horizon_row + rng.uniform(-8, 30)
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - cx) ** 2 + ((yy - cy) * 2.0) ** 2
    glare = 150.0 * np.exp(-d2 / (2.0 * (0.16 * W) ** 2))
    return img + glare[..., None]



# Monocular range estimation

def range_from_horizon(cam: CameraSpec, v_waterline: float,
                       v_horizon: float | None = None, sigma_px: float = 1.0):
    """
    Range from the vertical offset between waterline and horizon.

    R = h_cam * fx / (v_waterline - v_horizon)

    Returns (range_m, sigma_m) with sigma from a one-pixel localisation error;
    note sigma grows as R^2, which is the dominant limitation of the method.
    """
    v_hor = cam.horizon_row if v_horizon is None else float(v_horizon)
    # Work in the camera's own geometry so the estimate is consistent with the
    # spherical projection; a detected horizon that differs from the nominal one
    # is absorbed as an offset on the waterline row.
    v_eff = float(v_waterline) - (v_hor - cam.horizon_row)
    R = float(cam.row_to_range(v_eff))
    if not np.isfinite(R) or R <= 0:
        return np.inf, np.inf
    sigma = float(cam.range_sigma_from_pixel(R, sigma_px))
    if not np.isfinite(sigma):
        return np.inf, np.inf
    return R, sigma


def range_from_size(cam: CameraSpec, height_px: float, air_draught_prior_m: float,
                    prior_rel_sigma: float = 0.30):
    """
    Range from apparent height and a prior on the vessel's air draught.

    R = fx * H_ship / height_px

    Two error sources: pixel localisation (again quadratic in R) and the prior
    itself, which contributes a fixed *relative* error.
    """
    if height_px <= 1e-6:
        return np.inf, np.inf
    R = cam.fx * air_draught_prior_m / float(height_px)
    sigma_px = R ** 2 / (cam.fx * max(air_draught_prior_m, 1e-6))
    sigma_prior = R * prior_rel_sigma
    return float(R), float(np.hypot(sigma_px, sigma_prior))


def fuse_ranges(estimates):
    """
    Inverse-variance fusion of independent range estimates.
    `estimates` is a sequence of (range_m, sigma_m); non-finite entries drop out.
    """
    vals = [(r, s) for r, s in estimates
            if np.isfinite(r) and np.isfinite(s) and s > 0]
    if not vals:
        return np.inf, np.inf
    w = np.array([1.0 / s ** 2 for _, s in vals])
    r = np.array([r for r, _ in vals])
    R = float((w * r).sum() / w.sum())
    sigma = float(np.sqrt(1.0 / w.sum()))
    return R, sigma


def bearing_from_column(cam: CameraSpec, u: float, sigma_px: float = 2.0):
    """Relative bearing and its uncertainty from an image column."""
    b = float(cam.column_to_bearing(u))
    # d(bearing)/du = fx / (fx^2 + (u - cx)^2)
    du = float(u) - cam.cx
    dbdu = cam.fx / (cam.fx ** 2 + du ** 2)
    return b, float(abs(dbdu) * sigma_px)


__all__ = ["render_scene", "range_from_horizon", "range_from_size",
           "fuse_ranges", "bearing_from_column"]
