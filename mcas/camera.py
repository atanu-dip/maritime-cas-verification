"""
Own-ship camera model.

A pinhole camera mounted on own-ship's bridge, looking along her heading. Two
pieces of geometry matter for everything downstream.

**The horizon is the datum.** For a camera at height h above the waterline on a
*flat* sea the horizon projects exactly to the principal point. On a spherical
Earth it sits slightly below it, by the dip angle

    delta = sqrt(2h / R_e)

and the geometric horizon lies at range d_h = sqrt(2 h R_e). For a bridge 20 m
above the water that is about 16 km (8.6 nm). This is not a detail: it sets a
hard ceiling on monocular ranging, because beyond d_h a vessel's waterline is
*below the visible horizon* and the range cue the whole method depends on
simply is not in the image.

**Hull-down targets.** Past the geometric horizon a vessel is progressively
occluded from the waterline up. At range R the hidden height is

    h_hidden = (R - d_h)^2 / (2 R_e)

so a vessel whose air draught is less than h_hidden is invisible regardless of
sensor quality, and one only partly hidden presents a truncated silhouette
whose apparent height no longer corresponds to her real height. Both effects
are modelled here, because both put a physical floor under how well any
perception-driven collision avoidance system can possibly do.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class CameraSpec:
    """Intrinsics and mounting of own-ship's forward-looking camera."""
    width: int = 640
    height: int = 384
    hfov_deg: float = 60.0
    height_m: float = 20.0          # above waterline
    pitch_deg: float = 0.0          # positive = nose up
    yaw_offset_deg: float = 0.0     # relative to own-ship heading

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx                       # square pixels

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan((self.height / 2.0) / self.fx)))

    #  horizon 
    @property
    def horizon_dip_rad(self) -> float:
        """Angular depression of the true horizon below the level plane."""
        return float(np.sqrt(2.0 * self.height_m / EARTH_RADIUS_M))

    @property
    def horizon_range_m(self) -> float:
        """Distance to the geometric horizon."""
        return float(np.sqrt(2.0 * self.height_m * EARTH_RADIUS_M))

    @property
    def horizon_row(self) -> float:
        """Image row of the horizon (pixels from top)."""
        return self.cy + self.fx * np.tan(self.horizon_dip_rad
                                          + np.radians(self.pitch_deg))

    #  projection 
    def bearing_to_column(self, rel_bearing_rad):
        """Relative bearing (+ starboard) -> image column."""
        b = np.asarray(rel_bearing_rad, float) - np.radians(self.yaw_offset_deg)
        return self.cx + self.fx * np.tan(b)

    def column_to_bearing(self, u):
        """Image column -> relative bearing (+ starboard)."""
        return np.arctan2(np.asarray(u, float) - self.cx, self.fx) \
            + np.radians(self.yaw_offset_deg)

    def waterline_depression(self, range_m):
        """
        Depression angle of a sea-surface point at range R, below the level
        plane, on a spherical Earth:

            theta(R) = h/R + R/(2 R_e)

        The first term is the flat-Earth foreshortening, the second the drop of
        the surface away from the tangent plane. theta has a minimum at
        R = sqrt(2 h R_e), where it equals the horizon dip exactly -- so the
        waterline curve meets the horizon at the horizon range and can never
        appear above it. Using only the flat-Earth term makes distant waterlines
        project *above* the horizon, which is geometrically impossible.
        """
        R = np.maximum(np.asarray(range_m, float), 1.0)
        return self.height_m / R + R / (2.0 * EARTH_RADIUS_M)

    def waterline_row(self, range_m):
        """Image row of a target's waterline contact point at a given range."""
        theta = self.waterline_depression(range_m)
        theta = np.maximum(theta, self.horizon_dip_rad)   # never above horizon
        return self.cy + self.fx * np.tan(theta + np.radians(self.pitch_deg))

    def row_to_range(self, v):
        """
        Invert `waterline_row`. Solving R^2/(2 R_e) - theta R + h = 0 gives

            R = R_e * (theta -/+ sqrt(theta^2 - 2h/R_e))

        The near branch (minus sign) is the visible one; the far branch is
        occluded by the horizon. Rows at or above the horizon return inf,
        because no visible range projects there.
        """
        v = np.asarray(v, float)
        theta = np.arctan2(v - self.cy, self.fx) - np.radians(self.pitch_deg)
        disc = theta ** 2 - 2.0 * self.height_m / EARTH_RADIUS_M
        with np.errstate(invalid="ignore"):
            R = EARTH_RADIUS_M * (theta - np.sqrt(np.maximum(disc, 0.0)))
        return np.where(disc <= 0.0, np.inf, R)

    def range_sigma_from_pixel(self, range_m, sigma_px=1.0):
        """
        Range uncertainty implied by a pixel-level waterline localisation error,
        by finite difference on the exact projection. Diverges at the horizon,
        which is the honest answer: there the cue carries no range information.
        """
        R = np.asarray(range_m, float)
        v = self.waterline_row(R)
        lo = self.row_to_range(v + sigma_px)
        hi = self.row_to_range(v - sigma_px)
        with np.errstate(invalid="ignore"):
            sig = 0.5 * np.abs(np.where(np.isfinite(hi), hi, np.inf) - lo)
        return sig

    def hidden_height_m(self, range_m):
        """Height of a target concealed by the Earth's curvature at range R."""
        R = np.asarray(range_m, float)
        excess = np.maximum(R - self.horizon_range_m, 0.0)
        return excess ** 2 / (2.0 * EARTH_RADIUS_M)

    def pixels_per_metre_vertical(self, range_m):
        """Vertical image scale at a given range (px per metre of height)."""
        R = np.maximum(np.asarray(range_m, float), 1.0)
        return self.fx / R


def apparent_beam_m(length_m: float, beam_m: float, aspect_rad: float) -> float:
    """
    Silhouette width of a vessel seen at a given aspect angle.

    `aspect_rad` is the angle between the observer's line of sight and the
    target's heading: 0 or pi is bow/stern on (narrow), pi/2 is beam on (widest).
    """
    return float(abs(length_m * np.sin(aspect_rad))
                 + abs(beam_m * np.cos(aspect_rad)))


def project_vessel(cam: CameraSpec, rel_bearing_rad: float, range_m: float,
                   length_m: float, beam_m: float, air_draught_m: float,
                   aspect_rad: float):
    """
    Project a vessel into the image.

    Returns a dict with the bounding box in pixels and the visibility flags, or
    None if the vessel projects entirely outside the image or is fully hidden
    below the horizon.
    """
    if range_m <= 1.0:
        return None

    hidden = float(cam.hidden_height_m(range_m))
    if hidden >= air_draught_m:
        return None                                  # fully hull-down

    scale = float(cam.pixels_per_metre_vertical(range_m))
    v_water = float(cam.waterline_row(range_m))
    v_horizon = float(cam.horizon_row)

    # The visible bottom is the waterline, or the horizon if the hull is
    # partly concealed by the curvature of the Earth.
    v_bottom = v_water - hidden * scale
    v_bottom = min(v_bottom, v_water)
    v_top = v_water - air_draught_m * scale

    u_centre = float(cam.bearing_to_column(rel_bearing_rad))
    width_m = apparent_beam_m(length_m, beam_m, aspect_rad)
    half_w = 0.5 * width_m * scale

    box = np.array([u_centre - half_w, v_top, u_centre + half_w, v_bottom])
    if box[2] < 0 or box[0] > cam.width or box[3] < 0 or box[1] > cam.height:
        return None

    return {
        "box": box,
        "u_centre": u_centre,
        "v_waterline": v_water,
        "v_bottom": v_bottom,
        "v_top": v_top,
        "v_horizon": v_horizon,
        "height_px": float(v_bottom - v_top),
        "width_px": float(2 * half_w),
        "hidden_m": hidden,
        "hull_down": bool(hidden > 0.0),
        "apparent_width_m": width_m,
        "range_m": float(range_m),
        "rel_bearing_rad": float(rel_bearing_rad),
    }


def visibility_range_m(visibility: str) -> float:
    """Meteorological visibility, in metres, by qualitative band."""
    return {"good": 10 * 1852.0, "moderate": 3 * 1852.0,
            "restricted": 1 * 1852.0}.get(visibility, 10 * 1852.0)


__all__ = ["CameraSpec", "EARTH_RADIUS_M", "apparent_beam_m", "project_vessel",
           "visibility_range_m"]
