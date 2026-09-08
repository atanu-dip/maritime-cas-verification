"""
Maritime geometry primitives.

Coordinate convention (maritime standard):
    x = East (m), y = North (m)
    heading = radians clockwise from North  (0 = North, pi/2 = East)
    velocity = speed * (sin(heading), cos(heading))

All angles internally in radians; helpers provided for degrees.
"""
from __future__ import annotations

import numpy as np

KNOTS_TO_MS = 0.5144444
NM_TO_M = 1852.0


# Angle helpers

def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def wrap_2pi(a):
    """Wrap angle(s) to [0, 2pi)."""
    return np.asarray(a) % (2 * np.pi)


def heading_vector(heading):
    """Unit vector for a maritime heading (0 = North, clockwise +)."""
    return np.array([np.sin(heading), np.cos(heading)])


def velocity_vector(speed_kn, heading):
    """Velocity in m/s from speed in knots and heading in radians."""
    return speed_kn * KNOTS_TO_MS * heading_vector(heading)


# Relative geometry

def true_bearing(from_pos, to_pos):
    """True bearing (rad from North, clockwise) of `to_pos` seen from `from_pos`."""
    d = np.asarray(to_pos) - np.asarray(from_pos)
    return np.arctan2(d[0], d[1])


def relative_bearing(observer_pos, observer_heading, target_pos):
    """
    Bearing of target relative to observer's bow, in (-pi, pi].
    Positive = starboard side, negative = port side.
    """
    return wrap_pi(true_bearing(observer_pos, target_pos) - observer_heading)


def cpa(own_pos, own_vel, tgt_pos, tgt_vel):
    """
    Closest Point of Approach under constant velocity.

    Returns (tcpa_seconds, dcpa_metres). TCPA is clipped at >= 0: a negative
    analytic TCPA means the vessels are already opening, in which case the
    closest approach is *now*.
    """
    r0 = np.asarray(tgt_pos, dtype=float) - np.asarray(own_pos, dtype=float)
    w = np.asarray(tgt_vel, dtype=float) - np.asarray(own_vel, dtype=float)
    w2 = float(w @ w)
    if w2 < 1e-9:                       # parallel, same speed: range is constant
        return 0.0, float(np.linalg.norm(r0))
    t = -float(r0 @ w) / w2
    t = max(0.0, t)
    return t, float(np.linalg.norm(r0 + w * t))


def range_rate(own_pos, own_vel, tgt_pos, tgt_vel):
    """d|range|/dt in m/s. Negative = closing."""
    r0 = np.asarray(tgt_pos, float) - np.asarray(own_pos, float)
    w = np.asarray(tgt_vel, float) - np.asarray(own_vel, float)
    rng = np.linalg.norm(r0)
    if rng < 1e-6:
        return 0.0
    return float(r0 @ w) / rng


# COLREGS sector geometry

# Rule 13: an overtaking vessel approaches from more than 22.5 deg abaft the
# beam of the vessel being overtaken, i.e. bearing outside +/-112.5 deg.
SECTOR_HEAD_ON = np.radians(6.0)        # Rule 14: "nearly reciprocal" tolerance
SECTOR_BOW = np.radians(112.5)          # forward sector limit (2 points abaft beam)
SECTOR_OVERTAKING = np.radians(22.5)    # Rule 13 heading-alignment tolerance


def sector_name(rel_bearing):
    """Name the sector a relative bearing falls in (observer-centric)."""
    b = float(wrap_pi(rel_bearing))
    ab = abs(b)
    if ab <= np.radians(11.25):
        return "ahead"
    if ab >= np.radians(168.75):
        return "astern"
    side = "starboard" if b > 0 else "port"
    if ab <= SECTOR_BOW:
        return f"{side}_bow" if ab <= np.radians(67.5) else f"{side}_beam"
    return f"{side}_quarter"



# Ship domain (Fujii)

class FujiiDomain:
    """
    Elliptical ship domain after Fujii & Tanaka.

    The classical open-water domain is an ellipse of total length 8L and total
    breadth 3.2L centred on the vessel and aligned with her heading, i.e.
    semi-axes a = 4L (fore-and-aft) and b = 1.6L (athwartships).

    `intrusion(...)` returns the normalised elliptical radius of the intruder:
        < 1.0  -> inside the domain (violation)
        = 1.0  -> exactly on the boundary
        > 1.0  -> clear
    """

    def __init__(self, length_m: float, a_factor: float = 4.0, b_factor: float = 1.6):
        self.L = float(length_m)
        self.a = a_factor * self.L      # semi-axis along heading
        self.b = b_factor * self.L      # semi-axis abeam

    def intrusion(self, own_pos, own_heading, tgt_pos):
        """Normalised elliptical radius of `tgt_pos` in own-ship's domain frame."""
        d = np.asarray(tgt_pos, float) - np.asarray(own_pos, float)
        c, s = np.cos(own_heading), np.sin(own_heading)
        # rotate into (along-heading, abeam) frame
        along = d[0] * s + d[1] * c
        abeam = d[0] * c - d[1] * s
        return float(np.hypot(along / self.a, abeam / self.b))

    def violated(self, own_pos, own_heading, tgt_pos):
        return self.intrusion(own_pos, own_heading, tgt_pos) < 1.0

    def boundary(self, own_pos, own_heading, n=200):
        """Polygon of the domain boundary in world coordinates (for plotting)."""
        th = np.linspace(0, 2 * np.pi, n)
        along = self.a * np.cos(th)
        abeam = self.b * np.sin(th)
        c, s = np.cos(own_heading), np.sin(own_heading)
        x = own_pos[0] + along * s + abeam * c
        y = own_pos[1] + along * c - abeam * s
        return np.column_stack([x, y])


__all__ = [
    "KNOTS_TO_MS", "NM_TO_M", "wrap_pi", "wrap_2pi", "heading_vector",
    "velocity_vector", "true_bearing", "relative_bearing", "cpa", "range_rate",
    "sector_name", "FujiiDomain",
    "SECTOR_HEAD_ON", "SECTOR_BOW", "SECTOR_OVERTAKING",
]
