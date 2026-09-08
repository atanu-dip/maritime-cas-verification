"""
Vessel specifications and manoeuvring dynamics.

Two dynamics models are provided:

  KinematicModel  -- heading slews directly at a bounded rate. Cheap, but a
                     vessel can reverse her swing instantly, which no ship can.

  NomotoModel     -- first-order Nomoto response, the standard reduced-order
                     model for ship steering:

                         T * dr/dt + r = K * delta
                         dpsi/dt        = r

                     with r the yaw rate, delta the rudder angle, and a PD
                     autopilot driving delta towards the commanded heading.
                     Rudder angle and rudder *rate* are both saturated, and
                     speed is lost in the turn. This reproduces realistic
                     advance and tactical diameter, which the kinematic model
                     does not.

  Nondimensional gains K' = K*L/U and T' = T*U/L are held constant so that a
  spec scales sensibly with length and speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import KNOTS_TO_MS, velocity_vector, wrap_pi

# COLREGS Rule 18 vessel categories, ordered by privilege (higher = more privileged)
VESSEL_STATUS = {
    "power_driven": 0,
    "sailing": 1,
    "fishing": 2,
    "restricted_manoeuvrability": 3,
    "not_under_command": 4,
}


@dataclass(frozen=True)
class VesselSpec:
    """Static particulars of a vessel."""
    name: str = "vessel"
    length_m: float = 150.0
    beam_m: float = 24.0
    service_speed_kn: float = 12.0
    max_rudder_deg: float = 35.0
    rudder_rate_deg_s: float = 2.5
    K_prime: float = 0.80          # nondimensional Nomoto gain
    T_prime: float = 2.00          # nondimensional Nomoto time constant
    speed_loss_coeff: float = 0.30  # fraction of speed lost at full rudder
    status: str = "power_driven"

    def K(self, speed_kn: float) -> float:
        """Dimensional Nomoto gain [1/s] at a given speed."""
        U = max(speed_kn * KNOTS_TO_MS, 1e-3)
        return self.K_prime * U / self.length_m

    def T(self, speed_kn: float) -> float:
        """Dimensional Nomoto time constant [s] at a given speed."""
        U = max(speed_kn * KNOTS_TO_MS, 1e-3)
        return self.T_prime * self.length_m / U


# Library of representative vessels 
VESSEL_LIBRARY = {
    "container_feeder": VesselSpec("container_feeder", 150.0, 24.0, 14.0),
    "handysize_bulker": VesselSpec("handysize_bulker", 180.0, 28.0, 12.0,
                                   K_prime=0.80, T_prime=2.4),
    "vlcc": VesselSpec("vlcc", 330.0, 60.0, 11.0,
                       K_prime=0.86, T_prime=2.8, speed_loss_coeff=0.35),
    "coastal_tanker": VesselSpec("coastal_tanker", 110.0, 18.0, 11.0,
                                 K_prime=0.90, T_prime=1.7),
    "trawler": VesselSpec("trawler", 45.0, 10.0, 8.0,
                          K_prime=1.20, T_prime=1.1, status="fishing"),
}


@dataclass
class ShipState:
    """Instantaneous dynamic state of a vessel."""
    x: float
    y: float
    heading: float          # rad, 0 = North, clockwise +
    speed_kn: float
    yaw_rate: float = 0.0   # rad/s
    rudder: float = 0.0     # rad, positive = starboard
    spec: VesselSpec = field(default_factory=VesselSpec)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return velocity_vector(self.speed_kn, self.heading)

    def copy(self) -> "ShipState":
        return replace(self)



# Dynamics models

class KinematicModel:
    """Heading slews directly toward the command at a bounded rate."""
    name = "kinematic"

    def __init__(self, max_turn_rate_deg_s: float = 0.6):
        self.max_rate = np.radians(max_turn_rate_deg_s)

    def step(self, s: ShipState, heading_cmd: float, speed_cmd_kn: float, dt: float):
        dpsi = np.clip(wrap_pi(heading_cmd - s.heading), -self.max_rate * dt,
                       self.max_rate * dt)
        s.yaw_rate = dpsi / dt
        s.heading = float(wrap_pi(s.heading + dpsi))
        s.speed_kn += np.clip(speed_cmd_kn - s.speed_kn, -0.05 * dt, 0.05 * dt)
        v = s.velocity
        s.x += v[0] * dt
        s.y += v[1] * dt
        return s


class NomotoModel:
    """First-order Nomoto steering response with a PD autopilot."""
    name = "nomoto"

    def __init__(self, kp: float = 2.2, kd: float = 90.0):
        self.kp = kp      # rudder per radian of heading error
        self.kd = kd      # rudder per rad/s of yaw rate (damping)

    def step(self, s: ShipState, heading_cmd: float, speed_cmd_kn: float, dt: float):
        spec = s.spec
        max_rudder = np.radians(spec.max_rudder_deg)
        max_rudder_rate = np.radians(spec.rudder_rate_deg_s)

        # --- PD autopilot -> desired rudder, saturated in angle and rate ----
        err = float(wrap_pi(heading_cmd - s.heading))
        delta_des = np.clip(self.kp * err - self.kd * s.yaw_rate,
                            -max_rudder, max_rudder)
        d_delta = np.clip(delta_des - s.rudder,
                          -max_rudder_rate * dt, max_rudder_rate * dt)
        s.rudder = float(s.rudder + d_delta)

        #Nomoto yaw response 
        K = spec.K(s.speed_kn)
        T = spec.T(s.speed_kn)
        # semi-implicit Euler: unconditionally stable for this first-order ODE
        s.yaw_rate = float((s.yaw_rate + dt * K * s.rudder / T) / (1.0 + dt / T))
        s.heading = float(wrap_pi(s.heading + s.yaw_rate * dt))

        # speed: loss in the turn, first-order approach to commanded 
        turn_frac = abs(s.rudder) / max_rudder
        target_speed = speed_cmd_kn * (1.0 - spec.speed_loss_coeff * turn_frac ** 2)
        tau_speed = 60.0
        s.speed_kn += (target_speed - s.speed_kn) * min(dt / tau_speed, 1.0)
        s.speed_kn = max(s.speed_kn, 0.5)

        v = s.velocity
        s.x += v[0] * dt
        s.y += v[1] * dt
        return s


# 
# Manoeuvring validation: standard turning circle
# 
def turning_circle(spec: VesselSpec, rudder_deg: float = 35.0, dt: float = 1.0,
                   duration: float = 1800.0, model: NomotoModel | None = None):
    """
    Execute a standard turning-circle test: steady approach, then hard-over
    rudder held constant. Returns the track and the IMO manoeuvring measures.

      advance          -- distance advanced along the original course when the
                          heading has changed by 90 deg
      transfer         -- lateral distance at the same instant
      tactical_diameter-- lateral distance when heading has changed 180 deg

    IMO Res. MSC.137(76) requires advance <= 4.5 L and tactical diameter <= 5 L.
    """
    model = model or NomotoModel()
    s = ShipState(0.0, 0.0, 0.0, spec.service_speed_kn, spec=spec)
    s.rudder = 0.0
    delta = np.radians(rudder_deg)
    max_rate = np.radians(spec.rudder_rate_deg_s)

    n = int(duration / dt)
    track = np.zeros((n, 2))
    heads = np.zeros(n)
    speeds = np.zeros(n)
    rudders = np.zeros(n)

    for i in range(n):
        track[i] = (s.x, s.y)
        heads[i] = s.heading
        speeds[i] = s.speed_kn
        rudders[i] = s.rudder

        # ramp rudder to hard-over and hold (bypasses the autopilot)
        s.rudder = float(np.clip(delta, s.rudder - max_rate * dt,
                                 s.rudder + max_rate * dt))
        K, T = spec.K(s.speed_kn), spec.T(s.speed_kn)
        s.yaw_rate = float((s.yaw_rate + dt * K * s.rudder / T) / (1.0 + dt / T))
        s.heading = float(s.heading + s.yaw_rate * dt)   # unwrapped on purpose
        turn_frac = abs(s.rudder) / np.radians(spec.max_rudder_deg)
        tgt = spec.service_speed_kn * (1.0 - spec.speed_loss_coeff * turn_frac ** 2)
        s.speed_kn += (tgt - s.speed_kn) * min(dt / 60.0, 1.0)
        v = velocity_vector(s.speed_kn, s.heading)
        s.x += v[0] * dt
        s.y += v[1] * dt

    turned = heads - heads[0]
    out = {"track": track, "heading": heads, "speed_kn": speeds,
           "rudder": rudders, "dt": dt, "L": spec.length_m, "spec": spec}

    def at_turn(angle_deg):
        idx = np.argmax(np.abs(turned) >= np.radians(angle_deg))
        return None if idx == 0 else idx

    i90 = at_turn(90.0)
    i180 = at_turn(180.0)
    out["advance"] = float(track[i90, 1]) if i90 else np.nan
    out["transfer"] = float(abs(track[i90, 0])) if i90 else np.nan
    out["tactical_diameter"] = float(abs(track[i180, 0])) if i180 else np.nan
    out["advance_L"] = out["advance"] / spec.length_m
    out["transfer_L"] = out["transfer"] / spec.length_m
    out["tactical_diameter_L"] = out["tactical_diameter"] / spec.length_m
    out["steady_turn_rate_deg_s"] = float(np.degrees(np.median(
        np.diff(heads[int(0.5 * n):]) / dt)))
    return out


__all__ = ["VesselSpec", "VESSEL_LIBRARY", "VESSEL_STATUS", "ShipState",
           "KinematicModel", "NomotoModel", "turning_circle"]
