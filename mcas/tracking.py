"""
Target state estimation from camera measurements.

A single frame gives *where* a target is. It cannot give *course and speed* --
and those are exactly what a collision avoidance system needs, because CPA is a
function of relative velocity. Velocity only emerges from filtering detections
over time, so the filter's convergence time is not an implementation detail: it
is a hard delay between first detection and the first moment the CAS has a
usable track.

That delay is measured directly in this project, because COLREGS Rule 8(a)
requires avoiding action to be taken "in ample time". A filter that needs six
minutes to produce a trustworthy velocity has spent six minutes of the ample
time the Rules assume the mariner had.

Two filters are provided over the same constant-velocity model:

  EKF -- linearises the polar measurement about the current estimate. Cheap,
         and the standard choice.
  UKF -- propagates sigma points through the exact nonlinearity. Better when
         range uncertainty is large relative to range, which is precisely the
         regime monocular ranging lives in.

The measurement is polar (bearing, range) and strongly anisotropic: bearing is
accurate to a few milliradians while range may be 50% uncertain. Treating that
as an isotropic Cartesian error -- a common shortcut -- discards the structure
that makes the filter work at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import KNOTS_TO_MS, wrap_pi
from .vessels import VESSEL_LIBRARY, ShipState, VesselSpec


@dataclass
class Measurement:
    """One camera observation of a target."""
    t: float
    bearing_rad: float                 # relative to own-ship's bow
    range_m: float
    sigma_bearing_rad: float
    sigma_range_m: float
    own_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    own_heading: float = 0.0
    detected: bool = True
    extra: dict = field(default_factory=dict)

    def to_world(self):
        """Absolute position implied by this measurement."""
        b = self.bearing_rad + self.own_heading
        return self.own_pos + self.range_m * np.array([np.sin(b), np.cos(b)])


def _polar_cov_to_cartesian(bearing_world, rng, s_b, s_r):
    """Rotate the anisotropic polar measurement covariance into world axes."""
    # x = r sin(b), y = r cos(b)
    J = np.array([[np.sin(bearing_world), rng * np.cos(bearing_world)],
                  [np.cos(bearing_world), -rng * np.sin(bearing_world)]])
    S = np.diag([s_r ** 2, s_b ** 2])
    return J @ S @ J.T


class ConstantVelocityEKF:
    """
    EKF over state [x, y, vx, vy] in world coordinates, with a polar
    measurement model.
    """
    name = "EKF"

    def __init__(self, q_accel: float = 0.02, init_speed_sigma: float = 6.0):
        self.q = q_accel                    # process noise as accel PSD (m/s^2)
        self.init_speed_sigma = init_speed_sigma
        self.x = None
        self.P = None
        self.t_last = None
        self.n_updates = 0

    #  lifecycle 
    def initialise(self, z: Measurement):
        p = z.to_world()
        self.x = np.array([p[0], p[1], 0.0, 0.0])
        b_world = z.bearing_rad + z.own_heading
        Pp = _polar_cov_to_cartesian(b_world, z.range_m,
                                     z.sigma_bearing_rad, z.sigma_range_m)
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = Pp
        self.P[2, 2] = self.P[3, 3] = self.init_speed_sigma ** 2
        self.t_last = z.t
        self.n_updates = 1

    def predict(self, t: float):
        if self.x is None:
            return
        dt = max(t - self.t_last, 0.0)
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # piecewise-white acceleration process noise
        q = self.q ** 2
        Q = q * np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_last = t

    def _h(self, x, own_pos, own_heading):
        d = x[:2] - own_pos
        rng = float(np.hypot(d[0], d[1]))
        brg = float(wrap_pi(np.arctan2(d[0], d[1]) - own_heading))
        return np.array([brg, rng]), d, rng

    def _H(self, d, rng):
        x, y = d
        r2 = max(rng ** 2, 1e-6)
        return np.array([
            [y / r2, -x / r2, 0.0, 0.0],          # d(bearing)/d(pos)
            [x / max(rng, 1e-6), y / max(rng, 1e-6), 0.0, 0.0],
        ])

    def update(self, z: Measurement):
        if self.x is None:
            self.initialise(z)
            return
        self.predict(z.t)
        h, d, rng = self._h(self.x, z.own_pos, z.own_heading)
        H = self._H(d, rng)
        R = np.diag([z.sigma_bearing_rad ** 2, z.sigma_range_m ** 2])
        y = np.array([float(wrap_pi(z.bearing_rad - h[0])), z.range_m - h[1]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
        self.n_updates += 1

    # output 
    @property
    def ready(self) -> bool:
        """Enough updates and a tight enough velocity to be worth acting on."""
        if self.x is None or self.n_updates < 4:
            return False
        return float(np.sqrt(self.P[2, 2] + self.P[3, 3])) < 3.0   # m/s

    def state(self, spec: VesselSpec | None = None) -> ShipState | None:
        """Current estimate as a ShipState the CAS can consume."""
        if self.x is None:
            return None
        vx, vy = self.x[2], self.x[3]
        speed = float(np.hypot(vx, vy)) / KNOTS_TO_MS
        heading = float(np.arctan2(vx, vy)) if (abs(vx) + abs(vy)) > 1e-6 else 0.0
        return ShipState(float(self.x[0]), float(self.x[1]), heading, speed,
                         spec=spec or VESSEL_LIBRARY["container_feeder"])

    def velocity_sigma(self) -> float:
        if self.P is None:
            return np.inf
        return float(np.sqrt(self.P[2, 2] + self.P[3, 3]))

    def position_sigma(self) -> float:
        if self.P is None:
            return np.inf
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))


class ConstantVelocityUKF(ConstantVelocityEKF):
    """
    Unscented variant. Sigma points are propagated through the exact polar
    measurement, which matters when sigma_range / range is large -- the regime
    monocular ranging operates in, where the EKF's linearisation is poorest.
    """
    name = "UKF"

    def __init__(self, q_accel=0.02, init_speed_sigma=6.0,
                 alpha=0.5, beta=2.0, kappa=0.0):
        super().__init__(q_accel, init_speed_sigma)
        self.n = 4
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.lam = alpha ** 2 * (self.n + kappa) - self.n
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha ** 2 + beta)

    @staticmethod
    def _safe_cholesky(A):
        """
        Cholesky with escalating jitter, falling back to an eigenvalue floor.
        Repeated Joseph-form updates can leave the covariance marginally
        indefinite through rounding, which would otherwise abort the filter
        mid-track.
        """
        A = 0.5 * (A + A.T)
        scale = float(np.trace(A)) / max(A.shape[0], 1)
        for k in range(8):
            try:
                return np.linalg.cholesky(A + (0.0 if k == 0 else
                                               10.0 ** (k - 8) * scale)
                                          * np.eye(A.shape[0]))
            except np.linalg.LinAlgError:
                continue
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 1e-9 * max(scale, 1.0), None)
        return V @ np.diag(np.sqrt(w))

    def _sigma_points(self):
        S = self._safe_cholesky((self.n + self.lam) * self.P)
        pts = [self.x]
        for i in range(self.n):
            pts.append(self.x + S[:, i])
            pts.append(self.x - S[:, i])
        return np.array(pts)

    def update(self, z: Measurement):
        if self.x is None:
            self.initialise(z)
            return
        self.predict(z.t)
        X = self._sigma_points()
        Z = np.array([self._h(p, z.own_pos, z.own_heading)[0] for p in X])
        z_mean = np.array([
            float(np.arctan2((self.Wm * np.sin(Z[:, 0])).sum(),
                             (self.Wm * np.cos(Z[:, 0])).sum())),
            float((self.Wm * Z[:, 1]).sum())])
        dZ = Z - z_mean
        dZ[:, 0] = wrap_pi(dZ[:, 0])
        dX = X - self.x
        R = np.diag([z.sigma_bearing_rad ** 2, z.sigma_range_m ** 2])
        Pzz = (self.Wc[:, None, None] *
               np.einsum("ni,nj->nij", dZ, dZ)).sum(axis=0) + R
        Pxz = (self.Wc[:, None, None] *
               np.einsum("ni,nj->nij", dX, dZ)).sum(axis=0)
        K = Pxz @ np.linalg.inv(Pzz)
        y = np.array([float(wrap_pi(z.bearing_rad - z_mean[0])),
                      z.range_m - z_mean[1]])
        self.x = self.x + K @ y
        self.P = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        # keep the covariance strictly positive definite
        w, V = np.linalg.eigh(self.P)
        if np.any(w <= 0):
            self.P = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
        self.n_updates += 1


TRACKER_REGISTRY = {"ekf": ConstantVelocityEKF, "ukf": ConstantVelocityUKF}

__all__ = ["Measurement", "ConstantVelocityEKF", "ConstantVelocityUKF",
           "TRACKER_REGISTRY"]
