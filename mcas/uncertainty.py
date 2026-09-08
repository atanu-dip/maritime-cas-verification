"""
Uncertainty quantification for perception-driven collision avoidance.

Phase 2a produced a *point estimate* of the target's state and let the CAS act
on it as though it were true. That is the single most consequential
simplification in the whole pipeline: a CAS that does not know how wrong it
might be cannot be conservative in the right places, and cannot be efficient in
the rest.

Three layers here.

**Ensembles.** Detector disagreement separates two kinds of error. Members share
the same scene, so noise they *agree* on is aleatoric -- irreducible sensing
noise. Noise they *disagree* on is epistemic -- ignorance that more data or a
better model would remove. Only the second shrinks with training, and the
distinction matters for what a designer should do about it.

**Propagation.** The tracker gives a covariance over [x, y, vx, vy]. DCPA is a
nonlinear function of that state, so its uncertainty must be propagated, not
guessed. Two routes are given: a Jacobian (cheap, adequate when the covariance
is small) and Monte Carlo (exact, and the reference the Jacobian is checked
against). They disagree exactly where the linearisation fails, which is where
range uncertainty is large -- the regime monocular ranging lives in.

**Conformal prediction.** Neither route gives a *guarantee*. Split conformal
does: calibrate a nonconformity quantile on held-out encounters and obtain a
lower bound on DCPA that covers the truth with probability at least 1 - alpha,
with no assumption that the errors are Gaussian or that the model is
well-specified. That bound is what the risk-aware CAS in `riskaware` acts on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraSpec, project_vessel
from .geometry import KNOTS_TO_MS, cpa
from .perception import ModelledPerception
from .render import (bearing_from_column, fuse_ranges, range_from_horizon,
                     range_from_size)
from .tracking import Measurement



# DCPA uncertainty

def _dcpa_from_state(own_pos, own_vel, x):
    """DCPA for a target state x = [px, py, vx, vy]."""
    _, d = cpa(own_pos, own_vel, x[:2], x[2:])
    return d


def dcpa_jacobian(own_pos, own_vel, x, eps=(1.0, 1.0, 0.05, 0.05)):
    """Finite-difference gradient of DCPA with respect to the target state."""
    g = np.zeros(4)
    for i in range(4):
        h = eps[i]
        xp, xm = np.array(x, float), np.array(x, float)
        xp[i] += h
        xm[i] -= h
        g[i] = (_dcpa_from_state(own_pos, own_vel, xp)
                - _dcpa_from_state(own_pos, own_vel, xm)) / (2 * h)
    return g


def dcpa_sigma_linear(own_pos, own_vel, x, P):
    """First-order (delta method) standard deviation of DCPA."""
    g = dcpa_jacobian(own_pos, own_vel, x)
    var = float(g @ P @ g)
    return float(np.sqrt(max(var, 0.0)))


def dcpa_distribution(own_pos, own_vel, x, P, n_samples=256, rng=None):
    """
    Monte Carlo distribution of DCPA under the tracker's state uncertainty.
    Exact up to sampling error, and the reference for the linear approximation.
    """
    rng = rng or np.random.default_rng(0)
    P = 0.5 * (np.asarray(P, float) + np.asarray(P, float).T)
    w, V = np.linalg.eigh(P)
    L = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    draws = np.asarray(x, float)[None, :] + rng.standard_normal((n_samples, 4)) @ L.T
    vals = np.array([_dcpa_from_state(own_pos, own_vel, d) for d in draws])
    return vals



# Conformal prediction

@dataclass
class ConformalDCPA:
    """
    Split-conformal lower bound on DCPA.

    Nonconformity is the standardised over-prediction

        s = (dcpa_predicted - dcpa_true) / sigma_predicted

    and the bound is  L = dcpa_predicted - q * sigma_predicted, with q the
    (1-alpha)(1 + 1/n) empirical quantile of s on a held-out calibration set.
    Then P(dcpa_true >= L) >= 1 - alpha in finite samples, distribution-free.

    Standardising by sigma is what makes one scalar quantile work across the
    whole range band: raw residuals are heteroscedastic because sigma_DCPA
    grows with range, so an unstandardised bound would be far too loose close in
    and too tight far out.
    """
    alpha: float = 0.10
    q: float = np.nan
    n_calib: int = 0
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def calibrate(self, dcpa_pred, sigma, dcpa_true):
        pred = np.asarray(dcpa_pred, float)
        sig = np.clip(np.asarray(sigma, float), 1e-6, None)
        true = np.asarray(dcpa_true, float)
        ok = np.isfinite(pred) & np.isfinite(sig) & np.isfinite(true)
        s = (pred[ok] - true[ok]) / sig[ok]
        n = len(s)
        if n < 20:
            raise ValueError(f"need >= 20 calibration points, got {n}")
        level = min((1.0 - self.alpha) * (1.0 + 1.0 / n), 1.0)
        self.q = float(np.quantile(s, level))
        self.n_calib = n
        self.scores = s
        return self

    def lower_bound(self, dcpa_pred, sigma):
        if not np.isfinite(self.q):
            raise RuntimeError("ConformalDCPA is not calibrated")
        return np.asarray(dcpa_pred, float) - self.q * np.clip(
            np.asarray(sigma, float), 1e-6, None)

    def coverage(self, dcpa_pred, sigma, dcpa_true):
        """Empirical coverage of the bound on a test set."""
        lb = self.lower_bound(dcpa_pred, sigma)
        true = np.asarray(dcpa_true, float)
        ok = np.isfinite(lb) & np.isfinite(true)
        return float(np.mean(true[ok] >= lb[ok]))

    def mean_width(self, dcpa_pred, sigma):
        """Mean gap between the point estimate and the bound (conservatism)."""
        return float(np.nanmean(self.q * np.clip(np.asarray(sigma, float),
                                                 1e-6, None)))


# Calibration diagnostics

def z_scores(pred, sigma, true):
    """Standardised residuals; well-calibrated uncertainty gives N(0, 1)."""
    pred, sigma, true = (np.asarray(a, float) for a in (pred, sigma, true))
    ok = np.isfinite(pred) & np.isfinite(true) & (sigma > 0)
    return (pred[ok] - true[ok]) / sigma[ok]


def calibration_curve(pred, sigma, true, levels=None):
    """
    Expected vs observed coverage of central Gaussian intervals.
    A perfectly calibrated model traces the diagonal; below it means
    over-confidence, above means the uncertainty is too generous.
    """
    from math import erf, sqrt
    levels = np.asarray(levels if levels is not None
                        else np.linspace(0.05, 0.95, 19), float)
    z = z_scores(pred, sigma, true)
    obs = []
    for p in levels:
        # two-sided z for coverage p
        k = _norm_ppf(0.5 + p / 2.0)
        obs.append(float(np.mean(np.abs(z) <= k)))
    return levels, np.array(obs)


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_calibration_error(levels, observed):
    """Mean absolute deviation from perfect calibration."""
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(levels))))



# Ensemble perception

class EnsemblePerception(ModelledPerception):
    """
    An ensemble of detectors observing the same frame.

    Each member sees the same scene, so its error decomposes into a shared
    component (aleatoric: the scene really is ambiguous at that range) and an
    independent component (epistemic: the members disagree). The fused
    measurement uses the ensemble mean, and its covariance is inflated by the
    member spread -- which is what makes the downstream DCPA uncertainty
    reflect model ignorance rather than just assumed sensor noise.
    """
    name = "ensemble"

    def __init__(self, cam, error_model, n_members: int = 5,
                 epistemic_frac: float = 0.55, tracker="ekf", seed=0,
                 visibility="good", air_draught_prior_rel_sigma=0.30,
                 mc_dropout: bool = False):
        self.n_members = int(n_members)
        self.epistemic_frac = float(np.clip(epistemic_frac, 0.0, 1.0))
        self.mc_dropout = mc_dropout
        super().__init__(cam, error_model, tracker=tracker, seed=seed,
                         visibility=visibility,
                         air_draught_prior_rel_sigma=air_draught_prior_rel_sigma)

    def reset(self):
        super().reset()
        self.member_spread = []

    def observe(self, own, tgt, t, step):
        rng_true, brg_true, aspect = self.geometry(own, tgt)
        air_draught = 0.15 * tgt.spec.length_m
        proj = project_vessel(self.cam, brg_true, rng_true, tgt.spec.length_m,
                              tgt.spec.beam_m, air_draught, aspect)
        if proj is None:
            return None
        p_det = self.em.detection_probability(proj["height_px"], rng_true,
                                              self.cam, self.visibility)

        # Members vote on whether the target is there at all; a split vote is
        # itself information about how marginal the detection is.
        votes = self.rng.random(self.n_members) < p_det
        if votes.sum() < max(1, self.n_members // 2):
            return None

        s_u, s_v = self.em.sigma_u_px, self.em.sigma_v_px
        s_shared = np.sqrt(max(1.0 - self.epistemic_frac, 0.0))
        s_indep = np.sqrt(self.epistemic_frac)
        du0 = self.rng.normal(0.0, s_u * s_shared)
        dv0 = self.rng.normal(0.0, s_v * s_shared)

        bearings, ranges, sig_r = [], [], []
        prior_shared = air_draught * (1.0 + self.rng.normal(
            0.0, 0.6 * self.prior_rel_sigma))
        for m in range(self.n_members):
            if not votes[m]:
                continue
            u = proj["u_centre"] + du0 + self.rng.normal(0.0, s_u * s_indep)
            v_bot = proj["v_bottom"] + dv0 + self.rng.normal(0.0, s_v * s_indep)
            h_px = max(proj["height_px"] + dv0
                       + self.rng.normal(0.0, 1.4 * s_v * s_indep), 0.5)
            b, _ = bearing_from_column(self.cam, u, s_u)
            r_h, sh = range_from_horizon(self.cam, v_bot, sigma_px=s_v)
            prior = prior_shared * (1.0 + self.rng.normal(
                0.0, 0.4 * self.prior_rel_sigma))
            r_s, ss = range_from_size(self.cam, h_px, max(prior, 1.0),
                                      self.prior_rel_sigma)
            R, sR = fuse_ranges([(r_h, sh), (r_s, ss)])
            if np.isfinite(R):
                bearings.append(b)
                ranges.append(R)
                sig_r.append(sR)
        if not ranges:
            return None

        b_mean = float(np.arctan2(np.mean(np.sin(bearings)),
                                  np.mean(np.cos(bearings))))
        r_mean = float(np.mean(ranges))
        # aleatoric: mean member variance; epistemic: variance between members
        aleatoric_r = float(np.mean(np.square(sig_r)))
        epistemic_r = float(np.var(ranges)) if len(ranges) > 1 else 0.0
        epistemic_b = float(np.var(bearings)) if len(bearings) > 1 else 0.0
        s_R = float(np.sqrt(aleatoric_r + epistemic_r))
        s_B = float(np.sqrt((self.em.sigma_u_px / self.cam.fx) ** 2 + epistemic_b))

        self.member_spread.append({
            "t": t, "range_true": rng_true, "n_votes": int(votes.sum()),
            "range_mean": r_mean, "sigma_aleatoric": float(np.sqrt(aleatoric_r)),
            "sigma_epistemic": float(np.sqrt(epistemic_r)), "sigma_total": s_R})

        true_status = tgt.spec.status
        if self.rng.random() < self.em.class_confusion:
            options = [s for s in ("power_driven", "fishing",
                                   "restricted_manoeuvrability", "sailing")
                       if s != true_status]
            est_status = str(self.rng.choice(options))
        else:
            est_status = true_status
        self.status_estimates.append(est_status)

        return Measurement(t=t, bearing_rad=b_mean, range_m=r_mean,
                           sigma_bearing_rad=max(s_B, 1e-4),
                           sigma_range_m=max(s_R, 1.0),
                           own_pos=own.position.copy(), own_heading=own.heading,
                           detected=True,
                           extra={"height_px": proj["height_px"],
                                  "n_votes": int(votes.sum()),
                                  "sigma_epistemic": float(np.sqrt(epistemic_r)),
                                  "sigma_aleatoric": float(np.sqrt(aleatoric_r)),
                                  "status_est": est_status,
                                  "status_true": true_status})


__all__ = ["dcpa_jacobian", "dcpa_sigma_linear", "dcpa_distribution",
           "ConformalDCPA", "z_scores", "calibration_curve",
           "expected_calibration_error", "EnsemblePerception"]
