"""
The perception pipeline, and the detection error model it runs on.

Two ways of producing measurements are provided, and both feed the *same*
tracker so the comparison is clean:

  RenderedPerception   renders own-ship's camera view, runs a real detector,
                       estimates bearing and range from the image. Faithful,
                       and far too slow for a thousand-encounter campaign.

  ModelledPerception   samples from an error model fitted to the detector's
                       measured performance on real maritime imagery, skipping
                       rendering entirely.

This split is deliberate and is stated plainly in the results: **detection
performance is measured on real data; the closed-loop campaign runs on an error
model fitted to those measurements.** Rendering every frame of every encounter
would be roughly a million network evaluations, and no recorded video can show
what the camera would have seen after a manoeuvre that never happened. Fitting
an error model from real performance and sampling it is the standard way out,
and it keeps the expensive path available for case studies.

The error model captures what actually degrades with range:

  * detection probability, which falls as the target's apparent height shrinks
    toward the noise floor, and collapses past the geometric horizon;
  * bearing error, near-constant in angle;
  * range error, growing as R^2 because that is what the projection geometry
    dictates;
  * false alarms, from wave crests and glare;
  * classification confusion, which is what makes COLREGS role inversion
    possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraSpec, project_vessel, visibility_range_m
from .geometry import relative_bearing, wrap_pi
from .render import (bearing_from_column, fuse_ranges, range_from_horizon,
                     range_from_size, render_scene)
from .tracking import TRACKER_REGISTRY, Measurement
from .vessels import VESSEL_LIBRARY, ShipState

# COLREGS Rule 18 status implied by a recognised vessel category.
CLASS_TO_STATUS = {
    "Ferry": "power_driven", "Boat": "power_driven", "Vessel/ship": "power_driven",
    "Speed boat": "power_driven", "Other": "power_driven", "Buoy": "power_driven",
    "Kayak": "power_driven", "Swimming person": "power_driven",
    "Sail boat": "sailing", "Fishing boat": "fishing", "Trawler": "fishing",
}


@dataclass
class DetectionErrorModel:
    """
    Range-dependent detector performance.

    Defaults are physically motivated placeholders; `fit_from_detector` replaces
    them with values measured on real imagery.
    """
    pd_height_px_50: float = 3.0        # apparent height at 50% detection
    pd_slope: float = 2.2               # logistic sharpness
    pd_ceiling: float = 0.985
    sigma_u_px: float = 2.0             # horizontal localisation error
    sigma_v_px: float = 2.0             # vertical (waterline) localisation error
    false_alarm_rate: float = 0.02      # per frame
    class_confusion: float = 0.12       # P(wrong Rule 18 category)
    horizon_pd_falloff: float = 0.55    # extra suppression past the horizon
    fitted: bool = False
    notes: str = "default (not fitted to real data)"

    def detection_probability(self, height_px: float, range_m: float,
                              cam: CameraSpec, visibility: str = "good") -> float:
        if height_px <= 0:
            return 0.0
        z = self.pd_slope * (np.log(max(height_px, 1e-3))
                             - np.log(self.pd_height_px_50))
        p = self.pd_ceiling / (1.0 + np.exp(-z))
        # atmospheric extinction (Koschmieder)
        vis = visibility_range_m(visibility)
        p *= float(np.clip(np.exp(-2.2 * range_m / max(vis, 1.0)), 0.0, 1.0))
        if range_m > cam.horizon_range_m:
            p *= self.horizon_pd_falloff
        return float(np.clip(p, 0.0, 1.0))

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def fit_from_detector(records, cam: CameraSpec) -> DetectionErrorModel:
    """
    Fit the error model from per-object detector results on real imagery.

    `records` is a sequence of dicts with keys:
        height_px  -- ground-truth box height
        detected   -- bool
        du_px, dv_px -- localisation residuals for detected objects (optional)
        class_correct -- bool (optional)

    Detection probability is fitted as a logistic in log apparent height, which
    is the natural variable: apparent height is inversely proportional to range,
    so a logistic in log-height is a logistic in log-range.
    """
    m = DetectionErrorModel()
    h = np.array([r["height_px"] for r in records], float)
    d = np.array([bool(r["detected"]) for r in records])
    ok = h > 0
    h, d = h[ok], d[ok]
    if len(h) >= 30:
        # logistic regression on log height, plain Newton steps
        X = np.column_stack([np.ones(len(h)), np.log(h)])
        w = np.zeros(2)
        for _ in range(60):
            p = 1.0 / (1.0 + np.exp(-X @ w))
            W = np.clip(p * (1 - p), 1e-6, None)
            g = X.T @ (d.astype(float) - p)
            H = -(X * W[:, None]).T @ X
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            w = w + step
            if np.max(np.abs(step)) < 1e-8:
                break
        if abs(w[1]) > 1e-6:
            m.pd_slope = float(abs(w[1]))
            m.pd_height_px_50 = float(np.exp(-w[0] / w[1]))
        m.pd_ceiling = float(np.clip(d[h > np.percentile(h, 80)].mean()
                                     if (h > np.percentile(h, 80)).any() else 0.98,
                                     0.5, 0.999))

    du = [r["du_px"] for r in records if r.get("du_px") is not None]
    dv = [r["dv_px"] for r in records if r.get("dv_px") is not None]
    if len(du) >= 10:
        m.sigma_u_px = float(np.std(du))
    if len(dv) >= 10:
        m.sigma_v_px = float(np.std(dv))
    cls = [r["class_correct"] for r in records if r.get("class_correct") is not None]
    if len(cls) >= 20:
        m.class_confusion = float(1.0 - np.mean(cls))
    m.fitted = True
    m.notes = f"fitted from {len(records)} object instances on real imagery"
    return m



# Perception pipelines

class PerceptionBase:
    """Common interface: observe(own, tgt, t) -> Measurement or None."""
    name = "base"

    def __init__(self, cam: CameraSpec, tracker: str = "ekf",
                 seed: int = 0, visibility: str = "good"):
        self.cam = cam
        self.tracker_key = tracker
        self.visibility = visibility
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.tracker = TRACKER_REGISTRY[self.tracker_key]()
        self.history = []
        self.first_detection_idx = None
        self.first_ready_idx = None
        self.n_detections = 0
        self.n_false_alarms = 0
        self.status_estimates = []

    def geometry(self, own: ShipState, tgt: ShipState):
        d = tgt.position - own.position
        rng = float(np.hypot(d[0], d[1]))
        brg = float(relative_bearing(own.position, own.heading, tgt.position))
        aspect = float(wrap_pi(np.arctan2(-d[0], -d[1]) - tgt.heading))
        return rng, brg, aspect

    def observe(self, own, tgt, t, step) -> Measurement | None:
        raise NotImplementedError

    def step(self, own: ShipState, tgt: ShipState, t: float, step: int):
        """Observe, update the tracker, and return the estimated target state."""
        z = self.observe(own, tgt, t, step)
        if z is not None and z.detected:
            self.n_detections += 1
            if self.first_detection_idx is None:
                self.first_detection_idx = step
            self.tracker.update(z)
        else:
            self.tracker.predict(t)
        if self.tracker.ready and self.first_ready_idx is None:
            self.first_ready_idx = step
        est = self.tracker.state(spec=tgt.spec)
        self.history.append({
            "t": t, "detected": bool(z is not None and z.detected),
            "range_true": float(np.linalg.norm(tgt.position - own.position)),
            "range_meas": float(z.range_m) if z is not None else np.nan,
            "bearing_meas": float(z.bearing_rad) if z is not None else np.nan,
            "sigma_range": float(z.sigma_range_m) if z is not None else np.nan,
            "vel_sigma": self.tracker.velocity_sigma(),
            "pos_sigma": self.tracker.position_sigma(),
            "ready": bool(self.tracker.ready),
        })
        return est


class ModelledPerception(PerceptionBase):
    """Samples the fitted error model. Fast enough for the full campaign."""
    name = "modelled"

    def __init__(self, cam, error_model: DetectionErrorModel, tracker="ekf",
                 seed=0, visibility="good", air_draught_prior_rel_sigma=0.30):
        self.em = error_model
        self.prior_rel_sigma = air_draught_prior_rel_sigma
        super().__init__(cam, tracker, seed, visibility)

    def observe(self, own, tgt, t, step):
        rng_true, brg_true, aspect = self.geometry(own, tgt)
        air_draught = 0.15 * tgt.spec.length_m
        proj = project_vessel(self.cam, brg_true, rng_true, tgt.spec.length_m,
                              tgt.spec.beam_m, air_draught, aspect)
        if proj is None:
            return None
        p_det = self.em.detection_probability(proj["height_px"], rng_true,
                                              self.cam, self.visibility)
        if self.rng.random() > p_det:
            return None

        # perturb the image measurements, then invert the same geometry the
        # image-based pipeline would use
        u = proj["u_centre"] + self.rng.normal(0.0, self.em.sigma_u_px)
        v_bot = proj["v_bottom"] + self.rng.normal(0.0, self.em.sigma_v_px)
        h_px = max(proj["height_px"] + self.rng.normal(0.0, 1.4 * self.em.sigma_v_px),
                   0.5)

        b, s_b = bearing_from_column(self.cam, u, self.em.sigma_u_px)
        r_h, s_h = range_from_horizon(self.cam, v_bot, sigma_px=self.em.sigma_v_px)
        prior = air_draught * (1.0 + self.rng.normal(0.0, self.prior_rel_sigma))
        r_s, s_s = range_from_size(self.cam, h_px, max(prior, 1.0),
                                   self.prior_rel_sigma)
        R, s_R = fuse_ranges([(r_h, s_h), (r_s, s_s)])
        if not np.isfinite(R):
            return None

        # Rule 18 category, occasionally wrong -- this is what enables
        # perception-induced role inversion
        true_status = tgt.spec.status
        if self.rng.random() < self.em.class_confusion:
            options = [s for s in ("power_driven", "fishing",
                                   "restricted_manoeuvrability", "sailing")
                       if s != true_status]
            est_status = str(self.rng.choice(options))
        else:
            est_status = true_status
        self.status_estimates.append(est_status)

        return Measurement(t=t, bearing_rad=b, range_m=R,
                           sigma_bearing_rad=max(s_b, 1e-4),
                           sigma_range_m=max(s_R, 1.0),
                           own_pos=own.position.copy(), own_heading=own.heading,
                           detected=True,
                           extra={"height_px": proj["height_px"],
                                  "hull_down": proj["hull_down"],
                                  "status_est": est_status,
                                  "status_true": true_status})


class RenderedPerception(PerceptionBase):
    """
    Renders the frame and runs a detector. Used for case studies and for
    validating that the error model reproduces image-based behaviour.

    `detector` must expose detect(image) -> list of dicts with "box" and
    optionally "class_name" and "score". If None, a geometric oracle with
    pixel noise stands in so the pipeline is testable without a GPU.
    """
    name = "rendered"

    def __init__(self, cam, detector=None, segmenter=None, tracker="ekf",
                 seed=0, visibility="good", sea_state=0.35,
                 keep_frames_every=0):
        self.detector = detector
        self.segmenter = segmenter
        self.sea_state = sea_state
        self.keep_frames_every = keep_frames_every
        self.frames = []
        super().__init__(cam, tracker, seed, visibility)

    def reset(self):
        super().reset()
        self.frames = []

    def render(self, own, tgt):
        rng_true, brg_true, aspect = self.geometry(own, tgt)
        air_draught = 0.15 * tgt.spec.length_m
        proj = project_vessel(self.cam, brg_true, rng_true, tgt.spec.length_m,
                              tgt.spec.beam_m, air_draught, aspect)
        img = render_scene(self.cam, [proj] if proj else [],
                           visibility=self.visibility, sea_state=self.sea_state,
                           seed=int(self.rng.integers(0, 2 ** 31)))
        return img, proj

    def observe(self, own, tgt, t, step):
        img, proj = self.render(own, tgt)
        if self.keep_frames_every and step % self.keep_frames_every == 0:
            self.frames.append({"step": step, "image": img, "proj": proj})
        if proj is None:
            return None

        # horizon: from the segmenter if available, else the nominal geometry
        v_hor = None
        if self.segmenter is not None:
            try:
                v_hor = float(self.segmenter.horizon_row(img))
            except Exception:
                v_hor = None

        if self.detector is not None:
            dets = self.detector.detect(img)
            if not dets:
                return None
            box = max(dets, key=lambda d: d.get("score", 1.0))["box"]
            cls = max(dets, key=lambda d: d.get("score", 1.0)).get("class_name")
        else:
            # geometric oracle with pixel noise, so the loop is testable on CPU
            jitter = self.rng.normal(0.0, 2.0, size=4)
            box = np.asarray(proj["box"], float) + jitter
            cls = None

        u = 0.5 * (box[0] + box[2])
        v_bot = box[3]
        h_px = max(box[3] - box[1], 0.5)

        b, s_b = bearing_from_column(self.cam, u, 2.0)
        r_h, s_h = range_from_horizon(self.cam, v_bot, v_hor, sigma_px=2.0)
        air_draught = 0.15 * tgt.spec.length_m
        r_s, s_s = range_from_size(self.cam, h_px, air_draught, 0.30)
        R, s_R = fuse_ranges([(r_h, s_h), (r_s, s_s)])
        if not np.isfinite(R):
            return None

        return Measurement(t=t, bearing_rad=b, range_m=R,
                           sigma_bearing_rad=max(s_b, 1e-4),
                           sigma_range_m=max(s_R, 1.0),
                           own_pos=own.position.copy(), own_heading=own.heading,
                           detected=True,
                           extra={"height_px": h_px, "class_name": cls,
                                  "v_horizon": v_hor})


class PerfectPerception(PerceptionBase):
    """Ground truth passthrough: the Phase 1 control condition."""
    name = "perfect"

    def observe(self, own, tgt, t, step):
        rng_true, brg_true, _ = self.geometry(own, tgt)
        return Measurement(t=t, bearing_rad=brg_true, range_m=rng_true,
                           sigma_bearing_rad=1e-5, sigma_range_m=1.0,
                           own_pos=own.position.copy(), own_heading=own.heading)

    def step(self, own, tgt, t, step):
        super().step(own, tgt, t, step)
        return tgt.copy()


__all__ = ["DetectionErrorModel", "fit_from_detector", "PerceptionBase",
           "ModelledPerception", "RenderedPerception", "PerfectPerception",
           "CLASS_TO_STATUS"]
