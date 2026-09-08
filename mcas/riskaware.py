"""
The risk-aware CAS, COLREGS role inversion, and vessel-category estimation.

Two ideas, both consequences of taking perception uncertainty seriously.

**A CAS that knows what it does not know.** The four Phase 1 strategies test
risk against a point estimate of DCPA. The fifth tests it against a conformal
*lower bound*: it acts when the closest approach could plausibly be unsafe, not
only when the best guess says it will be. When the track is tight the bound sits
close to the estimate and behaviour is nearly unchanged; when the track is loose
the bound falls away and the CAS becomes conservative exactly where it should.
This buys safety with efficiency, and both sides of that trade are measured.

**Role inversion.** COLREGS Rule 18 makes the give-way obligation depend on
*vessel category*, not on geometry. So a classification error does not merely
blur the picture -- it can hand own-ship a right of way she does not have. The
two directions are not symmetric:

    give-way  -> stand-on    dangerous: own-ship holds course when obliged to
                             keep clear, against a vessel that is not going to
                             move either
    stand-on  -> give-way    inefficient and confusing to the other bridge, but
                             it errs toward manoeuvring

Separating those two directions is the point. An aggregate "flip rate" hides
the asymmetry that actually matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import colregs as cr
from .cas import CASBase, RuleBasedCOLREGS
from .geometry import KNOTS_TO_MS, cpa, relative_bearing, wrap_pi
from .uncertainty import ConformalDCPA, dcpa_sigma_linear
from .vessels import VESSEL_STATUS


# Risk-aware CAS

class RiskAwareCAS(RuleBasedCOLREGS):
    """
    COLREGS-compliant avoidance driven by a conformal lower bound on DCPA.

    Identical to `RuleBasedCOLREGS` in *how* it manoeuvres -- one substantial
    starboard alteration in ample time -- and different in *when*. Risk is
    declared on the lower confidence bound rather than the point estimate, so
    the trigger tightens automatically as the track degrades.

    `tracker_provider` is a callable returning the current (state, covariance)
    of the target track, or None. It is supplied by the perception runner.
    """
    name = "Risk-aware (conformal)"

    def __init__(self, dcpa_limit=926.0, tcpa_limit=1200.0, alteration_deg=35.0,
                 act_tcpa_s=900.0, stand_on_fallback_tcpa_s=300.0,
                 conformal: ConformalDCPA | None = None,
                 tracker_provider=None, sigma_floor_m: float = 20.0,
                 max_sigma_m: float = 1400.0, require_ready: bool = True):
        self.conformal = conformal
        self.tracker_provider = tracker_provider
        self.sigma_floor = sigma_floor_m
        self.max_sigma = max_sigma_m
        self.require_ready = require_ready
        self.ready_provider = None
        self.last_sigma = np.nan
        self.last_lower_bound = np.nan
        super().__init__(dcpa_limit, tcpa_limit, alteration_deg, act_tcpa_s,
                         stand_on_fallback_tcpa_s)

    def reset(self, own=None):
        super().reset(own)
        self.last_sigma = np.nan
        self.last_lower_bound = np.nan
        self.n_conservative_steps = 0

    def _risk(self, own, tgt):
        """
        Risk on the conformal lower bound of DCPA rather than its estimate.

        The bound is only used once the track is *ready*. Acting on a bound
        derived from a track that has not converged is worse than useless: the
        covariance is then so large that the lower bound is below the safety
        limit essentially always, so the CAS manoeuvres continuously on noise
        and ends up less safe than one using the point estimate. Uncertainty
        awareness has to include knowing when the uncertainty estimate itself is
        not yet meaningful.
        """
        t, d = cpa(own.position, own.velocity, tgt.position, tgt.velocity)
        sigma = np.nan
        if self.require_ready and self.ready_provider is not None \
                and not self.ready_provider():
            self.last_sigma, self.last_lower_bound = np.nan, d
            return (d < self.dcpa_limit and 0.0 < t < self.tcpa_limit), t, d
        if self.tracker_provider is not None:
            got = self.tracker_provider()
            if got is not None:
                x, P = got
                try:
                    sigma = dcpa_sigma_linear(own.position, own.velocity, x, P)
                except Exception:
                    sigma = np.nan
        if np.isfinite(sigma) and self.conformal is not None \
                and np.isfinite(self.conformal.q):
            sigma = float(np.clip(sigma, self.sigma_floor, self.max_sigma))
            lb = float(self.conformal.lower_bound(d, sigma))
            self.last_sigma, self.last_lower_bound = sigma, lb
            if lb < d - 1.0:
                self.n_conservative_steps += 1
            d_eff = lb
        else:
            self.last_sigma, self.last_lower_bound = sigma, d
            d_eff = d
        return (d_eff < self.dcpa_limit and 0.0 < t < self.tcpa_limit), t, d_eff



# Role inversion

def role_from_status(own, tgt, est_status: str):
    """Own-ship's obligation if the target's Rule 18 category were `est_status`."""
    shadow = type(tgt.spec)(**{**tgt.spec.__dict__, "status": est_status})
    ghost = tgt.copy()
    ghost.spec = shadow
    return cr.assign_role(own, ghost)


def classify_flip(true_role: str, est_role: str) -> str:
    """Name the direction of a role inversion."""
    if true_role == est_role:
        return "none"
    gives = (cr.GIVE_WAY, cr.BOTH_GIVE_WAY)
    if true_role in gives and est_role == cr.STAND_ON:
        return "dangerous"          # obliged to keep clear, believes otherwise
    if true_role == cr.STAND_ON and est_role in gives:
        return "conservative"       # manoeuvres when not required to
    return "other"


def status_confusion_matrix(true_status, est_status, labels=None):
    """Confusion matrix over COLREGS Rule 18 categories."""
    labels = labels or list(VESSEL_STATUS.keys())
    idx = {s: i for i, s in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for a, b in zip(true_status, est_status):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    return pd.DataFrame(M, index=labels, columns=labels)


def role_inversion_report(results, dt: float = 5.0):
    """
    Per-scenario summary of role inversion from a set of completed runs.

    `results` is a mapping key -> result dict produced by the perception runner,
    each carrying "role_flip_kind" produced during the run.
    """
    rows = []
    for key, r in results.items():
        kinds = r.get("role_flip_kind")
        if kinds is None:
            continue
        kinds = np.asarray(kinds)
        n = len(kinds)
        dangerous = float(np.mean(kinds == "dangerous"))
        conservative = float(np.mean(kinds == "conservative"))
        # was own-ship mistaken during the approach, when it mattered?
        approach = np.asarray(r["tcpa_hist"]) < 900.0
        danger_on_approach = float(np.mean((kinds == "dangerous") & approach)) \
            if approach.any() else 0.0
        rows.append({
            "key": key if isinstance(key, str) else "_".join(map(str, key)),
            "cas": r["cas_name"], "encounter": r["encounter"], "role": r["role"],
            "tgt_status": r["spec"].tgt_status if r.get("spec") else None,
            "flip_dangerous": dangerous, "flip_conservative": conservative,
            "flip_dangerous_on_approach": danger_on_approach,
            "min_dist_m": r["min_dist"], "safe": r["safe"],
            "compliance": r["compliance"].overall,
        })
    return pd.DataFrame(rows)


# CLIP zero-shot vessel category

CLIP_PROMPTS = {
    "power_driven": ["a photo of a cargo ship at sea",
                     "a photo of a container ship",
                     "a photo of a tanker at sea",
                     "a photo of a passenger ferry"],
    "fishing": ["a photo of a fishing trawler at sea",
                "a photo of a fishing boat with nets",
                "a photo of a small fishing vessel"],
    "sailing": ["a photo of a sailing yacht under sail",
                "a photo of a sailboat with white sails"],
    "restricted_manoeuvrability": ["a photo of a dredger at work",
                                   "a photo of a cable-laying vessel",
                                   "a photo of a tug towing a barge"],
}


class ClipVesselClassifier:
    """
    Zero-shot COLREGS Rule 18 category from a cropped vessel image.

    Used inference-only, which is the honest and practical choice on a free
    GPU tier: no fine-tuning, no labels for the categories the Rules care about
    (SMD does not annotate "restricted in manoeuvrability"), and a
    vision-language model is precisely the tool for a label set that exists in
    the regulations but not in the dataset.
    """
    name = "CLIP zero-shot"

    def __init__(self, model_name="openai/clip-vit-base-patch32",
                 prompts=None, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.prompts = prompts or CLIP_PROMPTS
        self.labels = list(self.prompts)
        self.flat = [p for lab in self.labels for p in self.prompts[lab]]
        self.owner = [i for i, lab in enumerate(self.labels)
                      for _ in self.prompts[lab]]

    def classify(self, image_crop):
        """Return (label, probability vector over labels)."""
        import torch
        from PIL import Image
        img = Image.fromarray(image_crop) if isinstance(image_crop, np.ndarray) \
            else image_crop
        inputs = self.processor(text=self.flat, images=img,
                                return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        agg = np.zeros(len(self.labels))
        for p, o in zip(probs, self.owner):
            agg[o] += float(p)
        agg = agg / max(agg.sum(), 1e-9)
        return self.labels[int(np.argmax(agg))], agg

    def confusion_from_crops(self, crops, true_labels):
        est = [self.classify(c)[0] for c in crops]
        return status_confusion_matrix(true_labels, est, labels=self.labels), est


def confusion_to_error_rate(conf: pd.DataFrame) -> float:
    """Overall Rule 18 misclassification rate implied by a confusion matrix."""
    M = conf.to_numpy(float)
    tot = M.sum()
    return float(1.0 - np.trace(M) / tot) if tot > 0 else np.nan


__all__ = ["RiskAwareCAS", "role_from_status", "classify_flip",
           "status_confusion_matrix", "role_inversion_report",
           "ClipVesselClassifier", "CLIP_PROMPTS", "confusion_to_error_rate"]
