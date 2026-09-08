"""
A learned avoidance policy, and what it takes to ship one.

Two purposes.

**Distillation.** MPC is the safest strategy in this harness and the most
expensive: it re-solves a discrete optimisation over a 1200 s horizon every few
steps. A shipboard CAS runs on an embedded box, not a workstation. Behaviour
cloning distils MPC's *decisions* into a small feedforward network that
evaluates in microseconds, and the interesting question is how much of MPC's
safety survives the distillation.

**A different kind of failure.** A rule-based CAS fails predictably: it does
what the Rules say, and the Rules are sometimes wrong about a non-cooperative
target. A learned policy fails differently -- it can be confidently wrong
off-distribution. That makes it the natural subject for the falsification
search in `falsification`: random testing flatters learned controllers, and
directed search does not.

The feature vector is deliberately *relative and dimensionless where possible*,
so the policy generalises across vessel sizes and speeds rather than memorising
the training distribution.
"""
from __future__ import annotations

import numpy as np

from .cas import CASBase, MPC
from .colregs import BOTH_GIVE_WAY, GIVE_WAY, STAND_ON, assign_role, classify
from .geometry import KNOTS_TO_MS, cpa, relative_bearing, wrap_pi

FEATURE_NAMES = [
    "range_norm",          # range / 5 nm
    "sin_bearing", "cos_bearing",
    "sin_rel_heading", "cos_rel_heading",
    "speed_ratio",
    "dcpa_norm",           # DCPA / safety limit
    "tcpa_norm",           # TCPA / 20 min
    "range_rate_norm",
    "is_give_way", "is_stand_on", "is_both_give_way",
    "own_speed_norm",
    "course_deviation",    # current heading error vs original track
]


def extract_features(own, tgt, base_heading, base_speed, safety_m=926.0):
    """Relative, scale-free description of the encounter."""
    d = tgt.position - own.position
    rng = float(np.hypot(d[0], d[1]))
    brg = float(relative_bearing(own.position, own.heading, tgt.position))
    dpsi = float(wrap_pi(tgt.heading - own.heading))
    t_c, d_c = cpa(own.position, own.velocity, tgt.position, tgt.velocity)
    w = tgt.velocity - own.velocity
    rr = float(d @ w / max(rng, 1e-6))
    role = assign_role(own, tgt)
    return np.array([
        min(rng / (5 * 1852.0), 3.0),
        np.sin(brg), np.cos(brg),
        np.sin(dpsi), np.cos(dpsi),
        np.clip(tgt.speed_kn / max(own.speed_kn, 1e-3), 0.0, 3.0),
        min(d_c / max(safety_m, 1.0), 5.0),
        min(t_c / 1200.0, 3.0),
        np.clip(rr / 10.0, -3.0, 3.0),
        1.0 if role == GIVE_WAY else 0.0,
        1.0 if role == STAND_ON else 0.0,
        1.0 if role == BOTH_GIVE_WAY else 0.0,
        np.clip(own.speed_kn / 16.0, 0.0, 2.0),
        float(wrap_pi(own.heading - base_heading)) / np.pi,
    ], dtype=float)



# Dataset generation

def collect_demonstrations(scenario_set, dt=5.0, horizon_s=3600.0,
                           safety_m=926.0, expert=None, max_scenarios=None,
                           progress=None):
    """
    Roll out the expert (MPC by default) and record (features, action) pairs.

    The action is the *commanded course change and speed factor* relative to the
    original track, not the absolute heading -- which is what makes the policy
    transferable to a track it never saw in training.
    """
    from .runner import run_scenario
    from .vessels import NomotoModel

    X, Y, meta = [], [], []
    items = [(e, it) for e, v in scenario_set.items() for it in v]
    if max_scenarios:
        items = items[:max_scenarios]
    for k, (enc, (spec, own0, tgt0, diag)) in enumerate(items):
        cas = expert() if expert else MPC(dcpa_limit=safety_m)
        own, tgt = own0.copy(), tgt0.copy()
        dyn = NomotoModel()
        cas.reset(own)
        base_h, base_s = own.heading, own.speed_kn
        n = int(horizon_s / dt)
        for i in range(n):
            f = extract_features(own, tgt, base_h, base_s, safety_m)
            h_cmd, v_cmd = cas.decide(own, tgt, i * dt, i)
            X.append(f)
            Y.append([float(wrap_pi(h_cmd - base_h)) / np.pi,
                      float(v_cmd / max(base_s, 1e-6))])
            meta.append({"encounter": enc, "step": i,
                         "role": assign_role(own, tgt)})
            dyn.step(own, h_cmd, v_cmd, dt)
            tv = tgt.velocity
            tgt.x += tv[0] * dt
            tgt.y += tv[1] * dt
        if progress and k % max(len(items) // 20, 1) == 0:
            progress(k, len(items))
    return np.array(X), np.array(Y), meta



# Models

class NumpyMLP:
    """
    Small feedforward regressor with Adam, implemented directly.

    Present so the phase runs without torch, and so the exported policy has no
    framework dependency at inference time -- which is the realistic constraint
    for embedded deployment.
    """
    name = "MLP (numpy)"

    def __init__(self, n_in, hidden=(64, 64), n_out=2, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b))
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def _fwd(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            acts.append(h)
        return acts

    def predict(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        return self._fwd((x - self.mu) / self.sd)[-1]

    def fit(self, X, Y, epochs=60, batch=256, lr=2e-3, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        Xn = (X - self.mu) / self.sd
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        hist = []
        for ep in range(epochs):
            idx = rng.permutation(len(Xn))
            tot = 0.0
            for s in range(0, len(idx), batch):
                sel = idx[s:s + batch]
                xb, yb = Xn[sel], Y[sel]
                acts = self._fwd(xb)
                out = acts[-1]
                err = (out - yb) / len(sel)
                tot += float(np.mean((out - yb) ** 2)) * len(sel)
                g = err
                t += 1
                for i in reversed(range(len(self.W))):
                    gW = acts[i].T @ g
                    gb = g.sum(0)
                    if i > 0:
                        g = (g @ self.W[i].T) * (1 - acts[i] ** 2)
                    for arr, m, v, grad in ((self.W, mW, vW, gW), (self.b, mb, vb, gb)):
                        m[i] = b1 * m[i] + (1 - b1) * grad
                        v[i] = b2 * v[i] + (1 - b2) * grad ** 2
                        arr[i] -= lr * (m[i] / (1 - b1 ** t)) / \
                            (np.sqrt(v[i] / (1 - b2 ** t)) + eps)
            hist.append(tot / len(Xn))
            if verbose and (ep % max(epochs // 8, 1) == 0 or ep == epochs - 1):
                print(f"  epoch {ep+1:3d}/{epochs}  mse={hist[-1]:.5f}")
        return hist

    def save(self, path: str):
        """
        Persist weights, biases and input normalisation to a single .npz file.

        This is the policy's checkpoint: everything `load` needs to
        reconstruct a functionally identical model, with no dependency on the
        training data or the random seed used to initialise it.
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 n_layers=len(self.W),
                 **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)},
                 mu=self.mu, sd=self.sd)
        return path

    @classmethod
    def load(cls, path: str) -> "NumpyMLP":
        """Reconstruct a NumpyMLP from a checkpoint written by `save`."""
        d = np.load(path)
        n_layers = int(d["n_layers"])
        W = [d[f"W{i}"] for i in range(n_layers)]
        b = [d[f"b{i}"] for i in range(n_layers)]
        m = cls.__new__(cls)
        m.W, m.b = W, b
        m.mu, m.sd = d["mu"], d["sd"]
        return m


def build_torch_mlp(n_in, hidden=(64, 64), n_out=2, dropout=0.0):
    """Equivalent architecture in torch, for GPU training and ONNX export."""
    import torch.nn as nn
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


class LearnedCAS(CASBase):
    """
    Behaviour-cloned policy wrapped in the CAS interface.

    The safety envelope is deliberately *not* enforced here: the point is to
    measure what the policy learned, including where it is confidently wrong.
    A production system would wrap this in a rule-based supervisor, and doing
    so before evaluation would hide exactly the failures worth finding.
    """
    name = "Learned policy (BC)"

    def __init__(self, model=None, dcpa_limit=926.0, tcpa_limit=1200.0,
                 max_course_change_deg=75.0):
        self.model = model
        self.max_course = np.radians(max_course_change_deg)
        super().__init__(dcpa_limit, tcpa_limit)

    def decide(self, own, tgt, t, step):
        if self.model is None:
            return self.base_heading, self.base_speed
        f = extract_features(own, tgt, self.base_heading, self.base_speed,
                             self.dcpa_limit)
        out = np.asarray(self.model.predict(f)).ravel()
        dpsi = float(np.clip(out[0] * np.pi, -self.max_course, self.max_course))
        vfac = float(np.clip(out[1], 0.5, 1.05))
        if abs(dpsi) > np.radians(2.0) or vfac < 0.98:
            self._note_manoeuvre(step)
        self.encounter = classify(own, tgt)
        self.role = assign_role(own, tgt, self.encounter)
        return float(wrap_pi(self.base_heading + dpsi)), self.base_speed * vfac



# Deployment

def export_onnx(torch_model, n_in, path="outputs/models/policy.onnx"):
    """Export to ONNX: a framework-independent artefact an embedded box can run."""
    import os
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch_model.eval()
    dummy = torch.zeros(1, n_in)
    torch.onnx.export(torch_model, dummy, path, input_names=["features"],
                      output_names=["action"],
                      dynamic_axes={"features": {0: "batch"},
                                    "action": {0: "batch"}},
                      opset_version=17)
    return path


def quantize_onnx(src, dst=None):
    """Dynamic INT8 quantisation. Smaller and faster; accuracy cost measured."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst = dst or src.replace(".onnx", "_int8.onnx")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    return dst


def benchmark_latency(predict_fn, n_in, n=2000, seed=0):
    """Per-decision latency, which is what a shipboard control loop budgets for."""
    import time
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_in))
    predict_fn(X[:1])
    t0 = time.perf_counter()
    for i in range(n):
        predict_fn(X[i:i + 1])
    dt = (time.perf_counter() - t0) / n
    return {"mean latency (us)": dt * 1e6, "throughput (Hz)": 1.0 / dt}


def model_size_kb(path):
    import os
    return os.path.getsize(path) / 1024.0 if os.path.exists(path) else np.nan


__all__ = ["FEATURE_NAMES", "extract_features", "collect_demonstrations",
           "NumpyMLP", "build_torch_mlp", "LearnedCAS", "export_onnx",
           "quantize_onnx", "benchmark_latency", "model_size_kb"]
