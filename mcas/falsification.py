"""
Falsification: searching the scenario space instead of sampling it.

Every campaign so far drew scenarios uniformly. That answers "how often does
this system fail?" -- a useful question, and the wrong one for certification.
A rare failure is still a failure, and uniform sampling finds rare failures
only by being lucky. Phase 1 measured that uniform sampling leaves parts of the
encounter space thin; Phase 2b added a second thing worth hunting for, the
combinations where perception misreads the COLREGS category in the dangerous
direction.

Falsification inverts the question: **given a system, find the scenario that
breaks it.** Formally, minimise a robustness score

    rho(theta) = min_t distance(t) - d_safe

over scenario parameters theta. rho < 0 is a counterexample. This is the
objective used in the VerifAI line of work built on Scenic, and it turns a
randomised test campaign into a directed search.

Three optimisers, so the comparison is meaningful:

  RandomSearch  -- the control. Any adaptive method must beat it, and the
                   comparison is the whole point: without it, "we found 40
                   failures" says nothing about search efficiency.
  CMA-ES        -- evolution strategy with full covariance adaptation. Robust
                   on noisy, non-differentiable, moderately-dimensioned
                   objectives, which is exactly what a simulation loop is.
  BayesOpt      -- Gaussian process with expected improvement. Sample-efficient
                   when each evaluation is expensive, at the cost of cubic
                   scaling in the number of evaluations.

The metric that matters is not how many failures were found but **how many
simulations were needed to find the first one**, and how bad the worst one is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scenarios import EncounterSpec

# Continuous search dimensions and their bounds.
DIMS = [
    ("dcpa_m", 0.0, 1500.0),
    ("tcpa_s", 600.0, 1800.0),
    ("own_speed_kn", 8.0, 16.0),
    ("tgt_speed_kn", 8.0, 16.0),
    ("tgt_heading_off", -1.0, 1.0),      # position within the encounter's band
]

HEADING_BANDS = {
    "head_on": (176.0, 184.0),
    "overtaking": (-12.0, 12.0),
    "crossing": (200.0, 340.0),
}


@dataclass
class ScenarioSpace:
    """
    Maps a unit hypercube to `EncounterSpec`s of one encounter type.

    Searching in [0,1]^n rather than in physical units keeps every dimension
    comparably scaled, which matters for CMA-ES's step size and for the GP's
    length scales.
    """
    encounter: str
    tgt_status: str = "power_driven"
    visibility: str = "good"
    passing_side: int = 1
    own_vessel: str = "container_feeder"
    tgt_vessel: str = "container_feeder"

    @property
    def n_dims(self):
        return len(DIMS)

    def decode(self, u) -> EncounterSpec:
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        vals = {}
        for (name, lo, hi), ui in zip(DIMS, u):
            vals[name] = lo + ui * (hi - lo)
        lo_h, hi_h = HEADING_BANDS[self.encounter]
        frac = 0.5 * (vals.pop("tgt_heading_off") + 1.0)
        heading = lo_h + frac * (hi_h - lo_h)
        own_s, tgt_s = vals["own_speed_kn"], vals["tgt_speed_kn"]
        if self.encounter == "overtaking" and own_s <= tgt_s + 2.5:
            # Rule 13 requires the overtaking vessel to be materially faster;
            # repair rather than reject so the search space stays connected.
            tgt_s = max(own_s - 3.0, 3.0)
        return EncounterSpec(
            encounter=self.encounter, dcpa_m=vals["dcpa_m"],
            tcpa_s=vals["tcpa_s"], own_speed_kn=own_s, tgt_speed_kn=tgt_s,
            passing_side=self.passing_side, own_vessel=self.own_vessel,
            tgt_vessel=self.tgt_vessel, own_status="power_driven",
            tgt_status=self.tgt_status, visibility=self.visibility,
            tgt_heading_deg=heading)

    def describe(self, u):
        s = self.decode(u)
        return (f"DCPA {s.dcpa_m:.0f} m, TCPA {s.tcpa_s/60:.1f} min, "
                f"own {s.own_speed_kn:.1f} kn, target {s.tgt_speed_kn:.1f} kn "
                f"@ {s.tgt_heading_deg:.0f} deg, {s.tgt_status}, {s.visibility}")


@dataclass
class FalsificationResult:
    method: str
    encounter: str
    history: list = field(default_factory=list)     # robustness per evaluation
    best_u: np.ndarray | None = None
    best_rho: float = np.inf
    n_evals: int = 0
    first_violation_at: int | None = None
    violations: list = field(default_factory=list)  # (eval index, u, rho)

    def record(self, u, rho):
        self.n_evals += 1
        self.history.append(float(rho))
        if rho < self.best_rho:
            self.best_rho, self.best_u = float(rho), np.array(u, float)
        if rho < 0:
            if self.first_violation_at is None:
                self.first_violation_at = self.n_evals
            self.violations.append((self.n_evals, np.array(u, float), float(rho)))

    @property
    def running_best(self):
        return np.minimum.accumulate(np.asarray(self.history, float))

    def summary(self):
        return {"method": self.method, "encounter": self.encounter,
                "evaluations": self.n_evals,
                "best robustness (m)": self.best_rho,
                "violations found": len(self.violations),
                "evals to first violation": self.first_violation_at
                if self.first_violation_at is not None else np.nan}


# Optimisers

def random_search(objective, space: ScenarioSpace, n_evals=150, seed=0):
    """Uniform sampling. The control every adaptive method is measured against."""
    rng = np.random.default_rng(seed)
    res = FalsificationResult("Random search", space.encounter)
    for _ in range(n_evals):
        u = rng.random(space.n_dims)
        res.record(u, objective(u))
    return res


def cma_es(objective, space: ScenarioSpace, n_evals=150, sigma0=0.30, seed=0,
           x0=None):
    """
    (mu/mu_w, lambda)-CMA-ES with rank-one and rank-mu covariance updates.

    Chosen because the objective is a simulation: noisy, non-differentiable,
    and with strong parameter interactions (a small DCPA matters far more at
    short TCPA). Covariance adaptation learns those interactions rather than
    assuming the axes are independent.
    """
    rng = np.random.default_rng(seed)
    n = space.n_dims
    res = FalsificationResult("CMA-ES", space.encounter)

    xmean = rng.random(n) if x0 is None else np.clip(np.asarray(x0, float), 0, 1)
    sigma = sigma0
    lam = int(4 + np.floor(3 * np.log(n)))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    pc, ps = np.zeros(n), np.zeros(n)
    C, invsqrtC = np.eye(n), np.eye(n)
    eigen_eval, counteval = 0, 0

    while res.n_evals < n_evals:
        try:
            B_ = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = C + 1e-8 * np.eye(n)
            B_ = np.linalg.cholesky(C)
        z = rng.standard_normal((lam, n))
        y = z @ B_.T
        X = np.clip(xmean + sigma * y, 0.0, 1.0)

        f = np.empty(lam)
        for i in range(lam):
            if res.n_evals >= n_evals:
                f[i:] = np.inf
                break
            f[i] = objective(X[i])
            res.record(X[i], f[i])
        counteval += lam

        order = np.argsort(f)
        X, y = X[order], y[order]
        xold = xmean.copy()
        xmean = w @ X[:mu]
        ymean = (xmean - xold) / max(sigma, 1e-12)

        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ ymean)
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam))
                / chiN) < (1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean

        artmp = y[:mu]
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
             + cmu * (artmp.T * w) @ artmp)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = float(np.clip(sigma, 1e-4, 0.6))

        if counteval - eigen_eval > lam / (c1 + cmu) / n / 10:
            eigen_eval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            d, B = np.linalg.eigh(C)
            d = np.clip(d, 1e-12, None)
            invsqrtC = B @ np.diag(d ** -0.5) @ B.T
    return res


def bayes_opt(objective, space: ScenarioSpace, n_evals=150, n_init=20, seed=0,
              n_candidates=2000):
    """
    Gaussian process with expected improvement.

    Sample-efficient per evaluation, which is the right trade when each
    evaluation is a full simulated encounter. The cost is cubic refitting, so
    it is run at a smaller budget than CMA-ES in practice.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    rng = np.random.default_rng(seed)
    n = space.n_dims
    res = FalsificationResult("Bayesian optimisation", space.encounter)

    X = rng.random((n_init, n))
    y = np.array([objective(x) for x in X])
    for xi, yi in zip(X, y):
        res.record(xi, yi)

    kernel = (ConstantKernel(1.0, (1e-2, 1e4))
              * Matern(length_scale=np.full(n, 0.3), nu=2.5,
                       length_scale_bounds=(1e-2, 1e1))
              + WhiteKernel(1e-2, (1e-6, 1e2)))

    while res.n_evals < n_evals:
        ys = (y - y.mean()) / max(y.std(), 1e-9)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                      n_restarts_optimizer=1,
                                      random_state=int(rng.integers(1e6)))
        try:
            gp.fit(X, ys)
        except Exception:
            xn = rng.random(n)
            fn = objective(xn)
            res.record(xn, fn)
            X = np.vstack([X, xn]); y = np.append(y, fn)
            continue

        cand = rng.random((n_candidates, n))
        mu, sd = gp.predict(cand, return_std=True)
        sd = np.maximum(sd, 1e-9)
        best = ys.min()
        z = (best - mu) / sd
        from math import erf, sqrt
        Phi = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
        phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        ei = (best - mu) * Phi + sd * phi          # expected improvement
        xn = cand[int(np.argmax(ei))]
        fn = objective(xn)
        res.record(xn, fn)
        X = np.vstack([X, xn]); y = np.append(y, fn)
    return res


OPTIMISERS = {"random": random_search, "cmaes": cma_es, "bayesopt": bayes_opt}



# Diversity of the counterexamples found

def counterexample_diversity(res: FalsificationResult, radius: float = 0.18):
    """
    Number of *distinct* counterexamples, greedily clustered in the unit cube.

    A search that returns forty variations of the same failure has found one
    bug, not forty. Reporting a raw violation count without this is the easiest
    way to overstate what a falsifier achieved.
    """
    if not res.violations:
        return 0, []
    pts = [v[1] for v in res.violations]
    reps = []
    for p in pts:
        if all(np.linalg.norm(p - q) > radius for q in reps):
            reps.append(p)
    return len(reps), reps


def compare_methods(results):
    """Tabulate optimisers against the random-search control."""
    import pandas as pd
    rows = []
    for r in results:
        n_distinct, _ = counterexample_diversity(r)
        d = r.summary()
        d["distinct counterexamples"] = n_distinct
        rows.append(d)
    return pd.DataFrame(rows).set_index("method")


__all__ = ["ScenarioSpace", "FalsificationResult", "random_search", "cma_es",
           "bayes_opt", "OPTIMISERS", "counterexample_diversity",
           "compare_methods", "DIMS", "HEADING_BANDS"]
