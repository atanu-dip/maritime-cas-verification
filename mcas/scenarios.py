"""
Maritime encounter scenario space.

The central idea: *parameterise the scenario space by encounter geometry, not
by position.*

Sampling ship positions uniformly is wasteful and unsafe as a test strategy --
most samples are not encounters at all, and the ones that are have
uncontrolled difficulty. Instead an encounter is specified abstractly by

    (encounter type, DCPA, TCPA, own speed, target speed, passing side,
     vessel categories, visibility)

and the geometry is solved *backwards* for the initial positions and headings
that realise it exactly.

Inverse solve
-------------
With own-ship at the origin, let w = v_target - v_own be the relative velocity
and t*, d the desired TCPA and DCPA. Writing w_hat = w/|w| and w_perp for
w_hat rotated 90 deg clockwise, the initial relative position

    r0 = -t* |w| w_hat  +  s d w_perp,      s in {+1, -1}

satisfies TCPA(r0, w) = t* and DCPA(r0, w) = d identically -- substitute into
t = -(r0.w)/|w|^2 to verify. `s` selects which side the target passes.

Two consequences worth noting:

  * Every sample is a genuine encounter with a *known* difficulty, so DCPA can
    be swept as an independent variable.
  * The overtaking geometry comes out correct by construction. Sampling
    positions directly makes it easy to place the "overtaken" vessel astern of
    a faster own-ship, which produces two vessels that simply diverge and never
    interact -- a silent failure that yields identical results for every CAS.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .colregs import CROSSING, HEAD_ON, OVERTAKING, assign_role, classify
from .geometry import KNOTS_TO_MS, cpa, relative_bearing, wrap_pi
from .vessels import VESSEL_LIBRARY, ShipState, VesselSpec

VISIBILITY = {"good": 10 * 1852.0, "moderate": 3 * 1852.0, "restricted": 1 * 1852.0}


@dataclass
class EncounterSpec:
    """Abstract, simulator-independent description of one encounter."""
    encounter: str
    dcpa_m: float
    tcpa_s: float
    own_speed_kn: float
    tgt_speed_kn: float
    passing_side: int                  # +1 target passes to starboard of the CPA line
    own_vessel: str = "container_feeder"
    tgt_vessel: str = "container_feeder"
    own_status: str = "power_driven"
    tgt_status: str = "power_driven"
    visibility: str = "good"
    tgt_heading_deg: float = 180.0
    seed: Optional[int] = None

    def as_dict(self):
        return asdict(self)



# Inverse geometry

def solve_initial_geometry(spec: EncounterSpec):
    """
    Solve for the initial states realising `spec` exactly.
    Own-ship starts at the origin heading due North.
    Returns (own_state, target_state, diagnostics).
    """
    own_spec = VESSEL_LIBRARY[spec.own_vessel]
    tgt_spec = VESSEL_LIBRARY[spec.tgt_vessel]
    own_spec = VesselSpec(**{**own_spec.__dict__, "status": spec.own_status})
    tgt_spec = VesselSpec(**{**tgt_spec.__dict__, "status": spec.tgt_status})

    psi_o = 0.0
    psi_t = np.radians(spec.tgt_heading_deg)

    v_o = spec.own_speed_kn * KNOTS_TO_MS * np.array([np.sin(psi_o), np.cos(psi_o)])
    v_t = spec.tgt_speed_kn * KNOTS_TO_MS * np.array([np.sin(psi_t), np.cos(psi_t)])
    w = v_t - v_o
    w_norm = float(np.linalg.norm(w))
    if w_norm < 1e-3:
        raise ValueError("degenerate encounter: zero relative velocity")

    w_hat = w / w_norm
    w_perp = np.array([w_hat[1], -w_hat[0]])          # 90 deg clockwise

    r0 = -spec.tcpa_s * w_norm * w_hat + spec.passing_side * spec.dcpa_m * w_perp

    own = ShipState(0.0, 0.0, psi_o, spec.own_speed_kn, spec=own_spec)
    tgt = ShipState(float(r0[0]), float(r0[1]), psi_t, spec.tgt_speed_kn,
                    spec=tgt_spec)

    t_check, d_check = cpa(own.position, own.velocity, tgt.position, tgt.velocity)
    diagnostics = {
        "initial_range_m": float(np.linalg.norm(r0)),
        "relative_bearing_deg": float(np.degrees(
            relative_bearing(own.position, own.heading, tgt.position))),
        "tcpa_check_s": t_check,
        "dcpa_check_m": d_check,
        "tcpa_error_s": abs(t_check - spec.tcpa_s),
        "dcpa_error_m": abs(d_check - spec.dcpa_m),
        "classified": classify(own, tgt),
        "own_role": assign_role(own, tgt),
    }
    return own, tgt, diagnostics



# Analytic sampler

class AnalyticEncounterSampler:
    """
    Seeded sampler over the abstract encounter space. Always available; used as
    the fallback when Scenic cannot be imported, and as the reference the
    Scenic sampler is checked against.
    """

    def __init__(self, seed: int = 0,
                 dcpa_range=(0.0, 1500.0), tcpa_range=(600.0, 1800.0),
                 speed_range=(8.0, 16.0),
                 vessels=("container_feeder", "handysize_bulker",
                          "coastal_tanker", "vlcc"),
                 status_weights=(("power_driven", 0.80), ("fishing", 0.10),
                                 ("restricted_manoeuvrability", 0.06),
                                 ("not_under_command", 0.04)),
                 visibility_weights=(("good", 0.75), ("moderate", 0.18),
                                     ("restricted", 0.07))):
        self.rng = np.random.default_rng(seed)
        self.dcpa_range = dcpa_range
        self.tcpa_range = tcpa_range
        self.speed_range = speed_range
        self.vessels = list(vessels)
        self.status_names = [s for s, _ in status_weights]
        self.status_p = np.array([p for _, p in status_weights], float)
        self.status_p /= self.status_p.sum()
        self.vis_names = [s for s, _ in visibility_weights]
        self.vis_p = np.array([p for _, p in visibility_weights], float)
        self.vis_p /= self.vis_p.sum()

    def _speeds(self, encounter):
        lo, hi = self.speed_range
        if encounter == OVERTAKING:
            # own-ship must be materially faster to overtake at all
            own = self.rng.uniform(0.6 * lo + 0.4 * hi, hi)
            tgt = self.rng.uniform(lo, own - 2.5)
            return own, max(tgt, 3.0)
        return self.rng.uniform(lo, hi), self.rng.uniform(lo, hi)

    def _target_heading(self, encounter):
        if encounter == HEAD_ON:
            return 180.0 + self.rng.uniform(-4.0, 4.0)
        if encounter == OVERTAKING:
            return self.rng.uniform(-12.0, 12.0)
        # crossing: target approaches from either bow, avoiding the head-on and
        # overtaking sectors
        from_starboard = self.rng.random() < 0.5
        return (self.rng.uniform(200.0, 340.0) if from_starboard
                else self.rng.uniform(20.0, 160.0))

    def sample_spec(self, encounter: str, index: int = 0) -> EncounterSpec:
        own_v, tgt_v = self.rng.choice(self.vessels, size=2, replace=True)
        own_speed, tgt_speed = self._speeds(encounter)
        return EncounterSpec(
            encounter=encounter,
            dcpa_m=float(self.rng.uniform(*self.dcpa_range)),
            tcpa_s=float(self.rng.uniform(*self.tcpa_range)),
            own_speed_kn=float(own_speed),
            tgt_speed_kn=float(tgt_speed),
            passing_side=int(self.rng.choice([-1, 1])),
            own_vessel=str(own_v),
            tgt_vessel=str(tgt_v),
            own_status="power_driven",
            tgt_status=str(self.rng.choice(self.status_names, p=self.status_p)),
            visibility=str(self.rng.choice(self.vis_names, p=self.vis_p)),
            tgt_heading_deg=float(self._target_heading(encounter)),
            seed=index,
        )

    def sample(self, encounter: str, n: int, verify: bool = True):
        """Draw `n` specs of the requested type, rejecting degenerate ones."""
        out, attempts = [], 0
        while len(out) < n and attempts < 60 * n:
            attempts += 1
            spec = self.sample_spec(encounter, index=len(out))
            try:
                own, tgt, diag = solve_initial_geometry(spec)
            except ValueError:
                continue
            if diag["initial_range_m"] < 800.0 or diag["initial_range_m"] > 30000.0:
                continue
            if verify and diag["classified"] != encounter:
                # the realised geometry must actually be the requested encounter
                continue
            out.append((spec, own, tgt, diag))
        if len(out) < n:
            warnings.warn(f"only {len(out)}/{n} {encounter} scenarios sampled")
        return out


# Scenic sampler

SCENIC_SOURCE = r'''
"""Maritime encounter scenario space (Scenic 3).

Scenic's role here is what a scenario description language is actually for:
declaring an *abstract space* of encounters -- distributions over the encounter
parameters plus constraints on which combinations are admissible -- rather than
a fixed list of test cases. Each sampled scene fixes one concrete assignment of
those parameters, which the Python domain bridge then instantiates into initial
positions and headings via the inverse DCPA/TCPA solve.

Parameterising by (DCPA, TCPA, relative geometry) rather than by position is
what makes the sampled set a controlled test suite: encounter difficulty is an
independent variable, and every sample is a genuine encounter.
"""

param encounter = 'head_on'          # head_on | crossing | overtaking

# encounter difficulty 
param dcpa_m = Range(0, 1500)        # metres at closest point of approach
param tcpa_s = Range(600, 1800)      # seconds to closest point of approach
param passing_side = Uniform(-1, 1)

# vessel particulars 
param own_vessel = Uniform('container_feeder', 'handysize_bulker',
                           'coastal_tanker', 'vlcc')
param tgt_vessel = Uniform('container_feeder', 'handysize_bulker',
                           'coastal_tanker', 'vlcc')

param own_speed_kn = Range(8, 16)
param tgt_speed_kn = Range(8, 16)

# COLREGS Rule 18 categories and Rule 19 visibility 
param own_status = 'power_driven'
param tgt_status = Discrete({'power_driven': 0.80,
                             'fishing': 0.10,
                             'restricted_manoeuvrability': 0.06,
                             'not_under_command': 0.04})
param visibility = Discrete({'good': 0.75, 'moderate': 0.18, 'restricted': 0.07})

#  target heading, conditioned on encounter type 
param head_on_heading    = Range(176, 184)
param overtaking_heading = Range(-12, 12)
param crossing_heading_s = Range(200, 340)   # target approaching from starboard
param crossing_heading_p = Range(20, 160)    # target approaching from port
param crossing_from_starboard = Discrete({1: 0.5, 0: 0.5})

# The Rule 13 speed constraint (an overtaking vessel must be materially
# faster than the vessel overtaken) is enforced by REPAIR in the Python
# bridge (see _scene_to_spec below), not by a `require` clause here. A
# `require` referencing globalParameters raised a NameError under the
# Scenic version this project targets, and rejection sampling would in any
# case make the acceptance rate depend on the encounter type and leave the
# search space disconnected -- repair keeps every draw usable.

# ---- a minimal concrete scene so the sample is inspectable in Scenic ----
class Vessel:
    width: 24
    length: 150
    allowCollisions: True
    requireVisible: False

ownShip = new Vessel at (0, 0), facing 0 deg
ego = ownShip
targetShip = new Vessel at (0, 3000), facing 180 deg
'''


class ScenicEncounterSampler:
    """
    Samples the abstract encounter space using Scenic, then instantiates the
    geometry through the same inverse solve as the analytic sampler.

    Scenic owns the *specification and sampling* of the scenario space; the
    domain bridge owns instantiation into the maritime simulator. This mirrors
    how Scenic is used with CARLA or Webots, where Scenic samples the scene and
    a simulator interface realises it -- no maritime simulator binding exists,
    so this project supplies one.
    """

    def __init__(self, scenic_path: str, seed: int = 0):
        import scenic                                   # noqa: F401  (import check)
        self.scenic_path = scenic_path
        self.seed = seed
        self._scenic = scenic

    @staticmethod
    def available() -> bool:
        try:
            import scenic  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def write_source(path: str) -> str:
        with open(path, "w") as fh:
            fh.write(SCENIC_SOURCE)
        return path

    def _scene_to_spec(self, params: dict, encounter: str, index: int) -> EncounterSpec:
        if encounter == HEAD_ON:
            hdg = float(params["head_on_heading"])
        elif encounter == OVERTAKING:
            hdg = float(params["overtaking_heading"])
        else:
            hdg = float(params["crossing_heading_s"]
                        if int(params["crossing_from_starboard"]) == 1
                        else params["crossing_heading_p"])
        own_speed = float(params["own_speed_kn"])
        tgt_speed = float(params["tgt_speed_kn"])
        if encounter == OVERTAKING and own_speed <= tgt_speed + 2.5:
            # Repair rather than reject: see the note above SCENIC_SOURCE.
            tgt_speed = max(own_speed - 3.0, 3.0)
        return EncounterSpec(
            encounter=encounter,
            dcpa_m=float(params["dcpa_m"]),
            tcpa_s=float(params["tcpa_s"]),
            own_speed_kn=own_speed,
            tgt_speed_kn=tgt_speed,
            passing_side=int(params["passing_side"]),
            own_vessel=str(params["own_vessel"]),
            tgt_vessel=str(params["tgt_vessel"]),
            own_status=str(params["own_status"]),
            tgt_status=str(params["tgt_status"]),
            visibility=str(params["visibility"]),
            tgt_heading_deg=hdg,
            seed=index,
        )

    def sample(self, encounter: str, n: int, verify: bool = True):
        scenario = self._scenic.scenarioFromFile(
            self.scenic_path, params={"encounter": encounter}, mode2D=True)
        out, attempts = [], 0
        while len(out) < n and attempts < 60 * n:
            attempts += 1
            scene, _ = scenario.generate(maxIterations=2000)
            spec = self._scene_to_spec(scene.params, encounter, len(out))
            try:
                own, tgt, diag = solve_initial_geometry(spec)
            except ValueError:
                continue
            if diag["initial_range_m"] < 800.0 or diag["initial_range_m"] > 30000.0:
                continue
            if verify and diag["classified"] != encounter:
                continue
            out.append((spec, own, tgt, diag))
        if len(out) < n:
            warnings.warn(f"Scenic produced only {len(out)}/{n} {encounter} scenarios")
        return out


def get_sampler(scenic_path: str | None = None, seed: int = 0, prefer_scenic=True):
    """
    Return (sampler, backend_name). Falls back to the analytic sampler.

    Compile-tests the scenario with a single probe sample before trusting it:
    constructing ScenicEncounterSampler only imports scenic and stores the
    path, so a scenario that fails to compile would otherwise still be
    reported as backend='scenic' and then raise the first time it is used.
    """
    if prefer_scenic and scenic_path and ScenicEncounterSampler.available():
        try:
            s = ScenicEncounterSampler(scenic_path, seed=seed)
            if s.sample("crossing", 1):
                return s, "scenic"
            warnings.warn("Scenic compiled but produced no samples; "
                          "using analytic sampler")
        except Exception as exc:                      # pragma: no cover
            warnings.warn(f"Scenic unusable ({exc}); using analytic sampler")
    return AnalyticEncounterSampler(seed=seed), "analytic"


__all__ = ["EncounterSpec", "solve_initial_geometry", "AnalyticEncounterSampler",
           "ScenicEncounterSampler", "get_sampler", "SCENIC_SOURCE", "VISIBILITY"]
