"""
Geometry correctness: closest-point-of-approach, the inverse DCPA/TCPA solver,
and the Fujii ship domain.

These are the checks that appear inline throughout the notebooks (e.g. the
"max |DCPA err| (m)" tables in Phase 1's scenario-verification cells) --
extracted here as real, independently runnable pytest tests rather than
notebook-only print statements.
"""
import numpy as np
import pytest

from mcas.geometry import cpa, wrap_pi, FujiiDomain, sector_name
from mcas.scenarios import EncounterSpec, solve_initial_geometry, AnalyticEncounterSampler


class TestCPA:
    def test_head_on_closing(self):
        """Two vessels approaching directly should show a decreasing range."""
        own_pos, own_vel = np.array([0.0, 0.0]), np.array([0.0, 5.0])
        tgt_pos, tgt_vel = np.array([0.0, 1000.0]), np.array([0.0, -5.0])
        t, d = cpa(own_pos, own_vel, tgt_pos, tgt_vel)
        assert t == pytest.approx(100.0, abs=1e-9)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_parallel_same_speed_constant_range(self):
        """Two vessels on parallel tracks at equal speed never close -- CPA
        is now, at the current range, not at some future or past time."""
        own_pos, own_vel = np.array([0.0, 0.0]), np.array([0.0, 5.0])
        tgt_pos, tgt_vel = np.array([500.0, 0.0]), np.array([0.0, 5.0])
        t, d = cpa(own_pos, own_vel, tgt_pos, tgt_vel)
        assert t == pytest.approx(0.0, abs=1e-9)
        assert d == pytest.approx(500.0, abs=1e-6)

    def test_negative_tcpa_clips_to_zero(self):
        """A target already past its closest approach should report the
        CURRENT range, not extrapolate backwards in time."""
        own_pos, own_vel = np.array([0.0, 0.0]), np.array([0.0, 5.0])
        tgt_pos, tgt_vel = np.array([0.0, -1000.0]), np.array([0.0, -5.0])
        t, d = cpa(own_pos, own_vel, tgt_pos, tgt_vel)
        assert t == 0.0
        assert d == pytest.approx(1000.0, abs=1e-6)


class TestInverseSolver:
    """
    The core design claim of the whole project: an encounter is specified by
    its DCPA/TCPA and the initial geometry is solved BACKWARDS to realise it
    exactly, rather than being sampled directly and hoped for.
    """

    @pytest.mark.parametrize("encounter", ["head_on", "crossing", "overtaking"])
    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_exact_to_machine_precision(self, encounter, seed):
        sampler = AnalyticEncounterSampler(seed=seed)
        for spec, own, tgt, diag in sampler.sample(encounter, 5):
            assert diag["dcpa_error_m"] < 1e-6, (
                f"DCPA error {diag['dcpa_error_m']} m exceeds tolerance "
                f"for {encounter} (seed={seed})"
            )
            assert diag["tcpa_error_s"] < 1e-6, (
                f"TCPA error {diag['tcpa_error_s']} s exceeds tolerance "
                f"for {encounter} (seed={seed})"
            )

    def test_realised_geometry_matches_requested_encounter_type(self):
        """
        Regression test for the original Phase 1 bug: sampling positions
        directly could place an 'overtaken' vessel astern of a faster
        own-ship, producing two ships that only diverge -- a scenario that
        silently fails to be an overtaking encounter at all. The inverse
        solver must make this structurally impossible.
        """
        sampler = AnalyticEncounterSampler(seed=3)
        for encounter in ("head_on", "crossing", "overtaking"):
            for spec, own, tgt, diag in sampler.sample(encounter, 20):
                assert diag["classified"] == encounter, (
                    f"requested {encounter} but geometry solved to "
                    f"{diag['classified']}"
                )

    def test_overtaking_speed_repair(self):
        """Rule 13 requires the overtaking vessel to be materially faster;
        the sampler must repair violating draws rather than silently emit
        an invalid overtaking scenario."""
        sampler = AnalyticEncounterSampler(seed=11)
        for spec, own, tgt, diag in sampler.sample("overtaking", 30):
            assert spec.own_speed_kn > spec.tgt_speed_kn + 2.4


class TestFujiiDomain:
    def test_own_position_is_inside_domain(self):
        """A vessel is always inside its own domain at zero range."""
        domain = FujiiDomain(length_m=150.0)
        assert domain.intrusion(np.array([0.0, 0.0]), 0.0, np.array([0.0, 0.0])) < 1.0

    def test_far_away_target_is_outside_domain(self):
        domain = FujiiDomain(length_m=150.0)
        far = np.array([0.0, 20 * 150.0])
        assert domain.intrusion(np.array([0.0, 0.0]), 0.0, far) > 1.0

    def test_domain_longer_ahead_than_abeam(self):
        """The Fujii domain is an ellipse extended ahead (8L) more than
        abeam (3.2L); a point exactly ahead at 3.5L should intrude, the
        same distance directly abeam should not."""
        domain = FujiiDomain(length_m=100.0)
        ahead = np.array([0.0, 350.0])     # 3.5 L ahead
        abeam = np.array([350.0, 0.0])     # 3.5 L abeam
        assert domain.intrusion(np.array([0.0, 0.0]), 0.0, ahead) < 1.0
        assert domain.intrusion(np.array([0.0, 0.0]), 0.0, abeam) > 1.0


class TestSectorGeometry:
    def test_dead_ahead(self):
        assert sector_name(0.0) == "ahead"

    def test_dead_astern(self):
        assert sector_name(np.radians(180)) == "astern"

    def test_starboard_bow(self):
        assert "starboard" in sector_name(np.radians(45))

    def test_port_quarter(self):
        name = sector_name(np.radians(-135))
        assert "port" in name and "quarter" in name
