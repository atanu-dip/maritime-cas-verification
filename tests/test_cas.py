"""
Collision avoidance strategies: the baseline must never manoeuvre, and the
rule-compliant strategy must alter to starboard when obliged to give way.

These are lightweight behavioural checks on one decision, not full closed-
loop runs (see notebooks/ for the full Monte Carlo campaigns) -- fast enough
to run on every commit in CI.
"""
import numpy as np
import pytest

from mcas.vessels import ShipState, VesselSpec
from mcas.geometry import wrap_pi
from mcas.cas import NoAction, RuleBasedCOLREGS, CAS_REGISTRY

FEEDER = VesselSpec("feeder", length_m=150.0, beam_m=24.0)


def ship(x, y, heading_deg, speed_kn=12.0):
    return ShipState(x, y, np.radians(heading_deg), speed_kn, spec=FEEDER)


class TestNoAction:
    def test_never_alters_course_or_speed(self):
        own = ship(0, 0, 0, speed_kn=12.0)
        tgt = ship(0, 500, 180, speed_kn=12.0)   # dangerously close head-on
        cas = NoAction(dcpa_limit=926.0)
        cas.reset(own)
        heading, speed = cas.decide(own, tgt, 0.0, 0)
        assert heading == pytest.approx(own.heading)
        assert speed == pytest.approx(own.speed_kn)


class TestRuleBasedCOLREGS:
    def test_give_way_vessel_alters_to_starboard(self):
        """Rule 15: own-ship has the target on her starboard bow, close
        enough for genuine risk (DCPA well inside the safety limit), and
        must give way. The commanded heading should be a starboard
        (clockwise, positive) alteration from her base course."""
        own = ship(0, 0, 0, speed_kn=12.0)
        tgt = ship(600, 1600, 270, speed_kn=12.0)  # crossing from starboard,
        cas = RuleBasedCOLREGS(dcpa_limit=926.0)   # DCPA well under 926 m
        cas.reset(own)
        heading, speed = cas.decide(own, tgt, 0.0, 0)
        alteration = float(wrap_pi(heading - own.heading))
        assert alteration > np.radians(10.0), (
            "give-way vessel should make a substantial starboard alteration"
        )

    def test_stand_on_vessel_holds_course_when_not_close_quarters(self):
        """Rule 17(a): a stand-on vessel keeps her course and speed while
        the situation is not yet close-quarters."""
        own = ship(0, 0, 0, speed_kn=12.0)
        tgt = ship(-1500, 3000, 90, speed_kn=12.0)  # own-ship stands on
        cas = RuleBasedCOLREGS(dcpa_limit=926.0)
        cas.reset(own)
        heading, speed = cas.decide(own, tgt, 0.0, 0)
        assert heading == pytest.approx(own.heading, abs=1e-9)


def test_registry_contains_all_four_baseline_strategies():
    assert set(CAS_REGISTRY.keys()) == {
        "no_action", "rule_based", "velocity_obstacle", "mpc"
    }


@pytest.mark.parametrize("key", list(CAS_REGISTRY.keys()))
def test_every_strategy_returns_a_finite_heading_and_speed(key):
    """Smoke test: every registered strategy must return well-formed output
    for a generic encounter, not NaN or an exception."""
    own = ship(0, 0, 0, speed_kn=12.0)
    tgt = ship(1000, 2000, 270, speed_kn=10.0)
    cas = CAS_REGISTRY[key](dcpa_limit=926.0)
    cas.reset(own)
    heading, speed = cas.decide(own, tgt, 0.0, 0)
    assert np.isfinite(heading)
    assert np.isfinite(speed)
    assert speed > 0
