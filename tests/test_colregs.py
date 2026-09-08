"""
COLREGS encounter classification and role assignment: Rules 13-15 (steering
rules by encounter geometry) and Rule 18 (precedence by vessel category).
"""
import numpy as np
import pytest

from mcas.vessels import ShipState, VesselSpec
from mcas.colregs import (
    classify, assign_role,
    HEAD_ON, CROSSING, OVERTAKING, OVERTAKEN,
    GIVE_WAY, STAND_ON, BOTH_GIVE_WAY,
)

FEEDER = VesselSpec("feeder", length_m=150.0, beam_m=24.0)


def ship(x, y, heading_deg, speed_kn=12.0, status="power_driven"):
    spec = VesselSpec(**{**FEEDER.__dict__, "status": status})
    return ShipState(x, y, np.radians(heading_deg), speed_kn, spec=spec)


class TestClassification:
    def test_reciprocal_headings_is_head_on(self):
        own = ship(0, 0, 0)
        tgt = ship(0, 5000, 180)
        assert classify(own, tgt) == HEAD_ON

    def test_crossing_from_starboard(self):
        own = ship(0, 0, 0)
        tgt = ship(3000, 3000, 270)   # approaching from the starboard bow
        assert classify(own, tgt) == CROSSING

    def test_faster_vessel_overtaking_from_astern(self):
        """Own-ship overtakes a slower vessel dead ahead, same heading."""
        own = ship(0, 0, 0, speed_kn=16.0)
        tgt = ship(0, 1000, 0, speed_kn=8.0)
        assert classify(own, tgt) == OVERTAKING

    def test_slower_vessel_is_being_overtaken(self):
        """From the slower vessel's own perspective, she is the one being
        overtaken -- the classification is symmetric in the right way."""
        own = ship(0, 0, 0, speed_kn=8.0)
        tgt = ship(0, -1000, 0, speed_kn=16.0)
        assert classify(own, tgt) == OVERTAKEN

    def test_overtaking_persists_even_as_bearing_opens(self):
        """Rule 13: a vessel that starts an overtake stays the overtaking
        vessel even if her bearing later draws into what would otherwise
        look like a crossing sector, as long as headings are still close
        to parallel."""
        own = ship(0, 0, 0, speed_kn=16.0)
        tgt = ship(200, 900, 5, speed_kn=8.0)   # bearing has opened, but
        assert classify(own, tgt) == OVERTAKING  # heading is still ~parallel


class TestRoleAssignment:
    def test_head_on_is_both_give_way(self):
        own, tgt = ship(0, 0, 0), ship(0, 5000, 180)
        assert assign_role(own, tgt) == BOTH_GIVE_WAY

    def test_crossing_target_on_starboard_gives_way(self):
        """Rule 15: the vessel with the other on her own starboard side
        keeps out of the way."""
        own = ship(0, 0, 0)
        tgt = ship(3000, 3000, 270)
        assert assign_role(own, tgt) == GIVE_WAY

    def test_crossing_target_on_port_stands_on(self):
        own = ship(0, 0, 0)
        tgt = ship(-3000, 3000, 90)
        assert assign_role(own, tgt) == STAND_ON

    def test_overtaking_vessel_gives_way(self):
        own = ship(0, 0, 0, speed_kn=16.0)
        tgt = ship(0, 1000, 0, speed_kn=8.0)
        assert assign_role(own, tgt) == GIVE_WAY

    def test_overtaken_vessel_stands_on(self):
        own = ship(0, 0, 0, speed_kn=8.0)
        tgt = ship(0, -1000, 0, speed_kn=16.0)
        assert assign_role(own, tgt) == STAND_ON

    def test_rule_18_overrides_geometry_for_fishing_vessel(self):
        """A power-driven vessel keeps out of the way of a fishing vessel
        regardless of the crossing geometry -- Rule 18 takes precedence
        over Rules 13-15."""
        own = ship(0, 0, 0, status="power_driven")
        # geometry alone would make own-ship stand-on (target on her port
        # side) but Rule 18 must override that because target is fishing
        tgt = ship(-3000, 3000, 90, status="fishing")
        assert assign_role(own, tgt) == GIVE_WAY

    def test_rule_18_not_under_command_outranks_fishing(self):
        own = ship(0, 0, 0, status="fishing")
        tgt = ship(-3000, 3000, 90, status="not_under_command")
        assert assign_role(own, tgt) == GIVE_WAY

    def test_equal_status_falls_back_to_geometry(self):
        """When both vessels share the same Rule 18 category, the steering
        rules (13-15) decide the role, not Rule 18."""
        own = ship(0, 0, 0, status="fishing")
        tgt = ship(-3000, 3000, 90, status="fishing")
        assert assign_role(own, tgt) == STAND_ON
