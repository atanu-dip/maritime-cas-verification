"""
Ship manoeuvring dynamics, validated against the IMO Standards for Ship
Manoeuvrability (Res. MSC.137(76)): advance <= 4.5 ship lengths and tactical
diameter <= 5.0 ship lengths on a standard 35-degree turning circle test.

This is the check behind every "PASS" in Phase 1's turning-circle validation
table -- run here as an actual assertion per hull, not just a printed table.
"""
import pytest

from mcas.vessels import VESSEL_LIBRARY, turning_circle


@pytest.mark.parametrize("name", list(VESSEL_LIBRARY.keys()))
def test_turning_circle_meets_imo_criteria(name):
    spec = VESSEL_LIBRARY[name]
    result = turning_circle(spec)
    assert result["advance_L"] <= 4.5, (
        f"{name}: advance {result['advance_L']:.2f} L exceeds IMO limit of 4.5 L"
    )
    assert result["tactical_diameter_L"] <= 5.0, (
        f"{name}: tactical diameter {result['tactical_diameter_L']:.2f} L "
        f"exceeds IMO limit of 5.0 L"
    )


def test_all_five_hull_types_present():
    """The validated fleet spans a 45 m trawler to a 330 m VLCC -- the result
    is not tuned to one ship size."""
    assert len(VESSEL_LIBRARY) == 5
    lengths = sorted(v.length_m for v in VESSEL_LIBRARY.values())
    assert lengths[0] < 60          # smallest hull is a small craft
    assert lengths[-1] > 300        # largest hull is a very large vessel


def test_larger_vessels_turn_less_tightly():
    """Sanity check on the dynamics model itself: a VLCC should have a wider
    tactical diameter, in absolute metres, than a small trawler."""
    trawler = turning_circle(VESSEL_LIBRARY["trawler"])
    vlcc = turning_circle(VESSEL_LIBRARY["vlcc"])
    trawler_diam_m = trawler["tactical_diameter_L"] * trawler["L"]
    vlcc_diam_m = vlcc["tactical_diameter_L"] * vlcc["L"]
    assert vlcc_diam_m > trawler_diam_m
