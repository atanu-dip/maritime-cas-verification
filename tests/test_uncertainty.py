"""
Split conformal prediction: the one property that actually matters is the
finite-sample coverage guarantee -- P(true value >= lower bound) >= 1 - alpha
-- and it must hold even when the underlying errors are not Gaussian, which
is the whole point of using conformal prediction here instead of a raw
Gaussian uncertainty band (see Phase 2b: measured residuals were badly
non-Gaussian, mean 4.18 / sd 26.43 in standardised units).
"""
import numpy as np
import pytest

from mcas.uncertainty import ConformalDCPA


def _synthetic_calibration_data(n, sigma_scale=1.0, seed=0, heavy_tailed=False):
    """Predicted DCPA, its sigma, and the true DCPA, with heteroscedastic
    (range-dependent) noise -- deliberately not Gaussian when
    heavy_tailed=True, to test that the guarantee is genuinely
    distribution-free rather than only working by accident on nice data."""
    rng = np.random.default_rng(seed)
    true = rng.uniform(50, 3000, n)
    sigma = sigma_scale * (50 + 0.15 * true)
    if heavy_tailed:
        noise = rng.standard_t(df=3, size=n) * sigma
    else:
        noise = rng.normal(0, sigma, n)
    pred = true + noise
    return pred, sigma, true


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
@pytest.mark.parametrize("heavy_tailed", [False, True])
def test_empirical_coverage_meets_target(alpha, heavy_tailed):
    pred, sigma, true = _synthetic_calibration_data(4000, heavy_tailed=heavy_tailed, seed=1)
    n = len(pred)
    idx = np.random.default_rng(0).permutation(n)
    cal, test = idx[: n // 2], idx[n // 2 :]

    cf = ConformalDCPA(alpha=alpha).calibrate(pred[cal], sigma[cal], true[cal])
    coverage = cf.coverage(pred[test], sigma[test], true[test])

    # finite-sample guarantee: coverage should meet the target within a
    # small slack for sampling noise at this test-set size
    assert coverage >= (1 - alpha) - 0.03, (
        f"empirical coverage {coverage:.3f} fell materially below the "
        f"{1 - alpha:.2f} target (heavy_tailed={heavy_tailed})"
    )


def test_lower_bound_is_never_above_the_point_estimate():
    pred, sigma, true = _synthetic_calibration_data(1000, seed=2)
    cf = ConformalDCPA(alpha=0.10).calibrate(pred, sigma, true)
    bound = cf.lower_bound(pred, sigma)
    assert np.all(bound <= pred + 1e-6)


def test_tighter_alpha_gives_a_wider_bound():
    """A smaller alpha (higher confidence) must be at least as conservative
    -- the bound should not get looser as the required coverage increases."""
    pred, sigma, true = _synthetic_calibration_data(2000, seed=3)
    cal = slice(0, 1000)
    cf_loose = ConformalDCPA(alpha=0.20).calibrate(pred[cal], sigma[cal], true[cal])
    cf_tight = ConformalDCPA(alpha=0.05).calibrate(pred[cal], sigma[cal], true[cal])
    assert cf_tight.q >= cf_loose.q


def test_raises_with_too_few_calibration_points():
    cf = ConformalDCPA(alpha=0.10)
    with pytest.raises(ValueError):
        cf.calibrate([100.0, 200.0], [10.0, 10.0], [90.0, 210.0])
