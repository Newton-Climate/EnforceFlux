"""M8 — background nuisance parameters in bayesian_linear_inversion."""
import numpy as np
import pytest

from enforceflux.inversion.bayesian import bayesian_linear_inversion


def _tiny_system(seed=0):
    rng = np.random.default_rng(seed)
    m, n = 12, 3
    g = rng.normal(size=(m, n))
    x_true = np.array([1.0, -0.5, 2.0])
    r_diag = np.full(m, 0.05 ** 2)
    y = g @ x_true + rng.normal(scale=0.05, size=m)
    x_prior = np.zeros(n)
    s_a = np.diag(np.full(n, 4.0))
    return g, y, x_prior, s_a, r_diag, x_true


def test_no_background_matches_current():
    g, y, xp, sa, r, _ = _tiny_system()
    base = bayesian_linear_inversion(g=g, y=y, x_prior=xp, s_a=sa, r=r)
    passthrough = bayesian_linear_inversion(
        g=g, y=y, x_prior=xp, s_a=sa, r=r, g_beta=None, sigma_beta=None,
    )
    np.testing.assert_array_equal(base.x_posterior, passthrough.x_posterior)
    np.testing.assert_array_equal(base.posterior_cov, passthrough.posterior_cov)
    assert base.posterior_beta_mean is None
    assert base.posterior_beta_cov is None


def test_zero_sigma_pins_beta():
    g, y, xp, sa, r, _ = _tiny_system(seed=1)
    base = bayesian_linear_inversion(g=g, y=y, x_prior=xp, s_a=sa, r=r)
    g_beta = np.ones((g.shape[0], 1))
    pinned = bayesian_linear_inversion(
        g=g, y=y, x_prior=xp, s_a=sa, r=r,
        g_beta=g_beta, sigma_beta=np.array([1e-8]),
    )
    np.testing.assert_allclose(pinned.x_posterior, base.x_posterior, atol=1e-6)
    assert pinned.posterior_beta_mean is not None
    assert pinned.posterior_beta_mean.shape == (1,)
    assert abs(float(pinned.posterior_beta_mean[0])) < 1e-6


def test_absorbs_constant_offset():
    g, y_clean, xp, sa, r, x_true = _tiny_system(seed=2)
    offset = 3.75
    y_off = y_clean + offset

    baseline = bayesian_linear_inversion(g=g, y=y_clean, x_prior=xp, s_a=sa, r=r)
    joint = bayesian_linear_inversion(
        g=g, y=y_off, x_prior=xp, s_a=sa, r=r,
        g_beta=np.ones((g.shape[0], 1)), sigma_beta=np.array([100.0]),
    )
    np.testing.assert_allclose(joint.x_posterior, baseline.x_posterior, atol=5e-3)
    assert joint.posterior_beta_mean is not None
    assert joint.posterior_beta_mean.shape == (1,)
    np.testing.assert_allclose(float(joint.posterior_beta_mean[0]), offset, atol=5e-2)


def test_shape_validation():
    g, y, xp, sa, r, _ = _tiny_system(seed=3)
    with pytest.raises(ValueError):
        bayesian_linear_inversion(
            g=g, y=y, x_prior=xp, s_a=sa, r=r,
            g_beta=np.ones((g.shape[0] + 1, 1)), sigma_beta=np.array([1.0]),
        )
    with pytest.raises(ValueError):
        bayesian_linear_inversion(
            g=g, y=y, x_prior=xp, s_a=sa, r=r,
            g_beta=np.ones((g.shape[0], 2)), sigma_beta=np.array([1.0]),
        )
