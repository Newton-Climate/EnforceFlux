"""Unit tests for enforceflux.inversion and enforceflux.flux."""
import numpy as np
import pytest

from enforceflux.core.base import FluxResult, IFluxEstimator
from enforceflux.inversion import (
    InversionResult, bayesian_linear_inversion, bounded_bayesian_linear_inversion,
    oe_from_linear, optimize_oe,
)
from enforceflux.utils.plugin_registry import get_plugin


def _linear_problem():
    g = np.array([[1.0], [1.0]])
    y = np.array([10.0, 12.0])
    x_prior = np.array([5.0])
    s_a = np.array([[100.0]])
    r = np.eye(2) * 1.0
    return g, y, x_prior, s_a, r


def test_bayesian_linear_inversion_shapes():
    g, y, x_prior, s_a, r = _linear_problem()
    result = bayesian_linear_inversion(g=g, y=y, x_prior=x_prior, s_a=s_a, r=r)
    assert isinstance(result, InversionResult)
    assert result.x_posterior.shape == (1,)
    assert result.posterior_cov.shape == (1, 1)
    assert result.fisher_information is not None
    assert result.residual is not None
    # posterior should move toward the observations, away from the (loose) prior
    assert abs(result.x_posterior[0] - 11.0) < abs(x_prior[0] - 11.0)


def test_oe_from_linear_matches_bayesian():
    g, y, x_prior, s_a, r = _linear_problem()
    bayes = bayesian_linear_inversion(g=g, y=y, x_prior=x_prior, s_a=s_a, r=r)
    oe = oe_from_linear(G=g, y=y, x_prior=x_prior, Sa=s_a, Se=r)
    np.testing.assert_allclose(oe.x_posterior, bayes.x_posterior)
    np.testing.assert_allclose(oe.posterior_cov, bayes.posterior_cov)


def test_bounded_bayesian_inversion_enforces_nonnegative_emissions():
    result = bounded_bayesian_linear_inversion(
        g=np.eye(2), y=np.array([-3.0, 2.0]), x_prior=np.zeros(2),
        s_a=np.eye(2) * 1e6, r=np.eye(2) * 1e-6,
    )
    assert result.converged
    assert np.all(result.x_posterior >= 0.0)
    assert np.isclose(result.x_posterior[0], 0.0, atol=1e-9)
    assert np.isclose(result.x_posterior[1], 2.0, atol=1e-5)


def test_bounded_bayesian_inversion_keeps_zero_operator_at_its_prior():
    result = bounded_bayesian_linear_inversion(
        g=np.zeros((2, 3)), y=np.array([1.0e6, 2.0e6]), x_prior=np.zeros(3),
        s_a=np.eye(3) * 1e-3, r=np.eye(2) * 1e-6,
    )
    assert result.converged
    np.testing.assert_allclose(result.x_posterior, 0.0)


def test_bayesian_linear_inversion_is_stable_for_extreme_operator_scaling():
    """Regression test for GP OSSE operators that overflow precision space."""
    g = np.array([[1.0e100, 0.0], [0.0, 1.0e100]])
    truth = np.array([1.0e-2, 2.0e-2])
    result = bayesian_linear_inversion(
        g=g,
        y=g @ truth,
        x_prior=np.zeros(2),
        s_a=np.array([[1.0e-4, 5.0e-5], [5.0e-5, 1.0e-4]]),
        r=np.eye(2),
    )
    assert np.all(np.isfinite(result.x_posterior))
    assert np.all(np.isfinite(result.posterior_cov))
    np.testing.assert_allclose(result.x_posterior, truth, rtol=1e-12, atol=1e-12)


def test_optimize_oe_linear_case_converges():
    g, y, x_prior, s_a, r = _linear_problem()
    result = optimize_oe(F=lambda x: g @ x, y=y, x_prior=x_prior, Sa=s_a, Se=r)
    assert isinstance(result, InversionResult)
    assert result.converged
    linear = oe_from_linear(G=g, y=y, x_prior=x_prior, Sa=s_a, Se=r)
    np.testing.assert_allclose(result.x_posterior, linear.x_posterior, atol=1e-4)


def test_inversion_flux_estimator_via_registry():
    g, y, x_prior, s_a, r = _linear_problem()
    estimator = get_plugin("enforceflux.flux", "inversion", IFluxEstimator)()
    result = estimator.estimate(
        {"g": g, "y": y, "x_prior": x_prior, "s_a": s_a, "r": r}, {}
    )
    assert isinstance(result, FluxResult)
    assert isinstance(result.meta["inversion"], InversionResult)
    np.testing.assert_allclose(result.flux, result.meta["inversion"].x_posterior)


@pytest.mark.parametrize("name", ["flux_gradient"])
def test_unimplemented_flux_estimators_raise(name):
    estimator = get_plugin("enforceflux.flux", name, IFluxEstimator)()
    with pytest.raises(NotImplementedError):
        estimator.estimate({}, {})
