"""Unit tests for enforceflux.inversion and enforceflux.flux."""
import numpy as np
import pytest

from enforceflux.core.base import FluxResult, IFluxEstimator
from enforceflux.inversion import InversionResult, bayesian_linear_inversion, oe_from_linear, optimize_oe
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
