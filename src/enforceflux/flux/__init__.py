"""enforceflux.flux — user-selectable flux estimation methods.

A flux estimate can come from fundamentally different paradigms:

- transport inversion (Bayesian linear or nonlinear OE against a Jacobian),
  delegating to an ``enforceflux.inversion`` engine (:class:`InversionFluxEstimator`)
- local micrometeorological methods that need no transport model at all,
  e.g. eddy covariance or flux-gradient (stubs for now; see
  :mod:`enforceflux.flux.eddy_covariance`, :mod:`enforceflux.flux.flux_gradient`)

Each is registered under the ``enforceflux.flux`` entry-point group as an
:class:`~enforceflux.core.base.IFluxEstimator`.
"""
from enforceflux.flux.inversion_estimator import InversionFluxEstimator
from enforceflux.flux.eddy_covariance import EddyCovarianceFluxEstimator, EddyCovarianceWindow
from enforceflux.flux.flux_gradient import FluxGradientFluxEstimator

__all__ = [
    "InversionFluxEstimator",
    "EddyCovarianceFluxEstimator",
    "EddyCovarianceWindow",
    "FluxGradientFluxEstimator",
]
