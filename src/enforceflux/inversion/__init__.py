"""enforceflux.inversion — Bayesian linear and nonlinear optimal-estimation algorithms.

These are the underlying math routines used by ``IInversionEngine`` registry
plugins (see ``enforceflux.plugins.inversion_*``) and by the ``IFluxEstimator``
inversion-based estimator (see ``enforceflux.flux``).
"""
from enforceflux.inversion.result import InversionResult
from enforceflux.inversion.bayesian import bayesian_linear_inversion
from enforceflux.inversion.optimal_linear import oe_from_linear
from enforceflux.inversion.optimal_nonlinear import optimize_oe

__all__ = [
    "InversionResult",
    "bayesian_linear_inversion",
    "oe_from_linear",
    "optimize_oe",
]
