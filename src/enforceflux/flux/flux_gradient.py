"""Flux-gradient (aerodynamic/profile method) flux estimator (not yet implemented)."""
from typing import Any

from enforceflux.core.base import FluxResult, IFluxEstimator


class FluxGradientFluxEstimator(IFluxEstimator):
    """Computes flux from vertical concentration gradients and eddy diffusivity.

    Not yet implemented — requires multi-height concentration profiles and a
    stability-dependent eddy diffusivity parameterization, distinct from the
    transport-inversion estimators.
    """

    def estimate(self, observations: Any, config: dict[str, Any]) -> FluxResult:
        raise NotImplementedError(
            "FluxGradientFluxEstimator is a stub; flux-gradient flux "
            "computation is not yet implemented."
        )
