"""Eddy covariance flux estimator (not yet implemented)."""
from typing import Any

from enforceflux.core.base import FluxResult, IFluxEstimator


class EddyCovarianceFluxEstimator(IFluxEstimator):
    """Computes flux from high-frequency turbulence covariances (w', c').

    Not yet implemented — requires raw sonic anemometer / gas analyzer time
    series and a turbulence-flux pipeline (despiking, coordinate rotation,
    spectral correction) distinct from the transport-inversion estimators.
    """

    def estimate(self, observations: Any, config: dict[str, Any]) -> FluxResult:
        raise NotImplementedError(
            "EddyCovarianceFluxEstimator is a stub; eddy covariance flux "
            "computation is not yet implemented."
        )
