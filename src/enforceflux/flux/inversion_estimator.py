"""Flux estimator that delegates to a transport-inversion engine."""
from typing import Any

import numpy as np

from enforceflux.core.base import FluxResult, IFluxEstimator, IInversionEngine
from enforceflux.utils.plugin_registry import get_plugin, normalize_plugin_name


class InversionFluxEstimator(IFluxEstimator):
    """Estimates flux by inverting observations against a transport Jacobian.

    Expects ``observations`` to be a mapping with keys ``g`` (Jacobian),
    ``y`` (observations), ``x_prior``, ``s_a`` (prior covariance), and ``r``
    (observation covariance). ``config["engine"]`` selects the
    ``enforceflux.inversion`` plugin (default ``"bayesian"``).
    """

    def estimate(self, observations: Any, config: dict[str, Any]) -> FluxResult:
        engine_name = normalize_plugin_name(
            "enforceflux.inversion", config.get("engine", "bayesian")
        )
        engine_cls = get_plugin("enforceflux.inversion", engine_name, IInversionEngine)
        engine = engine_cls()

        result = engine.invert(
            g=np.asarray(observations["g"], dtype=float),
            y=np.asarray(observations["y"], dtype=float),
            x_prior=np.asarray(observations["x_prior"], dtype=float),
            s_a=observations["s_a"],
            r=observations["r"],
        )

        return FluxResult(flux=np.asarray(result.x_posterior, dtype=float), meta={"inversion": result})
