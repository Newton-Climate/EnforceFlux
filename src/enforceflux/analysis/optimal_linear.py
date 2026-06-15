"""Linear optimal-estimation wrapper around Bayesian linear inversion."""
from __future__ import annotations

import numpy as np

from enforceflux.analysis.optimal_models import OEResult
from enforceflux.retrieval.inversion import bayesian_linear_inversion


def oe_from_linear(
    G: np.ndarray,
    y: np.ndarray,
    x_prior: np.ndarray,
    Sa: np.ndarray,
    Se: np.ndarray,
    source_names: list[str] | None = None,
) -> OEResult:
    """Run OE for a linear forward model F(x) = G @ x."""
    G = np.asarray(G, dtype=float)
    xa = np.asarray(x_prior, dtype=float)

    Sa_full = np.diag(np.asarray(Sa)) if np.asarray(Sa).ndim == 1 else np.asarray(Sa)
    Se_full = np.diag(np.asarray(Se)) if np.asarray(Se).ndim == 1 else np.asarray(Se)

    result = bayesian_linear_inversion(
        g=G, y=y, x_prior=xa, s_a=Sa_full, r=Se_full
    )

    y_prior = G @ xa
    y_opt = G @ result.x_posterior

    return OEResult(
        x_opt=result.x_posterior,
        x_prior=xa,
        Sx=result.posterior_cov,
        averaging_kernel=result.averaging_kernel,
        y_obs=np.asarray(y, dtype=float),
        y_prior=y_prior,
        y_opt=y_opt,
        cost_history=[],
        converged=True,
        n_iter=1,
        source_names=source_names,
    )
