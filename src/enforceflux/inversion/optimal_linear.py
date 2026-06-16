"""Linear optimal-estimation entry point (thin wrapper over Bayesian linear inversion)."""
import numpy as np

from enforceflux.inversion.bayesian import bayesian_linear_inversion
from enforceflux.inversion.result import InversionResult


def oe_from_linear(
    G: np.ndarray,
    y: np.ndarray,
    x_prior: np.ndarray,
    Sa: np.ndarray,
    Se: np.ndarray,
    source_names: list[str] | None = None,
) -> InversionResult:
    """Run OE for a linear forward model F(x) = G @ x."""
    return bayesian_linear_inversion(
        g=np.asarray(G, dtype=float),
        y=y,
        x_prior=np.asarray(x_prior, dtype=float),
        s_a=Sa,
        r=Se,
        source_names=source_names,
    )
