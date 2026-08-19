"""Unified result container for inversion algorithms (linear and nonlinear)."""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class InversionResult:
    """Posterior result of an inversion (Bayesian linear or nonlinear OE).

    ``fisher_information`` and ``residual`` are only populated by the linear
    Bayesian engine, where they fall out of the closed-form solution.
    """

    x_posterior: np.ndarray
    x_prior: np.ndarray
    posterior_cov: np.ndarray
    averaging_kernel: np.ndarray
    y_obs: np.ndarray
    y_prior: np.ndarray
    y_posterior: np.ndarray
    fisher_information: np.ndarray | None = None
    residual: np.ndarray | None = None
    cost_history: list = field(default_factory=list)
    converged: bool = True
    n_iter: int = 1
    source_names: list | None = None
    # --- source-heterogeneity OSSE (M8) ---
    posterior_beta_mean: np.ndarray | None = None
    posterior_beta_cov: np.ndarray | None = None
    # --- end source-heterogeneity OSSE (M8) ---
