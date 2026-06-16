"""Dataclasses for information-theoretic inversion analysis results."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FisherResult:
    """Fisher Information Matrix decomposed by observation group."""

    FIM_total: np.ndarray
    FIM_per_group: dict[str, np.ndarray]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    group_names: list[str]
    n_obs_per_group: dict[str, int] = field(default_factory=dict)


@dataclass
class DofResult:
    """Averaging kernel and degrees of freedom for signal."""

    dfs_total: float
    dfs_per_group: dict[str, float]
    dfs_per_source: np.ndarray
    averaging_kernel: np.ndarray
    averaging_kernel_per_group: dict[str, np.ndarray]
    source_names: list[str] | None = None


@dataclass
class PosteriorResult:
    """Gaussian posterior covariance and uncertainty reduction."""

    Sx: np.ndarray
    posterior_sigma: np.ndarray
    prior_sigma: np.ndarray
    uncertainty_reduction: np.ndarray
    correlation_matrix: np.ndarray
    Sx_per_group: dict[str, np.ndarray]
    source_names: list[str] | None = None


@dataclass
class AblationResult:
    """Result for a single observation-group scenario in an ablation study."""

    scenario: str
    groups: list[str]
    fisher: FisherResult
    dof: DofResult
    posterior: PosteriorResult
    dfs_total: float
