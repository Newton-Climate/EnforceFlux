from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResults:
    fisher_information: np.ndarray
    null_space_dimension: int
    jacobian_rank: int
    posterior_cov: np.ndarray
    posterior_std: np.ndarray
    averaging_kernel: np.ndarray
    condition_number: float


def compute_metrics(g: np.ndarray, fisher: np.ndarray, posterior_cov: np.ndarray, averaging_kernel: np.ndarray, r_cond: float) -> MetricResults:
    g = np.asarray(g)
    fisher = np.asarray(fisher)
    posterior_cov = np.asarray(posterior_cov)
    averaging_kernel = np.asarray(averaging_kernel)

    u, s, v = np.linalg.svd(g, full_matrices=False)
    if s.size == 0:
        rank = 0
    else:
        tol = r_cond * s.max()
        rank = int((s > tol).sum())

    null_space_dimension = g.shape[1] - rank
    condition_number = float(s.max() / s.min()) if s.size > 0 and s.min() > 0 else float("inf")
    posterior_std = np.sqrt(np.diag(posterior_cov))

    return MetricResults(
        fisher_information=fisher,
        null_space_dimension=null_space_dimension,
        jacobian_rank=rank,
        posterior_cov=posterior_cov,
        posterior_std=posterior_std,
        averaging_kernel=averaging_kernel,
        condition_number=condition_number,
    )
