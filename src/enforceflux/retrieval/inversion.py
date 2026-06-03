from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InversionResult:
    x_posterior: np.ndarray
    posterior_cov: np.ndarray
    averaging_kernel: np.ndarray
    fisher_information: np.ndarray
    residual: np.ndarray


def _as_covariance(matrix_or_diag: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix_or_diag)
    if arr.ndim == 1:
        return np.diag(arr)
    return arr


def bayesian_linear_inversion(
    g: np.ndarray,
    y: np.ndarray,
    x_prior: np.ndarray,
    s_a: np.ndarray,
    r: np.ndarray,
) -> InversionResult:
    """Compute the Bayesian linear inversion.

    Args:
        g: Forward operator (m x n)
        y: Observations (m)
        x_prior: Prior mean (n)
        s_a: Prior covariance (n x n) or diagonal (n)
        r: Observation covariance (m x m) or diagonal (m)
    """

    g = np.asarray(g)
    y = np.asarray(y).reshape(-1)
    x_prior = np.asarray(x_prior).reshape(-1)
    s_a = _as_covariance(s_a)
    r = _as_covariance(r)

    s_a_inv = np.linalg.inv(s_a)
    r_inv = np.linalg.inv(r)
    fisher = g.T @ r_inv @ g

    posterior_cov = np.linalg.inv(s_a_inv + fisher)
    gain = posterior_cov @ g.T @ r_inv
    x_post = x_prior + gain @ (y - g @ x_prior)
    averaging_kernel = gain @ g
    residual = y - g @ x_post

    return InversionResult(
        x_posterior=x_post,
        posterior_cov=posterior_cov,
        averaging_kernel=averaging_kernel,
        fisher_information=fisher,
        residual=residual,
    )
