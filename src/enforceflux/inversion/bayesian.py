"""Bayesian linear inversion (closed-form Gaussian posterior)."""
import numpy as np

from enforceflux.inversion.result import InversionResult


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
    source_names: list[str] | None = None,
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
    y_prior = g @ x_prior
    x_post = x_prior + gain @ (y - y_prior)
    averaging_kernel = gain @ g
    y_post = g @ x_post
    residual = y - y_post

    return InversionResult(
        x_posterior=x_post,
        x_prior=x_prior,
        posterior_cov=posterior_cov,
        averaging_kernel=averaging_kernel,
        y_obs=y,
        y_prior=y_prior,
        y_posterior=y_post,
        fisher_information=fisher,
        residual=residual,
        cost_history=[],
        converged=True,
        n_iter=1,
        source_names=source_names,
    )
