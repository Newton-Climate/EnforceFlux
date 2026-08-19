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
    *,
    g_beta: np.ndarray | None = None,
    sigma_beta: np.ndarray | None = None,
) -> InversionResult:
    """Compute the Bayesian linear inversion.

    Args:
        g: Forward operator (m x n)
        y: Observations (m)
        x_prior: Prior mean (n)
        s_a: Prior covariance (n x n) or diagonal (n)
        r: Observation covariance (m x m) or diagonal (m)
        g_beta: Optional observation x n_beta nuisance forward matrix.
        sigma_beta: Optional nuisance prior std, length n_beta.
    """

    g = np.asarray(g)
    y = np.asarray(y).reshape(-1)
    x_prior = np.asarray(x_prior).reshape(-1)
    s_a = _as_covariance(s_a)
    r = _as_covariance(r)

    # --- source-heterogeneity OSSE (M8) ---
    use_beta = g_beta is not None and sigma_beta is not None
    n_x = x_prior.size
    if use_beta:
        g_beta_arr = np.asarray(g_beta, dtype=float)
        sigma_beta_arr = np.asarray(sigma_beta, dtype=float).reshape(-1)
        if g_beta_arr.ndim != 2 or g_beta_arr.shape[0] != g.shape[0]:
            raise ValueError(
                f"g_beta must be (m, n_beta) with m={g.shape[0]}, got {g_beta_arr.shape}"
            )
        if g_beta_arr.shape[1] != sigma_beta_arr.size:
            raise ValueError(
                f"sigma_beta length {sigma_beta_arr.size} does not match g_beta columns {g_beta_arr.shape[1]}"
            )
        g = np.concatenate([g, g_beta_arr], axis=1)
        x_prior = np.concatenate([x_prior, np.zeros(sigma_beta_arr.size)])
        s_a_full = np.zeros((n_x + sigma_beta_arr.size, n_x + sigma_beta_arr.size))
        s_a_full[:n_x, :n_x] = s_a
        s_a_full[n_x:, n_x:] = np.diag(sigma_beta_arr ** 2)
        s_a = s_a_full
    # --- end source-heterogeneity OSSE (M8) ---

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

    # --- source-heterogeneity OSSE (M8) ---
    if use_beta:
        beta_mean = x_post[n_x:]
        beta_cov = posterior_cov[n_x:, n_x:]
        x_post = x_post[:n_x]
        posterior_cov = posterior_cov[:n_x, :n_x]
        averaging_kernel = averaging_kernel[:n_x, :n_x]
        fisher = fisher[:n_x, :n_x]
        x_prior = x_prior[:n_x]
    else:
        beta_mean = None
        beta_cov = None
    # --- end source-heterogeneity OSSE (M8) ---

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
        posterior_beta_mean=beta_mean,
        posterior_beta_cov=beta_cov,
    )
