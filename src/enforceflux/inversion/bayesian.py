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

    # Work in prior- and observation-whitened coordinates.  The former
    # precision-space expression, inv(Sa^-1 + G.T @ R^-1 @ G), is fragile for
    # dense GP priors and highly informative operators: the two terms can
    # differ by hundreds of orders of magnitude.  This SVD form evaluates the
    # same Gaussian posterior without explicitly inverting either covariance.
    s_a = 0.5 * (s_a + s_a.T)
    r = 0.5 * (r + r.T)
    prior_eigval, prior_eigvec = np.linalg.eigh(s_a)
    prior_scale = max(float(np.max(np.abs(prior_eigval))), 1.0)
    if np.min(prior_eigval) < -1e-10 * prior_scale:
        raise ValueError("s_a must be positive semidefinite")
    prior_eigval = np.clip(prior_eigval, 0.0, None)
    prior_factor = prior_eigvec * np.sqrt(prior_eigval)

    # R should be positive definite.  A tiny diagonal round-off repair keeps
    # user-supplied covariance matrices usable without silently changing their
    # stated uncertainty at a meaningful scale.
    try:
        r_chol = np.linalg.cholesky(r)
    except np.linalg.LinAlgError:
        r_scale = max(float(np.max(np.abs(np.diag(r)))), 1.0)
        r_chol = np.linalg.cholesky(r + np.eye(r.shape[0]) * (1e-12 * r_scale))

    y_prior = g @ x_prior
    innovation = y - y_prior
    g_white = np.linalg.solve(r_chol, g)
    innovation_white = np.linalg.solve(r_chol, innovation)
    operator_white = g_white @ prior_factor
    u, singular_values, vt = np.linalg.svd(operator_white, full_matrices=False)
    filter_factor = singular_values / (1.0 + singular_values ** 2)
    latent_increment = vt.T @ (filter_factor * (u.T @ innovation_white))
    x_post = x_prior + prior_factor @ latent_increment

    information_fraction = (singular_values ** 2) / (1.0 + singular_values ** 2)
    latent_covariance = np.eye(s_a.shape[0]) - (vt.T * information_fraction) @ vt
    posterior_cov = prior_factor @ latent_covariance @ prior_factor.T
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    gain_white = prior_factor @ ((vt.T * filter_factor) @ u.T)
    gain = np.linalg.solve(r_chol.T, gain_white.T).T
    averaging_kernel = gain @ g
    fisher = g_white.T @ g_white
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


def bounded_bayesian_linear_inversion(
    g: np.ndarray,
    y: np.ndarray,
    x_prior: np.ndarray,
    s_a: np.ndarray,
    r: np.ndarray,
    source_names: list[str] | None = None,
) -> InversionResult:
    """Gaussian linear MAP estimate constrained to non-negative emissions.

    The usual observation and prior Mahalanobis objective is solved with
    ``x >= 0``.  This prevents a weakly regularized spatial GP from cancelling
    positive and negative source cells to fit a sparse sensor network.
    """
    from scipy.optimize import lsq_linear

    g = np.asarray(g, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_prior = np.asarray(x_prior, dtype=float).reshape(-1)
    s_a = _as_covariance(s_a).astype(float)
    r = _as_covariance(r).astype(float)
    s_a = 0.5 * (s_a + s_a.T)
    r = 0.5 * (r + r.T)

    # Rows with exactly zero sensitivity contribute a constant residual but no
    # gradient.  Keeping their potentially very large residual in a bounded
    # least-squares solve causes premature relative-cost convergence. They
    # contain no information about emissions and are removed only for solving;
    # the returned modeled-observation vector retains every original row.
    full_g, full_y, full_r = g, y, r
    active_rows = np.any(np.abs(g) > 0.0, axis=1)
    if not np.any(active_rows):
        x_post = np.maximum(x_prior, 0.0)
        return InversionResult(
            x_posterior=x_post,
            x_prior=x_prior,
            posterior_cov=s_a,
            averaging_kernel=np.zeros((x_prior.size, x_prior.size)),
            y_obs=full_y,
            y_prior=full_g @ x_prior,
            y_posterior=full_g @ x_post,
            fisher_information=np.zeros((x_prior.size, x_prior.size)),
            residual=full_y - full_g @ x_post,
            converged=True,
            n_iter=0,
            source_names=source_names,
        )
    if not np.all(active_rows):
        g = g[active_rows]
        y = y[active_rows]
        r = r[np.ix_(active_rows, active_rows)]

    def _chol(matrix: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
            return np.linalg.cholesky(matrix + np.eye(matrix.shape[0]) * 1e-12 * scale)

    l_r = _chol(r)
    l_a = _chol(s_a)
    n_state = x_prior.size
    a_data = np.linalg.solve(l_r, g)
    b_data = np.linalg.solve(l_r, y)
    a_prior = np.linalg.solve(l_a, np.eye(n_state))
    b_prior = np.linalg.solve(l_a, x_prior)
    solution = lsq_linear(
        np.vstack((a_data, a_prior)), np.concatenate((b_data, b_prior)),
        bounds=(0.0, np.inf), method="trf", lsmr_tol="auto",
    )
    x_post = np.asarray(solution.x, dtype=float)
    sa_inv = np.linalg.solve(s_a, np.eye(n_state))
    r_inv_g = np.linalg.solve(r, g)
    fisher = g.T @ r_inv_g
    posterior_cov = np.linalg.pinv(fisher + sa_inv)
    gain = posterior_cov @ g.T @ np.linalg.solve(r, np.eye(r.shape[0]))
    y_prior = full_g @ x_prior
    y_post = full_g @ x_post
    return InversionResult(
        x_posterior=x_post,
        x_prior=x_prior,
        posterior_cov=posterior_cov,
        averaging_kernel=gain @ g,
        y_obs=full_y,
        y_prior=y_prior,
        y_posterior=y_post,
        fisher_information=fisher,
        residual=full_y - y_post,
        cost_history=[],
        converged=bool(solution.success),
        n_iter=int(solution.nit or 1),
        source_names=source_names,
    )
