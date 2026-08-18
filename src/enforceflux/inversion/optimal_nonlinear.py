"""Nonlinear optimal-estimation solver (Levenberg-Marquardt).

Convergence: Rodgers d-i-squared by default (Rodgers 2000, section 5.4):
    d = dx.T @ (K.T @ Se_inv @ K + Sa_inv) @ dx
    converged when d / n_state < rodgers_tol.
The legacy scale-dependent test max(abs(dx)) < eps remains available via
``convergence_test="max_abs"``. See the optimal-estimation skill for why.
"""
from typing import Callable, Literal

import numpy as np

from enforceflux.inversion.result import InversionResult


def _numerical_jacobian(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    step: float = 1e-5,
) -> np.ndarray:
    """Central finite differences Jacobian with relative parameter scaling."""
    n = len(x)
    y0 = np.asarray(F(x), dtype=float)
    m = len(y0)
    K = np.zeros((m, n))
    for j in range(n):
        h = step * max(abs(float(x[j])), 1.0)
        dx = np.zeros(n)
        dx[j] = h
        K[:, j] = (np.asarray(F(x + dx)) - np.asarray(F(x - dx))) / (2.0 * h)
    return K


def _full_inv(cov: np.ndarray) -> np.ndarray:
    """Invert a covariance matrix given as 1-D diagonal or 2-D full."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 1:
        return np.diag(1.0 / (cov + 1e-300))
    return np.linalg.inv(cov)


def optimize_oe(
    F: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x_prior: np.ndarray,
    Sa: np.ndarray,
    Se: np.ndarray,
    K_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_iter: int = 20,
    lam0: float = 1e-3,
    lam_factor: float = 10.0,
    eps: float = 1e-4,
    fd_step: float = 1e-5,
    source_names: list[str] | None = None,
    convergence_test: Literal["rodgers", "max_abs"] = "rodgers",
    rodgers_tol: float = 0.01,
) -> InversionResult:
    """Nonlinear Optimal Estimation via Levenberg-Marquardt.

    Parameters
    ----------
    convergence_test : "rodgers" (default) or "max_abs".
        "rodgers": terminate when d / n_state < rodgers_tol, where
            d = dx.T @ (K.T @ Se_inv @ K + Sa_inv) @ dx at the accepted iterate
            (Rodgers 2000, section 5.4). Scale-invariant.
        "max_abs": legacy behaviour, terminate when max(abs(dx)) < eps.
            Scale-dependent; retained for backward compatibility.
    rodgers_tol : threshold for the Rodgers test, default 0.01.
    """
    y = np.asarray(y, dtype=float)
    xa = np.asarray(x_prior, dtype=float)
    x = xa.copy()

    Sa_inv = _full_inv(Sa)
    Se_inv = _full_inv(Se)

    def _jacobian(x_: np.ndarray) -> np.ndarray:
        if K_fn is not None:
            return np.asarray(K_fn(x_), dtype=float)
        return _numerical_jacobian(F, x_, step=fd_step)

    def _cost(x_: np.ndarray) -> float:
        r = y - np.asarray(F(x_))
        pr = xa - x_
        return float(r @ Se_inv @ r + pr @ Sa_inv @ pr)

    y_prior = np.asarray(F(xa), dtype=float)
    lam = float(lam0)
    cost_hist: list[float] = []
    converged = False
    n_state = len(xa)
    conv_value: float = float("nan")

    for _ in range(n_iter):
        Fx = np.asarray(F(x), dtype=float)
        K = _jacobian(x)

        resid = y - Fx
        prior_r = xa - x

        KtSe = K.T @ Se_inv
        KtSeK = KtSe @ K
        KtSe_r = KtSe @ resid

        cost = float(resid @ Se_inv @ resid + prior_r @ Sa_inv @ prior_r)
        cost_hist.append(cost)

        # H_undamped is the current-iterate posterior precision (Sx^{-1}); used
        # both for the LM solve (with lam*I added) and, when the step is
        # accepted, for the Rodgers d-i-squared convergence test.
        H_undamped = KtSeK + Sa_inv
        H_mat = H_undamped + lam * np.eye(n_state)
        g_vec = KtSe_r + Sa_inv @ prior_r

        try:
            dx = np.linalg.solve(H_mat, g_vec)
        except np.linalg.LinAlgError:
            break

        x_new = x + dx
        cost_new = _cost(x_new)

        if cost_new < cost:
            x = x_new
            lam = max(lam / lam_factor, 1e-12)
            # Convergence test evaluated ONLY on accepted steps. The old code
            # tested on the trial dx unconditionally, which could return
            # converged=True on a rejected step.
            if convergence_test == "rodgers":
                d_sq = float(dx @ H_undamped @ dx)
                conv_value = d_sq / max(n_state, 1)
                if conv_value < rodgers_tol:
                    converged = True
                    break
            else:  # "max_abs"
                conv_value = float(np.max(np.abs(dx)))
                if conv_value < eps:
                    converged = True
                    break
        else:
            lam = min(lam * lam_factor, 1e10)

    K_f = _jacobian(x)
    KtSeK_f = K_f.T @ Se_inv @ K_f
    H_f = KtSeK_f + Sa_inv
    try:
        Sx = np.linalg.inv(H_f)
    except np.linalg.LinAlgError:
        Sx = np.linalg.pinv(H_f)
    A = Sx @ KtSeK_f

    result = InversionResult(
        x_posterior=x,
        x_prior=xa,
        posterior_cov=Sx,
        averaging_kernel=A,
        y_obs=y,
        y_prior=y_prior,
        y_posterior=np.asarray(F(x), dtype=float),
        cost_history=cost_hist,
        converged=converged,
        n_iter=len(cost_hist),
        source_names=source_names,
    )
    # Attach convergence-test metadata for downstream reporting.
    # InversionResult is a frozen dataclass, so use object.__setattr__.
    threshold = rodgers_tol if convergence_test == "rodgers" else eps
    object.__setattr__(result, "convergence_criterion", convergence_test)
    object.__setattr__(result, "convergence_value", conv_value)
    object.__setattr__(result, "convergence_threshold", threshold)
    return result
