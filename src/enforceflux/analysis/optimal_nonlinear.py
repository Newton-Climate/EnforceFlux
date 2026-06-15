"""Nonlinear optimal-estimation solver (Levenberg-Marquardt)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from enforceflux.analysis.optimal_models import OEResult


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
) -> OEResult:
    """Nonlinear Optimal Estimation via Levenberg-Marquardt."""
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

        H_mat = KtSeK + Sa_inv + lam * np.eye(len(xa))
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
        else:
            lam = min(lam * lam_factor, 1e10)

        if float(np.max(np.abs(dx))) < eps:
            converged = True
            break

    K_f = _jacobian(x)
    KtSeK_f = K_f.T @ Se_inv @ K_f
    H_f = KtSeK_f + Sa_inv
    try:
        Sx = np.linalg.inv(H_f)
    except np.linalg.LinAlgError:
        Sx = np.linalg.pinv(H_f)
    A = Sx @ KtSeK_f

    return OEResult(
        x_opt=x,
        x_prior=xa,
        Sx=Sx,
        averaging_kernel=A,
        y_obs=y,
        y_prior=y_prior,
        y_opt=np.asarray(F(x), dtype=float),
        cost_history=cost_hist,
        converged=converged,
        n_iter=len(cost_hist),
        source_names=source_names,
    )
