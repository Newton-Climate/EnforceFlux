"""Log-Gaussian random field generator on a regular grid via circulant embedding."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FieldGrid:
    nx: int
    ny: int
    dx_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.origin_x_m + (np.arange(self.nx) + 0.5) * self.dx_m
        y = self.origin_y_m + (np.arange(self.ny) + 0.5) * self.dx_m
        return x, y

    def cell_areas_m2(self) -> np.ndarray:
        return np.full((self.ny, self.nx), self.dx_m * self.dx_m, dtype=float)


@dataclass(frozen=True)
class LognormalFieldSpec:
    grid: FieldGrid
    Q_true_kg_s: float
    L_m: float
    cv: float
    covariance: Literal["exponential", "matern"] = "exponential"
    matern_nu: float = 1.5
    f_emit: float | None = None
    seed: int = 0


def _matern_correlation(r: np.ndarray, L: float, nu: float) -> np.ndarray:
    from scipy.special import gamma, kv

    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    mask = r > 0
    scaled = np.sqrt(2.0 * nu) * r[mask] / L
    coef = (2.0 ** (1.0 - nu)) / gamma(nu)
    out[mask] = coef * (scaled ** nu) * kv(nu, scaled)
    return out


def _correlation(r: np.ndarray, spec: LognormalFieldSpec) -> np.ndarray:
    if spec.covariance == "exponential":
        return np.exp(-r / spec.L_m)
    if spec.covariance == "matern":
        return _matern_correlation(r, spec.L_m, spec.matern_nu)
    raise ValueError(f"Unknown covariance model: {spec.covariance}")


def _padded_cholesky_sample(
    spec: LognormalFieldSpec, sigma2: float, rng: np.random.Generator
) -> np.ndarray:
    g = spec.grid
    xs = (np.arange(g.nx) - g.nx // 2) * g.dx_m
    ys = (np.arange(g.ny) - g.ny // 2) * g.dx_m
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    C = sigma2 * _correlation(r, spec)
    # Jitter to keep positive-definite under roundoff.
    C = C + 1e-10 * np.eye(C.shape[0])
    L = np.linalg.cholesky(C)
    z = L @ rng.standard_normal(C.shape[0])
    return z.reshape(g.ny, g.nx)


def _circulant_embedding_sample(
    spec: LognormalFieldSpec, sigma2: float, rng: np.random.Generator
) -> np.ndarray | None:
    """2D circulant embedding on a doubled grid. Returns None if not PSD."""
    g = spec.grid
    Mx, My = 2 * g.nx, 2 * g.ny
    # Signed lag on the doubled torus.
    lx = np.concatenate([np.arange(0, g.nx + 1), np.arange(-g.nx + 1, 0)]) * g.dx_m
    ly = np.concatenate([np.arange(0, g.ny + 1), np.arange(-g.ny + 1, 0)]) * g.dx_m
    LX, LY = np.meshgrid(lx, ly, indexing="xy")
    R = np.hypot(LX, LY)
    C = sigma2 * _correlation(R, spec)
    # Eigenvalues of the block-circulant matrix are the 2D FFT of the first row.
    lam = np.fft.fft2(C).real
    if lam.min() < -1e-8 * abs(lam.max()):
        return None
    lam = np.clip(lam, 0.0, None)
    # Complex normal in frequency domain; take real part for a real Gaussian sample.
    xi = rng.standard_normal((My, Mx)) + 1j * rng.standard_normal((My, Mx))
    freq = np.sqrt(lam) * xi
    field = np.fft.fft2(freq).real / np.sqrt(Mx * My)
    return field[: g.ny, : g.nx]


def sample_lognormal_field(
    spec: LognormalFieldSpec, rng: np.random.Generator
) -> np.ndarray:
    """Return F_true(y, x) in kg/s per cell with sum(F * area) == Q_true_kg_s."""
    sigma2 = np.log1p(spec.cv ** 2)
    Z = _circulant_embedding_sample(spec, sigma2, rng)
    if Z is None:
        warnings.warn(
            "Circulant embedding not PSD; falling back to padded Cholesky.",
            RuntimeWarning,
            stacklevel=2,
        )
        Z = _padded_cholesky_sample(spec, sigma2, rng)
    # Recenter so E[exp(Z)] = 1 in the limit, then renormalize exactly to Q_true.
    field = np.exp(Z - 0.5 * sigma2)
    if spec.f_emit is not None:
        if not (0.0 < spec.f_emit <= 1.0):
            raise ValueError("f_emit must be in (0, 1].")
        mask = rng.random(field.shape) < spec.f_emit
        field = field * mask
        if not mask.any():
            raise RuntimeError("f_emit produced an all-zero mask; retry with a different seed.")
    areas = spec.grid.cell_areas_m2()
    total = float((field * areas).sum())
    if total <= 0.0:
        raise RuntimeError("Sampled field has zero total emission; cannot renormalize.")
    F = field * (spec.Q_true_kg_s / total)
    return F
