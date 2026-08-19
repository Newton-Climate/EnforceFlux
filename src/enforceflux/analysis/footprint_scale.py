"""Footprint spatial-scale metric L_H (memo section 4.3).

Given a Jacobian ``H`` of shape (n_obs, n_cells) laid out on a regular 2D
``grid`` (row-major, ``y`` outer / ``x`` inner in ``(ny, nx)`` order), compute
a single correlation length ``L_H`` from the column-averaged 2D
autocorrelation of the per-observation footprints.

The estimator:

  1. Reshape each row of H (a footprint image) to (ny, nx).
  2. Compute its 2D autocorrelation via FFT and normalize to 1 at zero lag.
  3. Average the autocorrelations across observations (weighted by footprint
     total mass so near-zero footprints don't dominate).
  4. Integrate the isotropic radial autocorrelation from 0 out to the first
     zero-crossing (or the domain-half cutoff). For rho(r) = exp(-r/L) this
     yields L; for rho(r) = exp(-r^2/(4 sigma^2)) it yields sqrt(pi) * sigma.
     Both scale linearly with the underlying footprint width.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class _GridLike(Protocol):
    nx: int
    ny: int
    dx_m: float


def _autocorr_2d(img: np.ndarray) -> np.ndarray:
    """Normalized 2D autocorrelation of ``img`` via FFT, peak at center."""
    ny, nx = img.shape
    # Zero-pad to avoid circular wrap.
    pad_y, pad_x = 2 * ny, 2 * nx
    F = np.fft.fft2(img, s=(pad_y, pad_x))
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    # Trim to (2*ny-1, 2*nx-1) so lag 0 is exactly centered.
    cy, cx = pad_y // 2, pad_x // 2
    ac = ac[cy - (ny - 1) : cy + ny, cx - (nx - 1) : cx + nx]
    peak = ac[ny - 1, nx - 1]
    if peak > 0:
        ac = ac / peak
    return ac


def footprint_correlation_length(H: np.ndarray, grid: _GridLike) -> float:
    """Return L_H in metres from column-averaged footprint autocorrelation."""
    H = np.asarray(H, dtype=float)
    nx, ny = int(grid.nx), int(grid.ny)
    dx = float(grid.dx_m)
    if H.ndim != 2 or H.shape[1] != nx * ny:
        raise ValueError(
            f"H must be (n_obs, nx*ny={nx * ny}); got shape {H.shape}"
        )

    # Weight each footprint by its total mass so empty rows don't skew the mean.
    weights = np.abs(H).sum(axis=1)
    if not np.any(weights > 0):
        return float("nan")

    ac_sum = np.zeros((2 * ny - 1, 2 * nx - 1), dtype=float)
    wsum = 0.0
    for row, w in zip(H, weights):
        if w <= 0:
            continue
        img = row.reshape(ny, nx)
        ac = _autocorr_2d(img)
        ac_sum += w * ac
        wsum += w
    ac_avg = ac_sum / wsum

    # Radial average.
    ly = np.arange(-(ny - 1), ny) * dx
    lx = np.arange(-(nx - 1), nx) * dx
    LX, LY = np.meshgrid(lx, ly, indexing="xy")
    r = np.sqrt(LX**2 + LY**2)
    r_max = 0.5 * min(nx, ny) * dx
    n_bins = min(nx, ny)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rho = np.empty(n_bins, dtype=float)
    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        rho[i] = ac_avg[mask].mean() if mask.any() else np.nan

    # Integral length scale: L_H = integral_0^{r_cut} rho(r) dr, where r_cut
    # is the first zero-crossing (or the radial cutoff if rho stays positive).
    # Discard empty bins (some radii have no lattice points at this dx).
    finite = np.isfinite(rho)
    if finite.sum() < 2:
        return float("nan")
    r_f = centers[finite]
    rho_f = rho[finite]
    neg = np.where(rho_f <= 0)[0]
    cutoff = int(neg[0]) if neg.size > 0 else r_f.size
    if cutoff < 1:
        return float("nan")
    r_use = r_f[:cutoff]
    y_use = rho_f[:cutoff]
    # Prepend r=0, rho=1 to anchor the integral at zero lag.
    r_full = np.concatenate([[0.0], r_use])
    y_full = np.concatenate([[1.0], y_use])
    return float(np.trapezoid(y_full, r_full))
