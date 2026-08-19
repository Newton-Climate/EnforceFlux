"""Analytic Gaussian footprint recovery for L_H (M3)."""

from dataclasses import dataclass

import numpy as np
import pytest

from enforceflux.analysis.footprint_scale import footprint_correlation_length


@dataclass(frozen=True)
class _Grid:
    nx: int
    ny: int
    dx_m: float


def _gaussian_footprint(nx: int, ny: int, dx: float, sigma: float, cx, cy) -> np.ndarray:
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma**2))


def test_L_H_recovers_input_length():
    """A Gaussian footprint of width sigma has autocorrelation exp(-r^2/(4 sigma^2)).

    The exponential-fit estimator (rho ~ exp(-r/L)) then recovers an L that
    scales with sigma. Test that the estimator is a monotone, ~linear function
    of sigma across a factor-of-4 range.
    """
    nx = ny = 64
    dx = 25.0
    grid = _Grid(nx=nx, ny=ny, dx_m=dx)

    rng = np.random.default_rng(0)
    Ls = []
    sigmas = [100.0, 200.0, 400.0]
    for sigma in sigmas:
        rows = []
        # Several footprints centered at random points well inside the domain.
        for _ in range(8):
            cx = rng.uniform(nx * dx * 0.3, nx * dx * 0.7)
            cy = rng.uniform(ny * dx * 0.3, ny * dx * 0.7)
            rows.append(_gaussian_footprint(nx, ny, dx, sigma, cx, cy).ravel())
        H = np.array(rows)
        Ls.append(footprint_correlation_length(H, grid))

    # Monotone increasing with sigma.
    assert Ls[0] < Ls[1] < Ls[2]
    # And roughly linear in sigma (ratio ~ 2 within 40%).
    assert Ls[1] / Ls[0] == pytest.approx(2.0, rel=0.4)
    assert Ls[2] / Ls[1] == pytest.approx(2.0, rel=0.4)
