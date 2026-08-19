from __future__ import annotations

import numpy as np
import pytest

from enforceflux.source_fields.lognormal_gp import (
    FieldGrid,
    LognormalFieldSpec,
    sample_lognormal_field,
)


def _grid(nx=32, ny=32, dx=25.0):
    return FieldGrid(nx=nx, ny=ny, dx_m=dx)


@pytest.mark.parametrize("cv", [0.25, 1.0, 2.0])
def test_variance_relation_holds(cv):
    grid = _grid(nx=32, ny=32, dx=25.0)
    spec = LognormalFieldSpec(grid=grid, Q_true_kg_s=1.0, L_m=50.0, cv=cv, seed=0)
    rng = np.random.default_rng(0)
    # Use E[Var(F)]/mean^2 -> cv^2 rather than mean of per-realization std/mean,
    # which is heavily downward-biased for heavy-tailed lognormal at large cv.
    variances = []
    for _ in range(200):
        F = sample_lognormal_field(spec, rng)
        variances.append(F.var())
    mean_F = spec.Q_true_kg_s / (grid.nx * grid.ny * grid.dx_m * grid.dx_m)
    cv_est = np.sqrt(np.mean(variances)) / mean_F
    assert abs(cv_est - cv) / cv < 0.05


def test_total_emission_conserved():
    grid = _grid()
    spec = LognormalFieldSpec(grid=grid, Q_true_kg_s=2.7778e-2, L_m=100.0, cv=1.0)
    rng = np.random.default_rng(42)
    F = sample_lognormal_field(spec, rng)
    areas = grid.cell_areas_m2()
    total = float((F * areas).sum())
    assert abs(total - spec.Q_true_kg_s) / spec.Q_true_kg_s < 1e-12


def test_empirical_variogram_matches_input_L():
    from scipy.optimize import curve_fit

    L_true = 100.0
    grid = FieldGrid(nx=64, ny=64, dx_m=25.0)
    spec = LognormalFieldSpec(grid=grid, Q_true_kg_s=1.0, L_m=L_true, cv=0.5, seed=1)
    rng = np.random.default_rng(1)

    n_lags = 12
    dx = grid.dx_m
    lags = (np.arange(1, n_lags + 1)) * dx
    gamma = np.zeros(n_lags)
    for _ in range(30):
        F = sample_lognormal_field(spec, rng)
        Z = np.log(F)
        for k, h_cells in enumerate(range(1, n_lags + 1)):
            dxs = Z[:, h_cells:] - Z[:, :-h_cells]
            dys = Z[h_cells:, :] - Z[:-h_cells, :]
            gamma[k] += 0.5 * (dxs.var() + dys.var()) / 2.0
    gamma /= 30.0

    sill0 = gamma[-3:].mean()

    def model(h, sill, L):
        return sill * (1.0 - np.exp(-h / L))

    (sill, L_fit), _ = curve_fit(model, lags, gamma, p0=[sill0, L_true], maxfev=5000)
    assert (L_true / 1.5) <= L_fit <= (L_true * 1.5)


def test_reproducible_with_seed():
    grid = _grid()
    spec = LognormalFieldSpec(grid=grid, Q_true_kg_s=1.0, L_m=100.0, cv=1.0)
    F1 = sample_lognormal_field(spec, np.random.default_rng(123))
    F2 = sample_lognormal_field(spec, np.random.default_rng(123))
    np.testing.assert_array_equal(F1, F2)


def test_f_emit_sparsity():
    grid = FieldGrid(nx=48, ny=48, dx_m=25.0)
    f_emit = 0.3
    spec = LognormalFieldSpec(
        grid=grid, Q_true_kg_s=1.0, L_m=75.0, cv=1.0, f_emit=f_emit, seed=7
    )
    rng = np.random.default_rng(7)
    fracs = []
    for _ in range(40):
        F = sample_lognormal_field(spec, rng)
        fracs.append(float((F > 0).mean()))
    assert abs(np.mean(fracs) - f_emit) < 0.03
