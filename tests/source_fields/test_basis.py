from __future__ import annotations

import numpy as np
import pytest

from enforceflux.source_fields.basis import (
    load_mapping,
    polygon_basis,
    project_flux_to_coarse,
    save_mapping,
    uniform_coarse_basis,
)
from enforceflux.source_fields.lognormal_gp import (
    FieldGrid,
    LognormalFieldSpec,
    sample_lognormal_field,
)


def _fine():
    return FieldGrid(nx=16, ny=16, dx_m=50.0)


def test_uniform_basis_conserves_mass():
    grid = _fine()
    mapping = uniform_coarse_basis(grid, coarsen=4)
    spec = LognormalFieldSpec(grid=grid, Q_true_kg_s=1.0, L_m=100.0, cv=1.0)
    F = sample_lognormal_field(spec, np.random.default_rng(0))
    x = project_flux_to_coarse(F, mapping)
    fine_total = float((F * grid.cell_areas_m2()).sum())
    coarse_total = float(x.sum())
    assert abs(coarse_total - fine_total) / fine_total < 1e-12
    assert x.shape == (16,)


def test_uniform_basis_coarse_area_matches():
    grid = _fine()
    mapping = uniform_coarse_basis(grid, coarsen=4)
    fine_total_area = grid.nx * grid.ny * grid.dx_m ** 2
    assert abs(mapping.coarse_cell_areas_m2.sum() - fine_total_area) < 1e-9


def test_polygon_basis_partition_of_unity():
    # Deferred in M1; ensure a clear NotImplementedError is raised.
    grid = _fine()
    with pytest.raises(NotImplementedError):
        polygon_basis(grid, [])


def test_roundtrip_npz(tmp_path):
    grid = _fine()
    mapping = uniform_coarse_basis(grid, coarsen=4)
    path = tmp_path / "basis_mapping.npz"
    save_mapping(path, mapping)
    loaded = load_mapping(path)
    np.testing.assert_array_equal(loaded.W, mapping.W)
    np.testing.assert_array_equal(loaded.fine_cell_areas_m2, mapping.fine_cell_areas_m2)
    np.testing.assert_array_equal(
        loaded.coarse_cell_areas_m2, mapping.coarse_cell_areas_m2
    )
    np.testing.assert_array_equal(loaded.coarse_centers_m, mapping.coarse_centers_m)
