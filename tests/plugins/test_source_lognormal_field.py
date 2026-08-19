import numpy as np

from enforceflux.plugins.source_lognormal_field import (
    LognormalFieldSource,
    clear_pending_writes,
    drain_pending_writes,
)


def _cfg():
    return {
        "Q_true_kg_s": 1.0e-2,
        "grid": {"nx": 8, "ny": 8, "dx_m": 100.0,
                 "origin_x_m": -400.0, "origin_y_m": -400.0},
        "alt_m": 2.0,
        "covariance": {"model": "exponential", "L_m": 200.0},
        "cv": 0.5,
        "seed": 7,
        "basis": {"coarsen": 2},
        "prior": {"mean_kg_s_per_cell": 0.0},
    }


def test_expands_to_run_sources():
    clear_pending_writes()
    plugin = LognormalFieldSource()
    sources = plugin.build_sources(_cfg(), domain=None)
    assert len(sources) == 8 * 8
    ids = {s.id for s in sources}
    assert len(ids) == 64
    total = sum(s.emission_rate_kg_s for s in sources)
    assert abs(total - 1.0e-2) / 1.0e-2 < 1.0e-12
    xs = sorted({s.x_m for s in sources})
    ys = sorted({s.y_m for s in sources})
    assert xs[0] == -350.0 and xs[-1] == 350.0
    assert ys[0] == -350.0 and ys[-1] == 350.0
    for s in sources:
        assert s.altitude_m == 2.0
        assert s.prior_mean_kg_s == 0.0
    clear_pending_writes()


def test_persists_truth_and_basis(tmp_path):
    clear_pending_writes()
    plugin = LognormalFieldSource()
    plugin.build_sources(_cfg(), domain=None)
    written = drain_pending_writes(tmp_path)
    names = {relpath for relpath, _ in written}
    assert "truth_field.nc" in names
    assert "basis_mapping.npz" in names
    assert (tmp_path / "truth_field.nc").is_file()
    assert (tmp_path / "basis_mapping.npz").is_file()

    from enforceflux.source_fields.basis import load_mapping
    mapping = load_mapping(tmp_path / "basis_mapping.npz")
    assert mapping.W.shape == (16, 64)

    from netCDF4 import Dataset
    with Dataset(tmp_path / "truth_field.nc") as ds:
        assert ds.variables["F_true"].shape == (8, 8)
        assert float(getattr(ds, "L_true_m")) == 200.0
        assert float(getattr(ds, "cv")) == 0.5
        F = np.asarray(ds.variables["F_true"][:])
        area = np.asarray(ds.variables["cell_area_m2"][:])
        assert abs(float((F * area).sum()) - 1.0e-2) / 1.0e-2 < 1.0e-12
    clear_pending_writes()


def test_seed_reproducibility():
    clear_pending_writes()
    a = LognormalFieldSource().build_sources(_cfg(), domain=None)
    clear_pending_writes()
    b = LognormalFieldSource().build_sources(_cfg(), domain=None)
    clear_pending_writes()
    ra = np.array([s.emission_rate_kg_s for s in a])
    rb = np.array([s.emission_rate_kg_s for s in b])
    assert np.array_equal(ra, rb)
