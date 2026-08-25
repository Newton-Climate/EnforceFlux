"""Multiplicative representation error on a resolved state.

The term scales each observation's error by its sensitivity to the total
flux. For a one-cell state that sensitivity IS the state column, which is why
the term was originally restricted to total-only inversions. A resolved state
spreads the total across columns and no single column carries it, so the
input builder supplies the uniform-template sensitivity instead.

These tests pin that the one-state path is unchanged, that the resolved path
uses the supplied sensitivity, and that a missing sensitivity is refused
rather than silently dropping the term.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "apps", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _sigma(G, obs_meta, *, fraction=0.3, flux_scale=0.03, n_obs=None):
    """The multiplicative term exactly as flux_main assembles it."""
    from flux_main import total_flux_sensitivity

    n = len(G) if n_obs is None else n_obs
    return fraction * flux_scale * total_flux_sensitivity(G, obs_meta, n_obs=n)


def test_one_state_uses_its_own_column():
    """Unchanged behaviour: the state column is the total sensitivity."""
    G = np.array([[2.0e6], [5.0e6]])
    np.testing.assert_allclose(
        _sigma(G, {}), 0.3 * 0.03 * np.array([2.0e6, 5.0e6]))


def test_one_state_ignores_any_supplied_sensitivity():
    """The pre-existing path must not change behaviour when meta is present."""
    G = np.array([[2.0e6], [5.0e6]])
    np.testing.assert_allclose(
        _sigma(G, {"uniform_total_sensitivity": [9.9e9, 9.9e9]}),
        _sigma(G, {}))


def test_resolved_state_uses_the_supplied_sensitivity():
    G = np.array([[1.0e6, 3.0e6], [4.0e6, 6.0e6]])
    meta = {"uniform_total_sensitivity": [2.0e6, 5.0e6]}
    np.testing.assert_allclose(
        _sigma(G, meta), 0.3 * 0.03 * np.array([2.0e6, 5.0e6]))


def test_resolved_state_without_the_sensitivity_is_refused():
    G = np.array([[1.0e6, 3.0e6], [4.0e6, 6.0e6]])
    with pytest.raises(ValueError, match="sensitivity to a uniform total flux"):
        _sigma(G, {})


def test_a_wrong_length_sensitivity_is_refused():
    G = np.array([[1.0e6, 3.0e6], [4.0e6, 6.0e6]])
    with pytest.raises(ValueError, match="but there are 2 observations"):
        _sigma(G, {"uniform_total_sensitivity": [1.0, 2.0, 3.0]})


def test_input_builder_supplies_a_sensitivity_per_observation(tmp_path):
    """The builder's value must be the uniform-template row sum, per observation."""
    from netCDF4 import Dataset
    from flux_inputs import build_from_prebuilt_operator_with_instrument

    n_time, n_inst, n_fine, n_coarse = 2, 2, 4, 2
    W = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    areas = np.array([100.0, 300.0, 200.0, 400.0])       # deliberately unequal
    np.savez(tmp_path / "basis.npz", W=W, fine_cell_areas_m2=areas,
             coarse_cell_areas_m2=np.array([400.0, 600.0]),
             coarse_centers_m=np.zeros((n_coarse, 2)))
    rng = np.random.default_rng(0)
    G = rng.uniform(1e5, 1e6, (n_time * n_inst, n_fine))
    np.savez(tmp_path / "jac.npz", G=G,
             row_labels=np.array([f"r{i}" for i in range(n_time * n_inst)]),
             column_labels=np.array([f"c{j}" for j in range(n_fine)]),
             units=np.array("ng m-3 / (kg s-1)"))
    with Dataset(tmp_path / "truth.nc", "w") as ds:
        ds.createDimension("y", 2); ds.createDimension("x", 2)
        ds.createVariable("F_true", "f8", ("y", "x"))[:] = 1e-5
        ds.createVariable("cell_area_m2", "f8", ("y", "x"))[:] = areas.reshape(2, 2)
        ds.L_true_m = 100.0
    obs = tmp_path / "obs.nc"
    with Dataset(obs, "w") as ds:
        ds.createDimension("time", n_time); ds.createDimension("instrument", n_inst)
        v = ds.createVariable("y_obs", "f8", ("time", "instrument"))
        v[:] = 1.0
        v.units = "ng m-3"
        ds.createVariable("valid_mask", "i1", ("time", "instrument"))[:] = 1
        ds.createVariable("noise_variance", "f8", ("time", "instrument"))[:] = 1.0

    class _Up:
        def file(self, role):
            return {"jacobian": tmp_path / "jac.npz",
                    "basis_mapping": tmp_path / "basis.npz",
                    "truth_field": tmp_path / "truth.nc"}[role]

    *_head, obs_meta, _n, _xp, _sa, _diag = (
        build_from_prebuilt_operator_with_instrument(
            {"input": {"mode": "instrument_netcdf", "time_reduce": "none"},
             "observations": {"default_sigma": 1.0},
             "inversion": {"method": "nonnegative"}},
            _Up(), obs)
    )
    supplied = np.asarray(obs_meta["uniform_total_sensitivity"])
    assert supplied.shape == (n_time * n_inst,)

    # Area-weighted row sum, in the flux stage's instrument-major row order.
    template = areas / areas.sum()
    row_order = [t * n_inst + i for i in range(n_inst) for t in range(n_time)]
    np.testing.assert_allclose(supplied, G[row_order] @ template, rtol=1e-12)
    # It must be a genuine mixture, not any single column.
    for j in range(n_fine):
        assert not np.allclose(supplied, G[row_order][:, j])
