"""Pairing a time-resolved operator with the observation frames it covers.

A bLS operator built with ``interval_reduce: none`` has one row per (interval,
instrument) and can only cover the window its turbulence intervals were built
for. ``input.time_index_range`` selects the matching observation frames; these
tests pin the selection, the ordering contract, and the two ways the pairing
can silently go wrong.
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

from flux_inputs import build_from_prebuilt_operator_with_instrument  # noqa: E402

# The observation file spans more frames than the operator covers: a
# time-resolved bLS operator is only built for the window with usable met.
N_TIME, N_INST, N_FINE, N_COARSE = 6, 2, 4, 2
N_INTERVAL, FIRST_FRAME = 3, 1


class _Up:
    """Minimal stand-in for the dispersion RunDir accessor."""

    def __init__(self, files: dict[str, Path]) -> None:
        self._files = files

    def file(self, role: str) -> Path:
        return self._files[role]


@pytest.fixture
def upstream(tmp_path):
    from netCDF4 import Dataset

    # Two coarse cells over four fine cells.
    W = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    np.savez(tmp_path / "basis.npz", W=W,
             fine_cell_areas_m2=np.full(N_FINE, 100.0),
             coarse_cell_areas_m2=np.full(N_COARSE, 200.0),
             coarse_centers_m=np.zeros((N_COARSE, 2)))
    # Interval-major operator rows: row index is t * n_inst + i. Only
    # N_INTERVAL intervals, starting at observation frame FIRST_FRAME.
    rng = np.random.default_rng(0)
    G = rng.uniform(1e5, 1e6, (N_INTERVAL * N_INST, N_FINE))
    np.savez(tmp_path / "jac.npz", G=G,
             row_labels=np.array([f"t{t}|s{i}" for t in range(N_INTERVAL)
                                  for i in range(N_INST)]),
             column_labels=np.array([f"cell_{j}" for j in range(N_FINE)]),
             units=np.array("ng m-3 / (kg s-1)"))
    with Dataset(tmp_path / "truth.nc", "w") as ds:
        ds.createDimension("y", 2); ds.createDimension("x", 2)
        ds.createVariable("F_true", "f8", ("y", "x"))[:] = np.full((2, 2), 1e-5)
        ds.createVariable("cell_area_m2", "f8", ("y", "x"))[:] = np.full((2, 2), 100.0)
        ds.L_true_m = 100.0
    obs = tmp_path / "obs.nc"
    with Dataset(obs, "w") as ds:
        ds.createDimension("time", N_TIME); ds.createDimension("instrument", N_INST)
        v = ds.createVariable("y_obs", "f8", ("time", "instrument"))
        v[:] = np.arange(N_TIME * N_INST, dtype=float).reshape(N_TIME, N_INST)
        v.units = "ng m-3"
        ds.createVariable("valid_mask", "i1", ("time", "instrument"))[:] = 1
        ds.createVariable("noise_variance", "f8", ("time", "instrument"))[:] = 1.0
    return _Up({"jacobian": tmp_path / "jac.npz",
                "basis_mapping": tmp_path / "basis.npz",
                "truth_field": tmp_path / "truth.nc"}), obs, G


def _cfg(**input_kw):
    return {"input": {"mode": "instrument_netcdf", **input_kw},
            "observations": {"default_sigma": 1.0},
            "inversion": {"method": "nonnegative"}}


def test_range_selects_the_matching_frames(upstream):
    up, obs, _ = upstream
    G, y, Se, names, *_rest = build_from_prebuilt_operator_with_instrument(
        _cfg(time_reduce="none",
             time_index_range=[FIRST_FRAME, FIRST_FRAME + N_INTERVAL]), up, obs)
    assert G.shape == (N_INTERVAL * N_INST, N_COARSE)
    assert len(names) == N_COARSE
    # Frames 1..3 of a (time, instrument) grid numbered 0..11, flattened
    # instrument-major: all times of s0, then all times of s1.
    np.testing.assert_allclose(y, [2.0, 4.0, 6.0, 3.0, 5.0, 7.0])


def test_range_keeps_operator_rows_aligned_with_their_frames(upstream):
    """Row (t, i) of the operator must meet observation (t, i), not another."""
    up, obs, G_raw = upstream
    W = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    G, _y, *_rest = build_from_prebuilt_operator_with_instrument(
        _cfg(time_reduce="none",
             time_index_range=[FIRST_FRAME, FIRST_FRAME + N_INTERVAL]), up, obs)
    expected = np.stack([
        G_raw[t * N_INST + i] @ (W / W.sum(axis=1)[:, None]).T
        for i in range(N_INST) for t in range(N_INTERVAL)
    ])
    np.testing.assert_allclose(G, expected, rtol=1e-12)


def test_out_of_bounds_range_is_refused(upstream):
    up, obs, _ = upstream
    with pytest.raises(ValueError, match="not a valid half-open range"):
        build_from_prebuilt_operator_with_instrument(
            _cfg(time_reduce="none", time_index_range=[0, N_TIME + 1]), up, obs)


def test_stale_time_reduce_mean_is_refused(upstream):
    """'mean' cannot silently no-op against a time-resolved operator.

    The averaging branch only fires for a one-row-per-instrument operator, so
    against a time-resolved one it would leave the full series in place while
    the config claimed otherwise.
    """
    up, obs, _ = upstream
    # An operator covering every observation frame, so the row count matches
    # and only the guard stands between the config and a silent no-op.
    jac = up.file("jacobian")
    rng = np.random.default_rng(1)
    np.savez(jac, G=rng.uniform(1e5, 1e6, (N_TIME * N_INST, N_FINE)),
             row_labels=np.array([f"t{t}|s{i}" for t in range(N_TIME)
                                  for i in range(N_INST)]),
             column_labels=np.array([f"cell_{j}" for j in range(N_FINE)]),
             units=np.array("ng m-3 / (kg s-1)"))
    with pytest.raises(ValueError, match="operator is time-resolved"):
        build_from_prebuilt_operator_with_instrument(
            _cfg(time_reduce="mean"), up, obs)


def test_a_range_the_operator_does_not_cover_is_refused(upstream):
    """Selecting more frames than the operator has intervals must not pass."""
    up, obs, _ = upstream
    with pytest.raises(ValueError, match="time_index_range to the frames"):
        build_from_prebuilt_operator_with_instrument(
            _cfg(time_reduce="none", time_index_range=[0, N_TIME]), up, obs)
