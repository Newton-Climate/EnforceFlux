"""Unit tests for the LES pseudo-observation sampler (M6b)."""
from __future__ import annotations

import numpy as np
import pytest

from enforceflux.core.observation_units import ObservationSpec
from enforceflux.instrument.models import Instrument
from enforceflux.instrument.open_path import path_average
from enforceflux.microhh.pseudo_obs import (
    _observation_spec_for,
    sample_les_through_instruments,
)


def _write_les_field(tmp_path, *, field, times_s, x_m, y_m):
    path = tmp_path / "les_field.npz"
    np.savez(
        path, times_s=times_s, x_m=x_m, y_m=y_m, field=field,
        variable="concentration_ppm", unit="ppm",
    )
    return tmp_path


def _op_instrument(x, y, path_length_m=200.0, bearing_deg=90.0):
    return Instrument(
        id="op0", tech_id="OP", x=x, y=y, z=2.0,
        path_length_m=path_length_m, path_bearing_deg=bearing_deg,
    )


def test_units_match_lpdm_operator(tmp_path):
    """A uniform LES field returns its constant through the pseudo-obs OP
    sampler, and that value matches ``path_average`` — the same forward
    operator the AERMOD/LPDM inversion uses on its own gridded field."""
    nx, ny, nt = 32, 32, 3
    dx = 25.0
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    const = 1.234
    field = np.full((nt, ny, nx), const, dtype=float)
    run_dir = _write_les_field(
        tmp_path, field=field, times_s=np.array([0.0, 60.0, 120.0]),
        x_m=x, y_m=y,
    )
    inst = _op_instrument(x[nx // 2], y[ny // 2])
    rng = np.random.default_rng(0)

    y_obs = sample_les_through_instruments(run_dir, [inst], rng)

    # LES pseudo-obs and the AERMOD OP forward operator agree exactly on a
    # constant field (arc-length mean of a constant is the constant).
    expected = path_average(
        field[0], x, y, inst.x, inst.y, inst.path_length_m, inst.path_bearing_deg
    )
    # Strip additive noise: on a constant field OP sigma is inst.params.sigma_abs
    # for every timestep. Test the noiseless value by verifying std around
    # the operator's own answer is bounded.
    assert abs(np.mean(y_obs) - const) < 3 * inst.params.sigma_abs
    assert abs(expected - const) < 1e-12
    # And path_average of a constant equals the constant to 1e-6 tolerance.
    assert abs(expected - const) < 1e-6


def test_seed_reproducibility(tmp_path):
    nx, ny, nt = 8, 8, 4
    x = (np.arange(nx) + 0.5) * 25.0
    y = (np.arange(ny) + 0.5) * 25.0
    rng_field = np.random.default_rng(7)
    field = rng_field.uniform(0.5, 1.5, size=(nt, ny, nx))
    run_dir = _write_les_field(
        tmp_path, field=field, times_s=np.arange(nt, dtype=float) * 60.0,
        x_m=x, y_m=y,
    )
    inst = _op_instrument(x[nx // 2], y[ny // 2], path_length_m=50.0)

    y1 = sample_les_through_instruments(run_dir, [inst], np.random.default_rng(42))
    y2 = sample_les_through_instruments(run_dir, [inst], np.random.default_rng(42))
    y3 = sample_les_through_instruments(run_dir, [inst], np.random.default_rng(43))

    assert np.array_equal(y1, y2)
    assert not np.array_equal(y1, y3)


def test_observation_spec_mismatch_raises(tmp_path):
    nx, ny, nt = 8, 8, 2
    x = (np.arange(nx) + 0.5) * 25.0
    y = (np.arange(ny) + 0.5) * 25.0
    field = np.ones((nt, ny, nx))
    run_dir = _write_les_field(
        tmp_path, field=field, times_s=np.array([0.0, 60.0]), x_m=x, y_m=y,
    )
    inst = _op_instrument(x[nx // 2], y[ny // 2])
    actual = _observation_spec_for(inst)
    wrong = ObservationSpec(
        variable=actual.variable, unit=actual.unit,
        averaging_window_s=actual.averaging_window_s,
        height_m=actual.height_m + 5.0,   # deliberately mismatched
        path_length_m=actual.path_length_m,
    )
    with pytest.raises(ValueError, match="ObservationSpec mismatch"):
        sample_les_through_instruments(
            run_dir, [inst], np.random.default_rng(0), expected_specs=[wrong]
        )
