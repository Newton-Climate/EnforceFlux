"""Tagged-tracer LES transport operator: case construction and reconstruction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from enforceflux.microhh.tagged_operator import (
    BOT_STAMP,
    apply_operator,
    build_tagged_case,
    read_ini,
    read_operator,
    surface_field,
    TaggedCase,
)

ITOT, JTOT, KTOT = 8, 4, 6
LEVEL = 3

INI = f"""\
[grid]
itot={ITOT}
jtot={JTOT}
ktot={KTOT}

[advec]
swadvec=2i5
fluxlimit_list=ch4,h2o

[boundary]
sbot_2d_list=ch4
scalar_outflow=ch4,h2o

[fields]
slist=ch4,h2o

[limiter]
limitlist=ch4,h2o

[cross]
swcross=1
crosslist=ch4,ch4_path
xy=2
xz=960

[time]
starttime=3600
endtime=6300
"""


def _template(tmp_path: Path, field: np.ndarray) -> Path:
    """A minimal stand-in for a completed nature case."""
    case = tmp_path / "template"
    case.mkdir()
    (case / "transport_run.ini").write_text(INI)
    field.astype("<f4").tofile(case / f"ch4_bot_in.{BOT_STAMP}")

    import netCDF4 as nc

    with nc.Dataset(case / "transport_run_input.nc", "w") as ds:
        ds.createDimension("z", KTOT)
        ds.createVariable("z", "f8", ("z",))[:] = np.arange(KTOT, dtype=float)
        init = ds.createGroup("init")
        for var in ("u", "v", "th", "ch4", "ch4_inflow"):
            init.createVariable(var, "f8", ("z",))[:] = np.zeros(KTOT)
    return case


def _two_cell_field() -> np.ndarray:
    field = np.zeros((JTOT, ITOT))
    field[1, 2] = 3.0e-8
    field[2, 5] = 1.0e-8
    return field


def test_one_tracer_per_active_cell(tmp_path):
    case = build_tagged_case(_template(tmp_path, _two_cell_field()), tmp_path / "op")

    assert case.n_tracers == 2
    assert case.cells.tolist() == [[1, 2], [2, 5]]
    # Default reference flux is the mean over active cells.
    assert case.reference_kinematic_flux == pytest.approx(2.0e-8)


def test_each_tracer_emits_only_from_its_own_cell(tmp_path):
    case = build_tagged_case(_template(tmp_path, _two_cell_field()), tmp_path / "op")

    for tracer, (j, i) in zip(case.tracers, case.cells):
        bot = surface_field(case.case_dir, tracer, case.grid, case.dtype)
        assert bot[j, i] == pytest.approx(case.reference_kinematic_flux)
        assert np.count_nonzero(bot) == 1, f"{tracer} emits from more than one cell"


def test_every_scalar_list_names_all_tracers(tmp_path):
    case = build_tagged_case(_template(tmp_path, _two_cell_field()), tmp_path / "op")
    keys = read_ini(case.case_dir / f"{case.case_name}.ini")

    joined = ",".join(case.tracers)
    for key in ("slist", "sbot_2d_list", "scalar_outflow", "limitlist",
                "fluxlimit_list", "crosslist"):
        assert keys[key] == joined, f"{key} does not name every tracer"


def test_numerics_and_timing_are_inherited_verbatim(tmp_path):
    """The operator is only comparable to the nature runs if it matches them."""
    template = _template(tmp_path, _two_cell_field())
    case = build_tagged_case(template, tmp_path / "op")

    before, after = read_ini(template / "transport_run.ini"), read_ini(
        case.case_dir / f"{case.case_name}.ini")
    for key in ("itot", "jtot", "ktot", "swadvec", "starttime", "endtime", "xy"):
        assert after[key] == before[key], f"{key} drifted from the template"


def test_xz_plane_is_dropped_by_default(tmp_path):
    template = _template(tmp_path, _two_cell_field())
    dropped = build_tagged_case(template, tmp_path / "op")
    kept = build_tagged_case(template, tmp_path / "op_xz", keep_xz_cross=True)

    assert "xz" not in read_ini(dropped.case_dir / "transport_run.ini")
    assert read_ini(kept.case_dir / "transport_run.ini")["xz"] == "960"


def test_all_zero_surface_flux_is_refused(tmp_path):
    template = _template(tmp_path, np.zeros((JTOT, ITOT)))
    with pytest.raises(ValueError, match="all-zero surface flux"):
        build_tagged_case(template, tmp_path / "op")


# --- operator round-trip ----------------------------------------------------


def _write_cross_sections(case: TaggedCase, response: np.ndarray,
                          times: list[int]) -> None:
    """response[tracer, time, j, i] -> MicroHH cross-section files."""
    for k, tracer in enumerate(case.tracers):
        for t_idx, t in enumerate(times):
            path = case.case_dir / f"{tracer}.xy.000.{LEVEL:05d}.{t:07d}"
            response[k, t_idx].astype(case.dtype).tofile(path)


def test_operator_reconstructs_a_linear_superposition(tmp_path):
    """H·E must return exactly what a linear model would have produced.

    With MicroHH's limiters off, transport is linear, so this is the property
    the real validation measures the departure from.
    """
    case = build_tagged_case(_template(tmp_path, _two_cell_field()), tmp_path / "op")
    times = [3600, 3660, 3720]
    rng = np.random.default_rng(0)
    # Per-tracer response to the reference flux.
    response = rng.random((case.n_tracers, len(times), JTOT, ITOT)).astype("f4")
    _write_cross_sections(case, response, times)

    read_times, H = read_operator(case, level_index=LEVEL)
    assert read_times.tolist() == times
    assert H.shape == (len(times), JTOT, ITOT, case.n_tracers)

    # A field twice the reference in one cell and half in the other.
    ref = case.reference_kinematic_flux
    field = np.zeros((JTOT, ITOT))
    field[1, 2] = 2.0 * ref
    field[2, 5] = 0.5 * ref
    expected = 2.0 * response[0] + 0.5 * response[1]

    np.testing.assert_allclose(
        apply_operator(H, field, case.cells), expected, rtol=1e-5
    )


def test_reference_flux_is_divided_out(tmp_path):
    """H is per unit kinematic flux, so re-emitting the reference returns it."""
    case = build_tagged_case(_template(tmp_path, _two_cell_field()), tmp_path / "op")
    times = [3600]
    response = np.ones((case.n_tracers, 1, JTOT, ITOT), dtype="f4")
    _write_cross_sections(case, response, times)

    _, H = read_operator(case, level_index=LEVEL)
    uniform = np.zeros((JTOT, ITOT))
    uniform[case.cells[:, 0], case.cells[:, 1]] = case.reference_kinematic_flux

    np.testing.assert_allclose(
        apply_operator(H, uniform, case.cells),
        np.full((1, JTOT, ITOT), float(case.n_tracers)),
        rtol=1e-5,
    )


def test_frames_before_the_restart_are_excluded(tmp_path):
    case = build_tagged_case(_template(tmp_path, _two_cell_field()), tmp_path / "op")
    times = [0, 3600, 3660]
    response = np.ones((case.n_tracers, len(times), JTOT, ITOT), dtype="f4")
    _write_cross_sections(case, response, times)

    read_times, _ = read_operator(case, level_index=LEVEL, since_s=3600)
    assert read_times.tolist() == [3600, 3660]


def test_unrun_case_is_reported_clearly(tmp_path):
    case = build_tagged_case(_template(tmp_path, _two_cell_field()), tmp_path / "op")
    with pytest.raises(FileNotFoundError, match="must be run first"):
        read_operator(case, level_index=LEVEL)


def test_positivity_limiter_can_be_disabled(tmp_path):
    """The clip is the dominant nonlinearity; dropping it must not disturb
    `fluxlimit_list`, which sets the ghost-cell count and so the restart
    layout."""
    template = _template(tmp_path, _two_cell_field())
    on = build_tagged_case(template, tmp_path / "on")
    off = build_tagged_case(template, tmp_path / "off", positivity_limiter=False)

    joined = ",".join(on.tracers)
    assert read_ini(on.case_dir / "transport_run.ini")["limitlist"] == joined
    assert read_ini(off.case_dir / "transport_run.ini")["limitlist"] == ""
    # The advection flux limiter, and thus kgc, is unchanged either way.
    for case in (on, off):
        assert read_ini(case.case_dir / "transport_run.ini")["fluxlimit_list"] == joined
