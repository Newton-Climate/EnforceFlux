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


# ── Evaluating a case through a saved operator ───────────────────────────────


def _operator_npz(path: Path, *, H, cells, ref, times, level=LEVEL) -> Path:
    np.savez_compressed(
        path, times_s=np.asarray(times), H=H, cells=cells,
        reference_kinematic_flux=ref, level_index=level,
    )
    return path


def _case_config(tmp_path: Path, operator_npz: Path | None):
    """A MicroHHConfig on the operator's grid, with one surface-flux patch."""
    from datetime import datetime

    from enforceflux.microhh.sim_config import BoxGrid, Forcing, MicroHHConfig

    return MicroHHConfig(
        executable=tmp_path / "microhh-does-not-exist",
        case_dir=tmp_path / "case",
        case_name="transport_run",
        grid=BoxGrid(itot=ITOT, jtot=JTOT, ktot=KTOT,
                     xsize=ITOT * 80.0, ysize=JTOT * 80.0, zsize=1024.0),
        forcing=Forcing(),
        origin_lon=-121.75, origin_lat=39.15, x_bearing_deg=90.0,
        source_x0=0.5 * ITOT * 80.0, source_y0=0.5 * JTOT * 80.0,
        sources=[], receptors=[],
        output_path=tmp_path / "native.nc",
        start=datetime(2026, 7, 22, 15, 0, 0),
        include_h2o=False,
        operator_npz=operator_npz,
    )


def _write_bot(cfg, field: np.ndarray) -> None:
    cfg.case_dir.mkdir(parents=True, exist_ok=True)
    field.astype("<f4").tofile(cfg.case_dir / f"ch4_bot_in.{BOT_STAMP}")


def _unit_operator(cells: np.ndarray, times) -> np.ndarray:
    """H whose tracer k paints its own emission into cell (0, k)."""
    H = np.zeros((len(times), JTOT, ITOT, len(cells)), dtype=np.float32)
    for k in range(len(cells)):
        H[:, 0, k, k] = 1.0
    return H


def test_synthesized_cross_sections_are_h_times_e(tmp_path):
    from enforceflux.microhh.tagged_operator import synthesize_cross_sections

    cells = np.array([[1, 2], [2, 5]])
    times = [3660, 3720]
    H = _unit_operator(cells, times)
    npz = _operator_npz(tmp_path / "operator.npz", H=H, cells=cells, ref=1.0,
                        times=times)

    cfg = _case_config(tmp_path, npz)
    bot = np.zeros((JTOT, ITOT))
    bot[1, 2], bot[2, 5] = 3.0e-8, 1.0e-8
    _write_bot(cfg, bot)

    info = synthesize_cross_sections(cfg, npz)
    assert info["n_frames"] == len(times)

    for stamp in times:
        written = np.fromfile(
            cfg.case_dir / f"ch4.xy.000.{LEVEL:05d}.{stamp:07d}", dtype="<f4"
        ).reshape(JTOT, ITOT)
        expected = np.zeros((JTOT, ITOT), dtype="f4")
        expected[0, 0], expected[0, 1] = 3.0e-8, 1.0e-8
        np.testing.assert_allclose(written, expected, rtol=1e-6)


def test_emission_outside_the_tagged_cells_is_refused(tmp_path):
    """Untagged emission has no column in H and would silently go missing."""
    from enforceflux.microhh.tagged_operator import synthesize_cross_sections

    cells = np.array([[1, 2], [2, 5]])
    times = [3660]
    npz = _operator_npz(tmp_path / "operator.npz", H=_unit_operator(cells, times),
                        cells=cells, ref=1.0, times=times)

    cfg = _case_config(tmp_path, npz)
    bot = np.zeros((JTOT, ITOT))
    bot[1, 2], bot[3, 7] = 3.0e-8, 1.0e-8      # (3, 7) is not tagged
    _write_bot(cfg, bot)

    with pytest.raises(ValueError, match="outside the operator"):
        synthesize_cross_sections(cfg, npz)


def test_operator_on_a_different_grid_is_refused(tmp_path):
    from enforceflux.microhh.tagged_operator import synthesize_cross_sections

    cells = np.array([[0, 0]])
    times = [3660]
    H = np.zeros((1, JTOT + 1, ITOT, 1), dtype=np.float32)
    npz = _operator_npz(tmp_path / "operator.npz", H=H, cells=cells, ref=1.0,
                        times=times)

    cfg = _case_config(tmp_path, npz)
    _write_bot(cfg, np.zeros((JTOT, ITOT)))

    with pytest.raises(ValueError, match="only valid on the grid"):
        synthesize_cross_sections(cfg, npz)


def test_operator_without_a_level_index_is_refused(tmp_path):
    """An operator that does not say which plane it is for cannot be applied."""
    from enforceflux.microhh.tagged_operator import load_operator

    path = tmp_path / "operator.npz"
    np.savez_compressed(
        path, times_s=np.array([3660]), H=np.zeros((1, JTOT, ITOT, 1), "f4"),
        cells=np.array([[0, 0]]), reference_kinematic_flux=1.0,
    )
    with pytest.raises(ValueError, match="level_index"):
        load_operator(path)


def test_runner_evaluates_the_operator_instead_of_integrating(tmp_path):
    """With an operator set, no MicroHH binary is needed or invoked."""
    from enforceflux.microhh.runner import MicroHHRunner

    cells = np.array([[1, 2]])
    times = [3660, 3720]
    npz = _operator_npz(tmp_path / "operator.npz",
                        H=_unit_operator(cells, times), cells=cells, ref=1.0,
                        times=times)
    cfg = _case_config(tmp_path, npz)
    bot = np.zeros((JTOT, ITOT))
    bot[1, 2] = 3.0e-8
    _write_bot(cfg, bot)

    result = MicroHHRunner(cfg).run()

    assert not cfg.executable.exists()
    assert result.meta["integrated"] is False
    assert result.meta["operator"]["n_frames"] == len(times)
    assert sorted(p.name for p in cfg.case_dir.glob("ch4.xy.*")) == [
        f"ch4.xy.000.{LEVEL:05d}.{t:07d}" for t in times
    ]
