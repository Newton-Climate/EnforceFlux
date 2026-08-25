"""Contract tests for the time-resolved bLS operator.

The feature spans two modules that never import each other: the transport
runner decides what an operator row *means*, and the flux stage decides which
observation each row is paired with. Nothing in the type system ties them
together, so a change to either ordering convention would produce a wrong
inversion rather than an error — every row would still line up with *an*
observation, just the wrong one.

These tests pin that seam, plus the backward-compatible behaviour of the
paths that existed before ``interval_reduce: none``.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "apps", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from enforceflux.instrument import Instrument  # noqa: E402
from enforceflux.plugins.transport_blsmodelr import BlsTransportOperator  # noqa: E402

requires_rscript = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript not on PATH; needed even for the bLS dry-run shim.",
)

NX = NY = 2                       # 4 fine cells
N_COARSE = 2                      # left half / right half
N_INST = 2
N_INTERVAL = 3
# The observation file is longer than the operator's window, as it is in
# practice: usable wall-model met covers only part of the run.
N_FRAME = 5
FIRST_FRAME = 1


def _instruments() -> list[Instrument]:
    return [
        Instrument(id="s0", tech_id="OP", x=-200.0, y=-600.0, z=2.0),
        Instrument(id="s1", tech_id="OP", x=200.0, y=-600.0, z=2.0),
    ]


def _config(interval_reduce: str) -> dict:
    """Three intervals differing as real one-minute windows do.

    Wind speed is varied deliberately: the dry-run R shim is an analytic stub
    that responds to wind speed but not to direction or u*, so intervals that
    differed only in those would yield identical rows and quietly disarm the
    ordering assertions below. The real bLS responds to all of them; the
    ``_assert_rows_differ`` precondition is what keeps this dependence on stub
    behaviour from becoming a silent false pass.
    """
    base = dict(id="t0", u_star=0.30, L=-30.0, z0=0.1,
                wind_dir_deg=10.0, wind_speed=2.0, z_ref=2.0)
    intervals = []
    for k, (direction, ustar, speed) in enumerate(
        ((10.0, 0.30, 1.2), (35.0, 0.25, 2.0), (60.0, 0.40, 3.0))
    ):
        intervals.append({**base, "id": f"t{k}", "wind_dir_deg": direction,
                          "u_star": ustar, "wind_speed": speed})
    return {
        "source_grid": {"x_bounds": [-500.0, 500.0],
                        "y_bounds": [-500.0, 500.0], "nx": NX, "ny": NY},
        "intervals": intervals,
        "interval_reduce": interval_reduce,
        "wrapper": {"dry_run": True, "rscript": "Rscript"},
    }


def _basis() -> np.ndarray:
    """Left/right split of the 2x2 fine grid, row-major like build_source_grid."""
    return np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])


def _assert_rows_differ(g: np.ndarray) -> None:
    """Every interval block must be distinguishable, or the seam tests are vacuous."""
    blocks = g.reshape(N_INTERVAL, N_INST, -1)
    for a in range(N_INTERVAL):
        for b in range(a + 1, N_INTERVAL):
            assert not np.allclose(blocks[a], blocks[b]), (
                f"interval blocks {a} and {b} are identical, so row ordering "
                "cannot be tested; the bLS backend stopped responding to the "
                "parameters these intervals vary"
            )


class _Up:
    def __init__(self, files: dict[str, Path]) -> None:
        self._files = files

    def file(self, role: str) -> Path:
        return self._files[role]


def _write_upstream(tmp_path: Path, g: np.ndarray, x_true: np.ndarray,
                    y_grid: np.ndarray) -> tuple[_Up, Path]:
    """Write the three artefacts the flux stage reads, as the pipeline does."""
    from netCDF4 import Dataset

    W = _basis()
    np.savez(tmp_path / "basis.npz", W=W,
             fine_cell_areas_m2=np.full(NX * NY, 250_000.0),
             coarse_cell_areas_m2=np.full(N_COARSE, 500_000.0),
             coarse_centers_m=np.zeros((N_COARSE, 2)))
    np.savez(tmp_path / "jac.npz", G=g,
             row_labels=np.array([f"row{i}" for i in range(g.shape[0])]),
             column_labels=np.array([f"cell_{j}" for j in range(g.shape[1])]),
             units=np.array("ng m-3 / (kg s-1)"))
    with Dataset(tmp_path / "truth.nc", "w") as ds:
        ds.createDimension("y", NY)
        ds.createDimension("x", NX)
        # F_true is a flux density; the loader multiplies by cell area.
        ds.createVariable("F_true", "f8", ("y", "x"))[:] = (
            (x_true @ _basis()) / 250_000.0).reshape(NY, NX)
        ds.createVariable("cell_area_m2", "f8", ("y", "x"))[:] = np.full(
            (NY, NX), 250_000.0)
        ds.L_true_m = 100.0
    obs = tmp_path / "obs.nc"
    with Dataset(obs, "w") as ds:
        ds.createDimension("time", y_grid.shape[0])
        ds.createDimension("instrument", y_grid.shape[1])
        v = ds.createVariable("y_obs", "f8", ("time", "instrument"))
        v[:] = y_grid
        v.units = "ng m-3"
        ds.createVariable("valid_mask", "i1", ("time", "instrument"))[:] = 1
        ds.createVariable("noise_variance", "f8", ("time", "instrument"))[:] = 1.0
    return _Up({"jacobian": tmp_path / "jac.npz",
                "basis_mapping": tmp_path / "basis.npz",
                "truth_field": tmp_path / "truth.nc"}), obs


# ── the seam: operator rows must meet the observations they describe ───────


@requires_rscript
def test_operator_rows_pair_with_their_own_observation_frames(tmp_path):
    """End-to-end: a noiseless forward run must invert back exactly.

    Observations are generated FROM the operator, frame by frame, so any
    mismatch between the runner's row ordering and the flux stage's pairing
    shows up as a residual. This is the assertion that would catch either
    convention silently flipping.
    """
    from flux_inputs import build_from_prebuilt_operator_with_instrument

    g = BlsTransportOperator().build_forward_operator(
        sources=[], instruments=_instruments(), domain=None,
        config=_config("none"),
    ).g
    assert g.shape == (N_INTERVAL * N_INST, NX * NY)
    _assert_rows_differ(g)

    W = _basis()
    g_coarse = g @ (W / W.sum(axis=1)[:, None]).T
    x_true = np.array([0.02, 0.008])

    # Frame FIRST_FRAME + t of the observation grid is interval t. Every other
    # frame is filled with a decoy that must never be selected.
    y_grid = np.full((N_FRAME, N_INST), -999.0)
    for t in range(N_INTERVAL):
        for i in range(N_INST):
            y_grid[FIRST_FRAME + t, i] = g_coarse[t * N_INST + i] @ x_true

    up, obs = _write_upstream(tmp_path, g, x_true, y_grid)
    G, y, _Se, names, *_rest = build_from_prebuilt_operator_with_instrument(
        {"input": {"mode": "instrument_netcdf", "time_reduce": "none",
                   "time_index_range": [FIRST_FRAME, FIRST_FRAME + N_INTERVAL]},
         "observations": {"default_sigma": 1.0},
         "inversion": {"method": "nonnegative"}},
        up, obs,
    )

    assert G.shape == (N_INTERVAL * N_INST, N_COARSE)
    assert len(names) == N_COARSE
    assert not np.any(y == -999.0), "a frame outside the operator's window leaked in"
    np.testing.assert_allclose(G @ x_true, y, rtol=1e-9)


@requires_rscript
def test_shuffling_the_operator_rows_breaks_the_pairing(tmp_path):
    """Guard on the guard: the test above must be sensitive to row order."""
    from flux_inputs import build_from_prebuilt_operator_with_instrument

    g = BlsTransportOperator().build_forward_operator(
        sources=[], instruments=_instruments(), domain=None,
        config=_config("none"),
    ).g
    _assert_rows_differ(g)
    W = _basis()
    g_coarse = g @ (W / W.sum(axis=1)[:, None]).T
    x_true = np.array([0.02, 0.008])
    y_grid = np.full((N_FRAME, N_INST), -999.0)
    for t in range(N_INTERVAL):
        for i in range(N_INST):
            y_grid[FIRST_FRAME + t, i] = g_coarse[t * N_INST + i] @ x_true

    # Instrument-major instead of interval-major — the plausible wrong choice.
    wrong = g.reshape(N_INTERVAL, N_INST, -1).transpose(1, 0, 2).reshape(g.shape)
    up, obs = _write_upstream(tmp_path, wrong, x_true, y_grid)
    G, y, *_rest = build_from_prebuilt_operator_with_instrument(
        {"input": {"mode": "instrument_netcdf", "time_reduce": "none",
                   "time_index_range": [FIRST_FRAME, FIRST_FRAME + N_INTERVAL]},
         "observations": {"default_sigma": 1.0},
         "inversion": {"method": "nonnegative"}},
        up, obs,
    )
    assert not np.allclose(G @ x_true, y, rtol=1e-6)


# ── backward compatibility of the pre-existing paths ──────────────────────


@requires_rscript
def test_default_interval_reduce_is_still_mean_and_shape_is_unchanged():
    """Configs written before this feature must behave exactly as before."""
    config = _config("mean")
    del config["interval_reduce"]
    result = BlsTransportOperator().build_forward_operator(
        sources=[], instruments=_instruments(), domain=None, config=config,
    )
    assert result.g.shape == (N_INST, NX * NY)
    assert result.meta["interval_reduce"] == "mean"


@requires_rscript
def test_none_and_mean_agree_on_the_interval_average():
    """'none' must be a refinement of 'mean', not a different operator."""
    op = BlsTransportOperator()
    stacked = op.build_forward_operator(
        sources=[], instruments=_instruments(), domain=None,
        config=_config("none")).g
    averaged = op.build_forward_operator(
        sources=[], instruments=_instruments(), domain=None,
        config=_config("mean")).g
    np.testing.assert_allclose(
        stacked.reshape(N_INTERVAL, N_INST, -1).mean(axis=0), averaged, rtol=1e-9
    )


def test_interval_order_is_ignored_by_the_collapsing_paths():
    """Adding the kwarg must not perturb 'mean'/'sum' for existing callers."""
    from enforceflux.blsmodelr.footprint import jacobian_from_bls_result
    from enforceflux.blsmodelr.wrapper import BlsRunResult

    rows = [("s1", "a", "i0", 1.0), ("s1", "a", "i1", 3.0)]
    result = BlsRunResult(
        sensor=np.array([r[0] for r in rows], dtype=object),
        source=np.array([r[1] for r in rows], dtype=object),
        interval=np.array([r[2] for r in rows], dtype=object),
        cxe=np.array([r[3] for r in rows], dtype=float),
        cxe_se=np.zeros(len(rows)), n_particles_used=np.zeros(len(rows), int),
    )
    kw = dict(result=result, sensor_order=["s1"], source_order=["a"])
    for reduce_mode in ("mean", "sum"):
        without = jacobian_from_bls_result(**kw, interval_reduce=reduce_mode)
        with_order = jacobian_from_bls_result(
            **kw, interval_reduce=reduce_mode, interval_order=["i1", "i0"])
        np.testing.assert_allclose(without, with_order)


# ── config resolution contract ────────────────────────────────────────────


def test_intervals_from_nature_resolves_relative_to_the_config(monkeypatch):
    """The path is config-relative, not cwd-relative.

    A cwd-relative path made the same config resolve differently depending on
    where the command was run from, which surfaces as a missing-file error at
    best and a different met file at worst.
    """
    import enforceflux.transport.runner as runner_mod

    seen: dict[str, object] = {}

    class _Run:
        def resolve(self, value):
            seen["resolved"] = value
            return Path("/resolved") / str(value)

    monkeypatch.setattr(
        "enforceflux.microhh.sim_config.load_microhh_config",
        lambda path: seen.setdefault("loaded", path),
    )
    monkeypatch.setattr(
        "enforceflux.blsmodelr.met_from_les.intervals_from_microhh_output",
        lambda cfg, **kw: [],
    )
    runner_mod._bls_intervals(
        {"intervals_from_nature": {"microhh_config": "../nature/case.yaml",
                                   "ix": 1, "iy": 2, "z_ref": 2.0, "z0": 0.1,
                                   "window_s": 60}},
        _Run(),
    )
    assert seen["resolved"] == "../nature/case.yaml"
    assert seen["loaded"] == Path("/resolved/../nature/case.yaml")


def test_inline_and_from_nature_are_mutually_exclusive():
    import enforceflux.transport.runner as runner_mod

    with pytest.raises(ValueError, match="not both"):
        runner_mod._bls_intervals(
            {"intervals": [{"id": "t0"}],
             "intervals_from_nature": {"microhh_config": "x.yaml"}}, object()
        )
