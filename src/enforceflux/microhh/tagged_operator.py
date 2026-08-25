"""Build an LES transport operator by tagging each source cell as its own tracer.

A nature run bakes one emission field into ``<scalar>_bot_in.0000000`` and
returns ``H·E`` for that single field — a matrix-vector product, from which
``H`` itself cannot be recovered. Emitting a *separate* passive tracer from
each active surface cell recovers one column of ``H`` per tracer from a single
LES, because the emitted scalar is passive: it never touches momentum,
buoyancy, or the adaptive timestep, so all tracers share one flow and
concentration is exactly linear in the emission field.

The operator is exact rather than approximate. ``surface_flux_field`` rasterises
the 40 m truth field onto the LES surface grid, so the LES only ever sees the
rasterised boundary condition — two truth fields with the same raster produce
identical output, and an operator at LES cell resolution loses nothing.

Two nonlinearities remain, both limiters on the scalar: ``[advec]
fluxlimit_list`` and the ``[limiter]`` positivity clip
(``microhh/src/limiter.cxx``). Each clips *upward* where a field would go
negative, and fires more readily on an individually small tracer than on their
sum, so ``H·E`` can be biased slightly high. This module therefore reproduces
the nature runs' numerics exactly, so :mod:`compare_tagged_operator` measures
that error rather than hiding it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# MicroHH reads the static 2-D bottom boundary from `<name>_bot_in.0000000`
# regardless of the run's start time (microhh/src/boundary.cxx:640).
BOT_STAMP = "0000000"

# `.ini` keys that must name every transported scalar.
_SCALAR_LIST_KEYS = (
    "slist",
    "sbot_2d_list",
    "scalar_outflow",
    "limitlist",
    "fluxlimit_list",
)


@dataclass(frozen=True)
class TaggedCase:
    """A written multi-tracer MicroHH case and the cells its tracers represent."""

    case_dir: Path
    case_name: str
    tracers: tuple[str, ...]
    # (n_tracer, 2) array of (j, i) indices into the (jtot, itot) surface grid.
    cells: np.ndarray
    reference_kinematic_flux: float
    grid: tuple[int, int, int]      # (itot, jtot, ktot)
    dtype: np.dtype

    @property
    def n_tracers(self) -> int:
        return len(self.tracers)


def read_ini(path: Path) -> dict[str, str]:
    """Flatten a MicroHH ``.ini`` to ``{key: value}``.

    Section names are dropped: the keys this module needs (grid extents,
    timing, cross-section cadence) are unique across sections.
    """
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _set_ini_key(text: str, key: str, value: str) -> str:
    """Replace ``key=...`` in place, leaving the rest of the namelist alone."""
    pattern = rf"(?m)^(\s*{re.escape(key)}\s*=).*$"
    if not re.search(pattern, text):
        return text
    return re.sub(pattern, rf"\g<1>{value}", text)


def surface_field(case_dir: Path, scalar: str, grid: tuple[int, int, int],
                  dtype: np.dtype) -> np.ndarray:
    """Read ``<scalar>_bot_in.0000000`` as a ``(jtot, itot)`` kinematic flux."""
    itot, jtot, _ = grid
    path = case_dir / f"{scalar}_bot_in.{BOT_STAMP}"
    if not path.is_file():
        raise FileNotFoundError(f"No surface-flux boundary condition at {path}")
    return np.fromfile(path, dtype=dtype).reshape(jtot, itot)


def build_tagged_case(
    template_case_dir: Path,
    out_dir: Path,
    *,
    scalar: str = "ch4",
    case_name: str | None = None,
    reference_kinematic_flux: float | None = None,
    keep_xz_cross: bool = False,
    positivity_limiter: bool = True,
) -> TaggedCase:
    """Write a case emitting one tracer per active cell of the template's BC.

    ``template_case_dir`` is a completed nature case: its grid, forcing, and
    timing are adopted verbatim so the operator and the runs it is validated
    against are numerically comparable. Only the scalar list and the surface
    boundary conditions change.
    """
    template_case_dir = Path(template_case_dir)
    out_dir = Path(out_dir)

    inis = sorted(template_case_dir.glob("*.ini"))
    if len(inis) != 1:
        raise ValueError(
            f"Expected exactly one .ini in {template_case_dir}, found {len(inis)}"
        )
    template_ini = inis[0]
    name = case_name or template_ini.stem

    keys = read_ini(template_ini)
    grid = (int(keys["itot"]), int(keys["jtot"]), int(keys["ktot"]))
    # MicroHH's build precision, which the raw binary slices are written in.
    dtype = np.dtype("<f4") if _is_single_precision(template_case_dir) else np.dtype("<f8")

    bot = surface_field(template_case_dir, scalar, grid, dtype)
    cells = np.argwhere(bot > 0.0)
    if cells.size == 0:
        raise ValueError(
            f"Template {template_case_dir} has an all-zero surface flux; there "
            "are no source cells to tag."
        )
    ref = (
        float(reference_kinematic_flux)
        if reference_kinematic_flux is not None
        else float(bot[bot > 0.0].mean())
    )
    if not ref > 0.0:
        raise ValueError(f"reference_kinematic_flux must be positive, got {ref}")

    tracers = tuple(f"{scalar}_{k:04d}" for k in range(len(cells)))

    out_dir.mkdir(parents=True, exist_ok=True)
    # Inputs only: the template's own outputs and restart state must not be
    # inherited, or `init` will refuse to overwrite them.
    shutil.copy2(template_ini, out_dir / f"{name}.ini")
    shutil.copy2(
        template_case_dir / f"{template_ini.stem}_input.nc",
        out_dir / f"{name}_input.nc",
    )

    joined = ",".join(tracers)
    text = (out_dir / f"{name}.ini").read_text()
    for key in _SCALAR_LIST_KEYS:
        text = _set_ini_key(text, key, joined)
    if not positivity_limiter:
        # The `[limiter]` clip is the dominant departure from superposition: it
        # floors every tracer independently, so n tracers raise the far-field
        # floor n-fold and bias the sum upward. Emptying `limitlist` removes it
        # while leaving `fluxlimit_list` — and therefore the ghost-cell layout
        # (advec_2i5.cxx:44) and restart compatibility — untouched. The cost is
        # small negative undershoots in the far field.
        text = _set_ini_key(text, "limitlist", "")
    # Cross-sections are the operator's output: one xy plane per tracer. The
    # template's `ch4_path` column integral would double the file count for no
    # gain, so drop it.
    text = _set_ini_key(text, "crosslist", joined)
    if not keep_xz_cross:
        # One xz slice per tracer per output time is ~9k files the operator
        # never reads — it lives entirely on the xy measurement plane.
        text = re.sub(r"(?m)^\s*xz\s*=.*$\n?", "", text)
    text = (
        f"# Tagged-tracer transport operator built from {template_case_dir}\n"
        f"# {len(tracers)} tracers, one per active surface cell; "
        f"reference kinematic flux {ref:.6e}\n"
    ) + text
    (out_dir / f"{name}.ini").write_text(text)

    _extend_input_nc(out_dir / f"{name}_input.nc", tracers)

    # One indicator boundary condition per tracer: the reference flux in its own
    # cell, zero everywhere else.
    for tracer, (j, i) in zip(tracers, cells):
        field = np.zeros((grid[1], grid[0]), dtype=dtype)
        field[j, i] = ref
        field.tofile(out_dir / f"{tracer}_bot_in.{BOT_STAMP}")

    return TaggedCase(
        case_dir=out_dir,
        case_name=name,
        tracers=tracers,
        cells=cells,
        reference_kinematic_flux=ref,
        grid=grid,
        dtype=dtype,
    )


def _is_single_precision(case_dir: Path) -> bool:
    """Infer the build precision from the size of a written 2-D slice."""
    inis = sorted(case_dir.glob("*.ini"))
    keys = read_ini(inis[0])
    n = int(keys["itot"]) * int(keys["jtot"])
    for candidate in sorted(case_dir.glob("*_bot_in.*")):
        size = candidate.stat().st_size
        if size == n * 4:
            return True
        if size == n * 8:
            return False
    raise ValueError(
        f"Cannot infer MicroHH precision from {case_dir}: no *_bot_in.* slice "
        f"matches {n} cells at 4 or 8 bytes."
    )


def _extend_input_nc(path: Path, tracers: tuple[str, ...]) -> None:
    """Add a zero initial profile and inflow profile for every tracer."""
    import netCDF4 as nc

    with nc.Dataset(path, "a") as ds:
        init = ds.groups["init"]
        nz = len(ds.dimensions["z"])
        zeros = np.zeros(nz)
        for tracer in tracers:
            for var in (tracer, f"{tracer}_inflow"):
                if var not in init.variables:
                    init.createVariable(var, "f8", ("z",))[:] = zeros


def run_tagged_case(
    case: TaggedCase,
    *,
    executable: Path,
    num_workers: int = 1,
    restart_from: Path | None = None,
    restart_time_s: int | None = None,
) -> None:
    """Invoke ``microhh init`` then ``microhh run`` for a tagged case.

    Warm-starting from the donor that the nature runs used is what makes the
    operator comparable to them: it puts every column of ``H`` in the same
    turbulent realization the nature runs integrate.
    """
    from enforceflux.microhh.runner import seed_restart

    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(f"MicroHH executable not found at {executable}")

    launcher: list[str] = []
    if num_workers > 1:
        mpi = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpi is None:
            raise RuntimeError(
                f"num_workers={num_workers} needs mpirun or mpiexec on PATH."
            )
        launcher = [mpi, "-n", str(num_workers)]

    argv = [*launcher, str(executable)]
    subprocess.run([*argv, "init", case.case_name], cwd=case.case_dir, check=True)

    if restart_from is not None:
        if restart_time_s is None:
            raise ValueError("restart_from requires restart_time_s")
        seed_restart(
            case_dir=case.case_dir,
            donor=Path(restart_from),
            time_s=int(restart_time_s),
            emitted_scalars=case.tracers,
        )

    subprocess.run([*argv, "run", case.case_name], cwd=case.case_dir, check=True)


def read_operator(
    case: TaggedCase,
    *,
    level_index: int,
    since_s: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the tagged cross-sections into ``H``.

    Returns ``(times_s, H)`` where ``H`` has shape
    ``(n_time, jtot, itot, n_tracer)`` and maps a *kinematic surface flux* per
    source cell to the scalar mixing ratio on the cross-section plane. Applying
    it to a nature run's own ``<scalar>_bot_in.0000000`` therefore needs no unit
    conversion.
    """
    itot, jtot, _ = case.grid
    per_tracer: list[np.ndarray] = []
    times: np.ndarray | None = None

    for tracer in case.tracers:
        pattern = f"{tracer}.xy.000.{level_index:05d}.*"
        files = sorted(case.case_dir.glob(pattern))
        stamps = np.array([int(f.name.rsplit(".", 1)[-1]) for f in files])
        keep = stamps >= since_s
        files = [f for f, k in zip(files, keep) if k]
        stamps = stamps[keep]
        if not files:
            raise FileNotFoundError(
                f"No cross-sections matching {pattern} in {case.case_dir} at or "
                f"after t={since_s}s. The tagged case must be run first."
            )
        if times is None:
            times = stamps
        elif not np.array_equal(times, stamps):
            raise ValueError(
                f"Tracer {tracer} has cross-section times {stamps.tolist()}, "
                f"which differ from {times.tolist()}"
            )
        frames = np.stack(
            [np.fromfile(f, dtype=case.dtype).reshape(jtot, itot) for f in files]
        )
        per_tracer.append(frames.astype(np.float32))

    assert times is not None
    # (n_tracer, n_time, jtot, itot) -> (n_time, jtot, itot, n_tracer)
    H = np.moveaxis(np.stack(per_tracer), 0, -1)
    return times, H / np.float32(case.reference_kinematic_flux)


@dataclass(frozen=True)
class LoadedOperator:
    """A tagged-tracer operator read back from ``operator.npz``."""

    times_s: np.ndarray            # (n_time,) MicroHH iteration stamps
    H: np.ndarray                  # (n_time, jtot, itot, n_tracer)
    cells: np.ndarray              # (n_tracer, 2) as (j, i)
    reference_kinematic_flux: float
    level_index: int               # cross-section plane H was read from

    @property
    def grid(self) -> tuple[int, int]:
        """``(jtot, itot)`` of the plane the operator predicts."""
        return int(self.H.shape[1]), int(self.H.shape[2])


def load_operator(path: Path) -> LoadedOperator:
    """Read an ``operator.npz`` written by the tagged-tracer build."""
    path = Path(path)
    with np.load(path) as d:
        if "level_index" not in d.files:
            raise ValueError(
                f"{path} predates the level_index field, so the plane it was "
                "read from is unknown. Re-run `run_les_tagged_operator.py "
                "compare` to rewrite it."
            )
        return LoadedOperator(
            times_s=np.asarray(d["times_s"], dtype=int),
            H=np.asarray(d["H"], dtype=np.float32),
            cells=np.asarray(d["cells"], dtype=int),
            reference_kinematic_flux=float(d["reference_kinematic_flux"]),
            level_index=int(d["level_index"]),
        )


def synthesize_cross_sections(cfg, operator_path: Path) -> dict:
    """Write a case's xy cross-sections as ``H @ e`` instead of integrating it.

    The emitted scalar is passive, so a case that shares the operator's grid,
    flow, and source footprint has a cross-section time series that is an exact
    linear function of its own bottom boundary condition — up to the scalar
    limiters, whose cost is measured in the operator's ``validation.json``.
    This writes the same raw ``(jtot, itot)`` binaries MicroHH would have
    written, so every downstream reader is unchanged.

    ``cfg`` is a :class:`~enforceflux.microhh.sim_config.MicroHHConfig` whose
    case has already been written (the surface BC must exist on disk).
    """
    op = load_operator(operator_path)
    case_dir = Path(cfg.case_dir)
    dtype = np.dtype("<f4") if cfg.precision == "float32" else np.dtype("<f8")

    jtot, itot = op.grid
    if (cfg.grid.itot, cfg.grid.jtot) != (itot, jtot):
        raise ValueError(
            f"Operator {operator_path} is for a ({itot}, {jtot}) (itot, jtot) "
            f"grid, but this case is ({cfg.grid.itot}, {cfg.grid.jtot}). An "
            "operator is only valid on the grid it was integrated on."
        )

    bot_path = case_dir / f"{cfg.scalar_name}_bot_in.{BOT_STAMP}"
    if not bot_path.is_file():
        raise FileNotFoundError(
            f"{bot_path} is missing. The operator multiplies a case's own "
            "surface boundary condition, so the case must be written (and must "
            "define surface-flux sources) before it can be evaluated."
        )
    bot = np.fromfile(bot_path, dtype=dtype).reshape(jtot, itot)

    # Emission outside the tagged cells has no column in H and would be
    # silently dropped, turning a configuration error into a quiet low bias.
    tagged = np.zeros((jtot, itot), dtype=bool)
    tagged[op.cells[:, 0], op.cells[:, 1]] = True
    stray = np.argwhere((bot > 0.0) & ~tagged)
    if stray.size:
        raise ValueError(
            f"{len(stray)} emitting cell(s) fall outside the operator's "
            f"{len(op.cells)} tagged cells, e.g. (j, i)={tuple(stray[0])}. The "
            "source footprint must match the one the operator was built for."
        )

    # `canonical.from_microhh` discards cross-sections stamped before the
    # case's spinup. An operator whose window sits entirely inside that
    # discard would write files nobody reads, and fail two stages later.
    spinup_s = int(getattr(cfg, "spinup_s", 0))
    if not (op.times_s >= spinup_s).any():
        raise ValueError(
            f"The operator's frames span {int(op.times_s.min())}.."
            f"{int(op.times_s.max())} s, all before this case's "
            f"spinup_seconds={spinup_s}. Every synthesised cross-section "
            "would be discarded when the case is read back; set "
            "spinup_seconds to 0 for a case restarted into the operator's "
            "window."
        )

    values = apply_operator(op.H, bot, op.cells)

    # MicroHH stamps each cross-section with the iteration time; keeping the
    # operator's own stamps is what makes the output indistinguishable from a
    # real run to `canonical.from_microhh`.
    written = []
    for value, stamp in zip(values, op.times_s):
        out = case_dir / f"{cfg.scalar_name}.xy.000.{op.level_index:05d}.{int(stamp):07d}"
        value.astype(dtype).tofile(out)
        written.append(out)

    rho = _reference_density_for(cfg)
    cell_area = cfg.grid.dx * cfg.grid.dy
    return {
        "backend": "les_tagged_operator",
        "operator": str(operator_path),
        "level_index": op.level_index,
        "n_frames": len(written),
        "times_s": [int(s) for s in op.times_s],
        "n_source_cells": int(len(op.cells)),
        "emitted_kg_s": float(bot.sum() * rho * cell_area),
        "peak_mixing_ratio": float(values.max()),
    }


def _reference_density_for(cfg) -> float:
    from enforceflux.microhh.case import _reference_density

    return _reference_density(cfg)


def apply_operator(H: np.ndarray, surface_flux: np.ndarray,
                   cells: np.ndarray) -> np.ndarray:
    """Predict the cross-section time series for an emission field.

    ``surface_flux`` is a ``(jtot, itot)`` kinematic flux — a nature run's own
    ``<scalar>_bot_in.0000000`` — sampled at ``cells`` to form the emission
    vector that ``H`` multiplies.
    """
    e = surface_flux[cells[:, 0], cells[:, 1]].astype(np.float32)
    return H @ e


__all__ = [
    "LoadedOperator",
    "TaggedCase",
    "apply_operator",
    "build_tagged_case",
    "load_operator",
    "read_ini",
    "read_operator",
    "synthesize_cross_sections",
    "run_tagged_case",
    "surface_field",
]
