"""Run any transport model from the shared config, with one output contract.

``run_transport`` is the single entry point: it reads the meteorology once,
translates the shared config into the chosen model's native form, dispatches
through the plugin registry, and normalises whatever comes back into a
:class:`TransportRunResult`.

Both modes return the same type:

* ``mode: simulation`` fills ``field``/``output_path`` — a canonical
  ``concentration(time, y, x)`` NetCDF in ng m⁻³, identical in layout for all
  three models (see :mod:`enforceflux.transport.canonical`).
* ``mode: operator`` fills ``g`` with the observation × source Jacobian, plus
  ``row_labels``/``column_labels`` so the rows stay identifiable.

Backends that need a compiled binary (FLEXPART, MicroHH) are dispatched exactly
the same way; ``dry_run`` stops after their input files are generated, which is
also what happens when the binary is absent.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from enforceflux.core.base import ITransportOperator, ITransportSimulation
from enforceflux.meteo.record import MetSeries
from enforceflux.transport import canonical, translate
from enforceflux.transport.canonical import CanonicalField
from enforceflux.transport.run_config import TransportRunConfig
from enforceflux.utils.plugin_registry import get_plugin

OPERATOR_UNITS = "ng m-3 / (kg s-1)"


@dataclass(frozen=True)
class TransportRunResult:
    """What a transport run returns, whatever the model and mode."""

    model: str
    mode: str
    units: str
    output_path: Path | None = None
    field: CanonicalField | None = None
    g: np.ndarray | None = None
    row_labels: tuple[Any, ...] = ()
    column_labels: tuple[str, ...] = ()
    native_output: Path | None = None
    generated_config: Path | None = None
    met: MetSeries | None = None
    # dataclasses.field is qualified: the 'field' attribute above shadows it here.
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.model} / {self.mode} — units: {self.units}"]
        if self.g is not None:
            lines.append(f"  Jacobian: {self.g.shape[0]} observations × {self.g.shape[1]} sources")
            lines.append(f"  range: {self.g.min():.4g} … {self.g.max():.4g}")
        if self.field is not None:
            peak = self.field.peak()
            lines.append(f"  field: {self.field.shape} (time, y, x)")
            lines.append(
                f"  peak: {peak['value']:.4g} {self.units} at "
                f"x={peak['x_m']:.0f} m, y={peak['y_m']:.0f} m"
                + (f", {peak['timestamp']}" if peak["timestamp"] else "")
            )
        if self.output_path:
            lines.append(f"  canonical output: {self.output_path}")
        if self.native_output:
            lines.append(f"  native output:    {self.native_output}")
        if self.generated_config:
            lines.append(f"  generated config: {self.generated_config}")
        return "\n".join(lines)


def run_transport(
    run: TransportRunConfig, *, dry_run: bool = False
) -> TransportRunResult:
    """Execute a transport run described by the shared config."""
    series = translate.build_met_series(run)
    run_dir = run.output.path.parent / f"{run.output.path.stem}_{run.model}"

    if run.mode == "operator":
        return _run_operator(run, series, run_dir, dry_run=dry_run)
    return _run_simulation(run, series, run_dir, dry_run=dry_run)


# ── Operator mode ────────────────────────────────────────────────────────────


def _run_operator(
    run: TransportRunConfig, series: MetSeries, run_dir: Path, *, dry_run: bool
) -> TransportRunResult:
    operator = get_plugin("enforceflux.transport_operator", run.model, ITransportOperator)()
    sources = translate.projected_sources(run)
    column_labels = tuple(s.id for s in run.sources)

    if run.model == "aermod":
        config = translate.aermod_config(run, series)
        # Receptors come from the shared config, so no Instrument objects needed.
        result = operator.build_forward_operator(sources, [], None, config)
        row_labels = _aermod_row_labels(run, config)
    else:
        config, generated = _binary_model_config(run, series, run_dir)
        config["dry_run"] = dry_run
        result = operator.build_forward_operator(sources, [], None, config)
        row_labels = tuple(r.id for r in run.receptors)
        return TransportRunResult(
            model=run.model,
            mode="operator",
            units=str(result.meta.get("units", OPERATOR_UNITS)),
            g=np.asarray(result.g),
            row_labels=row_labels,
            column_labels=column_labels,
            generated_config=generated,
            met=series,
            meta=dict(result.meta),
        )

    return TransportRunResult(
        model=run.model,
        mode="operator",
        units=OPERATOR_UNITS,
        g=np.asarray(result.g),
        row_labels=row_labels,
        column_labels=column_labels,
        met=series,
        meta=dict(result.meta),
    )


def _aermod_row_labels(run: TransportRunConfig, config: dict[str, Any]) -> tuple:
    """``(timestamp, receptor)`` rows when stacked, receptor ids otherwise."""
    if config.get("reduce") != "stack":
        return tuple(r.id for r in run.receptors)
    return tuple(
        (met.timestamp, receptor.id)
        for met in config["met_objects"]
        for receptor in run.receptors
    )


# ── Simulation mode ──────────────────────────────────────────────────────────


def _run_simulation(
    run: TransportRunConfig, series: MetSeries, run_dir: Path, *, dry_run: bool
) -> TransportRunResult:
    simulation = get_plugin(
        "enforceflux.transport_simulation", run.model, ITransportSimulation
    )()
    sources = translate.projected_sources(run)
    projection = run.projection()

    if run.model == "aermod":
        config = translate.aermod_config(run, series)
        result = simulation.simulate(sources, None, config)
        grid_field = result.meta["field"]
        field = canonical.from_aermod(
            grid_field,
            projection=projection,
            timestamps=[m.timestamp for m in config["met_objects"]],
            meta={"mode": "simulation", "n_sources": len(sources)},
        )
        native = None
        generated = None
    else:
        config, generated = _binary_model_config(run, series, run_dir)
        config["dry_run"] = dry_run
        result = simulation.simulate(sources, None, config)
        native = result.output_path
        if dry_run or native is None:
            return TransportRunResult(
                model=run.model,
                mode="simulation",
                units=canonical.CANONICAL_UNITS,
                native_output=native,
                generated_config=generated,
                met=series,
                meta={**dict(result.meta), "dry_run": True},
            )
        field = _canonicalise_binary_output(run, native, projection, generated)

    output_path = canonical.write_canonical(
        field, run.output.path, compress=run.output.compress
    )
    return TransportRunResult(
        model=run.model,
        mode="simulation",
        units=field.units,
        output_path=output_path,
        field=field,
        native_output=native,
        generated_config=generated,
        met=series,
        meta=dict(result.meta),
    )


def _canonicalise_binary_output(
    run: TransportRunConfig, native: Path, projection, generated_config: Path
) -> CanonicalField:
    if run.model == "flexpart":
        return canonical.from_flexpart_netcdf(
            native,
            projection=projection,
            variable=str(run.option("variable", "ch4_mixing_ratio")),
            height_index=int(run.option("height_index", 0)),
            meta={"mode": "simulation"},
        )
    if run.model == "microhh":
        from enforceflux.microhh.sim_config import load_microhh_config

        # MicroHH's cross-sections live in the case directory, which only the
        # generated case config knows about.
        case = load_microhh_config(generated_config)
        return canonical.from_microhh(
            case, level=int(run.option("level_index", 0)), meta={"mode": "simulation"}
        )
    raise ValueError(f"No canonical converter for model {run.model!r}")


def _binary_model_config(
    run: TransportRunConfig, series: MetSeries, run_dir: Path
) -> tuple[dict[str, Any], Path]:
    """Generate the native YAML for a binary-backed model and wrap it for the plugin."""
    if run.model == "flexpart":
        generated = translate.write_flexpart_config(run, series, run_dir)
    elif run.model == "microhh":
        generated = translate.write_microhh_config(run, series, run_dir)
    else:
        raise ValueError(f"{run.model!r} is not a binary-backed model")
    return {"sim_config": str(generated)}, generated
