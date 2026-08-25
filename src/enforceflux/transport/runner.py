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
    column_labels = tuple(s.id for s in run.sources)

    # bLS has a native runner here because it operates on area-source polygons
    # rather than the point-source plugin contract.  Resolve it before the
    # entry-point registry so a source checkout does not require reinstalling
    # editable package metadata merely to expose the optional plugin name.
    if run.model == "blsmodelr":
        return _run_blsmodelr_operator(run, series, column_labels)

    operator = get_plugin("enforceflux.transport_operator", run.model, ITransportOperator)()
    sources = translate.projected_sources(run)

    if run.model == "aermod":
        config = translate.aermod_config(run, series)
        # Receptors come from the shared config, so no Instrument objects needed.
        result = operator.build_forward_operator(sources, [], None, config)
        row_labels = _aermod_row_labels(run, config)
    else:
        config, generated = _binary_model_config(run, series, run_dir)
        config["dry_run"] = dry_run
        if run.model == "flexpart":
            # Preserve binary-model options that select the backward LPDM
            # operator, its particle budget, and OP beam quadrature.
            config.update(run.options)
            config.setdefault("mode", "backward")
            config.setdefault("base_run_dir", str(run_dir / "backward"))
            config.setdefault("source_areas_m2", run.domain.spacing_m ** 2)
            config.setdefault("mixing_height_m", run.domain.heights_m[0])
            instruments = translate.projected_instruments(run)
            result = operator.build_forward_operator(sources, instruments, run.domain, config)
        else:
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
    receptor_ids = []
    for receptor in config["receptors"]:
        label = receptor.get("group", receptor["id"])
        if label not in receptor_ids:
            receptor_ids.append(label)
    if config.get("reduce") != "stack":
        return tuple(receptor_ids)
    return tuple(
        (met.timestamp, receptor_id)
        for met in config["met_objects"]
        for receptor_id in receptor_ids
    )


# ── bLSmodelR operator ───────────────────────────────────────────────────────

_BLS_UNITS = "(kg m-3) / (kg m-2 s-1)"


def _run_blsmodelr_operator(
    run: TransportRunConfig, series: MetSeries, column_labels: tuple[str, ...],
) -> TransportRunResult:
    """Build a bLSmodelR Jacobian aligned with ``run.sources`` ordering.

    Each :class:`RunSource` becomes a square polygon of side ``domain.spacing_m``
    centered on its (x_m, y_m); the Jacobian's column ``j`` corresponds to
    ``run.sources[j]``. Turbulence intervals come either inline (config
    ``blsmodelr.intervals``) or from an LES-diagnosed sonic on a MicroHH
    nature run (config ``blsmodelr.intervals_from_nature``).
    """
    from enforceflux.blsmodelr.footprint import (
        build_source_polygons, jacobian_from_bls_result,
    )
    from enforceflux.blsmodelr.wrapper import (
        BlsModelParams, BlsRequest, BlsSensor, BlsWrapper,
    )

    opts = dict(run.options)
    spacing = float(run.domain.spacing_m)
    half = spacing / 2.0
    polygons = [
        (s.id, [
            (s.x_m - half, s.y_m - half), (s.x_m + half, s.y_m - half),
            (s.x_m + half, s.y_m + half), (s.x_m - half, s.y_m + half),
        ])
        for s in run.sources
    ]
    bls_sources, cell_area = build_source_polygons(polygons=polygons)

    intervals = _bls_intervals(opts)

    # Beam quadrature: an open-path receptor is expanded into `n_path` equally
    # weighted subpoints along its beam. Each subpoint goes to bLS as its own
    # sensor; the corresponding rows are averaged back into one observation row
    # for the returned Jacobian so downstream code still sees one row per
    # instrument. Point sensors stay as-is (one subpoint, weight 1).
    n_path = int(opts.get("receptor_path_samples", 8))
    instruments = translate.projected_instruments(run)
    if n_path < 2 and any(inst.path_length_m > 0.0 for inst in instruments):
        raise ValueError(
            "bLS open-path instruments require receptor_path_samples >= 2; "
            "a path may not silently degrade to one point sensor"
        )
    bls_sensors: list[BlsSensor] = []
    sensor_groups: list[tuple[str, list[str]]] = []  # (instrument_id, subpoint_ids)
    for inst in instruments:
        if inst.path_length_m > 0.0:
            bearing = np.deg2rad(float(inst.path_bearing_deg))
            offsets = (np.arange(n_path) + 0.5) * float(inst.path_length_m) / n_path
            sub_ids: list[str] = []
            for k, off in enumerate(offsets):
                sub_id = f"{inst.id}_p{k:02d}"
                bls_sensors.append(BlsSensor(
                    name=sub_id,
                    x=float(inst.x + off * np.sin(bearing)),
                    y=float(inst.y + off * np.cos(bearing)),
                    z=float(inst.z),
                ))
                sub_ids.append(sub_id)
            sensor_groups.append((inst.id, sub_ids))
        else:
            bls_sensors.append(BlsSensor(
                name=inst.id, x=float(inst.x), y=float(inst.y), z=float(inst.z)
            ))
            sensor_groups.append((inst.id, [inst.id]))

    wrapper = BlsWrapper(dict(opts.get("wrapper") or {}))
    model_params = BlsModelParams(**dict(opts.get("model_params") or {}))
    request = BlsRequest(
        sensors=bls_sensors, sources=bls_sources,
        intervals=intervals, model=model_params,
    )
    bls_result = wrapper.run(request)

    interval_reduce = str(opts.get("interval_reduce", "mean"))
    g_bls_sub = jacobian_from_bls_result(
        result=bls_result,
        sensor_order=[s.name for s in bls_sensors],
        source_order=[s.name for s in bls_sources],
        interval_reduce=interval_reduce,
    )
    # Collapse the subpoint rows back into one row per instrument (equal-weight
    # arithmetic mean over the beam quadrature). Point sensors have a single
    # subpoint and pass through unchanged.
    sub_index = {s.name: k for k, s in enumerate(bls_sensors)}
    g_bls = np.zeros((len(sensor_groups), g_bls_sub.shape[1]), dtype=float)
    inst_row_labels: list[str] = []
    for r, (inst_id, sub_ids) in enumerate(sensor_groups):
        rows = np.stack([g_bls_sub[sub_index[s]] for s in sub_ids], axis=0)
        g_bls[r] = rows.mean(axis=0)
        inst_row_labels.append(inst_id)
    # bLSmodelR returns "CE" — (kg m-3) per (kg m-2 s-1) areal emission — one
    # column per source polygon. Convert to the canonical operator contract
    # (OPERATOR_UNITS = "ng m-3 / (kg s-1)") so downstream flux code can pair
    # the Jacobian with LES pseudo-observations without a units switch:
    #   * divide column j by cell_area_j (m²) → (kg m-3) / (kg s-1)
    #   * multiply by 1e12 to lift kg → ng
    # Both operations are exact for the flux stage's per-cell state vector.
    KG_TO_NG = 1.0e12
    area = np.asarray(cell_area, dtype=float)
    if area.shape[0] != g_bls.shape[1]:
        raise RuntimeError(
            f"bLS cell_area vector ({area.shape[0]}) does not match Jacobian "
            f"column count ({g_bls.shape[1]}); source-polygon ordering broke."
        )
    g = np.asarray(g_bls, dtype=float) * (KG_TO_NG / area[np.newaxis, :])
    return TransportRunResult(
        model=run.model, mode="operator", units=OPERATOR_UNITS,
        g=g,
        row_labels=tuple(inst_row_labels),
        column_labels=column_labels,
        met=series,
        meta={
            "n_sources": len(bls_sources),
            "cell_area_m2": cell_area.tolist(),
            "n_intervals": len(intervals),
            "interval_reduce": interval_reduce,
            "receptor_path_samples": n_path,
            "workdir": bls_result.meta.get("workdir"),
            "raw_units": _BLS_UNITS,
            "unit_conversion": {
                "from": _BLS_UNITS, "to": OPERATOR_UNITS,
                "operations": ["divide_by_cell_area_m2", "multiply_by_1e12_kg_to_ng"],
            },
        },
    )


def _bls_intervals(opts: dict[str, Any]):
    """Resolve turbulence intervals from either inline or LES-nature config."""
    from enforceflux.blsmodelr.wrapper import BlsInterval

    inline = opts.get("intervals")
    from_nature = opts.get("intervals_from_nature")
    if inline and from_nature:
        raise ValueError(
            "blsmodelr: pass either 'intervals' or 'intervals_from_nature', not both."
        )
    if inline:
        return [BlsInterval(**dict(row)) for row in inline]
    if from_nature:
        from enforceflux.blsmodelr.met_from_les import intervals_from_microhh_output
        from enforceflux.microhh.sim_config import load_microhh_config
        nature = dict(from_nature)
        cfg_path = Path(nature.pop("microhh_config"))
        cfg = load_microhh_config(cfg_path)
        return list(intervals_from_microhh_output(cfg, **nature))
    raise ValueError(
        "blsmodelr: config needs 'intervals' (inline list) or "
        "'intervals_from_nature' (points at a MicroHH run + receptor)."
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
