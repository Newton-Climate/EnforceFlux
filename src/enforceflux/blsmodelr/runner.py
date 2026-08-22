"""Thin driver composing :class:`BlsWrapper` with the footprint / met helpers.

:class:`BlsRunner` assembles a bLSmodelR forward operator (instrument × source)
in three steps: build the sensor list from :class:`~enforceflux.instrument.Instrument`
objects, build the source grid via :func:`enforceflux.blsmodelr.footprint.build_source_grid`,
invoke the wrapper, and reshape the long-form C/E table into a Jacobian via
:func:`enforceflux.blsmodelr.footprint.jacobian_from_bls_result`.

The runner intentionally has no config schema of its own beyond the arguments to
:meth:`build_jacobian` — plugin-facing config parsing lives in
:mod:`enforceflux.plugins.transport_blsmodelr`.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from enforceflux.blsmodelr.wrapper import (
    BlsInterval,
    BlsModelParams,
    BlsRequest,
    BlsSensor,
    BlsWrapper,
)
from enforceflux.core.base import ForwardModelResult
from enforceflux.instrument import Instrument


class BlsRunner:
    """Compose :class:`BlsWrapper` with the source-grid and Jacobian helpers."""

    def __init__(self, wrapper_config: dict[str, Any] | None = None) -> None:
        self.wrapper = BlsWrapper(wrapper_config or {})

    # ── main entry ────────────────────────────────────────────────────────

    def build_jacobian(
        self,
        instruments: Iterable[Instrument],
        sources_config: dict[str, Any],
        intervals: Sequence[BlsInterval],
        interval_reduce: str = "mean",
        model_params: BlsModelParams | dict | None = None,
    ) -> ForwardModelResult:
        """Build the (n_instruments × n_sources) Jacobian for the given intervals.

        Parameters
        ----------
        instruments : Iterable[Instrument]
            Point receptors; ``id``/``x``/``y``/``z`` map onto :class:`BlsSensor`.
        sources_config : dict
            Kwargs for :func:`enforceflux.blsmodelr.footprint.build_source_grid`
            (``x_bounds``, ``y_bounds``, ``nx``, ``ny``, ``name_prefix``).
        intervals : Sequence[BlsInterval]
            MOST windows to average across (see ``interval_reduce``).
        interval_reduce : {"mean", "sum"}
            How :func:`jacobian_from_bls_result` collapses the interval axis.
        """
        # Deferred imports: these siblings may not exist yet during parallel
        # development, and importing at module top would break wrapper-only use.
        from enforceflux.blsmodelr.footprint import (
            build_source_grid,
            jacobian_from_bls_result,
        )

        instruments = list(instruments)
        if not instruments:
            raise ValueError("BlsRunner.build_jacobian needs at least one instrument")
        if not intervals:
            raise ValueError("BlsRunner.build_jacobian needs at least one interval")

        sensors = [
            BlsSensor(name=inst.id, x=float(inst.x), y=float(inst.y), z=float(inst.z))
            for inst in instruments
        ]
        sensor_order = [s.name for s in sensors]

        sources, _grid_meta = build_source_grid(**sources_config)
        source_order = [s.name for s in sources]

        if isinstance(model_params, dict):
            model_params = BlsModelParams(**model_params)
        request = BlsRequest(
            sensors=sensors, sources=sources, intervals=list(intervals),
            model=model_params or BlsModelParams(),
        )
        result = self.wrapper.run(request)

        g = jacobian_from_bls_result(
            result=result,
            sensor_order=sensor_order,
            source_order=source_order,
            interval_reduce=interval_reduce,
        )
        g = np.asarray(g, dtype=float)

        cell_area = _cell_area_m2(sources_config)
        meta = {
            "backend": "blsmodelr",
            "n_sources": len(sources),
            "n_intervals": len(intervals),
            "cell_area_m2": cell_area,
            "interval_reduce": interval_reduce,
            "workdir": result.meta.get("workdir"),
            "dry_run": result.meta.get("dry_run", False),
        }
        return ForwardModelResult(g=g, meta=meta)


def _cell_area_m2(sources_config: dict[str, Any]) -> float:
    """Uniform grid cell area from the ``build_source_grid`` kwargs."""
    x0, x1 = sources_config["x_bounds"]
    y0, y1 = sources_config["y_bounds"]
    nx = int(sources_config["nx"])
    ny = int(sources_config["ny"])
    return abs((float(x1) - float(x0)) / nx * (float(y1) - float(y0)) / ny)
