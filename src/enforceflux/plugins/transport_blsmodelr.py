"""Registry-facing bLSmodelR transport operator (``ITransportOperator``).

Builds the forward Jacobian ``g`` (instrument × source-cell) with the R
package bLSmodelR via :class:`~enforceflux.blsmodelr.runner.BlsRunner`. The
source axis is a rectangular cell grid — one Jacobian column per cell —
regardless of the ``sources`` iterable passed in (which is ignored: bLS
models area sources, not the point-source :class:`Source` objects the rest
of the framework carries).

Config keys
-----------
source_grid : dict
    Kwargs for :func:`enforceflux.blsmodelr.footprint.build_source_grid`
    (``x_bounds``, ``y_bounds``, ``nx``, ``ny``; optional ``name_prefix``).
intervals : list[dict]
    Inline list of :class:`~enforceflux.blsmodelr.wrapper.BlsInterval` fields.
    Mutually exclusive with ``met_from_les``.
met_from_les : dict, optional
    Kwargs for :func:`enforceflux.blsmodelr.met_from_les.intervals_from_les`
    (``velocity_field``, ``z``, ``surface_theta``, ``theta_ref``, ``z_ref``,
    ``z0``, ``window_s``, ``dt``, optional ``id_prefix``).
interval_reduce : {"mean", "sum", "none"}, default "mean"
    ``"none"`` keeps one observation row per (interval, instrument) rather
    than collapsing the interval axis.
    Passed through to :func:`jacobian_from_bls_result`.
wrapper : dict, optional
    Forwarded verbatim to :class:`~enforceflux.blsmodelr.wrapper.BlsWrapper`
    (``rscript``, ``workdir``, ``dry_run``, ``timeout_s``, ``env``).
"""
from __future__ import annotations

from typing import Any, Iterable

from enforceflux.blsmodelr.runner import BlsRunner
from enforceflux.blsmodelr.wrapper import BlsInterval
from enforceflux.core.base import ForwardModelResult, ITransportOperator
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source


class BlsTransportOperator(ITransportOperator):
    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        if "source_grid" not in config:
            raise ValueError(
                "bLSmodelR transport operator requires config['source_grid'] "
                "(x_bounds, y_bounds, nx, ny)."
            )
        source_grid = dict(config["source_grid"])

        intervals = _build_intervals(config)
        interval_reduce = str(config.get("interval_reduce", "mean"))
        wrapper_config = dict(config.get("wrapper") or {})

        # The ``sources`` iterable is intentionally unused: bLS runs on the
        # gridded area-source discretisation carried in ``source_grid``.
        _ = list(sources)

        runner = BlsRunner(wrapper_config=wrapper_config)
        return runner.build_jacobian(
            instruments=instruments,
            sources_config=source_grid,
            intervals=intervals,
            interval_reduce=interval_reduce,
            model_params=config.get("model_params"),
            receptor_path_samples=int(config.get("receptor_path_samples", 8)),
        )


def _build_intervals(config: dict[str, Any]) -> list[BlsInterval]:
    inline = config.get("intervals")
    from_les = config.get("met_from_les")
    if inline and from_les:
        raise ValueError(
            "bLSmodelR config: pass either 'intervals' (inline list) or "
            "'met_from_les' (LES-diagnosed windows), not both."
        )
    if inline:
        return [BlsInterval(**dict(row)) for row in inline]
    if from_les:
        from enforceflux.blsmodelr.met_from_les import intervals_from_les
        return list(intervals_from_les(**dict(from_les)))
    raise ValueError(
        "bLSmodelR transport operator requires config['intervals'] or "
        "config['met_from_les']."
    )
