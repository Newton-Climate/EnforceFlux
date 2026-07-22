"""Registry-facing AERMOD forward simulation (``ITransportSimulation``).

The simulation counterpart to
:class:`~enforceflux.plugins.transport_aermod.AermodTransportOperator`: instead
of a Jacobian it evaluates the actual emissions on a receptor grid and writes a
``(time, y, x)`` concentration NetCDF, in the same shape the FLEXPART and
MicroHH simulation backends produce.

Because the AERMOD kernel is analytic, this runs in-process in seconds — it is
the cheap synthetic-truth and plume-visualization path.

Config keys
-----------
Everything :class:`~enforceflux.plugins.transport_aermod.AermodTransportOperator`
accepts, plus:

grid : dict
    Receptor grid (``x_min``/``x_max``/``y_min``/``y_max``/``spacing_m`` and
    optional ``height_m``). Defaults to the run's ``domain`` extents when
    omitted.
output_path : str
    Destination NetCDF. When omitted, no file is written and the field is
    returned in ``meta['field']`` instead.
emissions : list[float]
    Per-source emission rates. Defaults to each source's ``flux_true``.
"""
from __future__ import annotations

from typing import Any, Iterable

from enforceflux.core.base import ITransportSimulation, TransportSimulationResult
from enforceflux.models.source import Source
from enforceflux.plugins.transport_aermod import _load_config, _resolve_path


class AermodSimulationModel(ITransportSimulation):
    def simulate(
        self,
        sources: Iterable[Source],
        domain: Any,
        config: dict[str, Any],
    ) -> TransportSimulationResult:
        from enforceflux.aermod import AermodModel, write_grid_netcdf

        aermod_config = _load_config(config)
        sources = list(sources)
        if not sources:
            raise ValueError("AERMOD simulation requires at least one source")

        grid = aermod_config.grid or _grid_from_domain(domain)
        if grid is None:
            raise ValueError(
                "AERMOD simulation needs a receptor grid: set 'grid' in the config, "
                "or provide a domain with x_min/x_max/y_min/y_max/grid_spacing."
            )

        model = AermodModel(aermod_config)
        field = model.grid_field(sources, grid=grid, emissions=config.get("emissions"))

        meta: dict[str, Any] = {
            "backend": "aermod",
            "model": "aermod",
            "units": field.units,
            "n_met": len(aermod_config.met),
            "grid_shape": list(field.values.shape),
            "receptor_height_m": field.z,
        }

        output_ref = config.get("output_path")
        if not output_ref:
            meta["field"] = field
            return TransportSimulationResult(output_path=None, meta=meta)

        timestamps = [m.timestamp for m in aermod_config.met]
        output_path = write_grid_netcdf(
            field,
            _resolve_path(output_ref),
            timestamps=timestamps if all(t is not None for t in timestamps) else None,
            compress=bool(config.get("compress", True)),
        )
        meta["output_path"] = str(output_path)
        return TransportSimulationResult(output_path=output_path, meta=meta)


def _grid_from_domain(domain: Any) -> "Any | None":
    """Fall back to the run's domain extents when no AERMOD grid is configured."""
    from enforceflux.aermod import ReceptorGrid

    required = ("x_min", "x_max", "y_min", "y_max", "grid_spacing")
    if domain is None or not all(hasattr(domain, attr) for attr in required):
        return None
    return ReceptorGrid(
        x_min=float(domain.x_min),
        x_max=float(domain.x_max),
        y_min=float(domain.y_min),
        y_max=float(domain.y_max),
        spacing_m=float(domain.grid_spacing),
    )
