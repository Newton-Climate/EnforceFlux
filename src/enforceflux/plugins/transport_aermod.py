"""Registry-facing AERMOD transport operator (``ITransportOperator``).

Builds the forward Jacobian ``g`` (instrument × source) with the differentiable
AERMOD-style plume model in :mod:`enforceflux.aermod`. Unlike the FLEXPART and
MicroHH operators, nothing is executed out-of-process and no meteorology files
are needed: the boundary layer is specified directly in the config, and a full
Jacobian is one vectorized JAX kernel launch.

This is the default near-field operator, replacing the toy Gaussian plume.

Config keys
-----------
The config is an :class:`~enforceflux.aermod.config.AermodConfig` in dict form
(see that class for the full schema) — ``met`` is the only required entry::

    "config": {
        "met": [{"wind_speed_m_s": 3.0, "wind_direction_deg": 225.0,
                 "stability_class": "D", "mixing_height_m": 800.0}],
        "default_stack": {"height_m": 10.0},
        "concentration_units": "ug_m3_per_g_s",
        "reduce": "mean"
    }

Two extra keys are handled here rather than by ``AermodConfig``:

config_path : str
    Load the AERMOD config from a JSON/YAML file instead of inline. Inline keys
    (other than this one) are ignored when it is set.
receptor_path_samples : int
    Sample open-path instruments at N points along their path and average the
    result into one observation row (1 = treat every instrument as a point).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from enforceflux.core.base import ForwardModelResult, ITransportOperator
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source


class AermodTransportOperator(ITransportOperator):
    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        from enforceflux.aermod import AermodModel, receptors_from_instruments

        aermod_config = _load_config(config)
        sources = list(sources)
        instruments = list(instruments)
        if not sources:
            raise ValueError("AERMOD transport operator requires at least one source")

        receptors = list(aermod_config.receptors)
        if not receptors:
            if not instruments:
                raise ValueError(
                    "AERMOD transport operator needs receptors: pass instruments, or "
                    "set 'receptors' in the AERMOD config."
                )
            receptors = receptors_from_instruments(
                instruments, path_samples=aermod_config.receptor_path_samples
            )

        model = AermodModel(aermod_config)
        g = model.jacobian(sources, receptors)

        meta = {
            "model": "aermod",
            "backend": "jax",
            "units": aermod_config.concentration_units,
            "reduce": aermod_config.reduce,
            "n_met": len(aermod_config.met),
            "n_receptors": len(receptors),
            "receptor_path_samples": aermod_config.receptor_path_samples,
        }
        return ForwardModelResult(g=g, meta=meta)


def _load_config(config: dict[str, Any]):
    """Build an ``AermodConfig`` from an inline blob, an ERA5 block, or a file."""
    from enforceflux.aermod import AermodConfig

    config_path = config.get("config_path")
    if config_path:
        return AermodConfig.from_file(_resolve_path(config_path))

    # Already-built SurfaceMet objects, e.g. from enforceflux.meteo.to_aermod.
    met_objects = config.get("met_objects")
    if met_objects:
        return AermodConfig.from_dict(
            {k: v for k, v in config.items() if k != "met_objects"}, met=met_objects
        )

    era5 = config.get("era5")
    if era5:
        return AermodConfig.from_dict(
            {k: v for k, v in config.items() if k != "era5"},
            met=_met_from_era5(era5),
        )

    if "met" not in config:
        raise ValueError(
            "AERMOD transport operator requires config['met'] (a mapping or list of "
            "hourly boundary-layer conditions), config['era5'] naming ERA5 "
            "meteorology to read, or config['config_path'] pointing at an AERMOD "
            "config file."
        )
    return AermodConfig.from_dict(config)


def _met_from_era5(era5: dict[str, Any]):
    """Read ERA5 GRIB into AERMOD hours via the canonical met adapter."""
    from enforceflux.meteo import met_series_from_era5, to_aermod

    missing = [k for k in ("meteo_dir", "longitude", "latitude") if k not in era5]
    if missing:
        raise ValueError(
            f"config['era5'] is missing {missing}; it needs at least meteo_dir, "
            "longitude, and latitude."
        )
    options = {k: v for k, v in era5.items() if k != "meteo_dir"}
    series = met_series_from_era5(_resolve_path(era5["meteo_dir"]), **options)
    return to_aermod(series)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Mirror the other transport plugins: resolve relative to the repo root.
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / path).resolve()
