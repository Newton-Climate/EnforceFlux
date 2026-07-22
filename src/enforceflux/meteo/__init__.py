"""Meteorological forcing: one canonical format, adapted to every model.

:class:`~enforceflux.meteo.record.MetSeries` is the single representation of
boundary-layer forcing in EnforceFlux. ERA5 is the entry point; the adapters
convert it to whatever a given transport model wants::

    from enforceflux.meteo import met_series_from_era5, to_aermod, to_microhh_forcing

    series = met_series_from_era5("runs/.../meteo_april_week", -121.75, 39.15)
    aermod_met = to_aermod(series)               # per-hour SurfaceMet
    les_forcing = to_microhh_forcing(series)     # single steady Forcing
"""

from enforceflux.meteo.adapters import (
    FlexpartMetSource,
    microhh_box_bearing,
    to_aermod,
    to_flexpart,
    to_microhh_forcing,
)
from enforceflux.meteo.era5 import ERA5Downloader
from enforceflux.meteo.era5_profile import met_series_from_era5
from enforceflux.meteo.record import MetRecord, MetSeries

__all__ = [
    "ERA5Downloader",
    "FlexpartMetSource",
    "MetRecord",
    "MetSeries",
    "met_series_from_era5",
    "microhh_box_bearing",
    "to_aermod",
    "to_flexpart",
    "to_microhh_forcing",
]
