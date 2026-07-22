"""Read ERA5 GRIB forcing into the canonical :class:`~enforceflux.meteo.record.MetSeries`.

The same ERA5 files that drive FLEXPART (as fetched by
:class:`~enforceflux.meteo.era5.ERA5Downloader`) already carry everything a
boundary-layer dispersion model needs, so this extracts a single-point time
series from them rather than introducing a second met source:

===========================  ============================================
ERA5 field                   canonical quantity
===========================  ============================================
``10u`` / ``10v``            wind speed and direction at 10 m
``2t``                       temperature
``sp``                       surface pressure (→ potential temperature)
``blh``                      mixing height
``sshf``                     sensible heat flux (sign flipped to upward)
``ewss`` / ``nsss``          surface stress (→ ``u*``, optional)
===========================  ============================================

Two ERA5 conventions are handled here so that nothing downstream has to think
about them: accumulated fields (``sshf``, ``ewss``, ``nsss``) are divided by
their own accumulation window, which is read per message from ``stepRange``;
and ERA5's downward-positive heat-flux sign is flipped to the upward-positive
convention used everywhere in this package.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from enforceflux.meteo.record import AIR_DENSITY, MetRecord, MetSeries

_SURFACE_FIELDS = ("10u", "10v", "2t", "sp", "blh", "sshf", "ewss", "nsss")

FrictionVelocitySource = Literal["log_law", "stress"]


def met_series_from_era5(
    meteo_dir: str | Path,
    longitude: float,
    latitude: float,
    *,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    surface_roughness_m: float = 0.1,
    friction_velocity: FrictionVelocitySource = "log_law",
    potential_temperature_gradient_k_m: float = 0.01,
    min_mixing_height_m: float = 50.0,
    min_wind_speed_m_s: float = 0.3,
    pattern: str = "EA*",
) -> MetSeries:
    """Extract a single-point met series from a directory of ERA5 GRIB files.

    Parameters
    ----------
    meteo_dir:
        Directory of ERA5 GRIB files (the ``meteo_dir`` a FLEXPART run uses).
    longitude, latitude:
        Site coordinates; the nearest ERA5 grid point is sampled and recorded in
        ``provenance``.
    start, end:
        Optional time window (inclusive). Defaults to every file present.
    surface_roughness_m:
        Local aerodynamic roughness at the site. ERA5 has no usable local ``z0``
        at dispersion scales, so this is the user's, and it feeds both the
        log-law ``u*`` and the models downstream.
    friction_velocity:
        ``"log_law"`` (default) derives ``u*`` from the 10 m wind and
        ``surface_roughness_m``; ``"stress"`` uses ERA5's own surface stress.
        The default is deliberate: ERA5's stress includes subgrid **orographic
        form drag**, which over complex terrain makes ``u*`` roughly 1.5-2×
        larger than the local shear that actually mixes a plume. Use
        ``"stress"`` over flat, homogeneous terrain, where it is the better
        estimate.
    potential_temperature_gradient_k_m:
        Free-atmosphere stability above the mixed layer. Not derived from ERA5
        here (that needs the hybrid-level coefficients); the default is a
        typical value and only affects stable-layer plume rise.
    min_mixing_height_m, min_wind_speed_m_s:
        Floors applied to ERA5's nocturnal extremes, which routinely go below
        what a steady-state plume model can represent.

    Returns
    -------
    MetSeries
        One record per ERA5 timestep at the sampled grid point.
    """
    import eccodes as ec  # imported lazily: eccodes is an optional extra

    meteo_path = Path(meteo_dir)
    if not meteo_path.is_dir():
        raise FileNotFoundError(f"ERA5 meteo directory not found: {meteo_path}")

    if friction_velocity not in ("log_law", "stress"):
        raise ValueError(
            f"friction_velocity must be 'log_law' or 'stress', got {friction_velocity!r}"
        )

    window = (_parse_time(start), _parse_time(end))
    files = sorted(p for p in meteo_path.glob(pattern) if p.is_file() and "static" not in p.name)
    if not files:
        raise FileNotFoundError(
            f"No ERA5 GRIB files matching {pattern!r} in {meteo_path}. "
            "Fetch them with ERA5Downloader first."
        )

    records: list[MetRecord] = []
    sampled_point: tuple[float, float] | None = None
    for path in files:
        # The surface fields sit at the very end of a FLEXPART-ready ERA5 file,
        # behind ~950 hybrid-level messages, so opening one is expensive. The
        # filename carries the timestamp — use it to skip files outside the
        # window without reading them.
        if not _filename_in_window(path, window):
            continue
        fields, valid_time, point = _read_surface_point(ec, path, longitude, latitude)
        if not fields or valid_time is None:
            continue
        if window[0] is not None and valid_time < window[0]:
            continue
        if window[1] is not None and valid_time > window[1]:
            continue
        sampled_point = sampled_point or point
        records.append(
            _to_record(
                fields,
                valid_time,
                surface_roughness_m=surface_roughness_m,
                friction_velocity=friction_velocity,
                potential_temperature_gradient_k_m=potential_temperature_gradient_k_m,
                min_mixing_height_m=min_mixing_height_m,
                min_wind_speed_m_s=min_wind_speed_m_s,
            )
        )

    if not records:
        raise ValueError(
            f"No ERA5 records in {meteo_path} within the requested window "
            f"({start} → {end}). Files found: {len(files)}."
        )

    provenance: dict[str, Any] = {
        "source": "era5",
        "meteo_dir": str(meteo_path),
        "n_files": len(files),
        "requested_lon": longitude,
        "requested_lat": latitude,
        "friction_velocity": friction_velocity,
        "surface_roughness_m": surface_roughness_m,
    }
    if sampled_point is not None:
        provenance["grid_lat"], provenance["grid_lon"] = sampled_point

    return MetSeries(
        records=tuple(records),
        longitude=longitude,
        latitude=latitude,
        provenance=provenance,
    )


# ── GRIB plumbing ────────────────────────────────────────────────────────────


def _read_surface_point(
    ec, path: Path, longitude: float, latitude: float
) -> tuple[dict[str, tuple[float, float]], datetime | None, tuple[float, float] | None]:
    """Nearest-point values for the surface fields in one GRIB file.

    Returns ``{shortName: (value, accumulation_seconds)}``, the valid time, and
    the sampled grid point.
    """
    fields: dict[str, tuple[float, float]] = {}
    valid_time: datetime | None = None
    point: tuple[float, float] | None = None
    index: int | None = None

    with open(path, "rb") as handle:
        while True:
            gid = ec.codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                short_name = ec.codes_get(gid, "shortName")
                if (
                    short_name not in _SURFACE_FIELDS
                    or short_name in fields
                    or ec.codes_get(gid, "typeOfLevel") != "surface"
                ):
                    continue
                if index is None:
                    index, point = _nearest_index(ec, gid, longitude, latitude)
                values = ec.codes_get_values(gid)
                seconds = _accumulation_seconds(ec, gid)
                fields[short_name] = (float(values[index]), seconds)
                if short_name == "2t":
                    valid_time = _valid_time(ec, gid)
            finally:
                ec.codes_release(gid)

    return fields, valid_time, point


def _nearest_index(ec, gid, longitude: float, latitude: float) -> tuple[int, tuple[float, float]]:
    import numpy as np

    lats = np.asarray(ec.codes_get_array(gid, "latitudes"), dtype=float)
    lons = np.asarray(ec.codes_get_array(gid, "longitudes"), dtype=float)
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    index = int(np.argmin((lats - latitude) ** 2 + (lons - longitude) ** 2))
    return index, (float(lats[index]), float(lons[index]))


def _filename_in_window(
    path: Path, window: tuple[datetime | None, datetime | None]
) -> bool:
    """Whether an ``EA<YYYYMMDDHH>`` filename falls inside the requested window.

    Unparseable names return True so that unusual naming falls back to reading
    the file and checking its actual valid time.
    """
    if window[0] is None and window[1] is None:
        return True
    stamp = path.name.lstrip("EA")
    if len(stamp) < 10 or not stamp[:10].isdigit():
        return True
    try:
        file_time = datetime.strptime(stamp[:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    if window[0] is not None and file_time < window[0]:
        return False
    if window[1] is not None and file_time > window[1]:
        return False
    return True


def _accumulation_seconds(ec, gid) -> float:
    """Length of an accumulated field's window; 0 for instantaneous fields."""
    if ec.codes_get(gid, "stepType") != "accum":
        return 0.0
    step_range = str(ec.codes_get(gid, "stepRange"))
    if "-" in step_range:
        first, last = step_range.split("-", 1)
        hours = float(last) - float(first)
    else:
        hours = float(step_range)
    return max(hours, 1.0) * 3600.0


def _valid_time(ec, gid) -> datetime:
    date = str(ec.codes_get(gid, "dataDate"))
    time_of_day = int(ec.codes_get(gid, "dataTime"))
    base = datetime(
        int(date[:4]), int(date[4:6]), int(date[6:8]),
        time_of_day // 100, time_of_day % 100, tzinfo=timezone.utc,
    )
    return base + timedelta(hours=float(ec.codes_get(gid, "step")))


def _to_record(
    fields: dict[str, tuple[float, float]],
    valid_time: datetime,
    *,
    surface_roughness_m: float,
    friction_velocity: FrictionVelocitySource,
    potential_temperature_gradient_k_m: float,
    min_mixing_height_m: float,
    min_wind_speed_m_s: float,
) -> MetRecord:
    missing = [f for f in ("10u", "10v", "2t", "blh") if f not in fields]
    if missing:
        raise ValueError(
            f"ERA5 file for {valid_time:%Y-%m-%d %H:%M} is missing required "
            f"surface fields: {missing}"
        )

    u10, _ = fields["10u"]
    v10, _ = fields["10v"]
    speed = max(math.hypot(u10, v10), min_wind_speed_m_s)
    # Meteorological direction: where the wind comes from.
    direction = (math.degrees(math.atan2(-u10, -v10))) % 360.0

    temperature = fields["2t"][0]
    pressure = fields["sp"][0] if "sp" in fields else 100000.0
    mixing_height = max(fields["blh"][0], min_mixing_height_m)

    # ERA5 accumulates fluxes in J m-2 and is positive *downward*; we want the
    # mean flux over the window, positive upward.
    heat_flux = 0.0
    if "sshf" in fields:
        value, seconds = fields["sshf"]
        heat_flux = -value / seconds if seconds else -value

    if friction_velocity == "stress" and "ewss" in fields and "nsss" in fields:
        ewss, ewss_seconds = fields["ewss"]
        nsss, nsss_seconds = fields["nsss"]
        stress = math.hypot(
            ewss / (ewss_seconds or 1.0), nsss / (nsss_seconds or 1.0)
        )
        u_star = math.sqrt(max(stress, 1.0e-6) / AIR_DENSITY)
    else:
        u_star = 0.4 * speed / math.log(10.0 / surface_roughness_m)

    return MetRecord(
        time=valid_time,
        wind_speed_m_s=speed,
        wind_direction_deg=direction,
        temperature_k=temperature,
        mixing_height_m=mixing_height,
        friction_velocity_m_s=max(u_star, 1.0e-3),
        sensible_heat_flux_w_m2=heat_flux,
        surface_roughness_m=surface_roughness_m,
        surface_pressure_pa=pressure,
        potential_temperature_gradient_k_m=potential_temperature_gradient_k_m,
    )


def _parse_time(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).rstrip("Z"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
