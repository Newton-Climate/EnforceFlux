"""The canonical concentration field every transport model is normalised into.

The three backends write very different files — FLEXPART a six-dimensional
``(nageclass, pointspec, time, height, latitude, longitude)`` NetCDF, MicroHH
raw Fortran binary cross-sections in box coordinates, AERMOD an analytic grid —
so anything downstream would otherwise branch per model. Each is converted here
into one shape::

    concentration(time, y, x)   [ng m-3]
    x(x), y(y)                  [m, local azimuthal-equidistant frame]
    longitude(y, x), latitude(y, x)
    timestamp(time)

``ng m-3`` is the common unit because it is FLEXPART's native output and both
of the others convert into it exactly (MicroHH from a mass mixing ratio, AERMOD
from χ/Q × emission).

Conversion is lossy where a model carries axes the canonical form does not:
FLEXPART's age classes, release points, and vertical levels are selected down
to one slice each (see :func:`from_flexpart_netcdf`), and MicroHH's vertical
structure is reduced to the requested horizontal plane. The originating file is
kept alongside by default so nothing is destroyed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CANONICAL_UNITS = "ng m-3"
KG_M3_TO_NG_M3 = 1.0e12


@dataclass(frozen=True)
class CanonicalField:
    """A concentration field in the canonical layout."""

    x: np.ndarray  # (nx,) metres, local frame
    y: np.ndarray  # (ny,) metres, local frame
    values: np.ndarray  # (time, ny, nx) in ng m-3
    timestamps: tuple[str, ...] = ()
    longitude: np.ndarray | None = None  # (ny, nx)
    latitude: np.ndarray | None = None  # (ny, nx)
    units: str = CANONICAL_UNITS
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(
                f"Canonical values must be (time, y, x); got shape {self.values.shape}"
            )
        n_time, n_y, n_x = self.values.shape
        if len(self.x) != n_x or len(self.y) != n_y:
            raise ValueError(
                f"Axis mismatch: values {self.values.shape} vs x={len(self.x)}, y={len(self.y)}"
            )
        if self.timestamps and len(self.timestamps) != n_time:
            raise ValueError(
                f"{len(self.timestamps)} timestamps for {n_time} time steps"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape  # type: ignore[return-value]

    def peak(self) -> dict[str, Any]:
        """Location and value of the maximum — the quickest sanity check."""
        index = np.unravel_index(int(np.argmax(self.values)), self.values.shape)
        return {
            "value": float(self.values[index]),
            "time_index": int(index[0]),
            "timestamp": self.timestamps[index[0]] if self.timestamps else None,
            "x_m": float(self.x[index[2]]),
            "y_m": float(self.y[index[1]]),
        }


# ── Writing ──────────────────────────────────────────────────────────────────


def write_canonical(field: CanonicalField, path: str | Path, *, compress: bool = True) -> Path:
    """Write a :class:`CanonicalField` to NetCDF and return the path."""
    from netCDF4 import Dataset

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_time, n_y, n_x = field.values.shape

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_time)
        ds.createDimension("y", n_y)
        ds.createDimension("x", n_x)

        var_x = ds.createVariable("x", "f8", ("x",))
        var_x.units = "m"
        var_x.long_name = "easting in the local projection"
        var_x[:] = field.x

        var_y = ds.createVariable("y", "f8", ("y",))
        var_y.units = "m"
        var_y.long_name = "northing in the local projection"
        var_y[:] = field.y

        var_t = ds.createVariable("time", "i4", ("time",))
        var_t.long_name = "time step index"
        var_t[:] = np.arange(n_time)

        if field.timestamps:
            stamps = ds.createVariable("timestamp", str, ("time",))
            stamps.long_name = "ISO-8601 valid time"
            for i, stamp in enumerate(field.timestamps):
                stamps[i] = stamp

        if field.longitude is not None and field.latitude is not None:
            lon = ds.createVariable("longitude", "f8", ("y", "x"))
            lon.units = "degrees_east"
            lon[:] = field.longitude
            lat = ds.createVariable("latitude", "f8", ("y", "x"))
            lat.units = "degrees_north"
            lat[:] = field.latitude

        conc = ds.createVariable(
            "concentration",
            "f4",
            ("time", "y", "x"),
            zlib=compress,
            complevel=4 if compress else 0,
        )
        conc.units = field.units
        conc.long_name = "CH4 concentration"
        conc.coordinates = "longitude latitude"
        conc[:] = field.values

        ds.Conventions = "EnforceFlux-canonical-1"
        for key, value in field.meta.items():
            setattr(ds, str(key), value if isinstance(value, (int, float, str)) else str(value))

    return path


def read_canonical(path: str | Path) -> CanonicalField:
    """Read a canonical NetCDF back into a :class:`CanonicalField`."""
    from netCDF4 import Dataset

    with Dataset(Path(path)) as ds:
        timestamps = (
            tuple(str(s) for s in ds.variables["timestamp"][:])
            if "timestamp" in ds.variables
            else ()
        )
        return CanonicalField(
            x=np.array(ds.variables["x"][:]),
            y=np.array(ds.variables["y"][:]),
            values=np.array(ds.variables["concentration"][:]),
            timestamps=timestamps,
            longitude=(
                np.array(ds.variables["longitude"][:]) if "longitude" in ds.variables else None
            ),
            latitude=(
                np.array(ds.variables["latitude"][:]) if "latitude" in ds.variables else None
            ),
            units=getattr(ds.variables["concentration"], "units", CANONICAL_UNITS),
            meta={k: ds.getncattr(k) for k in ds.ncattrs()},
        )


# ── Per-model conversion ─────────────────────────────────────────────────────


def _lonlat_grid(projection, x: np.ndarray, y: np.ndarray):
    """2-D longitude/latitude for a metric grid, or ``(None, None)`` without a projection."""
    if projection is None:
        return None, None
    grid_x, grid_y = np.meshgrid(x, y)
    lon, lat = projection.to_lonlat(grid_x, grid_y)
    return np.asarray(lon), np.asarray(lat)


def from_aermod(
    grid_field,
    *,
    projection=None,
    timestamps: Sequence[str | None] = (),
    meta: dict[str, Any] | None = None,
) -> CanonicalField:
    """AERMOD :class:`~enforceflux.aermod.model.GridField` → canonical field.

    AERMOD already produces ``(time, y, x)`` on a metric grid, so this only
    attaches geographic coordinates and checks the unit convention.
    """
    if grid_field.units not in ("ng_m3_per_kg_s", "ng m-3"):
        raise ValueError(
            f"Canonicalising AERMOD output needs ng m-3; the field is in "
            f"{grid_field.units!r}. Run with concentration_units='ng_m3_per_kg_s'."
        )
    lon, lat = _lonlat_grid(projection, grid_field.x, grid_field.y)
    return CanonicalField(
        x=np.asarray(grid_field.x, dtype=float),
        y=np.asarray(grid_field.y, dtype=float),
        values=np.asarray(grid_field.values, dtype=float),
        timestamps=tuple(str(t) for t in timestamps if t is not None),
        longitude=lon,
        latitude=lat,
        meta={"model": "aermod", "receptor_height_m": grid_field.z, **(meta or {})},
    )


def from_flexpart_netcdf(
    path: str | Path,
    *,
    projection=None,
    variable: str = "ch4_mixing_ratio",
    height_index: int = 0,
    age_index: int = 0,
    release_reduction: str = "sum",
    meta: dict[str, Any] | None = None,
) -> CanonicalField:
    """FLEXPART concentration NetCDF → canonical field.

    FLEXPART's grid is ``(nageclass, pointspec, time, height, latitude,
    longitude)`` in ng m-3. The age class and vertical level are *selected*
    (defaults: first of each — the surface layer of the first age class), and
    the release dimension is summed by default so a multi-release run gives the
    total field. Pass ``release_reduction="first"`` to keep a single release.
    """
    from netCDF4 import Dataset

    with Dataset(Path(path)) as ds:
        if variable not in ds.variables:
            raise ValueError(
                f"{path} has no variable {variable!r}. Available: {sorted(ds.variables)}"
            )
        var = ds.variables[variable]
        data = np.asarray(var[:])
        dims = list(var.dimensions)
        units = getattr(var, "units", CANONICAL_UNITS)
        longitude = np.asarray(ds.variables["longitude"][:], dtype=float)
        latitude = np.asarray(ds.variables["latitude"][:], dtype=float)

    # Collapse everything that is not (time, latitude, longitude).
    for name, index in (("nageclass", age_index), ("height", height_index)):
        if name in dims:
            axis = dims.index(name)
            data = np.take(data, index, axis=axis)
            dims.pop(axis)
    if "pointspec" in dims:
        axis = dims.index("pointspec")
        data = data.sum(axis=axis) if release_reduction == "sum" else np.take(data, 0, axis=axis)
        dims.pop(axis)

    expected = ["time", "latitude", "longitude"]
    if dims != expected:
        raise ValueError(
            f"After reduction the FLEXPART field has dims {dims}, expected {expected}."
        )

    if units.strip() not in ("ng m-3", "ng/m3"):
        raise ValueError(
            f"Expected FLEXPART output in ng m-3, found {units!r}; refusing to guess a "
            "conversion."
        )

    if projection is not None:
        centre_lon = float(np.mean(longitude))
        centre_lat = float(np.mean(latitude))
        x, _ = projection.to_xy(longitude, np.full_like(longitude, centre_lat))
        _, y = projection.to_xy(np.full_like(latitude, centre_lon), latitude)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    else:
        x, y = longitude, latitude

    grid_lon, grid_lat = np.meshgrid(longitude, latitude)
    return CanonicalField(
        x=x,
        y=y,
        values=np.asarray(data, dtype=float),
        longitude=grid_lon,
        latitude=grid_lat,
        meta={
            "model": "flexpart",
            "source_variable": variable,
            "height_index": height_index,
            "release_reduction": release_reduction,
            **(meta or {}),
        },
    )


def from_microhh(
    microhh_config,
    *,
    level: int = 0,
    variable: str | None = None,
    meta: dict[str, Any] | None = None,
) -> CanonicalField:
    """MicroHH horizontal cross-sections → canonical field.

    MicroHH writes each cross-section as a raw ``(jtot, itot)`` binary per output
    time, carrying a mass mixing ratio [kg/kg]; those are stacked into the time
    axis and converted to ng m-3. Coordinates are the LES box's own metric axes,
    with longitude/latitude from the case's wind-aligned box projection.
    """
    import glob

    from enforceflux.microhh.geometry import BoxProjection
    from enforceflux.microhh.units import mixing_ratio_to_mass_conc

    name = variable or microhh_config.scalar_name
    pattern = str(microhh_config.case_dir / f"{name}.xy.000.{level:05d}.*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No MicroHH xy cross-sections matching {pattern}. The case must be run "
            "with cross-section output enabled before it can be canonicalised."
        )

    grid = microhh_config.grid
    frames = [
        np.fromfile(f, dtype="<f8").reshape(grid.jtot, grid.itot) for f in files
    ]
    values = mixing_ratio_to_mass_conc(np.stack(frames)) * KG_M3_TO_NG_M3

    # Box-relative axes, centred on the source like the rest of the MicroHH code.
    x = (np.arange(grid.itot) + 0.5) * grid.dx - microhh_config.source_x0
    y = (np.arange(grid.jtot) + 0.5) * grid.dy - microhh_config.source_y0

    projection = BoxProjection(
        microhh_config.origin_lon,
        microhh_config.origin_lat,
        microhh_config.x_bearing_deg,
        microhh_config.source_x0,
        microhh_config.source_y0,
    )
    grid_x, grid_y = np.meshgrid(x + microhh_config.source_x0, y + microhh_config.source_y0)
    lon, lat = projection.to_lonlat(grid_x, grid_y)

    timestamps = tuple(Path(f).name.rsplit(".", 1)[-1] for f in files)
    return CanonicalField(
        x=x,
        y=y,
        values=values,
        timestamps=timestamps,
        longitude=lon,
        latitude=lat,
        meta={
            "model": "microhh",
            "level_index": level,
            "case_name": microhh_config.case_name,
            **(meta or {}),
        },
    )
