"""NetCDF writer for AERMOD-style concentration fields.

Mirrors the layout the FLEXPART/MicroHH simulation backends produce — a
``(time, y, x)`` concentration variable with coordinate axes — so the existing
analysis and visualization code can read AERMOD output without special-casing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from enforceflux.aermod.model import GridField


def write_grid_netcdf(
    field: GridField,
    path: str | Path,
    *,
    timestamps: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
    compress: bool = True,
) -> Path:
    """Write ``field`` to ``path`` and return the resolved path."""
    from netCDF4 import Dataset  # imported lazily; netCDF4 is a heavy import

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_time, n_y, n_x = field.values.shape
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_time)
        ds.createDimension("y", n_y)
        ds.createDimension("x", n_x)

        var_x = ds.createVariable("x", "f8", ("x",))
        var_x.units = "m"
        var_x.long_name = "projected easting"
        var_x[:] = field.x

        var_y = ds.createVariable("y", "f8", ("y",))
        var_y.units = "m"
        var_y.long_name = "projected northing"
        var_y[:] = field.y

        var_t = ds.createVariable("time", "i4", ("time",))
        var_t.long_name = "meteorological record index"
        var_t[:] = np.arange(n_time)
        if timestamps is not None and len(timestamps) == n_time:
            label = ds.createVariable("timestamp", str, ("time",))
            label.long_name = "meteorological record timestamp"
            for i, stamp in enumerate(timestamps):
                label[i] = stamp

        var_c = ds.createVariable(
            "concentration",
            "f4",
            ("time", "y", "x"),
            zlib=compress,
            complevel=4 if compress else 0,
        )
        var_c.units = field.units
        var_c.long_name = "surface concentration from AERMOD-style dispersion"
        var_c.receptor_height_m = field.z
        var_c[:] = field.values

        ds.model = "enforceflux-aermod"
        ds.receptor_height_m = field.z
        for key, value in {**field.meta, **(attributes or {})}.items():
            setattr(ds, str(key), value if isinstance(value, (int, float, str)) else str(value))

    return path
