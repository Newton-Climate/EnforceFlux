"""Read MicroHH output into receptor series and plume cross-sections.

MicroHH writes two output kinds this module consumes:

- **Columns** (``[column]``): one NetCDF per sampled location, named
  ``<case>.column.<ix>.<iy>.<iter>.nc`` (5-digit grid indices). Each holds the
  full vertical profile of every field vs. time at that column — the receptor
  time series the instrument operator needs. Times use ``seconds since start``,
  so ``decode_times=False`` is required.

- **Cross-sections** (``[cross]``): raw little-endian float64 binaries named
  ``<var>.xy.<n>.<k>.<iter>`` (horizontal slice, shape ``(jtot, itot)``) and
  ``<var>.xz.<n>.<j>.<iter>`` (vertical slice, shape ``(ktot, itot)``) — the 2D
  plume fields for visualization. There is no header; the dtype/shape come from
  the grid in the config.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enforceflux.microhh.geometry import BoxProjection
from enforceflux.microhh.sim_config import MicroHHConfig


@dataclass(frozen=True)
class ReceptorSeries:
    """Sampled scalar time series for each receptor."""

    receptor_ids: tuple[str, ...]
    times_s: np.ndarray          # (t,)
    values: np.ndarray           # (t, n_receptors) scalar mixing ratio


def _proj(cfg: MicroHHConfig) -> BoxProjection:
    return BoxProjection(
        origin_lon=cfg.origin_lon, origin_lat=cfg.origin_lat,
        x_bearing_deg=cfg.x_bearing_deg,
        source_x0=cfg.source_x0, source_y0=cfg.source_y0,
    )


def _column_index(x_m: float, y_m: float, cfg: MicroHHConfig) -> tuple[int, int]:
    """Grid index of the column MicroHH actually wrote for this location.

    ``case.py`` rounds the projected coordinate before writing it into the
    ``.ini``, and MicroHH derives the column index from that rounded value. So
    the index must be recomputed the same way: truncating the unrounded
    projection instead disagrees whenever a receptor lands within half a metre
    below a cell boundary (e.g. x=659.7 -> 32 by truncation, but the file on
    disk is 33), and the read fails with a missing-column error.
    """
    return int(round(x_m) / cfg.grid.dx), int(round(y_m) / cfg.grid.dy)


def find_column_file(cfg: MicroHHConfig, ix: int, iy: int) -> Path | None:
    """Locate the column NetCDF for a grid-index location (any start iter)."""
    pattern = str(cfg.case_dir / f"{cfg.case_name}.column.{ix:05d}.{iy:05d}.*.nc")
    matches = sorted(glob.glob(pattern))
    return Path(matches[0]) if matches else None


def read_receptor_series(cfg: MicroHHConfig, sample_level: int = 0) -> ReceptorSeries:
    """Read each receptor's column file → a near-surface scalar time series.

    ``sample_level`` selects the vertical index (0 = first model level).
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading MicroHH output needs the 'analysis' extra (xarray/netCDF4): "
            "pip install enforceflux[analysis]"
        ) from exc

    proj = _proj(cfg)
    times: np.ndarray | None = None
    series: list[np.ndarray] = []
    ids: list[str] = []

    for r in cfg.receptors:
        x, y = proj.to_box(r.lon, r.lat)
        ix, iy = _column_index(x, y, cfg)
        path = find_column_file(cfg, ix, iy)
        if path is None:
            raise FileNotFoundError(
                f"No column file for receptor {r.id!r} at grid index "
                f"({ix:05d},{iy:05d}) in {cfg.case_dir}. Run the case first."
            )
        ds = xr.open_dataset(path, decode_times=False)
        col = np.asarray(ds[cfg.scalar_name].isel(z=sample_level).values, dtype=float)
        if times is None:
            times = np.asarray(ds["time"].values, dtype=float)
        series.append(col)
        ids.append(r.id)
        ds.close()

    return ReceptorSeries(
        receptor_ids=tuple(ids),
        times_s=times if times is not None else np.empty(0),
        values=np.stack(series, axis=1) if series else np.empty((0, 0)),
    )


def _latest_iter(cfg: MicroHHConfig, var: str, plane: str) -> str:
    """Highest available time-stamp string for a cross-section variable."""
    files = sorted(glob.glob(str(cfg.case_dir / f"{var}.{plane}.*")))
    if not files:
        raise FileNotFoundError(
            f"No {plane} cross-section files for {var!r} in {cfg.case_dir}."
        )
    return files[-1].rsplit(".", 1)[-1]


def read_cross_xy(
    cfg: MicroHHConfig, var: str | None = None, k: int = 0, iter_s: str | None = None
) -> np.ndarray:
    """Read a horizontal cross-section as a ``(jtot, itot)`` array."""
    var = var or cfg.scalar_name
    plane = "xy"
    iter_s = iter_s or _latest_iter(cfg, var, plane)
    path = cfg.case_dir / f"{var}.{plane}.000.{k:05d}.{iter_s}"
    g = cfg.grid
    return np.fromfile(path, dtype="<f8").reshape(g.jtot, g.itot)


def read_cross_xz(
    cfg: MicroHHConfig, var: str | None = None, j: int | None = None, iter_s: str | None = None
) -> np.ndarray:
    """Read a vertical cross-section as a ``(ktot, itot)`` array."""
    var = var or cfg.scalar_name
    plane = "xz"
    if j is None:
        # The slice index MicroHH used is encoded in the filename.
        sample = sorted(glob.glob(str(cfg.case_dir / f"{var}.{plane}.000.*")))[0]
        j = int(Path(sample).name.split(".")[3])
    iter_s = iter_s or _latest_iter(cfg, var, plane)
    path = cfg.case_dir / f"{var}.{plane}.000.{j:05d}.{iter_s}"
    g = cfg.grid
    return np.fromfile(path, dtype="<f8").reshape(g.ktot, g.itot)
