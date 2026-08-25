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


@dataclass(frozen=True)
class LesColumnData:
    """Full atmospheric state at one MicroHH column, all fields on the same
    (time, z) grid.

    Native to the MicroHH column NetCDF: ``times_s``, ``z``, ``u``, ``v``,
    ``w``, ``theta``, ``ch4``, ``h2o`` (when the case includes H2O), and the
    SGS-inclusive surface-layer diagnostics ``ustar`` and ``obuk``.
    Derived from the anelastic base state via
    :mod:`enforceflux.microhh.thermo`: ``pressure`` (shape ``(nz,)``) and
    ``temperature`` (shape ``(nt, nz)``).
    """

    ix: int
    iy: int
    times_s: np.ndarray            # (nt,)
    z: np.ndarray                  # (nz,)
    u: np.ndarray                  # (nt, nz)
    v: np.ndarray                  # (nt, nz)
    w: np.ndarray                  # (nt, nz)
    theta: np.ndarray              # (nt, nz)
    pressure: np.ndarray           # (nz,)   base-state hydrostatic
    temperature: np.ndarray        # (nt, nz)  T = θ (p/p0)^κ
    ch4: np.ndarray                # (nt, nz)
    h2o: np.ndarray | None         # (nt, nz) or None when include_h2o=False
    ustar: np.ndarray | None       # (nt,) SGS-inclusive surface friction velocity
    obuk: np.ndarray | None        # (nt,) surface Obukhov length


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


def read_column_full(cfg: MicroHHConfig, ix: int, iy: int) -> LesColumnData:
    """Read the full atmospheric state at one MicroHH column.

    Loads u, v, w, θ, CH4, and (if configured) H2O from the column NetCDF at
    grid index ``(ix, iy)``, then reconstructs the anelastic base-state
    pressure profile from the case's initial θ (see
    :func:`enforceflux.microhh.thermo.base_state_pressure`) and derives
    absolute temperature T = θ (p/p0)^κ on the (nt, nz) grid.

    Returns
    -------
    LesColumnData
        Everything a sonic / OP-FTIR pseudo-instrument at this column needs.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading MicroHH output needs the 'analysis' extra (xarray/netCDF4): "
            "pip install enforceflux[analysis]"
        ) from exc

    from enforceflux.microhh.case import initial_profiles
    from enforceflux.microhh.thermo import base_state_pressure, temperature_from_theta

    path = find_column_file(cfg, ix, iy)
    if path is None:
        raise FileNotFoundError(
            f"No column file at grid index ({ix:05d},{iy:05d}) in "
            f"{cfg.case_dir}. Run the case first."
        )
    ds = xr.open_dataset(path, decode_times=False)
    try:
        times_s = np.asarray(ds["time"].values, dtype=float)
        z = np.asarray(ds["z"].values, dtype=float)
        u = np.asarray(ds["u"].values, dtype=float)
        v = np.asarray(ds["v"].values, dtype=float)
        # MicroHH stores w on half-levels ("zh"); if that variable exists,
        # interpolate to full levels so w shares (nt, nz) with u/v.
        if "w" in ds.variables and ds["w"].shape == u.shape:
            w = np.asarray(ds["w"].values, dtype=float)
        elif "w" in ds.variables:
            wh = np.asarray(ds["w"].values, dtype=float)  # (nt, nzh)
            w = 0.5 * (wh[:, :-1] + wh[:, 1:]) if wh.shape[1] == z.size + 1 else wh
        else:
            w = np.zeros_like(u)
        theta = np.asarray(ds["th"].values, dtype=float)
        ch4 = np.asarray(ds[cfg.scalar_name].values, dtype=float)
        h2o = (np.asarray(ds[cfg.h2o_name].values, dtype=float)
               if cfg.include_h2o and cfg.h2o_name in ds.variables else None)
        ustar = (np.asarray(ds["ustar"].values, dtype=float)
                 if "ustar" in ds.variables else None)
        obuk = (np.asarray(ds["obuk"].values, dtype=float)
                if "obuk" in ds.variables else None)
    finally:
        ds.close()

    # Base-state pressure from the case's initial θ profile. MicroHH's
    # anelastic base state is fixed at t=0, so this is exact.
    prof = initial_profiles(cfg)
    p = base_state_pressure(prof["z"], prof["th"], p_bot=1.0e5)
    if prof["z"].shape == z.shape and np.allclose(prof["z"], z):
        p_col = p
    else:
        p_col = np.interp(z, prof["z"], p)
    T = temperature_from_theta(theta, p_col)

    return LesColumnData(
        ix=ix, iy=iy, times_s=times_s, z=z,
        u=u, v=v, w=w, theta=theta,
        pressure=p_col, temperature=T,
        ch4=ch4, h2o=h2o, ustar=ustar, obuk=obuk,
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
    dtype = "<f4" if cfg.precision == "float32" else "<f8"
    return np.fromfile(path, dtype=dtype).reshape(g.jtot, g.itot)


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
    dtype = "<f4" if cfg.precision == "float32" else "<f8"
    return np.fromfile(path, dtype=dtype).reshape(g.ktot, g.itot)
