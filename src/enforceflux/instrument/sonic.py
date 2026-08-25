"""Sonic-anemometer observation type and LES sampler.

Defines the canonical met observation :class:`SonicObservation` — a raw
timeseries of ``u, v, w, θ`` at a sensor location over one averaging window
— and one instrument-operator function, :func:`sample_sonic_from_les`, that
turns an LES velocity field into that observation.

The point of putting this in ``instrument/`` (not in ``blsmodelr/``) is
architectural: any downstream that needs turbulence stats (bLSmodelR today,
another LPDM tomorrow) consumes :class:`SonicObservation`. Real sonic loaders
and LES samplers both emit this shape, so real-data and OSSE code paths share
every step downstream of the instrument.

This module intentionally does **no** EC processing. Deapotchka rotation,
u*, L, σ, and stationarity live in the consumer (see
:mod:`enforceflux.blsmodelr.met_from_sonic`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SonicObservation:
    """One averaging window of raw sonic-anemometer data.

    Fields
    ------
    instrument_id, interval_id : str
        Identifiers used downstream (e.g. as ``BlsInterval.id``).
    x, y, z : float
        Sensor location in the local metric frame; ``z`` is height AGL [m].
    times_s : (n,) float
        Timestamps [s] at sonic native rate. Non-uniform rates are accepted;
        a warning is emitted below 5 Hz median rate (turbulence spectrum
        under-resolved).
    u, v, w : (n,) float
        Wind components [m s⁻¹] in the site frame (i.e., *not* rotated into
        streamwise coordinates — that is the processor's job).
    theta : (n,) float
        Potential temperature [K]. For OSSE use this is exact; for real
        sonics it is the sonic temperature and callers should apply the
        humidity/pressure correction upstream if bias matters.
    z0 : float
        Roughness length [m]. Site metadata — not measured by the sonic —
        attached here so the observation is self-contained downstream.
    valid_mask : (n,) bool or None
        Optional spike/gap flag; ``None`` means "all valid".
    meta : dict
        Free-form provenance (nature-run seed, real-site ID, pre-processed
        stats if the loader received EC output instead of raw timeseries).
    """

    instrument_id: str
    interval_id: str
    x: float
    y: float
    z: float
    times_s: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    theta: np.ndarray
    z0: float
    valid_mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.times_s.shape[0]
        for name in ("u", "v", "w", "theta"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(
                    f"SonicObservation field {name!r} shape {arr.shape} "
                    f"does not match times_s length {n}."
                )
        if n >= 2:
            median_hz = 1.0 / float(np.median(np.diff(self.times_s)))
            if median_hz < 5.0:
                warnings.warn(
                    f"Sonic rate {median_hz:.2f} Hz on {self.instrument_id!r} "
                    f"is below 5 Hz; turbulence spectrum is under-resolved and "
                    f"u* / σ estimates will be biased low.",
                    stacklevel=2,
                )


# ── LES sampler ──────────────────────────────────────────────────────────


def sample_sonic_from_les(
    *,
    velocity_field,          # duck-typed .u/.v/.w of shape (nt, nz, ny, nx)
    theta_field,             # (nt, nz, ny, nx) OR (nt, ny, nx) OR (nt,)
    x_grid: np.ndarray,      # (nx,) LES cell-center x [m], local frame
    y_grid: np.ndarray,      # (ny,)
    z_grid: np.ndarray,      # (nz,) heights AGL [m]
    times_s: np.ndarray,     # (nt,) snapshot timestamps [s]
    sensor_x: float,
    sensor_y: float,
    sensor_z: float,
    z0: float,
    instrument_id: str,
    interval_id: str = "t0000",
    meta: dict | None = None,
) -> SonicObservation:
    """Sample an LES field at the sensor location and return a
    :class:`SonicObservation`.

    Horizontal: nearest LES grid point (a sonic is a point instrument;
    trilinear interpolation over sub-grid distances is spurious precision).
    Vertical: linear interpolation between the two straddling grid levels.
    Time: no resampling — the LES snapshot cadence is the sonic rate.
    """
    u = np.asarray(velocity_field.u, dtype=float)
    v = np.asarray(velocity_field.v, dtype=float)
    w = np.asarray(velocity_field.w, dtype=float)
    if not (u.shape == v.shape == w.shape) or u.ndim != 4:
        raise ValueError("u/v/w must share shape (nt, nz, ny, nx).")
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    times_s = np.asarray(times_s, dtype=float)

    ix = int(np.argmin(np.abs(x_grid - sensor_x)))
    iy = int(np.argmin(np.abs(y_grid - sensor_y)))

    def _z_interp(vol: np.ndarray) -> np.ndarray:
        col = vol[:, :, iy, ix]  # (nt, nz)
        return _lin_interp_axis1(col, z_grid, sensor_z)

    u_s = _z_interp(u)
    v_s = _z_interp(v)
    w_s = _z_interp(w)

    theta_field = np.asarray(theta_field, dtype=float)
    if theta_field.ndim == 4:
        theta_s = _z_interp(theta_field)
    elif theta_field.ndim == 3:
        theta_s = theta_field[:, iy, ix]
    elif theta_field.ndim == 1:
        theta_s = theta_field
    else:
        raise ValueError(
            f"theta_field must be (nt,nz,ny,nx), (nt,ny,nx), or (nt,); "
            f"got shape {theta_field.shape}"
        )

    return SonicObservation(
        instrument_id=instrument_id,
        interval_id=interval_id,
        x=float(sensor_x), y=float(sensor_y), z=float(sensor_z),
        times_s=times_s,
        u=u_s, v=v_s, w=w_s, theta=theta_s.astype(float),
        z0=float(z0),
        meta=dict(meta or {}, source="les_sample", ix=ix, iy=iy),
    )


def sonic_from_microhh(
    cfg,                           # microhh.sim_config.MicroHHConfig
    *,
    receptor_id: str | None = None,
    ix: int | None = None,
    iy: int | None = None,
    z_ref: float,
    z0: float,
    instrument_id: str | None = None,
    interval_id: str = "t0000",
    meta: dict | None = None,
) -> SonicObservation:
    """Public sonic-from-MicroHH entry point: instrument at a fixed height.

    Provide either ``receptor_id`` (resolved against ``cfg.receptors``) or an
    explicit column grid index ``(ix, iy)``. The instrument observes at
    ``z_ref`` only — mirroring the real observational schema of a sonic at a
    fixed mast height. Multi-height towers or beam-integrating instruments
    that legitimately need more than one level can call
    :func:`enforceflux.microhh.output.read_column_full` directly and pass the
    result to the internal helper below.
    """
    from enforceflux.microhh.geometry import BoxProjection
    from enforceflux.microhh.output import read_column_full

    if receptor_id is not None:
        matches = [r for r in cfg.receptors if r.id == receptor_id]
        if not matches:
            raise ValueError(f"No receptor {receptor_id!r} in cfg.receptors.")
        r = matches[0]
        proj = BoxProjection(
            origin_lon=cfg.origin_lon, origin_lat=cfg.origin_lat,
            x_bearing_deg=cfg.x_bearing_deg,
            source_x0=cfg.source_x0, source_y0=cfg.source_y0,
        )
        x_m, y_m = proj.to_box(r.lon, r.lat)
        ix = int(round(x_m) / cfg.grid.dx)
        iy = int(round(y_m) / cfg.grid.dy)
        instrument_id = instrument_id or receptor_id
        sensor_x, sensor_y = float(x_m), float(y_m)
    elif ix is None or iy is None:
        raise ValueError("Provide either receptor_id or both ix and iy.")
    else:
        instrument_id = instrument_id or f"col_{ix:05d}_{iy:05d}"
        sensor_x = (ix + 0.5) * cfg.grid.dx
        sensor_y = (iy + 0.5) * cfg.grid.dy

    column = read_column_full(cfg, ix, iy)
    return _sonic_from_column_data(
        column, z_ref=z_ref, z0=z0,
        instrument_id=instrument_id, interval_id=interval_id,
        sensor_x=sensor_x, sensor_y=sensor_y, meta=meta,
    )


def _sonic_from_column_data(
    column,                        # microhh.output.LesColumnData
    *,
    z_ref: float,
    z0: float,
    instrument_id: str,
    interval_id: str = "t0000",
    sensor_x: float = 0.0,
    sensor_y: float = 0.0,
    meta: dict | None = None,
) -> SonicObservation:
    """Internal helper: interpolate a pre-loaded column to ``z_ref``.

    Kept accessible for multi-height / beam-integrating pseudo-instruments
    that read the full column once and sample it at several heights.
    """
    zc = np.asarray(column.z, dtype=float)
    def _col_to_ref(field2d):
        return _lin_interp_axis1(np.asarray(field2d, dtype=float), zc, z_ref)

    obs_meta = dict(meta or {})
    obs_meta.update(
        source="microhh_column",
        ix=int(column.ix), iy=int(column.iy),
        pressure_at_zref=float(np.interp(z_ref, zc, column.pressure)),
        ch4_at_zref=_col_to_ref(column.ch4),
        h2o_at_zref=(_col_to_ref(column.h2o) if column.h2o is not None else None),
        temperature_at_zref=_col_to_ref(column.temperature),
        # These are SGS-inclusive wall-model diagnostics. Resolved u'w'/v'w'
        # alone tends to zero near the LES wall and must not be interpreted as
        # the total surface friction velocity used by MOST/bLS.
        surface_ustar=(
            np.asarray(column.ustar, dtype=float) if column.ustar is not None else None
        ),
        surface_obuk=(
            np.asarray(column.obuk, dtype=float) if column.obuk is not None else None
        ),
    )
    return SonicObservation(
        instrument_id=instrument_id,
        interval_id=interval_id,
        x=float(sensor_x), y=float(sensor_y), z=float(z_ref),
        times_s=np.asarray(column.times_s, dtype=float),
        u=_col_to_ref(column.u),
        v=_col_to_ref(column.v),
        w=_col_to_ref(column.w),
        theta=_col_to_ref(column.theta),
        z0=float(z0),
        meta=obs_meta,
    )


def _lin_interp_axis1(col: np.ndarray, z: np.ndarray, z_ref: float) -> np.ndarray:
    """Linear interpolation of ``col`` (nt, nz) to ``z_ref``, clamped at ends."""
    if z_ref <= z[0]:
        return col[:, 0]
    if z_ref >= z[-1]:
        return col[:, -1]
    k1 = int(np.searchsorted(z, z_ref))
    k0 = k1 - 1
    w1 = (z_ref - z[k0]) / (z[k1] - z[k0])
    return (1.0 - w1) * col[:, k0] + w1 * col[:, k1]
