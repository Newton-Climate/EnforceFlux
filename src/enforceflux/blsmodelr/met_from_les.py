"""LES → :class:`BlsInterval` convenience shim.

The real work is now split across two modules:

- :func:`enforceflux.instrument.sonic.sample_sonic_from_les` samples the LES
  field at a sensor location and returns a :class:`SonicObservation`.
- :func:`enforceflux.blsmodelr.met_from_sonic.interval_from_sonic` runs the
  EC processor on that observation and returns a :class:`BlsInterval`.

Both a real sonic and an LES pseudo-sonic feed the same processor, so the
inversion path is identical for OSSE and real-data runs.

This module keeps the older :func:`intervals_from_les` entry point for
back-compat: it now (i) bins the LES time axis into windows, (ii) builds a
one-column-per-window ``SonicObservation`` at the domain-center reference
column, and (iii) processes each with :func:`interval_from_sonic`. If you
have a sensor location, prefer calling
:func:`sample_sonic_from_les` + :func:`interval_from_sonic` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enforceflux.blsmodelr import BlsInterval
from enforceflux.blsmodelr.met_from_sonic import KAPPA, G, interval_from_sonic
from enforceflux.instrument.sonic import SonicObservation

__all__ = [
    "KAPPA", "G", "VelocityField",
    "intervals_from_les", "intervals_from_microhh_output",
]


@dataclass(frozen=True)
class VelocityField:
    """Container matching the ``.u/.v/.w`` duck-typed contract."""

    u: np.ndarray  # (nt, nz, ny, nx)
    v: np.ndarray
    w: np.ndarray


def _interp_z(field: np.ndarray, z: np.ndarray, z_ref: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z_ref <= z[0]:
        return field[:, 0]
    if z_ref >= z[-1]:
        return field[:, -1]
    k1 = int(np.searchsorted(z, z_ref))
    k0 = k1 - 1
    w1 = (z_ref - z[k0]) / (z[k1] - z[k0])
    return (1.0 - w1) * field[:, k0] + w1 * field[:, k1]


def intervals_from_les(
    *,
    velocity_field,
    z: np.ndarray,
    surface_theta: np.ndarray,   # kept for signature back-compat; unused
    theta_ref: np.ndarray,
    z_ref: float,
    z0: float,
    window_s: float,
    dt: float,
    id_prefix: str = "t",
) -> list[BlsInterval]:
    """Bin LES output into per-window :class:`BlsInterval`s (back-compat entry).

    Delegates to :func:`interval_from_sonic` after synthesising one
    domain-averaged :class:`SonicObservation` per window.
    """
    u = np.asarray(velocity_field.u, dtype=float)
    v = np.asarray(velocity_field.v, dtype=float)
    w = np.asarray(velocity_field.w, dtype=float)
    if not (u.shape == v.shape == w.shape) or u.ndim != 4:
        raise ValueError("u/v/w must share shape (nt, nz, ny, nx).")
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.shape[0] != u.shape[1]:
        raise ValueError("z must be 1-D with length nz.")

    n_per = int(round(window_s / dt))
    if n_per <= 0:
        raise ValueError("window_s/dt must yield at least one sample per window.")
    nt = u.shape[0]
    n_windows = nt // n_per
    if n_windows == 0:
        raise ValueError("Not enough time steps for one full window.")

    # Horizontally averaged column at z_ref — approximates a domain-center sonic.
    u_zr = _interp_z(u, z, z_ref)
    v_zr = _interp_z(v, z, z_ref)
    w_zr = _interp_z(w, z, z_ref)
    u_col = u_zr.reshape(nt, -1).mean(axis=1) if u_zr.ndim > 1 else u_zr
    v_col = v_zr.reshape(nt, -1).mean(axis=1) if v_zr.ndim > 1 else v_zr
    w_col = w_zr.reshape(nt, -1).mean(axis=1) if w_zr.ndim > 1 else w_zr

    theta_ref = np.asarray(theta_ref, dtype=float)
    if theta_ref.ndim == 1:
        theta_col = theta_ref
    else:
        theta_col = theta_ref.reshape(theta_ref.shape[0], -1).mean(axis=1)

    times_s = np.arange(nt, dtype=float) * dt

    _ = surface_theta  # unused; retained for signature back-compat

    intervals: list[BlsInterval] = []
    for k in range(n_windows):
        sl = slice(k * n_per, (k + 1) * n_per)
        obs = SonicObservation(
            instrument_id="les_domain_avg",
            interval_id=f"{id_prefix}{k:04d}",
            x=0.0, y=0.0, z=float(z_ref),
            times_s=times_s[sl],
            u=u_col[sl], v=v_col[sl], w=w_col[sl],
            theta=theta_col[sl],
            z0=float(z0),
            meta={"source": "les_domain_avg_shim"},
        )
        intervals.append(interval_from_sonic(obs))
    return intervals


def intervals_from_microhh_output(
    cfg,                       # MicroHHConfig
    *,
    receptor_id: str | None = None,
    ix: int | None = None,
    iy: int | None = None,
    z_ref: float,
    z0: float,
    window_s: float,
    id_prefix: str = "t",
) -> list[BlsInterval]:
    """Sample a MicroHH column as a fixed-height sonic and window into
    :class:`BlsInterval`s.

    Provide either ``receptor_id`` (resolved via ``cfg.receptors``) or
    explicit ``(ix, iy)`` grid indices. The instrument observes at ``z_ref``
    only — the column is a MicroHH file-format detail, not part of the
    observation model. Pipeline:

        MicroHH column NetCDF  ─▶  SonicObservation (at z_ref)
                                ─▶  interval_from_sonic (per window)
    """
    from enforceflux.instrument.sonic import sonic_from_microhh

    # One read → one SonicObservation covering the whole run.
    obs_full = sonic_from_microhh(
        cfg, receptor_id=receptor_id, ix=ix, iy=iy,
        z_ref=z_ref, z0=z0,
    )
    times_s = obs_full.times_s
    if times_s.size < 2:
        raise ValueError("MicroHH column has fewer than 2 timesteps.")
    dt = float(np.median(np.diff(times_s)))
    n_per = int(round(window_s / dt))
    if n_per <= 0:
        raise ValueError("window_s/dt must yield at least one sample per window.")
    n_windows = times_s.size // n_per
    if n_windows == 0:
        raise ValueError("Not enough MicroHH snapshots for one full window.")

    intervals: list[BlsInterval] = []
    for k in range(n_windows):
        sl = slice(k * n_per, (k + 1) * n_per)
        obs = SonicObservation(
            instrument_id=obs_full.instrument_id,
            interval_id=f"{id_prefix}{k:04d}",
            x=obs_full.x, y=obs_full.y, z=obs_full.z,
            times_s=times_s[sl],
            u=obs_full.u[sl], v=obs_full.v[sl],
            w=obs_full.w[sl], theta=obs_full.theta[sl],
            z0=obs_full.z0,
            meta={**obs_full.meta, "window_index": k},
        )
        intervals.append(interval_from_sonic(obs))
    return intervals
