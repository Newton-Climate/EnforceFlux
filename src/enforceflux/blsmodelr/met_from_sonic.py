"""Turn a :class:`SonicObservation` into a :class:`BlsInterval`.

This is the single EC processor that both real sonic hardware and LES
pseudo-sonics feed into. Splitting the pipeline this way makes the
fraternal-twin OSSE discipline structural: the inversion literally cannot
see the LES field, only what a sonic would have measured.

Physics
-------
- Double rotation (Kaimal & Finnigan 1994): rotate the site-frame (u, v, w)
  into streamwise coordinates so ⟨v⟩ = ⟨w⟩ = 0 over the window.
- Momentum flux and friction velocity:
    u* = (⟨u'w'⟩² + ⟨v'w'⟩²)^(1/4)         (site frame; rotation-invariant)
- Sensible heat flux and Obukhov length:
    L  = -u*³ · ⟨θ⟩ / (κ · g · ⟨w'θ'⟩)
- Standard deviations of the site-frame components (bLSmodelR consumes
  σ_u, σ_v, σ_w without a rotation assumption).

Saturation
----------
|L| is clipped to 1e6 m when ⟨w'θ'⟩ is ~0, matching :mod:`met_from_les`.
"""
from __future__ import annotations

import numpy as np

from enforceflux.blsmodelr import BlsInterval
from enforceflux.instrument.sonic import SonicObservation

KAPPA = 0.4
G = 9.81
_L_SATURATION = 1.0e6


def interval_from_sonic(
    obs: SonicObservation,
    *,
    interval_id: str | None = None,
) -> BlsInterval:
    """Derive one :class:`BlsInterval` from a sonic timeseries."""
    mask = obs.valid_mask if obs.valid_mask is not None else np.ones_like(obs.u, dtype=bool)
    u = obs.u[mask]; v = obs.v[mask]; w = obs.w[mask]; theta = obs.theta[mask]
    if u.size < 8:
        raise ValueError(
            f"SonicObservation {obs.instrument_id!r} has {u.size} valid samples "
            f"in interval {obs.interval_id!r}; refuse to compute turbulence stats."
        )

    u_bar = float(u.mean()); v_bar = float(v.mean()); w_bar = float(w.mean())
    up = u - u_bar; vp = v - v_bar; wp = w - w_bar

    upwp = float((up * wp).mean())
    vpwp = float((vp * wp).mean())
    u_star = float((upwp * upwp + vpwp * vpwp) ** 0.25)

    theta_bar = float(theta.mean())
    tp = theta - theta_bar
    wptp = float((wp * tp).mean())

    if abs(wptp) < 1e-12:
        L = _L_SATURATION if wptp >= 0 else -_L_SATURATION
    else:
        L = -(u_star ** 3) * theta_bar / (KAPPA * G * wptp)
        if not np.isfinite(L) or abs(L) > _L_SATURATION:
            L = float(np.sign(L) * _L_SATURATION) if L != 0 else _L_SATURATION

    wind_speed = float(np.hypot(u_bar, v_bar))
    # Meteorological convention: direction FROM which wind blows, 0=N, 90=E.
    wind_dir = float((np.degrees(np.arctan2(-u_bar, -v_bar)) + 360.0) % 360.0)

    return BlsInterval(
        id=interval_id or obs.interval_id,
        u_star=u_star,
        L=float(L),
        z0=float(obs.z0),
        wind_dir_deg=wind_dir,
        wind_speed=wind_speed,
        z_ref=float(obs.z),
        sd_u=float(up.std()),
        sd_v=float(vp.std()),
        sd_w=float(wp.std()),
    )
