"""Base-state pressure and temperature reconstruction for MicroHH runs.

MicroHH is anelastic: it prognoses θ (potential temperature) and carries a
hydrostatic base-state (``pbot``, base-state θ profile) plus a diagnostic
perturbation pressure. The column output does *not* include absolute
pressure or absolute temperature. For pseudo-instruments that need them
(sonic WPL corrections, OP-FTIR mixing-ratio conversions), we reconstruct
p(z) and T(θ, p) here from the same base-state MicroHH uses.

Physics
-------
- Dry hydrostatic base state:  dp/dz = -ρ g,  ρ = p / (R_d · T),  T = θ·(p/p0)^κ
  with κ = R_d / c_p. Substituting gives dp/dz = -g p^(1-κ) / (R_d · θ · p0^(-κ)),
  which integrates numerically level-by-level using MicroHH's initial θ profile
  (this matches the base state ``[thermo] swbasestate=anelastic`` computes).
- Temperature at a level:      T(z, t) = θ(z, t) · (p(z) / p0)^κ.
  θ varies in time but p(z) is base-state (time-invariant in the anelastic
  approximation), so T inherits θ's time variation directly.

This module is intentionally physics-only: it does not touch NetCDF files or
config objects. Callers pass in numpy arrays.
"""
from __future__ import annotations

import numpy as np

R_D = 287.04        # J kg-1 K-1, dry-air gas constant
C_P = 1005.0        # J kg-1 K-1, dry-air specific heat at constant pressure
G = 9.81            # m s-2
KAPPA = R_D / C_P   # ≈ 0.286
P0 = 1.0e5          # Pa, reference pressure for potential temperature


def base_state_pressure(z: np.ndarray, theta_base: np.ndarray, p_bot: float = P0) -> np.ndarray:
    """Hydrostatic base-state pressure profile [Pa] on the levels ``z``.

    Integrates ``dp/dz = -g / (R_d · θ · (p/p0)^κ)`` upward from ``p_bot``,
    which is the anelastic base state MicroHH constructs from the initial θ
    profile (see MicroHH ``[thermo] swbasestate=anelastic``, ``pbot``).

    Parameters
    ----------
    z, theta_base : (nz,)
        Level heights [m] and base-state potential temperature [K]. Must be
        monotonically increasing in z.
    p_bot : float
        Surface pressure [Pa] (MicroHH's ``pbot``; default 1.0e5).
    """
    z = np.asarray(z, dtype=float)
    theta = np.asarray(theta_base, dtype=float)
    if z.shape != theta.shape or z.ndim != 1:
        raise ValueError("z and theta_base must be 1-D arrays of the same length.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing.")

    nz = z.size
    p = np.empty(nz, dtype=float)
    p[0] = p_bot
    for k in range(nz - 1):
        dz = z[k + 1] - z[k]
        # Midpoint θ for a small trapezoidal step; base state is smooth so
        # single-step Euler on an interpolated θ is enough here.
        th_mid = 0.5 * (theta[k] + theta[k + 1])
        # Substitute ρ = p·(p/p0)^-κ / (R_d · θ) into hydrostatic balance.
        # Solve dp/dz = -g · p^(1-κ) · p0^κ / (R_d · θ) by treating the RHS
        # coefficient as constant across the step.
        coef = G * (P0 ** KAPPA) / (R_D * th_mid)
        # (1-κ) integration: d(p^κ)/dz = -κ · coef  →  p^κ decreases linearly.
        pk_kappa = p[k] ** KAPPA
        pk1_kappa = pk_kappa - KAPPA * coef * dz
        if pk1_kappa <= 0:
            raise ValueError(
                f"Base-state pressure went non-positive at level {k+1} "
                f"(z={z[k+1]:g}); check theta_base and p_bot."
            )
        p[k + 1] = pk1_kappa ** (1.0 / KAPPA)
    return p


def temperature_from_theta(theta: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Absolute temperature T = θ · (p/p0)^κ.

    ``theta`` and ``pressure`` broadcast against each other; typically
    ``theta`` is shape ``(nt, nz)`` and ``pressure`` is shape ``(nz,)``.
    """
    return np.asarray(theta, dtype=float) * (np.asarray(pressure, dtype=float) / P0) ** KAPPA
