"""Unit conversions for MicroHH CH4 output and an analytical plume reference.

MicroHH carries CH4 as a passive tracer in **mass mixing ratio** ``c`` [kg CH4
/ kg air] (``swvmr=false``). These helpers convert to the units the other
transport backends report, and provide a textbook Gaussian-plume concentration
for like-for-like comparison at a downwind point.
"""
from __future__ import annotations

import numpy as np

M_AIR = 28.9647      # g/mol
M_CH4 = 16.043       # g/mol
RHO_AIR = 1.2        # kg/m3, near-surface reference


def mixing_ratio_to_mass_conc(c_kg_kg: np.ndarray | float, rho: float = RHO_AIR):
    """Mass mixing ratio [kg/kg] → mass concentration [kg/m3]."""
    return np.asarray(c_kg_kg) * rho


def mixing_ratio_to_ppb(c_kg_kg: np.ndarray | float):
    """Mass mixing ratio [kg/kg] → mole fraction [ppb]."""
    return np.asarray(c_kg_kg) * (M_AIR / M_CH4) * 1.0e9


def mass_conc_to_ppb(chi_kg_m3: np.ndarray | float, rho: float = RHO_AIR):
    """Mass concentration [kg/m3] → mole fraction [ppb]."""
    return mixing_ratio_to_ppb(np.asarray(chi_kg_m3) / rho)


# ─── Pasquill–Gifford dispersion (Briggs, open-country) ──────────────────────

_PG_Y = {  # sigma_y(x) = a x / sqrt(1 + b x)
    "A": (0.22, 1e-4), "B": (0.16, 1e-4), "C": (0.11, 1e-4),
    "D": (0.08, 1e-4), "E": (0.06, 1e-4), "F": (0.04, 1e-4),
}
_PG_Z = {  # sigma_z(x) = a x / (1 + b x)^c
    "A": (0.20, 0.0, 1.0), "B": (0.12, 0.0, 1.0),
    "C": (0.08, 2e-4, 0.5), "D": (0.06, 1.5e-3, 0.5),
    "E": (0.03, 3e-4, 1.0), "F": (0.016, 3e-4, 1.0),
}


def pg_sigma(x_m: float, stability: str = "B") -> tuple[float, float]:
    """Briggs open-country σy, σz [m] at downwind distance ``x_m``."""
    ay, by = _PG_Y[stability]
    az, bz, cz = _PG_Z[stability]
    sy = ay * x_m / np.sqrt(1.0 + by * x_m)
    sz = az * x_m / (1.0 + bz * x_m) ** cz
    return float(sy), float(sz)


def gaussian_plume_ground_conc(
    Q_kg_s: float, x_m: float, u_m_s: float, h_m: float = 0.0,
    y_m: float = 0.0, stability: str = "B",
) -> float:
    """Ground-level Gaussian-plume mass concentration [kg/m3].

    Continuous point source at height ``h_m`` with total ground reflection:

        C(x,y,0) = Q / (2π u σy σz) · exp(-y²/2σy²) · 2·exp(-h²/2σz²)
    """
    sy, sz = pg_sigma(x_m, stability)
    c = Q_kg_s / (2.0 * np.pi * u_m_s * sy * sz)
    c *= np.exp(-(y_m**2) / (2.0 * sy**2))
    c *= 2.0 * np.exp(-(h_m**2) / (2.0 * sz**2))
    return float(c)
