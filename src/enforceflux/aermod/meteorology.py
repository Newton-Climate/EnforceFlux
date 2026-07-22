"""Boundary-layer state derivation for the AERMOD-style dispersion kernel.

AERMOD is driven by similarity-theory boundary-layer parameters rather than by
Pasquill classes directly: the surface friction velocity ``u*``, the
Monin-Obukhov length ``L``, the mixing height ``zi``, and (in the convective
boundary layer) the convective velocity scale ``w*``. This module turns the
user-facing :class:`~enforceflux.aermod.config.SurfaceMet` — which may specify
nothing more than a wind speed and a stability class — into that parameter set.

Everything here is written in ``jax.numpy`` and returns a
:class:`MetState` (a ``NamedTuple``, hence a JAX pytree), so the whole chain
from meteorology to concentration stays differentiable: ``jax.grad`` through a
concentration with respect to wind speed, roughness, or mixing height works
without any special handling.

References
----------
Golder, D. (1972), Relations among stability parameters in the surface layer.
Businger et al. (1971), Flux-profile relationships in the atmospheric surface layer.
Zilitinkevich, S. (1972), On the determination of the height of the Ekman layer.
Cimorelli et al. (2005), AERMOD: A dispersion model for industrial source
applications. Part I. J. Appl. Meteor. 44, 682-693.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from enforceflux.aermod.config import SurfaceMet

VON_KARMAN = 0.4
GRAVITY = 9.80665
AIR_DENSITY = 1.2  # kg m-3
AIR_CP = 1005.0  # J kg-1 K-1
CORIOLIS = 1.0e-4  # s-1, mid-latitude

# Golder (1972): 1/L = a + b log10(z0), per Pasquill-Gifford class.
_GOLDER_COEFFS = {
    "A": (-0.096, 0.029),
    "B": (-0.037, 0.029),
    "C": (-0.002, 0.018),
    "D": (0.0, 0.0),
    "E": (0.004, -0.018),
    "F": (0.035, -0.036),
}

MIN_MIXING_HEIGHT_M = 50.0
MAX_MIXING_HEIGHT_M = 4000.0


class MetState(NamedTuple):
    """Resolved boundary-layer parameters for one hour.

    Stability is carried as ``inv_obukhov_length`` (1/L) rather than L so the
    neutral limit is the finite value 0 instead of an infinity.
    """

    wind_speed: jnp.ndarray  # m s-1 at reference_height
    wind_dir_rad: jnp.ndarray  # radians, direction wind blows *from*
    reference_height: jnp.ndarray  # m
    temperature: jnp.ndarray  # K
    roughness: jnp.ndarray  # m
    u_star: jnp.ndarray  # m s-1
    inv_obukhov_length: jnp.ndarray  # m-1 (>0 stable, <0 convective)
    mixing_height: jnp.ndarray  # m
    w_star: jnp.ndarray  # m s-1 (0 in the stable boundary layer)
    dtheta_dz: jnp.ndarray  # K m-1, above the mixed layer

    @property
    def is_convective(self) -> jnp.ndarray:
        """1.0 where the boundary layer is convective, 0.0 otherwise."""
        return jnp.where(self.inv_obukhov_length < 0.0, 1.0, 0.0)


def golder_inv_obukhov_length(stability_class: str, roughness_m: float) -> float:
    """Monin-Obukhov 1/L from a Pasquill-Gifford class and surface roughness."""
    try:
        a, b = _GOLDER_COEFFS[stability_class]
    except KeyError:  # pragma: no cover - validated upstream in SurfaceMet
        raise ValueError(f"Unknown stability class {stability_class!r}") from None
    return a + b * jnp.log10(jnp.asarray(roughness_m, dtype=float))


def _psi_m(zeta: jnp.ndarray) -> jnp.ndarray:
    """Businger-Dyer stability correction for the momentum profile.

    ``zeta = z/L``. Negative (convective) and positive (stable) branches are
    both evaluated and selected with ``where``; the unstable branch argument is
    clipped so the unused branch cannot emit NaNs into the gradient.
    """
    zeta_unstable = jnp.minimum(zeta, 0.0)
    x = (1.0 - 16.0 * zeta_unstable) ** 0.25
    psi_unstable = (
        2.0 * jnp.log(0.5 * (1.0 + x))
        + jnp.log(0.5 * (1.0 + x * x))
        - 2.0 * jnp.arctan(x)
        + 0.5 * jnp.pi
    )
    # Stable branch saturates: beyond zeta ~ 1 the log-linear law is unreliable
    # and AERMOD effectively caps the shear enhancement.
    psi_stable = -5.0 * jnp.minimum(jnp.maximum(zeta, 0.0), 2.0)
    return jnp.where(zeta < 0.0, psi_unstable, psi_stable)


def friction_velocity(
    wind_speed: jnp.ndarray,
    reference_height: jnp.ndarray,
    roughness: jnp.ndarray,
    inv_l: jnp.ndarray,
) -> jnp.ndarray:
    """``u*`` from the diabatic log-law, inverted at the reference height."""
    denom = (
        jnp.log(reference_height / roughness)
        - _psi_m(reference_height * inv_l)
        + _psi_m(roughness * inv_l)
    )
    # A vanishing/negative denominator means the profile law has broken down
    # (very strong stability, very rough surface); floor it rather than blow up.
    denom = jnp.maximum(denom, 0.5)
    return jnp.maximum(VON_KARMAN * wind_speed / denom, 1.0e-3)


def wind_speed_at(state: MetState, height: jnp.ndarray) -> jnp.ndarray:
    """Diabatic log-law wind speed at ``height`` (m), floored at the roughness."""
    z = jnp.maximum(height, state.roughness * 2.0)
    profile = (
        jnp.log(z / state.roughness)
        - _psi_m(z * state.inv_obukhov_length)
        + _psi_m(state.roughness * state.inv_obukhov_length)
    )
    return jnp.maximum(state.u_star * jnp.maximum(profile, 0.5) / VON_KARMAN, 1.0e-3)


def sensible_heat_flux(
    u_star: jnp.ndarray, inv_l: jnp.ndarray, temperature: jnp.ndarray
) -> jnp.ndarray:
    """Surface sensible heat flux [W m-2] implied by ``u*`` and ``L``."""
    return -(
        AIR_DENSITY
        * AIR_CP
        * temperature
        * u_star**3
        * inv_l
        / (VON_KARMAN * GRAVITY)
    )


def default_mixing_height(
    u_star: jnp.ndarray, inv_l: jnp.ndarray
) -> jnp.ndarray:
    """Diagnostic mixing height when none is supplied.

    Neutral/convective: the Ekman-scaling depth ``0.3 u*/f``. Stable: the
    Zilitinkevich (1972) depth ``0.4 sqrt(u* L / f)``. Both are clamped to a
    physically sane range.
    """
    neutral = 0.3 * u_star / CORIOLIS
    # |L| keeps the unused convective branch finite (and NaN-free under grad).
    abs_l = 1.0 / jnp.maximum(jnp.abs(inv_l), 1.0e-6)
    stable = 0.4 * jnp.sqrt(u_star * abs_l / CORIOLIS)
    zi = jnp.where(inv_l > 0.0, jnp.minimum(stable, neutral), neutral)
    return jnp.clip(zi, MIN_MIXING_HEIGHT_M, MAX_MIXING_HEIGHT_M)


def convective_velocity(
    heat_flux: jnp.ndarray, mixing_height: jnp.ndarray, temperature: jnp.ndarray
) -> jnp.ndarray:
    """``w* = (g H zi / (ρ cp T))^(1/3)``; zero when the surface flux is downward.

    The cube root has an infinite derivative at zero, so the argument is floored
    before the root and the result is masked — a neutral or stable hour then
    yields ``w* = 0`` with a well-defined (zero) gradient rather than a NaN.
    """
    forcing = GRAVITY * heat_flux * mixing_height / (AIR_DENSITY * AIR_CP * temperature)
    safe = jnp.maximum(forcing, 1.0e-9)
    return jnp.where(forcing > 0.0, safe ** (1.0 / 3.0), 0.0)


def derive_met_state(met: SurfaceMet) -> MetState:
    """Resolve a :class:`SurfaceMet` into a fully specified :class:`MetState`.

    Explicit values in ``met`` always win; anything left as ``None`` is derived.
    """
    roughness = jnp.asarray(met.surface_roughness_m, dtype=float)
    reference_height = jnp.asarray(met.reference_height_m, dtype=float)
    temperature = jnp.asarray(met.temperature_k, dtype=float)
    wind_speed = jnp.asarray(met.wind_speed_m_s, dtype=float)

    if met.monin_obukhov_length_m is not None:
        inv_l = jnp.asarray(1.0 / met.monin_obukhov_length_m, dtype=float)
    else:
        inv_l = jnp.asarray(
            golder_inv_obukhov_length(met.stability_class, met.surface_roughness_m),
            dtype=float,
        )

    if met.friction_velocity_m_s is not None:
        u_star = jnp.asarray(met.friction_velocity_m_s, dtype=float)
    else:
        u_star = friction_velocity(wind_speed, reference_height, roughness, inv_l)

    if met.mixing_height_m is not None:
        mixing_height = jnp.asarray(met.mixing_height_m, dtype=float)
    else:
        mixing_height = default_mixing_height(u_star, inv_l)

    if met.convective_velocity_m_s is not None:
        w_star = jnp.asarray(met.convective_velocity_m_s, dtype=float)
    else:
        if met.sensible_heat_flux_w_m2 is not None:
            heat_flux = jnp.asarray(met.sensible_heat_flux_w_m2, dtype=float)
        else:
            heat_flux = sensible_heat_flux(u_star, inv_l, temperature)
        w_star = convective_velocity(heat_flux, mixing_height, temperature)

    return MetState(
        wind_speed=wind_speed,
        wind_dir_rad=jnp.deg2rad(jnp.asarray(met.wind_direction_deg, dtype=float)),
        reference_height=reference_height,
        temperature=temperature,
        roughness=roughness,
        u_star=u_star,
        inv_obukhov_length=inv_l,
        mixing_height=mixing_height,
        w_star=w_star,
        dtheta_dz=jnp.asarray(met.potential_temperature_gradient_k_m, dtype=float),
    )


def stack_met_states(mets) -> MetState:
    """Stack per-hour :class:`MetState`s into one batched state (leading time axis)."""
    states = [derive_met_state(m) for m in mets]
    return MetState(*(jnp.stack([getattr(s, f) for s in states]) for f in MetState._fields))
