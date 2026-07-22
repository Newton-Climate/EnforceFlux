"""Differentiable AERMOD-style steady-state plume kernel (JAX).

What this implements
--------------------
The AERMOD dispersion formulation (Cimorelli et al. 2005), written in
``jax.numpy`` so that the concentration is differentiable with respect to
*every* input — emission rate, source position, stack geometry, wind speed and
direction, roughness, mixing height, stability:

* similarity-scaled turbulence, ``σ_v`` and ``σ_w`` from ``u*``/``w*`` profiles;
* Taylor (1921) growth of ``σ_y``/``σ_z`` with a Lagrangian time scale, plus
  buoyancy-induced dispersion;
* Briggs momentum and buoyancy plume rise, gradual (distance-dependent) or
  final, with the stable-layer forms;
* a **stable/neutral** boundary layer treated as a reflected Gaussian, and a
  **convective** boundary layer treated with AERMOD's bi-Gaussian updraft /
  downdraft PDF (Weil et al. 1997 closure);
* no-flux boundaries at the ground and the mixing height via an image series,
  so the far-field well-mixed limit ``χ/Q → 1/(u·zi·√(2π)σ_y)`` is recovered.

What this does not implement
----------------------------
This is a research reimplementation, **not** EPA regulatory AERMOD. Absent
are: the penetrated-plume source for material injected above the CBL top,
terrain treatment (AERMAP receptor-elevation weighting between terrain-
following and horizontal plume states), building downwash (PRIME), area and
volume source integration, deposition/depletion, and NO₂ chemistry options. Do
not use it for regulatory demonstrations; do use it wherever a fast,
vectorized, autodifferentiable near-field plume operator is wanted.

All quantities are SI. ``χ/Q`` is returned in s m⁻³, i.e. kg m⁻³ of
concentration per kg s⁻¹ of emission.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from enforceflux.aermod.config import DispersionOptions
from enforceflux.aermod.meteorology import GRAVITY, MetState, wind_speed_at

_TINY = 1.0e-12
_SQRT_2PI = jnp.sqrt(2.0 * jnp.pi)


class StackArray(NamedTuple):
    """Release parameters as JAX arrays (differentiable stack geometry)."""

    height: jnp.ndarray  # m above ground
    diameter: jnp.ndarray  # m
    exit_velocity: jnp.ndarray  # m s-1
    exit_temperature: jnp.ndarray  # K (<=0 → ambient, no buoyancy)


class SourceArray(NamedTuple):
    """Source location as JAX arrays."""

    x: jnp.ndarray
    y: jnp.ndarray
    base_elevation: jnp.ndarray  # m, added to the stack height


class ReceptorArray(NamedTuple):
    """Receptor location as JAX arrays."""

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray


# ── Turbulence profiles ──────────────────────────────────────────────────────


def sigma_v(state: MetState, height: jnp.ndarray, options: DispersionOptions) -> jnp.ndarray:
    """Lateral turbulence intensity at ``height``.

    Convective and mechanical contributions add in quadrature:
    ``σ_vc = 0.35 w*`` (height-independent through the mixed layer) and
    ``σ_vm = 1.9 u* (1 - 0.8 z/zi)``.
    """
    frac = jnp.clip(height / jnp.maximum(state.mixing_height, _TINY), 0.0, 1.0)
    conv = 0.35 * state.w_star
    mech = 1.9 * state.u_star * (1.0 - 0.8 * frac)
    return jnp.maximum(
        jnp.sqrt(conv**2 + jnp.maximum(mech, 0.0) ** 2), options.min_sigma_v_m_s
    )


def sigma_w_components(
    state: MetState, height: jnp.ndarray, options: DispersionOptions
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convective and mechanical vertical turbulence at ``height``.

    ``σ_wc² = min(1.6 (z/zi)^{2/3}, 0.35) w*²`` grows through the surface layer
    and saturates in the mixed layer; ``σ_wm = 1.3 u* (1 - z/zi)^{3/4}`` decays
    to the inversion. They are returned separately because the convective
    boundary-layer treatment needs the convective part on its own.
    """
    zi = jnp.maximum(state.mixing_height, _TINY)
    frac = jnp.clip(height / zi, 0.0, 1.0)
    conv_sq = jnp.minimum(1.6 * jnp.maximum(frac, _TINY) ** (2.0 / 3.0), 0.35) * state.w_star**2
    # The +TINY keeps d(√x)/dx finite at w* = 0 (stable/neutral hours).
    conv = jnp.sqrt(conv_sq + _TINY)
    mech = 1.3 * state.u_star * jnp.maximum(1.0 - frac, _TINY) ** 0.75
    return conv, jnp.maximum(mech, options.min_sigma_w_m_s)


# ── Plume rise ───────────────────────────────────────────────────────────────


def _safe_pow(value: jnp.ndarray, exponent: float) -> jnp.ndarray:
    """``value**exponent`` with the singular derivative at 0 held finite."""
    return jnp.maximum(value, _TINY) ** exponent


def buoyancy_flux(stack: StackArray, ambient_temperature: jnp.ndarray) -> jnp.ndarray:
    """Briggs buoyancy flux ``Fb`` [m⁴ s⁻³]; zero for an ambient-temperature release."""
    t_exit = jnp.where(stack.exit_temperature > 0.0, stack.exit_temperature, ambient_temperature)
    delta = jnp.maximum(t_exit - ambient_temperature, 0.0)
    return (
        GRAVITY
        * stack.exit_velocity
        * stack.diameter**2
        * delta
        / (4.0 * jnp.maximum(t_exit, _TINY))
    )


def momentum_flux(stack: StackArray, ambient_temperature: jnp.ndarray) -> jnp.ndarray:
    """Briggs momentum flux ``Fm`` [m⁴ s⁻²]."""
    t_exit = jnp.where(stack.exit_temperature > 0.0, stack.exit_temperature, ambient_temperature)
    return (
        stack.exit_velocity**2
        * stack.diameter**2
        * ambient_temperature
        / (4.0 * jnp.maximum(t_exit, _TINY))
    )


def plume_rise(
    stack: StackArray,
    state: MetState,
    wind: jnp.ndarray,
    downwind_distance: jnp.ndarray,
    options: DispersionOptions,
) -> jnp.ndarray:
    """Briggs plume rise Δh [m] at ``downwind_distance``.

    Buoyancy and momentum rise are computed separately and the larger is taken
    (standard Briggs practice). In the stable boundary layer both are limited by
    the stability parameter ``s = (g/T)(∂θ/∂z)``; in the unstable/neutral layer
    buoyant rise levels off at the Briggs final-rise distance. With
    ``options.gradual_plume_rise`` the ``x^{2/3}`` transitional law applies
    until final rise is reached.
    """
    fb = buoyancy_flux(stack, state.temperature)
    fm = momentum_flux(stack, state.temperature)
    u = jnp.maximum(wind, options.min_wind_speed_m_s)
    x = jnp.maximum(downwind_distance, 0.0)

    has_buoyancy = jnp.where(fb > 0.0, 1.0, 0.0)
    has_momentum = jnp.where(fm > 0.0, 1.0, 0.0)

    # Unstable / neutral final rise (Briggs 1975).
    final_unstable = jnp.where(
        fb < 55.0,
        21.425 * _safe_pow(fb, 0.75) / u,
        38.71 * _safe_pow(fb, 0.6) / u,
    )
    # Stable final rise; s from the above-ML potential-temperature gradient.
    s = jnp.maximum(GRAVITY * state.dtheta_dz / jnp.maximum(state.temperature, _TINY), 1.0e-6)
    final_stable = 2.6 * _safe_pow(fb / (u * s), 1.0 / 3.0)
    is_stable = jnp.where(state.inv_obukhov_length > 0.0, 1.0, 0.0)
    final_buoyant = is_stable * final_stable + (1.0 - is_stable) * final_unstable

    if options.gradual_plume_rise:
        gradual = 1.6 * _safe_pow(fb, 1.0 / 3.0) * _safe_pow(x, 2.0 / 3.0) / u
        rise_buoyant = jnp.minimum(gradual, final_buoyant)
    else:
        rise_buoyant = final_buoyant
    rise_buoyant = has_buoyancy * rise_buoyant

    # Momentum rise: jet penetration, capped in stable air.
    mom_neutral = 3.0 * stack.diameter * stack.exit_velocity / u
    mom_stable = 1.5 * _safe_pow(fm / (u * jnp.sqrt(s)), 1.0 / 3.0)
    rise_momentum = has_momentum * (
        is_stable * jnp.minimum(mom_stable, mom_neutral) + (1.0 - is_stable) * mom_neutral
    )

    return jnp.maximum(rise_buoyant, rise_momentum)


# ── Plume spread ─────────────────────────────────────────────────────────────


def _taylor_growth(
    sigma_turbulence: jnp.ndarray, travel_time: jnp.ndarray, time_scale: jnp.ndarray
) -> jnp.ndarray:
    """Taylor (1921) dispersion: linear growth near-field, ``√t`` far-field."""
    return sigma_turbulence * travel_time / jnp.sqrt(1.0 + travel_time / (2.0 * time_scale))


def lateral_spread(
    state: MetState,
    height: jnp.ndarray,
    travel_time: jnp.ndarray,
    plume_rise_m: jnp.ndarray,
    options: DispersionOptions,
) -> jnp.ndarray:
    """``σ_y`` including buoyancy-induced dispersion (Δh/3.5, in quadrature)."""
    sv = sigma_v(state, height, options)
    time_scale = jnp.clip(state.mixing_height / jnp.maximum(sv, _TINY), 50.0, 5000.0)
    sigma = _taylor_growth(sv, travel_time, time_scale)
    buoyant = plume_rise_m / 3.5
    return jnp.sqrt(sigma**2 + buoyant**2) + _TINY


def vertical_spread(
    sigma_turbulence: jnp.ndarray,
    state: MetState,
    travel_time: jnp.ndarray,
    plume_rise_m: jnp.ndarray,
) -> jnp.ndarray:
    """``σ_z`` for a given vertical turbulence scale, plus buoyant spread."""
    time_scale = jnp.clip(
        0.5 * state.mixing_height / jnp.maximum(sigma_turbulence, _TINY), 10.0, 2000.0
    )
    sigma = _taylor_growth(sigma_turbulence, travel_time, time_scale)
    buoyant = plume_rise_m / 3.5
    return jnp.sqrt(sigma**2 + buoyant**2) + _TINY


# ── Vertical concentration structure ─────────────────────────────────────────


def _reflected_gaussian(
    receptor_z: jnp.ndarray,
    plume_z: jnp.ndarray,
    sigma_z: jnp.ndarray,
    mixing_height: jnp.ndarray,
    reflections: int,
) -> jnp.ndarray:
    """Vertical density [m⁻¹] with no-flux boundaries at 0 and ``zi``.

    The image series ``Σ_m [G(z - h - 2m zi) + G(z + h - 2m zi)]`` enforces both
    boundaries simultaneously and converges to the uniform value ``1/zi`` once
    ``σ_z ≳ zi``.
    """
    m = jnp.arange(-reflections, reflections + 1, dtype=float)
    offsets = 2.0 * m * mixing_height
    direct = receptor_z - plume_z - offsets
    image = receptor_z + plume_z - offsets
    terms = jnp.exp(-0.5 * (direct / sigma_z) ** 2) + jnp.exp(-0.5 * (image / sigma_z) ** 2)
    return jnp.sum(terms) / (_SQRT_2PI * sigma_z)


def bigaussian_parameters(
    sigma_w: jnp.ndarray, options: DispersionOptions
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Updraft/downdraft PDF parameters for the convective boundary layer.

    AERMOD represents the skewed CBL vertical-velocity distribution as two
    Gaussians (Weil et al. 1997). Writing the updraft mean as ``a > 0`` and the
    downdraft mean as ``-b < 0``, with the closure ``σ_wj = R |w̄j|`` and
    weights fixed by zero mean, the first three moments give

    ``λ1 = b/(a+b)``, ``λ2 = a/(a+b)``,
    ``a·b = σ_w²/(1+R²)``, and
    ``a - b = S σ_w³ / ((1+3R²) a b)``,

    where ``S`` is the vertical-velocity skewness. Solving the quadratic yields
    ``a`` and ``b`` in closed form.

    Returns ``(λ1, w̄1, λ2, w̄2)`` with ``w̄2`` negative.
    """
    r = options.cbl_sigma_ratio
    product = sigma_w**2 / (1.0 + r**2)
    diff = options.cbl_skewness * sigma_w**3 / ((1.0 + 3.0 * r**2) * jnp.maximum(product, _TINY))
    a = 0.5 * (diff + jnp.sqrt(diff**2 + 4.0 * product + _TINY))
    b = jnp.maximum(a - diff, _TINY)
    total = a + b
    return b / total, a, a / total, -b


def _convective_vertical(
    receptor_z: jnp.ndarray,
    plume_height: jnp.ndarray,
    sigma_wc: jnp.ndarray,
    sigma_z_mech: jnp.ndarray,
    travel_time: jnp.ndarray,
    plume_rise_m: jnp.ndarray,
    state: MetState,
    options: DispersionOptions,
    lid: jnp.ndarray,
) -> jnp.ndarray:
    """Bi-Gaussian CBL vertical density: updraft and downdraft sub-plumes.

    Each sub-plume is advected vertically at its own mean velocity
    (``h + w̄_j t``), spreads at its own rate, and is reflected at the ground and
    the inversion. Note the penetrated source (material injected into the stable
    layer above ``zi``) is not modelled — see the module docstring.
    """
    lam1, w1, lam2, w2 = bigaussian_parameters(sigma_wc, options)
    total = jnp.zeros_like(receptor_z)
    for weight, w_mean in ((lam1, w1), (lam2, w2)):
        sigma_j = jnp.sqrt(
            vertical_spread(options.cbl_sigma_ratio * jnp.abs(w_mean), state, travel_time, 0.0) ** 2
            + sigma_z_mech**2
            + (plume_rise_m / 3.5) ** 2
        ) + _TINY
        centre = plume_height + w_mean * travel_time
        total = total + weight * _reflected_gaussian(
            receptor_z, centre, sigma_j, lid, options.reflections
        )
    return total


# ── Main kernel ──────────────────────────────────────────────────────────────


def chi_over_q(
    receptor: ReceptorArray,
    source: SourceArray,
    stack: StackArray,
    state: MetState,
    options: DispersionOptions,
) -> jnp.ndarray:
    """Concentration per unit emission [s m⁻³] at one receptor from one source.

    Multiply by an emission rate in kg s⁻¹ to get kg m⁻³. Everything is a
    scalar-in/scalar-out JAX computation, so this composes with ``vmap`` (see
    :mod:`enforceflux.aermod.model`) and with ``grad`` in any argument.
    """
    # Rotate into plume coordinates. Meteorological wind direction is the
    # direction the wind blows *from*, so the downwind unit vector is
    # (-sin φ, -cos φ) and the crosswind axis is perpendicular to it.
    sin_d, cos_d = jnp.sin(state.wind_dir_rad), jnp.cos(state.wind_dir_rad)
    dx = receptor.x - source.x
    dy = receptor.y - source.y
    downwind = -dx * sin_d - dy * cos_d
    crosswind = -dx * cos_d + dy * sin_d

    stack_height = jnp.maximum(stack.height + source.base_elevation, 0.0)

    # Transport wind: evaluated at stack height for the rise calculation, then
    # re-evaluated at the final plume height (one fixed-point iteration).
    u_stack = jnp.maximum(
        wind_speed_at(state, jnp.maximum(stack_height, 2.0)), options.min_wind_speed_m_s
    )
    rise = plume_rise(stack, state, u_stack, downwind, options)
    plume_height = stack_height + rise
    u_eff = jnp.maximum(
        wind_speed_at(state, jnp.maximum(plume_height, 2.0)), options.min_wind_speed_m_s
    )

    travel_time = jnp.maximum(downwind, 0.0) / u_eff

    sigma_y = lateral_spread(state, plume_height, travel_time, rise, options)
    sigma_wc, sigma_wm = sigma_w_components(state, plume_height, options)
    sigma_z_mech = vertical_spread(sigma_wm, state, travel_time, rise)

    lateral = jnp.exp(-0.5 * (crosswind / sigma_y) ** 2) / (_SQRT_2PI * sigma_y)

    # A plume that rises past the inversion is in the stable air above it and is
    # no longer capped by the lid; reflecting it would spuriously mix it back
    # down to the ground. Lifting the lid out of range is the degenerate stand-in
    # for AERMOD's penetrated-source treatment.
    lid = jnp.where(plume_height < state.mixing_height, state.mixing_height, 1.0e5)

    stable_vertical = _reflected_gaussian(
        receptor.z,
        plume_height,
        jnp.sqrt(sigma_z_mech**2 + (sigma_wc * travel_time) ** 2) + _TINY,
        lid,
        options.reflections,
    )
    if options.convective_bigaussian:
        convective_vertical = _convective_vertical(
            receptor.z,
            plume_height,
            sigma_wc,
            sigma_z_mech,
            travel_time,
            rise,
            state,
            options,
            lid,
        )
        # Blend on the convective flag: w* > 0 only in an unstable boundary layer.
        is_convective = jnp.where(
            (state.inv_obukhov_length < 0.0) & (state.w_star > 0.0), 1.0, 0.0
        )
        vertical = is_convective * convective_vertical + (1.0 - is_convective) * stable_vertical
    else:
        vertical = stable_vertical

    chi = lateral * vertical / u_eff
    # No upwind transport in a steady-state plume model, and nothing above the
    # inversion that this formulation can represent.
    valid = jnp.where(downwind > 0.0, 1.0, 0.0)
    return valid * chi
