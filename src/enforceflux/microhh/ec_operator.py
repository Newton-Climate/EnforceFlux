"""LES-resolved eddy-covariance observation operators.

An EC analyser does not observe concentration or a backward residence-time
proxy.  Its resolved kinematic scalar flux is the averaging-window covariance

    <w' q'>,

where ``w`` is vertical velocity and ``q`` is methane mass mixing ratio.  For
unit-emission passive-scalar LES runs this covariance is linear in source
strength and therefore supplies one Jacobian column per source.

The routines here operate on arrays so they can be tested without MicroHH.  A
small reader converts MicroHH column NetCDF files into those arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from enforceflux.microhh.output import _column_index, _proj, find_column_file
from enforceflux.microhh.sim_config import MicroHHConfig


CH4_NMOL_PER_KG = 1e9 / 16.04e-3


@dataclass(frozen=True)
class LESECObservationOperatorResult:
    """Windowed EC Jacobian derived from resolved LES covariance.

    ``g[window, tower, source]`` maps source emission rate in kg s-1 to
    methane flux in nmol m-2 s-1. ``random_error`` is the standard error among
    sub-window covariance estimates in the same units.
    """

    times_s: np.ndarray
    g: np.ndarray
    valid_mask: np.ndarray
    random_error: np.ndarray
    n_samples: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LESColumnSeries:
    """Co-located tower variables interpolated to measurement height."""

    receptor_ids: tuple[str, ...]
    times_s: np.ndarray
    w_m_s: np.ndarray
    scalar_kg_kg: np.ndarray
    ustar_m_s: np.ndarray
    obukhov_m: np.ndarray


def _as_tower_series(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be (time,) or (time, tower), got {arr.shape}.")
    return arr


def _detrend(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Remove a linear trend independently from every column."""
    x = np.asarray(times_s, dtype=float)
    x = x - x.mean()
    denom = float(x @ x)
    centred = values - values.mean(axis=0, keepdims=True)
    if denom <= 0.0:
        return centred
    slopes = (x[:, None] * centred).sum(axis=0) / denom
    return centred - x[:, None] * slopes[None, :]


def _block_covariance(
    w: np.ndarray,
    q: np.ndarray,
    times_s: np.ndarray,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-window covariance and block-based standard error."""
    wp = _detrend(w[:, None], times_s)[:, 0]
    qp = _detrend(q, times_s)
    covariance = np.mean(wp[:, None] * qp, axis=0)

    blocks = [idx for idx in np.array_split(np.arange(len(times_s)), n_blocks) if len(idx) >= 2]
    if len(blocks) < 2:
        return covariance, np.full(q.shape[1], np.nan)
    block_cov = []
    for idx in blocks:
        wb = _detrend(w[idx, None], times_s[idx])[:, 0]
        qb = _detrend(q[idx], times_s[idx])
        block_cov.append(np.mean(wb[:, None] * qb, axis=0))
    block_cov = np.asarray(block_cov)
    standard_error = np.std(block_cov, axis=0, ddof=1) / np.sqrt(len(block_cov))
    return covariance, standard_error


def build_ec_observation_operator_from_les(
    *,
    w_m_s: np.ndarray,
    scalar_responses_kg_kg: np.ndarray,
    times_s: np.ndarray,
    source_emission_rates_kg_s: np.ndarray,
    air_density_kg_m3: float | np.ndarray = 1.2,
    window_s: float = 1800.0,
    min_samples: int = 120,
    n_error_blocks: int = 6,
    max_missing_fraction: float = 0.1,
    min_ustar_m_s: float | None = None,
    ustar_m_s: np.ndarray | None = None,
    subgrid_flux_response_kg_m2_s: np.ndarray | None = None,
) -> LESECObservationOperatorResult:
    """Build an EC Jacobian from co-located LES ``w`` and methane time series.

    Parameters
    ----------
    w_m_s
        Resolved vertical velocity, shape ``(time, tower)``.
    scalar_responses_kg_kg
        Methane mass-mixing-ratio responses from one LES run per source, shape
        ``(time, tower, source)``. Each run may use any non-zero emission rate;
        ``source_emission_rates_kg_s`` scales it to a unit response.
    air_density_kg_m3
        Scalar, ``(time,)``, or ``(time, tower)`` density. Density is applied
        before covariance so anelastic density variation is retained.
    subgrid_flux_response_kg_m2_s
        Optional SGS vertical methane-flux response with the same shape as the
        scalar responses. It is window-averaged and added to resolved flux.

    Notes
    -----
    Windows are non-overlapping. Samples exactly on a shared boundary belong
    to the later window, preventing double counting. Random uncertainty is
    estimated from covariance variation among sub-windows; it is not treated
    as independent instrument white noise.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be a strictly increasing 1-D array with at least 2 samples.")
    if window_s <= 0.0:
        raise ValueError("window_s must be positive.")

    w = _as_tower_series(w_m_s, "w_m_s")
    q = np.asarray(scalar_responses_kg_kg, dtype=float)
    if q.ndim == 2:
        q = q[:, None, :]
    if q.ndim != 3 or q.shape[:2] != w.shape or q.shape[0] != len(times):
        raise ValueError(
            "scalar_responses_kg_kg must be (time, tower, source) and align with w_m_s."
        )

    rates = np.asarray(source_emission_rates_kg_s, dtype=float)
    if rates.shape != (q.shape[2],) or np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("source_emission_rates_kg_s must be positive and match source count.")

    density = np.asarray(air_density_kg_m3, dtype=float)
    if density.ndim == 0:
        density = np.full(w.shape, float(density))
    elif density.ndim == 1:
        density = np.broadcast_to(density[:, None], w.shape)
    else:
        density = np.broadcast_to(density, w.shape)
    if np.any(density <= 0.0):
        raise ValueError("air_density_kg_m3 must be positive.")

    ustar = None if ustar_m_s is None else _as_tower_series(ustar_m_s, "ustar_m_s")
    if ustar is not None and ustar.shape != w.shape:
        raise ValueError("ustar_m_s must align with w_m_s.")

    sgs = None
    if subgrid_flux_response_kg_m2_s is not None:
        sgs = np.asarray(subgrid_flux_response_kg_m2_s, dtype=float)
        if sgs.shape != q.shape:
            raise ValueError("subgrid_flux_response_kg_m2_s must match scalar response shape.")

    first_end = times[0] + window_s
    end_times = np.arange(first_end, times[-1] + 1e-9, window_s)
    n_window, n_tower, n_source = len(end_times), w.shape[1], q.shape[2]
    g = np.full((n_window, n_tower, n_source), np.nan)
    error = np.full_like(g, np.nan)
    valid = np.zeros((n_window, n_tower), dtype=bool)
    counts = np.zeros((n_window, n_tower), dtype=int)

    for k, end in enumerate(end_times):
        start = end - window_s
        window = (times >= start) & (times < end)
        for tower in range(n_tower):
            finite = window & np.isfinite(w[:, tower]) & np.isfinite(density[:, tower])
            finite &= np.all(np.isfinite(q[:, tower, :]), axis=1)
            expected = int(window.sum())
            counts[k, tower] = int(finite.sum())
            if expected == 0 or counts[k, tower] < min_samples:
                continue
            if 1.0 - counts[k, tower] / expected > max_missing_fraction:
                continue
            if min_ustar_m_s is not None:
                if ustar is None:
                    raise ValueError("min_ustar_m_s requires ustar_m_s.")
                if float(np.nanmean(ustar[finite, tower])) < min_ustar_m_s:
                    continue

            idx = np.flatnonzero(finite)
            # rho*q is methane mass concentration [kg CH4 m-3]. Covariance
            # with w gives resolved vertical mass flux [kg CH4 m-2 s-1].
            mass_conc = density[idx, tower, None] * q[idx, tower, :]
            cov, cov_se = _block_covariance(w[idx, tower], mass_conc, times[idx], n_error_blocks)
            flux = cov
            flux_se = cov_se
            if sgs is not None:
                flux = flux + np.mean(sgs[idx, tower, :], axis=0)

            scale = CH4_NMOL_PER_KG / rates
            g[k, tower] = flux * scale
            error[k, tower] = np.abs(flux_se * scale)
            valid[k, tower] = np.all(np.isfinite(g[k, tower]))

    return LESECObservationOperatorResult(
        times_s=end_times,
        g=g,
        valid_mask=valid,
        random_error=error,
        n_samples=counts,
        meta={
            "method": "LES resolved covariance",
            "window_s": float(window_s),
            "units": "nmol m-2 s-1 / (kg s-1)",
            "includes_subgrid_flux": sgs is not None,
            "detrending": "linear",
            "error_method": "sub-window covariance standard error",
        },
    )


def _interp_vertical(values: np.ndarray, levels_m: np.ndarray, height_m: float) -> np.ndarray:
    """Linearly interpolate ``(time, level)`` data to one height."""
    levels = np.asarray(levels_m, dtype=float)
    if height_m < levels[0] or height_m > levels[-1]:
        raise ValueError(
            f"measurement_height_m={height_m:g} lies outside model levels "
            f"[{levels[0]:g}, {levels[-1]:g}] m."
        )
    hi = int(np.searchsorted(levels, height_m, side="right"))
    if hi == 0:
        return values[:, 0]
    if hi >= len(levels):
        return values[:, -1]
    lo = hi - 1
    fraction = (height_m - levels[lo]) / (levels[hi] - levels[lo])
    return values[:, lo] * (1.0 - fraction) + values[:, hi] * fraction


def read_microhh_ec_columns(
    cfg: MicroHHConfig,
    *,
    measurement_height_m: float,
    scalar_name: str | None = None,
) -> LESColumnSeries:
    """Read ``w`` and methane from MicroHH columns at a common physical height."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Reading MicroHH EC columns requires xarray.") from exc

    scalar_name = scalar_name or cfg.scalar_name
    projection = _proj(cfg)
    ids: list[str] = []
    times_ref: np.ndarray | None = None
    w_columns: list[np.ndarray] = []
    q_columns: list[np.ndarray] = []
    ustar_columns: list[np.ndarray] = []
    obuk_columns: list[np.ndarray] = []

    for receptor in cfg.receptors:
        x, y = projection.to_box(receptor.lon, receptor.lat)
        ix, iy = _column_index(x, y, cfg)
        path = find_column_file(cfg, ix, iy)
        if path is None:
            raise FileNotFoundError(
                f"No MicroHH column for {receptor.id!r} at ({ix:05d},{iy:05d})."
            )
        with xr.open_dataset(path, decode_times=False) as ds:
            times = np.asarray(ds["time"].values, dtype=float)
            if times_ref is None:
                times_ref = times
            elif not np.array_equal(times_ref, times):
                raise ValueError("MicroHH EC column times do not align across receptors.")
            w_columns.append(
                _interp_vertical(np.asarray(ds["w"].values), np.asarray(ds["zh"].values), measurement_height_m)
            )
            q_columns.append(
                _interp_vertical(
                    np.asarray(ds[scalar_name].values), np.asarray(ds["z"].values), measurement_height_m
                )
            )
            ustar_columns.append(np.asarray(ds["ustar"].values, dtype=float))
            obuk_columns.append(np.asarray(ds["obuk"].values, dtype=float))
        ids.append(receptor.id)

    return LESColumnSeries(
        receptor_ids=tuple(ids),
        times_s=times_ref if times_ref is not None else np.empty(0),
        w_m_s=np.stack(w_columns, axis=1),
        scalar_kg_kg=np.stack(q_columns, axis=1),
        ustar_m_s=np.stack(ustar_columns, axis=1),
        obukhov_m=np.stack(obuk_columns, axis=1),
    )


def build_ec_observation_operator_from_microhh_runs(
    configs: Sequence[MicroHHConfig],
    *,
    measurement_height_m: float,
    source_emission_rates_kg_s: np.ndarray,
    **kwargs: Any,
) -> LESECObservationOperatorResult:
    """Build an EC operator from one source-specific MicroHH run per source."""
    if not configs:
        raise ValueError("configs must contain at least one source-specific LES run.")
    columns = [
        read_microhh_ec_columns(cfg, measurement_height_m=measurement_height_m)
        for cfg in configs
    ]
    reference = columns[0]
    for item in columns[1:]:
        if item.receptor_ids != reference.receptor_ids or not np.array_equal(item.times_s, reference.times_s):
            raise ValueError("Source-specific MicroHH runs must share receptors and sample times.")
        if not np.allclose(item.w_m_s, reference.w_m_s, rtol=1e-5, atol=1e-7):
            raise ValueError(
                "Source-specific runs do not share the same turbulent flow realization; "
                "use identical forcing, initialization, and output cadence."
            )
    scalar = np.stack([item.scalar_kg_kg for item in columns], axis=2)
    return build_ec_observation_operator_from_les(
        w_m_s=reference.w_m_s,
        scalar_responses_kg_kg=scalar,
        times_s=reference.times_s,
        source_emission_rates_kg_s=source_emission_rates_kg_s,
        ustar_m_s=reference.ustar_m_s,
        **kwargs,
    )
