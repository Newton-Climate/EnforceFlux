"""FLEXPART-backed EC observation operators for OSSE and inversion workflows."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from enforceflux.flexpart.backward import FlexpartBackwardRunner
from enforceflux.flexpart.sim_config import SimulationConfig
from enforceflux.instrument import Instrument
from enforceflux.models.config import DomainConfig
from enforceflux.models.source import Source

CH4_NMOL_PER_KG = 1e9 / 16.04e-3


def _valid_backward_sync_seconds(output_step_s: int, sync_s: int) -> int:
    """Return a FLEXPART-compatible sync interval for backward runs."""
    if output_step_s <= 0:
        raise ValueError("output_step_s must be positive for backward EC runs.")
    if sync_s <= 0:
        raise ValueError("n_sync_s must be positive for backward EC runs.")
    return max(1, min(int(sync_s), int(output_step_s) // 2))


@dataclass(frozen=True)
class ECObservationOperatorResult:
    """Time-dependent EC observation operator in instrument-by-source space."""

    times_s: np.ndarray
    g: np.ndarray
    valid_mask: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def build_ec_observation_operator_from_flexpart(
    g_raw_series: np.ndarray,
    source_areas_m2: np.ndarray,
    times_s: np.ndarray,
) -> ECObservationOperatorResult:
    """Convert sampled FLEXPART backward footprints into EC flux operators.

    Parameters
    ----------
    g_raw_series : (t, m, n) or (m, n) array
        Sampled backward footprint amplitudes at the source support for each
        time and EC tower. Rows are converted into footprint-weighted surface
        flux operators.
    source_areas_m2 : (n,) array
        Horizontal area of each source element. The state is assumed to be
        emission rate per source element in ``kg s-1``.
    times_s : (t,) array
        Simulation timestamps in seconds.

    Returns
    -------
    ECObservationOperatorResult
        ``g[t, i, j]`` has units ``nmol m-2 s-1 / (kg s-1)``.
        Invalid rows are filled with ``NaN`` and marked in ``valid_mask``.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times_s must be a non-empty 1-D array.")

    g_raw = np.asarray(g_raw_series, dtype=float)
    if g_raw.ndim == 2:
        g_raw = np.broadcast_to(g_raw[None, :, :], (len(times),) + g_raw.shape)
    elif g_raw.ndim != 3:
        raise ValueError("g_raw_series must be a 2-D or 3-D array.")
    if g_raw.shape[0] != len(times):
        raise ValueError("g_raw_series and times_s must have the same number of timesteps.")

    areas = np.asarray(source_areas_m2, dtype=float)
    if areas.ndim != 1 or areas.shape[0] != g_raw.shape[2]:
        raise ValueError("source_areas_m2 must be a 1-D array matching the source dimension.")
    if np.any(areas <= 0.0):
        raise ValueError("source_areas_m2 must contain strictly positive values.")

    g_ec = np.full_like(g_raw, np.nan, dtype=float)
    valid = np.zeros(g_raw.shape[:2], dtype=bool)

    for t_idx in range(g_raw.shape[0]):
        for i_idx in range(g_raw.shape[1]):
            row = np.clip(g_raw[t_idx, i_idx], 0.0, None)
            overlap = row * areas
            total_overlap = float(overlap.sum())
            if not np.isfinite(total_overlap) or total_overlap <= 0.0:
                continue
            g_ec[t_idx, i_idx] = (row / total_overlap) * CH4_NMOL_PER_KG
            valid[t_idx, i_idx] = np.all(np.isfinite(g_ec[t_idx, i_idx]))

    return ECObservationOperatorResult(
        times_s=times.copy(),
        g=g_ec,
        valid_mask=valid,
        meta={"units": "nmol m-2 s-1 / (kg s-1)"},
    )


def build_ec_observation_operator_from_backward_runs(
    *,
    base_config: SimulationConfig,
    domain: DomainConfig,
    instruments: Iterable[Instrument],
    sources: Iterable[Source],
    source_areas_m2: np.ndarray,
    sample_times_s: np.ndarray,
    lookback_s: float,
    runner_config: dict[str, Any] | None = None,
) -> ECObservationOperatorResult:
    """Run time-dependent FLEXPART backward footprints and convert them to EC rows."""
    instruments = list(instruments)
    sources = list(sources)
    times = np.asarray(sample_times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("sample_times_s must be a non-empty 1-D array.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("sample_times_s must be monotonically non-decreasing.")
    if lookback_s <= 0.0:
        raise ValueError("lookback_s must be positive.")

    base_run_dir = Path((runner_config or {}).get("base_run_dir", base_config.run_dir / "ec_backward"))
    g_raw_series = np.zeros((len(times), len(instruments), len(sources)), dtype=float)
    meta_runs: list[dict[str, Any]] = []

    for t_idx, sample_s in enumerate(times):
        sample_end = base_config.start + timedelta(seconds=float(sample_s))
        sample_start = sample_end - timedelta(seconds=float(lookback_s))
        window_run_dir = base_run_dir / f"ec_t{t_idx:04d}"
        window_cfg = dataclasses.replace(
            base_config,
            start=sample_start,
            end=sample_end,
            run_dir=window_run_dir,
            output_path=window_run_dir / "output" / "_unused.nc",
            n_sync_s=_valid_backward_sync_seconds(base_config.output_step_s, base_config.n_sync_s),
            sources=[],
        )
        window_runner_config = dict(runner_config or {})
        window_runner_config["base_run_dir"] = str(window_run_dir)
        runner = FlexpartBackwardRunner(
            base_config=window_cfg,
            domain=domain,
            config=window_runner_config,
        )
        result = runner.run(instruments, sources)
        g_raw_series[t_idx] = np.asarray(result.g, dtype=float)
        meta_runs.append({"sample_time_s": float(sample_s), **dict(result.meta)})

    result = build_ec_observation_operator_from_flexpart(
        g_raw_series=g_raw_series,
        source_areas_m2=source_areas_m2,
        times_s=times,
    )
    meta = dict(result.meta)
    meta.update(
        {
            "lookback_s": float(lookback_s),
            "raw_units": "s",
            "runs": meta_runs,
        }
    )
    return ECObservationOperatorResult(
        times_s=result.times_s,
        g=result.g,
        valid_mask=result.valid_mask,
        meta=meta,
    )
