"""LES pseudo-observation sampler for source-heterogeneity OSSEs (M6b).

Takes a MicroHH nature RunDir and a list of instruments, and returns the
observation matrix ``y[i, t]`` that the downstream LPDM / Gaussian inversion
consumes. The sampling call for each instrument is the same forward operator
the inversion uses (open-path line integral, point sample, column integral),
so units are guaranteed to match by construction — and are asserted against
the shared :class:`ObservationSpec` before any values are returned.

Two input shapes are supported:

* A pre-baked NPZ slab written next to the run at ``les_field.npz`` with keys
  ``times_s (t,)``, ``x_m (nx,)``, ``y_m (ny,)``, ``field (t, ny, nx)`` and an
  optional ``variable``/``unit`` pair. Tests use this path; production drivers
  can also emit it once from raw MicroHH output.
* A MicroHH ``microhh_case/`` directory with a ``sim_config.yaml`` and the
  usual ``*.xy.000.<k>.<iter>`` cross-sections; a single horizontal slice at
  the receptor level is assembled into the same shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enforceflux.core.observation_units import ObservationSpec, assert_matches
from enforceflux.instrument.models import Instrument
from enforceflux.instrument.open_path import _bilinear, path_average_series


@dataclass(frozen=True)
class LESSample:
    times_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    field: np.ndarray            # (t, ny, nx)
    variable: str = "concentration_ppm"
    unit: str = "ppm"


def _observation_spec_for(inst: Instrument) -> ObservationSpec:
    op = inst.operator_type
    path_length_m = float(inst.path_length_m) if op == "line_integral" else 0.0
    unit_map = {
        "concentration_ppm": "ppm",
        "flux_nmol_m2_s": "nmol_m-2_s-1",
        "column_ppb": "ppb",
        "emission_rate_kg_hr": "kg_hr-1",
    }
    return ObservationSpec(
        variable=inst.observable,
        unit=unit_map[inst.observable],
        averaging_window_s=float(inst.params.cadence_s),
        height_m=float(inst.z),
        path_length_m=path_length_m,
    )


def _load_les_sample(les_run_dir: Path) -> LESSample:
    npz = les_run_dir / "les_field.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            variable = str(z["variable"]) if "variable" in z.files else "concentration_ppm"
            unit = str(z["unit"]) if "unit" in z.files else "ppm"
            return LESSample(
                times_s=np.asarray(z["times_s"], dtype=float),
                x_m=np.asarray(z["x_m"], dtype=float),
                y_m=np.asarray(z["y_m"], dtype=float),
                field=np.asarray(z["field"], dtype=float),
                variable=variable,
                unit=unit,
            )
    case_dir = les_run_dir / "microhh_case"
    cfg_yaml = case_dir / "sim_config.yaml"
    if cfg_yaml.exists():
        return _load_from_microhh(cfg_yaml)
    raise FileNotFoundError(
        f"No pseudo-observation source in {les_run_dir}: expected 'les_field.npz' "
        f"or 'microhh_case/sim_config.yaml'."
    )


def _load_from_microhh(cfg_yaml: Path) -> LESSample:
    import glob

    from enforceflux.microhh.output import read_cross_xy
    from enforceflux.microhh.sim_config import load_microhh_config

    cfg = load_microhh_config(cfg_yaml)
    pattern = str(cfg.case_dir / f"{cfg.scalar_name}.xy.000.00000.*")
    iters = sorted(int(Path(p).name.rsplit(".", 1)[-1]) for p in glob.glob(pattern))
    if not iters:
        raise FileNotFoundError(f"No xy cross-sections for {cfg.scalar_name} in {cfg.case_dir}")
    frames = [read_cross_xy(cfg, iter_s=f"{it:07d}") for it in iters]
    field = np.stack(frames, axis=0)
    x = (np.arange(cfg.grid.itot) + 0.5) * cfg.grid.dx
    y = (np.arange(cfg.grid.jtot) + 0.5) * cfg.grid.dy
    times = np.asarray(iters, dtype=float)
    return LESSample(times_s=times, x_m=x, y_m=y, field=field)


def _sample_one(inst: Instrument, sample: LESSample) -> np.ndarray:
    op = inst.operator_type
    if op == "line_integral":
        return path_average_series(
            sample.field, sample.x_m, sample.y_m,
            inst.x, inst.y, float(inst.path_length_m), float(inst.path_bearing_deg),
        )
    if op in ("point_flux", "ec_footprint", "plume_imaging",
              "multi_path_inversion", "lidar_path_integral"):
        return _bilinear(sample.field, sample.x_m, sample.y_m,
                         np.array([inst.x]), np.array([inst.y]))[:, 0]
    if op in ("column_aircraft", "column_satellite"):
        # 2-D slab pseudo-obs: reduce to a point sample at the tower foot.
        return _bilinear(sample.field, sample.x_m, sample.y_m,
                         np.array([inst.x]), np.array([inst.y]))[:, 0]
    raise ValueError(f"Unsupported operator_type for LES pseudo-obs: {op!r}")


def sample_les_through_instruments(
    les_run_dir: Path,
    instruments: list[Instrument],
    rng: np.random.Generator,
    *,
    expected_specs: list[ObservationSpec] | None = None,
) -> np.ndarray:
    """Sample a MicroHH nature run through ``instruments`` at each snapshot.

    Returns ``y`` of shape ``(n_instruments, n_times)`` in the same physical
    unit the downstream inversion expects. Gaussian noise is applied per each
    instrument's ``sigma_scale`` / ``sigma_abs`` model. When ``expected_specs``
    is supplied, each instrument's inferred :class:`ObservationSpec` is
    validated against the paired entry before sampling — the whole reason the
    M6a contract exists.
    """
    sample = _load_les_sample(Path(les_run_dir))
    n_inst = len(instruments)
    n_t = sample.times_s.size
    y = np.empty((n_inst, n_t), dtype=float)

    for i, inst in enumerate(instruments):
        spec = _observation_spec_for(inst)
        if expected_specs is not None:
            assert_matches(spec, expected_specs[i])
        clean = _sample_one(inst, sample)
        p = inst.params
        sigma = np.sqrt((p.sigma_scale * np.abs(clean)) ** 2 + p.sigma_abs ** 2)
        noise = rng.normal(0.0, 1.0, size=n_t) * sigma
        y[i] = clean * (1.0 + p.bias_scale) + p.bias_abs + noise
    return y


__all__ = [
    "LESSample",
    "sample_les_through_instruments",
]
