#!/usr/bin/env python3
"""Paired observability pilot using the bLSmodelR transport operator.

This is the sibling of :mod:`run_paired_observability_experiment` for the
bLSmodelR inversion path. It exercises the full new pipeline end-to-end
without going through the ``enforceflux`` CLI dispatch, so it does not
require the framework's ``transport/runner.py`` to grow a
``model == "blsmodelr"`` branch first (that wiring is the natural next
step to promote this from a pilot to the swept experiment).

Pipeline
--------
1. Load a MicroHH nature run (or synthesise sonic data with ``--synthetic``).
2. Sample the sonic instrument operator at each receptor
   (``enforceflux.instrument.sonic.sonic_from_microhh``) — one fixed height
   per sensor, mirroring the real observational schema.
3. Reduce each sonic to a :class:`BlsInterval` with the shared EC processor.
4. Build a source-grid Jacobian by driving bLSmodelR through
   :class:`BlsTransportOperator`.
5. Convolve with the truth field, add noise, invert with
   :func:`optimal_linear.solve_linear`, report ``E_Q``.

Defaults to ``--dry-run`` so it runs without R installed; when R + bLSmodelR
are available, pass ``--no-dry-run`` for a real inversion.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from enforceflux.blsmodelr import BlsInterval, BlsRunner
from enforceflux.blsmodelr.footprint import build_source_grid, jacobian_from_bls_result
from enforceflux.blsmodelr.met_from_sonic import interval_from_sonic
from enforceflux.instrument.sonic import SonicObservation
from enforceflux.inversion.optimal_linear import oe_from_linear
from enforceflux.plugins.transport_blsmodelr import BlsTransportOperator

ROOT = Path(__file__).resolve().parents[2]


# ── Sonic acquisition ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class SensorSpec:
    id: str
    x: float
    y: float
    z: float


def _synthetic_sonic(sensor: SensorSpec, *, interval_id: str, n: int = 1200,
                     dt: float = 0.1, seed: int = 0) -> SonicObservation:
    """Fabricate a plausible convective-BL sonic timeseries for pilot runs."""
    rng = np.random.default_rng(seed)
    u_bar, v_bar = 3.0, 0.0
    theta_bar = 300.0
    w = rng.normal(0.0, 0.6, size=n)
    u = u_bar + rng.normal(0.0, 0.9, size=n) - 0.3 * w
    v = v_bar + rng.normal(0.0, 0.7, size=n)
    theta = theta_bar + rng.normal(0.0, 0.05, size=n) + 0.08 * w
    return SonicObservation(
        instrument_id=sensor.id, interval_id=interval_id,
        x=sensor.x, y=sensor.y, z=sensor.z,
        times_s=np.arange(n) * dt,
        u=u, v=v, w=w, theta=theta, z0=0.05,
        meta={"source": "synthetic_pilot", "seed": seed},
    )


def sonic_from_nature_run(
    nature_run_dir: Path, sensor: SensorSpec, *, interval_id: str, z_ref: float, z0: float,
) -> SonicObservation:
    """Sample the sonic operator on an existing MicroHH nature run."""
    from enforceflux.instrument.sonic import sonic_from_microhh
    from enforceflux.microhh.sim_config import load_microhh_config
    cfg = load_microhh_config(nature_run_dir / "dispersion/concentration_microhh/microhh_generated.yaml")
    # Match receptor by nearest x/y in the box frame.
    from enforceflux.microhh.geometry import BoxProjection
    proj = BoxProjection(origin_lon=cfg.origin_lon, origin_lat=cfg.origin_lat,
                         x_bearing_deg=cfg.x_bearing_deg,
                         source_x0=cfg.source_x0, source_y0=cfg.source_y0)
    lon, lat = proj.to_lonlat(sensor.x, sensor.y)
    receptor = min(cfg.receptors, key=lambda r: (r.lon - lon) ** 2 + (r.lat - lat) ** 2)
    return sonic_from_microhh(
        cfg, receptor_id=receptor.id, z_ref=z_ref, z0=z0,
        instrument_id=sensor.id, interval_id=interval_id,
    )


# ── Truth field ───────────────────────────────────────────────────────────


def load_truth_field(nature_run_dir: Path, source_grid_bounds) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_true_kg_s_per_cell, cell_area_m2) on the bLS source grid.

    Reads the LES truth field and bins onto the coarser bLS source grid via
    nearest-cell assignment (adequate for a pilot; a proper areal average
    should replace this before publication).
    """
    from netCDF4 import Dataset
    with Dataset(nature_run_dir / "dispersion/truth_field.nc") as ds:
        F = np.asarray(ds["F_true"][:], float)               # (ny, nx) kg m-2 s-1
        area = np.asarray(ds["cell_area_m2"][:], float)      # (ny, nx) m2
        x_les = np.asarray(ds["x_m"][:], float)              # (nx,)
        y_les = np.asarray(ds["y_m"][:], float)              # (ny,)
    x0, x1 = source_grid_bounds["x_bounds"]
    y0, y1 = source_grid_bounds["y_bounds"]
    nx, ny = source_grid_bounds["nx"], source_grid_bounds["ny"]
    xc = np.linspace(x0, x1, nx + 1); xc = 0.5 * (xc[:-1] + xc[1:])
    yc = np.linspace(y0, y1, ny + 1); yc = 0.5 * (yc[:-1] + yc[1:])
    x_true = np.zeros(nx * ny, dtype=float)
    cell_area = np.zeros(nx * ny, dtype=float)
    for i, xi in enumerate(xc):
        for j, yj in enumerate(yc):
            ix = int(np.argmin(np.abs(x_les - xi)))
            iy = int(np.argmin(np.abs(y_les - yj)))
            k = i * ny + j
            x_true[k] = F[iy, ix] * area[iy, ix]
            cell_area[k] = area[iy, ix]
    return x_true, cell_area


def synthetic_truth(source_grid_bounds, *, Q_true_kg_s: float, seed: int) -> np.ndarray:
    """A random-walk positive field renormalised to Q_true — pilot only."""
    n = source_grid_bounds["nx"] * source_grid_bounds["ny"]
    rng = np.random.default_rng(seed)
    f = np.exp(rng.normal(0.0, 0.8, size=n))
    return Q_true_kg_s * f / f.sum()


# ── Pipeline ──────────────────────────────────────────────────────────────


def run_pipeline(
    *,
    sensors: list[SensorSpec],
    source_grid: dict,
    nature_run_dir: Path | None,
    dry_run: bool,
    noise_frac: float = 0.05,
    prior_sigma_frac: float = 1.0,
    seed: int = 0,
) -> dict:
    """Return {'Q_true','Q_hat','E_Q','n_sensors','n_sources','g_rank'}."""
    # 1–3: one representative sonic → one interval. All sensors sample under
    # the same turbulence field (mirrors reality: at a single instant the
    # whole domain sees one meteorology). For multi-window experiments,
    # loop the pipeline over successive averaging windows.
    ref_sensor = sensors[len(sensors) // 2]
    if nature_run_dir is not None:
        obs = sonic_from_nature_run(nature_run_dir, ref_sensor, interval_id="t0000",
                                    z_ref=ref_sensor.z, z0=0.05)
    else:
        obs = _synthetic_sonic(ref_sensor, interval_id="t0000", seed=seed)
    intervals: list[BlsInterval] = [interval_from_sonic(obs)]

    # 4: bLS Jacobian.
    op = BlsTransportOperator()
    result = op.build_forward_operator(
        [], sensors, None,   # sources/domain unused; source_grid drives it
        {
            "source_grid": source_grid,
            "intervals": [_interval_to_dict(iv) for iv in intervals],
            "interval_reduce": "mean",
            "wrapper": {"dry_run": dry_run},
            "sensor_order": [s.id for s in sensors],
            # Big particle count so each of nx*ny cells gets solid statistics
            # under the live path; ignored by --dry-run.
            "model_params": {"n_particles": 50_000, "max_traj_s": 500},
        },
    )
    g = np.asarray(result.g, float)
    n_inst, n_src = g.shape

    # 5: truth field.
    if nature_run_dir is not None:
        x_true, cell_area = load_truth_field(nature_run_dir, source_grid)
    else:
        x_true = synthetic_truth(source_grid, Q_true_kg_s=2.7778e-2, seed=seed + 999)
        cell_area = np.asarray(result.meta["cell_area_m2"], float)

    # 6: forward + noise.
    y_clean = g @ x_true
    rng = np.random.default_rng(seed + 42)
    noise_std = max(noise_frac * float(np.mean(np.abs(y_clean))), 1e-12)
    y_obs = y_clean + rng.normal(0.0, noise_std, size=y_clean.size)

    # 7: Bayesian inversion (diagonal B, diagonal R).
    x_a = np.zeros_like(x_true)
    B = np.diag((prior_sigma_frac * (np.mean(x_true) if x_true.mean() > 0 else 1e-6)) ** 2
                * np.ones_like(x_true))
    R = np.diag((noise_std ** 2) * np.ones_like(y_obs))
    sol = oe_from_linear(G=g, y=y_obs, x_prior=x_a, Sa=B, Se=R)
    x_hat = np.asarray(sol.x_posterior, float)

    q_true = float(x_true.sum())
    q_hat = float(x_hat.sum())
    e_q = abs(q_hat - q_true) / max(q_true, 1e-12)
    return {
        "Q_true_kg_s": q_true, "Q_hat_kg_s": q_hat, "E_Q": e_q,
        "n_sensors": n_inst, "n_sources": n_src,
        "g_rank": int(np.linalg.matrix_rank(g)),
        "workdir": result.meta.get("workdir"),
    }


def _interval_to_dict(iv: BlsInterval) -> dict:
    d = {"id": iv.id, "u_star": iv.u_star, "L": iv.L, "z0": iv.z0,
         "wind_dir_deg": iv.wind_dir_deg, "wind_speed": iv.wind_speed, "z_ref": iv.z_ref}
    for k in ("sd_u", "sd_v", "sd_w"):
        v = getattr(iv, k)
        if v is not None:
            d[k] = v
    return d


# ── CLI ───────────────────────────────────────────────────────────────────


def _default_sensors(n: int) -> list[SensorSpec]:
    """N sensors placed 50 m downwind of a compact source patch — this puts
    every sensor inside the plume for a west-wind interval, which the pilot
    needs to see nonzero touch-down statistics. Real experiments must add
    additional wind directions (rotating the intervals) to give sensors on
    the plume-edge a chance to sample.
    """
    ys = np.linspace(-30.0, 30.0, n)
    return [SensorSpec(id=f"S{k:02d}", x=50.0, y=float(y), z=2.0) for k, y in enumerate(ys)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nature-run", type=Path, default=None,
                    help="Directory of an existing MicroHH run (with dispersion/…). "
                         "Omit for synthetic sonic mode.")
    ap.add_argument("--n-sensors", type=int, default=6)
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--x-bounds", type=float, nargs=2, default=(-500.0, 500.0))
    ap.add_argument("--y-bounds", type=float, nargs=2, default=(-500.0, 500.0))
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                    help="Use the bLSmodelR shim's analytic stub (default).")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                    help="Call the real bLSmodelR (requires R + bLSmodelR installed).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-csv", type=Path,
                    default=ROOT / "notebooks/hetero_experiments/paired_observability_blsmodelr.csv")
    args = ap.parse_args()

    if not args.dry_run and shutil.which("Rscript") is None:
        raise SystemExit("Rscript not on PATH; either install R or use --dry-run.")

    sensors = _default_sensors(args.n_sensors)
    source_grid = {
        "x_bounds": tuple(args.x_bounds), "y_bounds": tuple(args.y_bounds),
        "nx": args.nx, "ny": args.ny,
    }

    result = run_pipeline(
        sensors=sensors, source_grid=source_grid,
        nature_run_dir=args.nature_run, dry_run=args.dry_run, seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    with args.output_csv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result.keys()), lineterminator="\n")
        if write_header:
            w.writeheader()
        w.writerow(result)
    print(f"appended row to {args.output_csv}")


if __name__ == "__main__":
    main()
