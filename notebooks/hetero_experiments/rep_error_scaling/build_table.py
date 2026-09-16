#!/usr/bin/env python3
"""Assemble the 576 source-heterogeneity inversions into one tidy table.

Reads only what the run directories themselves record. Nothing is inferred from
a run name: L, CV and the seed come from the nature run's ``truth_field.nc``
attributes, the sensor count and geometry from the instrument stage's config
snapshot, and every retrieval quantity from the flux stage's ``summary.json``
and ``matrices.npz``. The run name is carried along only as provenance, and is
cross-checked against the authoritative values at the end.

Writes ``inversions.csv`` and ``integrity.md``. Fails loudly on any gate.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "runs"

# The seeded rice-paddy sweep. The unseeded base runs (no _s<d>_) are excluded:
# SWEEPS.md records that they carry 46 cross-section frames rather than 45 and
# must not be pooled with the seeded ones.
RUN_GLOB = "source_heterogeneity_les_rice_paddy_l*_cv*_s[0-7]_wind3_n[1-4]_*"
NAME_RE = re.compile(
    r"^source_heterogeneity_les_rice_paddy_l(?P<L>\d+)_cv(?P<cv>[0-9p]+)"
    r"_s(?P<seed>\d)_wind3_n(?P<n>\d)_(?P<net>op|point)$"
)

EXPECTED_N_RUNS = 576
EXPECTED_CV = (0.5, 1.0, 2.0)
EXPECTED_L = (100.0, 250.0, 500.0)
EXPECTED_SEEDS = tuple(range(8))
EXPECTED_N = (1, 2, 3, 4)
EXPECTED_NET = ("op", "point")

GEOMETRY = {"OP": "open_path", "PS": "point"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _nature_dir(flux_summary: dict, run_dir: Path) -> Path:
    """The nature run supplying this inversion's truth field.

    Taken from the instrument stage's own recorded upstream, not from the name.
    """
    inst_cfg = yaml.safe_load(
        (run_dir / "instrument" / "config.snapshot.yaml").read_text()
    )
    rel = inst_cfg["inputs"]["dispersion"]
    return (run_dir / "instrument" / rel).resolve()


def _instruments(run_dir: Path) -> list[dict]:
    cfg = yaml.safe_load((run_dir / "instrument" / "config.snapshot.yaml").read_text())
    return cfg["instrument"]["instruments"]


def collect(run_dir: Path) -> dict:
    name = run_dir.name
    flux = _load_json(run_dir / "flux" / "summary.json")
    analysis = _load_json(run_dir / "analysis" / "summary.json")
    mats = np.load(run_dir / "flux" / "matrices.npz")
    inst = _instruments(run_dir)

    nature = _nature_dir(flux, run_dir)
    with xr.open_dataset(nature / "truth_field.nc") as ds:
        attrs = dict(ds.attrs)
        F = ds["F_true"].values
        area = ds["cell_area_m2"].values
        Q_true_check = float((F * area).sum())

    # --- design, from the truth field's own attributes
    L_m = float(attrs["L_true_m"])
    cv = float(attrs["cv"])
    seed = int(attrs["seed"])
    covariance = str(attrs["covariance_model"])
    Q_true = float(attrs["Q_true_kg_s"])
    dx_m = float(attrs["dx_m"])
    nx, ny = int(attrs["nx"]), int(attrs["ny"])

    techs = {i["tech_id"] for i in inst}
    if len(techs) != 1:
        raise ValueError(f"{name}: mixed instrument technologies {techs}")
    geometry = GEOMETRY[techs.pop()]
    n_inst = len(inst)
    path_lengths = [float(i.get("path_length_m", 0.0)) for i in inst]

    # --- retrieval
    x_opt = float(np.asarray(flux["x_opt_kg_s"]).ravel()[0])
    sigma_post = float(np.asarray(flux["posterior_sigma_kg_s"]).ravel()[0])
    x_prior = float(np.asarray(flux["x_prior_kg_s"]).ravel()[0])

    # --- Rodgers diagnostics recomputed from the stored matrices
    G = mats["G"]
    y_obs = mats["y_obs"]
    Se = mats["Se_diag"]
    y_opt = mats["y_opt"]
    A_k = mats["averaging_kernel"]
    dfs = float(np.trace(np.atleast_2d(A_k)))
    resid = y_obs - y_opt
    chi2 = float(np.sum(resid**2 / Se))
    n_obs = int(y_obs.size)
    n_state = int(np.atleast_2d(A_k).shape[0])
    dof = n_obs - n_state
    obs_mode = flux.get("observation_mode", {})
    sigma_repr = np.asarray(obs_mode.get("sigma_repr_by_observation", [np.nan]), float)

    e = (x_opt - Q_true) / Q_true

    return {
        "run": name,
        # design
        "L_m": L_m,
        "CV": cv,
        "seed": seed,
        "n": n_inst,
        "geometry": geometry,
        "covariance_model": covariance,
        "source_dx_m": dx_m,
        "source_nx": nx,
        "source_ny": ny,
        "domain_area_m2": (nx * dx_m) * (ny * dx_m),
        "path_length_total_m": float(np.sum(path_lengths)),
        "path_length_each_m": float(path_lengths[0]) if path_lengths else 0.0,
        # truth
        "Q_true_kg_s": Q_true,
        "Q_true_recomputed_kg_s": Q_true_check,
        # retrieval
        "Q_prior_kg_s": x_prior,
        "Q_hat_kg_s": x_opt,
        "posterior_sigma_kg_s": sigma_post,
        "e_signed": e,
        "e_abs": abs(e),
        "converged": bool(flux["converged"]),
        "n_iter": int(flux["n_iter"]),
        "method": flux["method"],
        # diagnostics
        "n_obs": n_obs,
        "n_state": n_state,
        "dof": dof,
        "dfs": dfs,
        "prior_influence": n_state - dfs,
        "ak_diag": float(np.atleast_2d(A_k)[0, 0]),
        "chi2": chi2,
        "chi2_per_dof": chi2 / dof if dof > 0 else np.nan,
        "sigma_repr_mean_ng_m3": float(np.mean(sigma_repr)),
        "inverse_crime_flag": bool(flux["inverse_crime_flag"]),
        "total_only": bool(flux.get("total_only", False)),
        "inversion_template": flux.get("inversion_template"),
        "n_fine_cells": int(flux.get("n_fine_cells", -1)),
        # units, recorded not assumed
        "state_units": obs_mode.get("units", {}).get("state_units"),
        "obs_units": obs_mode.get("units", {}).get("obs_units"),
        "jacobian_units": obs_mode.get("units", {}).get("jacobian_units"),
        # provenance
        "nature_run": nature.parent.name,
        "E_Q_reported": float(analysis["source_heterogeneity"]["E_Q"]),
    }


def main() -> int:
    dirs = sorted(
        d for d in RUNS.glob(RUN_GLOB)
        if (d / "flux" / "summary.json").is_file() and NAME_RE.match(d.name)
    )
    print(f"found {len(dirs)} run directories")
    rows = [collect(d) for d in dirs]
    df = pd.DataFrame(rows).sort_values(
        ["geometry", "n", "CV", "L_m", "seed"]
    ).reset_index(drop=True)

    checks: list[tuple[str, bool, str]] = []

    def lst(values) -> str:
        """Plain list text; numpy scalar reprs are unreadable in a report."""
        return "[" + ", ".join(f"{v:g}" if isinstance(v, (int, float, np.number))
                               else str(v) for v in values) + "]"

    def gate(label: str, ok: bool, detail: str = "") -> None:
        checks.append((label, bool(ok), detail))

    gate("row count is 576", len(df) == EXPECTED_N_RUNS, f"got {len(df)}")

    key = ["L_m", "CV", "seed", "n", "geometry"]
    dup = df.duplicated(key).sum()
    gate("no duplicate design cells", dup == 0, f"{dup} duplicates")

    gate("CV levels", sorted(df.CV.unique()) == list(EXPECTED_CV),
         lst(sorted(df.CV.unique())))
    gate("L levels", sorted(df.L_m.unique()) == list(EXPECTED_L),
         lst(sorted(df.L_m.unique())))
    gate("seeds", sorted(df.seed.unique()) == list(EXPECTED_SEEDS),
         lst(sorted(df.seed.unique())))
    gate("n levels", sorted(df["n"].unique()) == list(EXPECTED_N),
         lst(sorted(df["n"].unique())))
    gate("geometries", sorted(df.geometry.unique()) == ["open_path", "point"],
         lst(sorted(df.geometry.unique())))

    cell = df.groupby(key, observed=True).size()
    gate("full 3x3x8x4x2 crossing, one run each",
         len(cell) == EXPECTED_N_RUNS and cell.max() == 1 and cell.min() == 1,
         f"{len(cell)} cells, min {cell.min()}, max {cell.max()}")

    gate("all inversions converged", bool(df.converged.all()),
         f"{(~df.converged).sum()} not converged")

    spread = df.Q_true_kg_s.max() - df.Q_true_kg_s.min()
    gate("Q_true identical across runs", spread == 0.0, f"range {spread:.3e} kg/s")

    recon = float(np.abs(df.Q_true_recomputed_kg_s - df.Q_true_kg_s).max())
    gate("sum(F_true*area) reproduces Q_true", recon < 1e-12,
         f"max abs diff {recon:.3e} kg/s")

    # analysis/metrics.py defines E_Q as the *absolute* relative error, so the
    # signed error is recomputed here from x_opt and Q_true and checked against
    # it in magnitude.
    dE = float(np.abs(df.e_abs - df.E_Q_reported).max())
    gate("abs(e_signed) matches analysis-stage E_Q", dE < 1e-9,
         f"max abs diff {dE:.3e}")

    # Cross-check against the pre-existing aggregation, which is not an input.
    ext = pd.read_csv(HERE.parent / "seed_sweep_results.csv")
    ext["geometry"] = ext.network.map({"op": "open_path", "point": "point"})
    m = df.merge(ext, on=["L_m", "CV", "seed", "n", "geometry"],
                 how="inner", suffixes=("", "_ext"))
    d_ext = float(np.abs(m.e_signed - m.q_rel_error).max())
    gate("e_signed matches seed_sweep_results.csv",
         len(m) == EXPECTED_N_RUNS and d_ext < 1e-5,
         f"{len(m)} matched rows, max abs diff {d_ext:.3e}")

    # Name-vs-metadata consistency: the run name is provenance only, but if it
    # disagrees with the authoritative attributes something is badly wrong.
    parsed = df.run.str.extract(NAME_RE)
    name_ok = (
        (parsed["L"].astype(float) == df.L_m).all()
        and (parsed["cv"].str.replace("p", ".").astype(float) == df.CV).all()
        and (parsed["seed"].astype(int) == df.seed).all()
        and (parsed["n"].astype(int) == df["n"]).all()
        and (parsed["net"].map({"op": "open_path", "point": "point"}) == df.geometry).all()
    )
    gate("run names agree with recorded metadata", name_ok)

    gate("no inverse crime flagged", not df.inverse_crime_flag.any(),
         f"{df.inverse_crime_flag.sum()} flagged")
    gate("single-state total-flux retrievals", (df.n_state == 1).all()
         and df.total_only.all())
    gate("n_obs equals sensor count", (df.n_obs == df["n"]).all())
    gate("state units kg s-1", set(df.state_units.dropna()) == {"kg s-1"},
         lst(sorted(set(df.state_units.dropna()))))
    gate("covariance model is exponential everywhere",
         set(df.covariance_model) == {"exponential"},
         lst(sorted(set(df.covariance_model))))
    gate("source domain is 1e6 m2", (df.domain_area_m2 == 1e6).all(),
         lst(sorted(df.domain_area_m2.unique())))

    # Report before raising, so a failure is legible.
    lines = ["# Integrity checks — 576-inversion table", ""]
    lines.append("| check | result | detail |")
    lines.append("|---|---|---|")
    for label, ok, detail in checks:
        lines.append(f"| {label} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")
    lines.append(f"Assembled {len(df)} rows from `runs/{RUN_GLOB}`.")
    lines.append("")
    lines.append("Design cells: " + " x ".join([
        f"L {lst(sorted(df.L_m.unique()))} m", f"CV {lst(sorted(df.CV.unique()))}",
        "seed 0-7", f"n {lst(sorted(df['n'].unique()))}",
        "geometry (open path, point)",
    ]) + f" = {EXPECTED_N_RUNS} inversions.")
    (HERE / "integrity.md").write_text("\n".join(lines) + "\n")
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} {detail}")

    df.to_csv(HERE / "inversions.csv", index=False)
    print(f"wrote inversions.csv ({len(df)} rows, {df.shape[1]} columns)")

    failed = [c for c in checks if not c[1]]
    if failed:
        raise SystemExit(f"{len(failed)} integrity gate(s) failed; see integrity.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
