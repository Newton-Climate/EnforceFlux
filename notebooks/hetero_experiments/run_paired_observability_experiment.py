#!/usr/bin/env python3
"""Paired point/OP observability experiment without per-run YAML files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "configs/paired_observability_templates"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((TEMPLATES / name).read_text())


def cv_slug(cv: float) -> str:
    return str(float(cv)).replace(".", "p")


def nature_name(spec: dict, L: int, cv: float, source_seed: int) -> str:
    if source_seed != 0:
        raise ValueError(
            "Only source seed 0 has a completed LES nature run. Add seeded nature "
            "runs before extending experiment.source_seeds."
        )
    return spec["experiment"]["nature_run_pattern"].format(
        L=L, cv_slug=cv_slug(cv), source_seed=source_seed
    )


def affine_lonlat(nc_path: Path):
    with Dataset(nc_path) as ds:
        x = np.asarray(ds["x"][:], float)
        y = np.asarray(ds["y"][:], float)
        lon = np.asarray(ds["longitude"][:], float)
        lat = np.asarray(ds["latitude"][:], float)
    xx, yy = np.meshgrid(x, y)
    design = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
    transform, *_ = np.linalg.lstsq(
        design, np.column_stack((lon.ravel(), lat.ravel())), rcond=None
    )

    def convert(x_m: float, y_m: float) -> tuple[float, float]:
        out = np.array([x_m, y_m, 1.0]) @ transform
        return float(out[0]), float(out[1])

    return convert


def nested_centres(spec: dict, layout_seed: int) -> list[tuple[int, float]]:
    network = spec["network"]
    nmax = max(int(n) for n in spec["experiment"]["n_instruments"])
    rng = np.random.default_rng(7300 + layout_seed)
    centres = np.linspace(
        float(network["crosswind_min_m"]),
        float(network["crosswind_max_m"]), nmax,
    )
    centres += rng.uniform(
        -float(network["layout_jitter_m"]),
        float(network["layout_jitter_m"]), nmax,
    )
    centres.sort()
    candidates = list(range(nmax))
    chosen = [
        min(candidates, key=lambda i: abs(centres[i] + 200.0)),
        min(candidates, key=lambda i: abs(centres[i] - 200.0)),
    ]
    while len(chosen) < nmax:
        remaining = [i for i in candidates if i not in chosen]
        scores = np.array([
            min(abs(centres[i] - centres[j]) for j in chosen) for i in remaining
        ])
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        chosen.append(remaining[int(rng.choice(best))])
    return [(i, float(centres[i])) for i in chosen]


def paired_geometry(
    spec: dict, nature: str, n: int, layout_seed: int, technology: str,
) -> tuple[list[dict], list[dict], float, float]:
    nc = ROOT / "runs" / nature / "dispersion/concentration.nc"
    generated = ROOT / "runs" / nature / (
        "dispersion/concentration_microhh/microhh_generated.yaml"
    )
    to_lonlat = affine_lonlat(nc)
    bearing = float(yaml.safe_load(generated.read_text())["domain"]["x_bearing_deg"])
    beam_bearing = (bearing - 90.0) % 360.0
    x_m = float(spec["network"]["downwind_x_m"])
    length = float(spec["network"]["beam_length_m"])

    from enforceflux.transport.run_config import DomainProjection
    projection = DomainProjection(-121.75, 39.15)
    instruments, receptors = [], []
    for candidate, centre_y in nested_centres(spec, layout_seed)[:n]:
        is_op = technology == "op"
        local_y = centre_y - 0.5 * length if is_op else centre_y
        lon, lat = to_lonlat(x_m, local_y)
        east, north = projection.to_xy(lon, lat)
        rid = f"layout{layout_seed}_c{candidate:02d}"
        inst = {
            "id": rid, "tech_id": "OP" if is_op else "PS", "mode": "good",
            "lon": lon, "lat": lat, "z": 3.0, "response_scale": 1.0,
        }
        receptor = {"id": rid, "x_m": float(east), "y_m": float(north), "alt_m": 3.0}
        if is_op:
            inst.update(path_length_m=length, path_bearing_deg=0.0)
            receptor.update(path_length_m=length, path_bearing_deg=beam_bearing)
        instruments.append(inst)
        receptors.append(receptor)
    return instruments, receptors, bearing, beam_bearing


def stage(stage_name: str, blob: dict, tmp: Path) -> None:
    path = tmp / f"{stage_name}.yaml"
    path.write_text(yaml.safe_dump(blob, sort_keys=False))
    command = "dispersion" if stage_name == "operator" else stage_name
    result = subprocess.run(
        ["enforceflux", command, "--config", str(path)], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout[-12000:])
        result.check_returncode()


def configure_operator_paths(blob: dict, met_dir: Path) -> None:
    flex = blob["dispersion"]["flexpart"]
    flex["executable"] = str(ROOT / "flexpart/src/FLEXPART")
    flex["options_dir"] = str(ROOT / "flexpart/options")
    blob["dispersion"]["met"]["era5"]["meteo_dir"] = str(met_dir)


def run_cell(
    spec: dict, L: int, cv: float, source_seed: int, layout_seed: int,
    n: int, technology: str, tmp: Path,
) -> None:
    nature = nature_name(spec, L, cv, source_seed)
    tag = f"l{L}_cv{cv_slug(cv)}_s{source_seed}_layout{layout_seed}_n{n}_{technology}"
    base = f"paired_observability_{tag}"
    operator_name = f"{base}_operator"
    instruments, receptors, bearing, beam_bearing = paired_geometry(
        spec, nature, n, layout_seed, technology
    )
    print(f"Running {tag}", flush=True)

    inst = load_yaml("instrument.yaml")
    inst["run"]["name"] = base
    inst["inputs"]["dispersion"] = str(ROOT / "runs" / nature / "dispersion")
    inst["instrument"]["operator"]["random_seed"] = 100000 + 100 * layout_seed + n
    inst["instrument"]["instruments"] = instruments

    operator = load_yaml("operator.yaml")
    met_ref = spec["meteorology"]["cases"][0]["directory"]
    met_dir = (TEMPLATES / met_ref).resolve()
    configure_operator_paths(operator, met_dir)
    operator["run"]["name"] = operator_name
    source_cfg = operator["dispersion"]["sources"]["config"]
    source_cfg["covariance"]["L_m"] = float(L)
    source_cfg["cv"] = float(cv)
    source_cfg["seed"] = int(source_seed)
    source_cfg["basis"]["coarsen"] = int(spec["inversion"]["gp_coarsen"])
    operator["dispersion"]["receptors"] = receptors
    flex = operator["dispersion"]["flexpart"]
    flex["source_x_bearing_deg"] = bearing
    flex["physical_beam_bearing_deg"] = beam_bearing
    flex["path_samples"] = int(spec["network"]["beam_quadrature_points"])
    flex["base_run_dir"] = str(
        ROOT / "runs/paired_observability_flexpart_cache" / technology /
        f"layout{layout_seed}"
    )

    stage("instrument", inst, tmp)
    stage("operator", operator, tmp)

    n_coarse = (48 // int(spec["inversion"]["gp_coarsen"])) * (
        24 // int(spec["inversion"]["gp_coarsen"])
    )
    for mode, template in (("total_uniform", "flux_total.yaml"), ("spatial_gp", "flux_gp.yaml")):
        run_name = f"{base}_{mode}"
        flux = load_yaml(template)
        flux["run"]["name"] = run_name
        flux["inputs"]["dispersion"] = str(ROOT / "runs" / operator_name / "dispersion")
        flux["inputs"]["obs"] = str(ROOT / "runs" / base / "instrument")
        if mode == "spatial_gp":
            inv = flux["flux"]["inversion"]
            inv["prior_flux_kg_s"] = float(spec["inversion"]["gp_prior_total_kg_s"]) / n_coarse
            inv["prior_covariance"]["sigma_kg_s"] = float(
                spec["inversion"]["gp_sigma_per_cell_kg_s"]
            )
            inv["prior_covariance"]["L_B_m"] = float(spec["inversion"]["gp_length_m"])

        analysis = load_yaml("analysis.yaml")
        analysis["run"]["name"] = run_name
        analysis["inputs"]["dispersion"] = str(ROOT / "runs" / operator_name / "dispersion")
        analysis["inputs"]["flux"] = str(ROOT / "runs" / run_name / "flux")
        stage("flux", flux, tmp)
        stage("analysis", analysis, tmp)


def cells(spec: dict):
    e = spec["experiment"]
    for L in e["L_source_m"]:
        for cv in e["cv"]:
            for source_seed in e["source_seeds"]:
                for layout_seed in e["layout_seeds"]:
                    for n in e["n_instruments"]:
                        for technology in e["technologies"]:
                            yield int(L), float(cv), int(source_seed), int(layout_seed), int(n), str(technology)


def validate(spec: dict) -> None:
    missing = []
    for L in spec["experiment"]["L_source_m"]:
        for cv in spec["experiment"]["cv"]:
            for seed in spec["experiment"]["source_seeds"]:
                nature = nature_name(spec, int(L), float(cv), int(seed))
                path = ROOT / "runs" / nature / "dispersion/concentration.nc"
                if not path.exists():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing LES nature runs:\n" + "\n".join(missing))
    n_cells = sum(1 for _ in cells(spec))
    print(f"Validated 9 LES nature runs; {n_cells} paired sensor/operator cells")
    print(f"Planned inversions: {2 * n_cells} (total_uniform + spatial_gp)")


def summarize(spec: dict) -> None:
    output = ROOT / "notebooks/hetero_experiments/paired_observability_results.csv"
    fields = [
        "L_source_m", "cv", "source_seed", "layout_seed", "n_instruments",
        "technology", "mode", "Q_true_kg_s", "Q_hat_kg_s", "E_Q",
        "posterior_sigma_total_kg_s", "n_observations", "operator_rank",
        "zero_sensitivity_rows", "physical_nonnegative", "success",
    ]
    rows = []
    for L, cv, source_seed, layout_seed, n, technology in cells(spec):
        tag = f"l{L}_cv{cv_slug(cv)}_s{source_seed}_layout{layout_seed}_n{n}_{technology}"
        for mode in spec["experiment"]["inversion_modes"]:
            run_name = f"paired_observability_{tag}_{mode}"
            flux_path = ROOT / "runs" / run_name / "flux/summary.json"
            analysis_path = ROOT / "runs" / run_name / "analysis/summary.json"
            matrices_path = ROOT / "runs" / run_name / "flux/matrices.npz"
            if not (flux_path.exists() and analysis_path.exists() and matrices_path.exists()):
                continue
            flux = json.loads(flux_path.read_text())
            analysis = json.loads(analysis_path.read_text())["source_heterogeneity"]
            q_hat = float(np.asarray(flux["x_opt_kg_s"], float).sum())
            q_true = q_hat / (1.0 + float(analysis["E_Q"])) if q_hat >= 0 else 0.027778
            with Dataset(ROOT / "runs" / nature_name(spec, L, cv, source_seed) /
                         "dispersion/truth_field.nc") as ds:
                q_true = float(np.sum(ds["F_true"][:] * ds["cell_area_m2"][:]))
            matrices = np.load(matrices_path)
            sx = matrices["Sx"]
            g = np.asarray(matrices["G"], float)
            sigma_total = float(math.sqrt(max(float(np.ones(sx.shape[0]) @ sx @ np.ones(sx.shape[0])), 0.0)))
            error = abs(q_hat - q_true) / q_true
            rows.append({
                "L_source_m": L, "cv": cv, "source_seed": source_seed,
                "layout_seed": layout_seed, "n_instruments": n,
                "technology": technology, "mode": mode, "Q_true_kg_s": q_true,
                "Q_hat_kg_s": q_hat, "E_Q": error,
                "posterior_sigma_total_kg_s": sigma_total,
                "n_observations": int(g.shape[0]),
                "operator_rank": int(np.linalg.matrix_rank(g)),
                "zero_sensitivity_rows": int(np.sum(np.all(g == 0.0, axis=1))),
                "physical_nonnegative": int(np.all(np.asarray(flux["x_opt_kg_s"]) >= 0.0)),
                "success": int(error <= float(spec["inversion"]["success_error_fraction"])),
            })
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} completed inversions to {output}")

    paired = {}
    for row in rows:
        key = (
            row["L_source_m"], row["cv"], row["source_seed"],
            row["layout_seed"], row["n_instruments"], row["mode"],
        )
        paired.setdefault(key, {})[row["technology"]] = row
    groups = {}
    for key, technologies in paired.items():
        if {"point", "op"} <= technologies.keys():
            L, cv, _, _, n, mode = key
            groups.setdefault((L, cv, n, mode), []).append(technologies)

    equality_rows = []
    rng = np.random.default_rng(20260701)
    ratio_bounds = [float(v) for v in spec["inversion"]["equivalence_error_ratio"]]
    for (L, cv, n, mode), samples in sorted(groups.items()):
        point_error = np.array([s["point"]["E_Q"] for s in samples], float)
        op_error = np.array([s["op"]["E_Q"] for s in samples], float)
        log_ratio = np.log((op_error + 1.0e-12) / (point_error + 1.0e-12))
        if len(log_ratio) > 1:
            boot = np.median(
                rng.choice(log_ratio, size=(2000, len(log_ratio)), replace=True), axis=1
            )
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        else:
            ci_low = ci_high = float(log_ratio[0])
        median_log = float(np.median(log_ratio))
        median_ratio = float(np.exp(median_log))
        equality_rows.append({
            "L_source_m": L, "cv": cv, "n_instruments": n, "mode": mode,
            "n_paired_replicates": len(samples),
            "median_E_point": float(np.median(point_error)),
            "median_E_op": float(np.median(op_error)),
            "P_success_point": float(np.mean([s["point"]["success"] for s in samples])),
            "P_success_op": float(np.mean([s["op"]["success"] for s in samples])),
            "median_operator_rank_point": float(np.median([s["point"]["operator_rank"] for s in samples])),
            "median_operator_rank_op": float(np.median([s["op"]["operator_rank"] for s in samples])),
            "zero_sensitivity_rows_point": int(sum(s["point"]["zero_sensitivity_rows"] for s in samples)),
            "zero_sensitivity_rows_op": int(sum(s["op"]["zero_sensitivity_rows"] for s in samples)),
            "median_error_ratio_op_over_point": median_ratio,
            "log_ratio_ci_low": float(ci_low), "log_ratio_ci_high": float(ci_high),
            "equivalent": int(
                ci_low <= 0.0 <= ci_high
                and ratio_bounds[0] <= median_ratio <= ratio_bounds[1]
            ),
        })
    equality_output = ROOT / "notebooks/hetero_experiments/paired_observability_equality.csv"
    if equality_rows:
        with equality_output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(equality_rows[0]))
            writer.writeheader(); writer.writerows(equality_rows)
    print(f"Wrote {len(equality_rows)} paired equality cells to {equality_output}")

    required_rows = []
    probability_target = float(spec["inversion"]["success_probability"])
    for L in spec["experiment"]["L_source_m"]:
        for cv in spec["experiment"]["cv"]:
            for mode in spec["experiment"]["inversion_modes"]:
                relevant = [r for r in equality_rows if r["L_source_m"] == L
                            and r["cv"] == cv and r["mode"] == mode]
                for technology in ("point", "op"):
                    key = f"P_success_{technology}"
                    qualifying = [r["n_instruments"] for r in relevant
                                  if r[key] >= probability_target]
                    required_rows.append({
                        "L_source_m": L, "cv": cv, "mode": mode,
                        "technology": technology,
                        "N_min": min(qualifying) if qualifying else "",
                        "success_probability_target": probability_target,
                    })
    required_output = ROOT / "notebooks/hetero_experiments/paired_observability_required_network.csv"
    with required_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(required_rows[0]))
        writer.writeheader(); writer.writerows(required_rows)
    print(f"Wrote required-network summary to {required_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pilot", action="store_true", help="Run one paired cell in both inversion modes")
    group.add_argument("--full", action="store_true", help="Run the complete configured matrix")
    group.add_argument("--summarize", action="store_true", help="Collect completed runs into CSV")
    parser.add_argument("--L-source", type=int, choices=(200, 500, 1000),
                        help="Restrict --full to one source correlation length")
    parser.add_argument("--cv", type=float, choices=(0.5, 1.0, 2.0),
                        help="Restrict --full to one coefficient of variation")
    parser.add_argument("--n-instruments", type=int, choices=(2, 3, 4, 8, 16),
                        help="Restrict --full to one network size")
    args = parser.parse_args()
    spec = load_yaml("experiment.yaml")
    validate(spec)
    if not (args.pilot or args.full or args.summarize):
        return
    if args.summarize:
        summarize(spec); return
    selected = list(cells(spec))
    if args.pilot:
        selected = [cell for cell in selected if cell[:5] == (200, 0.5, 0, 0, 2)]
    else:
        if args.L_source is not None:
            selected = [cell for cell in selected if cell[0] == args.L_source]
        if args.cv is not None:
            selected = [cell for cell in selected if cell[1] == args.cv]
        if args.n_instruments is not None:
            selected = [cell for cell in selected if cell[4] == args.n_instruments]
    with tempfile.TemporaryDirectory(prefix="paired_observability_") as tmp_dir:
        for cell in selected:
            run_cell(spec, *cell, Path(tmp_dir))
    summarize(spec)


if __name__ == "__main__":
    main()
