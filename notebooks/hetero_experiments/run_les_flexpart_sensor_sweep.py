#!/usr/bin/env python3
"""Run the July LES/FLEXPART point-versus-OP OSSE without per-run configs."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "configs/july01_2026_osse_templates"
COUNTS = (2, 3, 4, 8, 16)
LS = (200, 500, 1000)
CVS = (0.5, 1.0, 2.0)


def slug(L: int, cv: float) -> str:
    return f"l{L}_cv{str(cv).replace('.', 'p')}"


def load_template(name: str) -> dict:
    return yaml.safe_load((TEMPLATES / f"{name}.yaml").read_text())


def nested_y_indices(y: np.ndarray, nmax: int = 16) -> list[int]:
    # OP beams start at each sensor and extend 400 m in +y.  Keeping starts at
    # or below 520 m leaves every beam inside this LES domain (y <= 920 m).
    candidates = np.flatnonzero(
        (y >= float(y.min()) + 240.0) & (y <= float(y.max()) - 400.0)
    ).tolist()
    if len(candidates) < nmax:
        raise ValueError(f"Need {nmax} crosswind sites, found {len(candidates)}")
    chosen = [
        min(candidates, key=lambda i: abs(y[i] + 200.0)),
        min(candidates, key=lambda i: abs(y[i] - 200.0)),
    ]
    while len(chosen) < nmax:
        remaining = [i for i in candidates if i not in chosen]
        chosen.append(max(remaining, key=lambda i: min(abs(y[i] - y[j]) for j in chosen)))
    return chosen


def geometry(nature: str, n: int, technology: str) -> tuple[list[dict], list[dict], float]:
    nc = ROOT / "runs" / nature / "dispersion/concentration.nc"
    generated = ROOT / "runs" / nature / "dispersion/concentration_microhh/microhh_generated.yaml"
    with Dataset(nc) as ds:
        x = np.asarray(ds["x"][:], float)
        y = np.asarray(ds["y"][:], float)
        lon = np.asarray(ds["longitude"][:], float)
        lat = np.asarray(ds["latitude"][:], float)
    bearing = float(yaml.safe_load(generated.read_text())["domain"]["x_bearing_deg"])
    # LES +y is the left-crosswind axis, i.e. 90 degrees counter-clockwise
    # from the downwind +x axis.
    beam_bearing = (bearing - 90.0) % 360.0
    ix = int(np.argmin(abs(x - 1440.0)))
    iy = nested_y_indices(y)[:n]

    from enforceflux.transport.run_config import DomainProjection
    projection = DomainProjection(-121.75, 39.15)
    instruments, receptors = [], []
    for k, j in enumerate(iy):
        rid = f"r_y{j:02d}"
        rlon, rlat = float(lon[j, ix]), float(lat[j, ix])
        east, north = projection.to_xy(rlon, rlat)
        inst = {"id": rid, "tech_id": technology, "mode": "good", "lon": rlon,
                "lat": rlat, "z": 3.0, "response_scale": 1.0}
        receptor = {"id": rid, "x_m": float(east), "y_m": float(north), "alt_m": 3.0}
        if technology == "OP":
            inst.update(path_length_m=400.0, path_bearing_deg=0.0)
            receptor.update(path_length_m=400.0, path_bearing_deg=beam_bearing)
        instruments.append(inst)
        receptors.append(receptor)
    return instruments, receptors, beam_bearing


def run_stage(stage: str, blob: dict, temp_dir: Path) -> None:
    path = temp_dir / f"{stage}.yaml"
    path.write_text(yaml.safe_dump(blob, sort_keys=False))
    command = "dispersion" if stage == "operator" else stage
    result = subprocess.run(
        ["enforceflux", command, "--config", str(path)], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout[-12000:])
        result.check_returncode()


def configure_paths(blob: dict) -> None:
    if blob.get("stage") != "dispersion":
        return
    fp = blob["dispersion"]["flexpart"]
    fp["executable"] = str(ROOT / "flexpart/src/FLEXPART")
    fp["options_dir"] = str(ROOT / "flexpart/options")
    blob["dispersion"]["met"]["era5"]["meteo_dir"] = str(
        ROOT / "runs/sacramento_valley_2026_july01_point/met")


def execute_case(L: int, cv: float, n: int, technology: str, temp_dir: Path) -> str:
    case = slug(L, cv)
    nature = f"source_heterogeneity_les_source_{case}"
    tech = "op" if technology == "OP" else "point"
    name = f"{nature}_n{n}_july_flex_{tech}"
    operator_name = f"{name}_operator"
    print(f"Running L={L} m, CV={cv:g}, N={n}, {technology}", flush=True)
    instruments, receptors, beam_bearing = geometry(nature, n, technology)

    inst = load_template("instrument")
    inst["run"]["name"] = name
    inst["inputs"]["dispersion"] = str(ROOT / "runs" / nature / "dispersion")
    inst["instrument"]["operator"]["random_seed"] = 1000 + n
    inst["instrument"]["instruments"] = instruments

    operator = load_template("operator")
    configure_paths(operator)
    operator["run"]["name"] = operator_name
    operator["dispersion"]["sources"]["config"]["covariance"]["L_m"] = float(L)
    operator["dispersion"]["sources"]["config"]["cv"] = float(cv)
    operator["dispersion"]["receptors"] = receptors
    operator["dispersion"]["flexpart"]["source_x_bearing_deg"] = (
        beam_bearing + 90.0
    ) % 360.0
    operator["dispersion"]["flexpart"]["base_run_dir"] = str(
        ROOT / "runs/july01_flexpart_footprint_cache_aligned_v2" / tech)
    operator["dispersion"]["flexpart"]["physical_beam_bearing_deg"] = beam_bearing

    flux = load_template("flux")
    flux["run"]["name"] = name
    flux["inputs"]["dispersion"] = str(ROOT / "runs" / operator_name / "dispersion")
    flux["inputs"]["obs"] = str(ROOT / "runs" / name / "instrument")

    analysis = load_template("analysis")
    analysis["run"]["name"] = name
    analysis["inputs"]["dispersion"] = str(ROOT / "runs" / operator_name / "dispersion")
    analysis["inputs"]["flux"] = str(ROOT / "runs" / name / "flux")

    for stage, blob in (("instrument", inst), ("operator", operator), ("flux", flux), ("analysis", analysis)):
        run_stage(stage, blob, temp_dir)
    return name


def make_figure() -> Path:
    colors = {200: "#0072B2", 500: "#E69F00", 1000: "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, cv in zip(axes, CVS):
        for L in LS:
            for tech, style in (("point", "-"), ("op", "--")):
                vals = []
                for n in COUNTS:
                    name = f"source_heterogeneity_les_source_{slug(L, cv)}_n{n}_july_flex_{tech}"
                    summary = ROOT / "runs" / name / "analysis/summary.json"
                    vals.append(abs(float(json.loads(summary.read_text())["source_heterogeneity"]["E_Q"])))
                ax.plot(COUNTS, vals, style, color=colors[L], marker="o", ms=4)
        ax.axhline(0.2, color=".2", ls=":", lw=1)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(COUNTS, [str(v) for v in COUNTS]); ax.set_title(f"CV = {cv:g}")
        ax.set_xlabel("Number of sensors")
    axes[0].set_ylabel(r"Total-flux error, $|E_Q|$")
    handles = [plt.Line2D([], [], color=colors[L], marker="o", label=f"L={L} m") for L in LS]
    handles += [plt.Line2D([], [], color=".15", ls="-", label="Point"),
                plt.Line2D([], [], color=".15", ls="--", label="400 m OP"),
                plt.Line2D([], [], color=".2", ls=":", label="20% target")]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.10), fontsize=8)
    fig.suptitle("July LES nature runs; FLEXPART total-flux inversions", y=1.22)
    out = ROOT / "notebooks/hetero_experiments/july_les_flexpart_point_vs_op.pdf"
    fig.savefig(out, bbox_inches="tight")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    cases = [(200, 0.5)] if args.pilot else [(L, cv) for L in LS for cv in CVS]
    counts = (2,) if args.pilot else COUNTS
    with tempfile.TemporaryDirectory(prefix="hetero_osse_") as tmp:
        temp_dir = Path(tmp)
        for technology in ("PS", "OP"):
            for L, cv in cases:
                for n in counts:
                    execute_case(L, cv, n, technology, temp_dir)
    if not args.pilot:
        print(make_figure())


if __name__ == "__main__":
    main()
