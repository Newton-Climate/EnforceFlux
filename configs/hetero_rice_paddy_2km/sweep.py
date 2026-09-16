#!/usr/bin/env python3
"""Run the 2 km rice-paddy L, CV, seed, sensor-count and layout sweep from code.

The 2 km counterpart of configs/hetero_rice_paddy_test/sweep.py. The grid lives
here, not in YAML files: every stage config is rendered in memory, written to a
throwaway file for the length of one ``enforceflux`` call, and deleted. Each run
directory keeps the resolved config as ``config.snapshot.yaml``.

    python sweep.py --bls-donors        # the 16 bLS builds (needs R/bLSmodelR)
    python sweep.py --nature            # H.e for every field (needs operator.npz)
    python sweep.py --inversions        # instrument/flux/analysis, reusing H
    python sweep.py --nature --inversions --shard 0 --nshards 16
    python sweep.py --nature --inversions --L 300 --cv 0.75 --seeds 0-7
    python sweep.py --count             # grid size and completion
    python sweep.py --write DIR --L 500 --cv 1.0 --seeds 3   # inspect rendered configs

Completed stages (manifest present) are skipped, so any mode can be resumed.

Nature cases are evaluated through the 2 km tagged-tracer operator (H.e), not
integrated; see ../hetero_rice_paddy_test/SWEEPS.md for what that implies.
Inversion configs are built from the 1 km templates in
configs/hetero_rice_paddy_test/inversions/, with the domain, source grid,
sensor geometry and bLS met replaced.

Two sensor layouts, both fences perpendicular to the wind and centred 20 m
beyond the field's projected downwind edge (the 1 km rule):
  f1000  the 1 km sweep's 1000 m fence, unchanged, on the 2 km field
  s2000  a fence scaled with the field to 2000 m
f1000 changes how much of the field the sensors see as well as how well the
field is represented; s2000 holds the covered fraction fixed.

The bLS Jacobian depends only on sensor geometry and met, so it is built once
per (layout, n, network) — 16 builds — and copied into every other (L, CV, seed).

Requires bls_met.json from derive_bls_met.py run on the 2 km donor.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
ROOT = CONFIG_DIR.parents[1]
NATURE_TEMPLATE = CONFIG_DIR / "les2km_l200_cv2p0_wind3.yaml"
INVERSION_TEMPLATE_DIR = ROOT / "configs" / "hetero_rice_paddy_test" / "inversions"
MET_JSON = CONFIG_DIR / "bls_met.json"

# Denser L axis than 1 km: the operator makes each point nearly free, and the
# footprint scale (~90-185 m) sits inside this range, so curvature is visible.
L_VALUES_M = (100, 150, 200, 250, 350, 500)
CV_VALUES = (0.5, 1.0, 2.0)
SEEDS = tuple(range(8))
N_VALUES = (1, 2, 3, 4)
NETWORKS = ("op", "point")
LAYOUTS = {"f1000": 1000.0, "s2000": 2000.0}
STAGES = (
    ("operator", "dispersion"),
    ("instrument", "instrument"),
    ("flux", "flux"),
    ("analysis", "analysis"),
)

PREFIX = "source_heterogeneity_les2km_rice_paddy"
BASE_RUN = f"{PREFIX}_l200_cv2p0_wind3"
DONOR_RUN = f"{BASE_RUN}_surface"
FIELD_HALF_M = 1000.0
FENCE_OFFSET_M = 20.0
# Rendered configs sit one directory below this one, three below ROOT.
UP = "../../../"
OPERATOR_NPZ = f"{UP}runs/{DONOR_RUN}_tagged_operator/operator.npz"
DONOR_CASE = f"{UP}runs/{DONOR_RUN}/dispersion/concentration_microhh/microhh_case"
DONOR_TIME_S = 3600
# The combination whose 16 bLS operators are built for real.
DONOR = (100, 0.5, 0)
DONOR_TAG = "l100_cv0p5_s0"


def cv_text(cv: float) -> str:
    """CV as written into configs: one decimal when exact (0.5, 1.0), else in full.

    One decimal keeps existing run names and configs unchanged; a fixed
    one-decimal format would silently turn 0.75 into 0.8.
    """
    return f"{cv:.1f}" if round(cv, 1) == cv else f"{cv:g}"


def cv_tag(cv: float) -> str:
    return cv_text(cv).replace(".", "p")


def tag_of(L: int, cv: float, seed: int) -> str:
    return f"l{L}_cv{cv_tag(cv)}_s{seed}"


def run_prefix(tag: str) -> str:
    return f"{PREFIX}_{tag}_wind3"


# ------------------------------------------------------------------ geometry


def fence(toward_deg: float, length_m: float, n: int) -> list[dict]:
    """Sensor positions for n segments of a crosswind fence.

    Open paths: each segment's start point, running along the fence bearing.
    Point sensors: segment centres. Reproduces the 1 km templates exactly.
    """
    th = np.deg2rad(toward_deg)
    d = np.array([np.sin(th), np.cos(th)])
    proj = FIELD_HALF_M * (abs(d[0]) + abs(d[1])) + FENCE_OFFSET_M
    centre = proj * d
    bearing = (toward_deg + 90.0) % 360.0
    ph = np.deg2rad(bearing)
    u = np.array([np.sin(ph), np.cos(ph)])
    start = centre - 0.5 * length_m * u
    seg = length_m / n
    return [{"k": k, "op_start": start + k * seg * u,
             "point": start + (k + 0.5) * seg * u,
             "seg_m": seg, "bearing": bearing} for k in range(n)]


def receptors(toward_deg: float, length_m: float, n: int, network: str) -> list[dict]:
    out = []
    for s in fence(toward_deg, length_m, n):
        rid = f"r_{s['k']:05d}"
        if network == "op":
            x, y = s["op_start"]
            out.append({"id": rid, "x_m": float(x), "y_m": float(y), "alt_m": 2.0,
                        "path_length_m": s["seg_m"], "path_bearing_deg": s["bearing"]})
        else:
            x, y = s["point"]
            out.append({"id": rid, "x_m": float(x), "y_m": float(y), "alt_m": 2.0})
    return out


def instruments(recs: list[dict], network: str) -> list[dict]:
    out = []
    for r in recs:
        i = {"id": r["id"], "tech_id": "OP" if network == "op" else "PS",
             "mode": "good", "x_m": r["x_m"], "y_m": r["y_m"], "z": 2.0}
        if network == "op":
            i["path_length_m"] = r["path_length_m"]
            i["path_bearing_deg"] = r["path_bearing_deg"]
        out.append(i)
    return out


# ------------------------------------------------------------------ templates


@lru_cache(maxsize=None)
def _met() -> dict:
    if not MET_JSON.is_file():
        raise FileNotFoundError(
            f"{MET_JSON} is missing. Run derive_bls_met.py on the 2 km donor's "
            "microhh_case first; the sensor geometry depends on its wind.")
    return json.loads(MET_JSON.read_text())


@lru_cache(maxsize=None)
def _nature_template() -> str:
    return NATURE_TEMPLATE.read_text()


@lru_cache(maxsize=None)
def _inversion_template(n: int, network: str, stage: str) -> str:
    return (INVERSION_TEMPLATE_DIR / f"n{n}_{network}_{stage}.yaml").read_text()


# ------------------------------------------------------------------ rendering


def render_nature(L: int, cv: float, seed: int) -> str:
    tag = tag_of(L, cv, seed)
    text = _nature_template().replace(BASE_RUN, run_prefix(tag))
    text = re.sub(r"(?m)^(\s*L_m:)\s*[-+0-9.eE]+", rf"\g<1> {float(L):.1f}", text)
    text = re.sub(r"(?m)^(\s*cv:)\s*[-+0-9.eE]+", rf"\g<1> {cv_text(cv)}", text)
    text = re.sub(r"(?m)^(\s*seed:)\s*0(\s)", rf"\g<1> {seed:d}\g<2>", text)
    for old, new in (
        ("meteo_dir: ../../runs/", f"meteo_dir: {UP}runs/"),
        ("executable: ../../microhh/", f"executable: {UP}microhh/"),
        ("    spinup_seconds: 2700 # Spinup period, s",
         "    spinup_seconds: 0 # Reuse the completed LES turbulent restart"),
        ("    column_sampletime: 0.1 # Column met interval, s",
         "    column_sampletime: 20 # Donor run already contains 10 Hz wind columns"),
        ("        swdump: 1 # Enable dumps",
         "        swdump: 0 # Donor run already contains the wind-field dumps"),
    ):
        if old not in text:
            raise ValueError(f"nature template no longer contains {old!r}")
        text = text.replace(old, new)
    anchor = "    num_workers: 4 # Parallel workers\n"
    if anchor not in text:
        raise ValueError("nature template lost its num_workers anchor")
    return text.replace(anchor, anchor
                        + "    restart_from:\n"
                        + f"      case_dir: {DONOR_CASE}\n"
                        + f"      time_s: {DONOR_TIME_S}\n"
                        + f"    operator_npz: {OPERATOR_NPZ} # Evaluate H.e, do not integrate\n")


def render_inversion(L: int, cv: float, seed: int, layout: str, n: int, network: str,
                     stage: str) -> str:
    c = yaml.safe_load(_inversion_template(n, network, stage))
    met = _met()
    tag = tag_of(L, cv, seed)
    base = f"{run_prefix(tag)}_{layout}_n{n}_{network}"
    nature_run = f"{run_prefix(tag)}_surface"
    runs = f"{UP}runs"
    recs = receptors(met["toward_bearing_deg"], LAYOUTS[layout], n, network)

    if stage == "operator":
        nature_blob = yaml.safe_load(_nature_template())
        c["run"]["name"] = f"{base}_gp"
        disp = c["dispersion"]
        disp["domain"] = copy.deepcopy(nature_blob["dispersion"]["domain"])
        src = copy.deepcopy(nature_blob["dispersion"]["sources"])
        src["config"]["covariance"]["L_m"] = float(L)
        src["config"]["cv"] = float(cv)
        src["config"]["seed"] = int(seed)
        disp["sources"] = src
        disp["receptors"] = recs
        g = src["config"]["grid"]
        bls = disp["blsmodelr"]
        bls["source_grid"] = {
            "x_bounds": [g["origin_x_m"], g["origin_x_m"] + g["nx"] * g["dx_m"]],
            "y_bounds": [g["origin_y_m"], g["origin_y_m"] + g["ny"] * g["dx_m"]],
            "nx": g["nx"], "ny": g["ny"]}
        bls["intervals"] = [dict(met["interval"])]
    elif stage == "instrument":
        c["run"]["name"] = base
        c["inputs"]["dispersion"] = f"{runs}/{nature_run}/dispersion"
        c["instrument"]["instruments"] = instruments(recs, network)
    elif stage == "flux":
        c["run"]["name"] = base
        c["inputs"] = {"dispersion": f"{runs}/{base}_gp/dispersion",
                       "obs": f"{runs}/{base}/instrument"}
    elif stage == "analysis":
        c["run"]["name"] = base
        c["inputs"] = {"dispersion": f"{runs}/{base}_gp/dispersion",
                       "flux": f"{runs}/{base}/flux"}
    return yaml.safe_dump(c, sort_keys=False)


# ------------------------------------------------------------------ running


def complete(text: str) -> bool:
    blob = yaml.safe_load(text)
    return (ROOT / "runs" / blob["run"]["name"] / blob["stage"] / "manifest.json").is_file()


class Renderer:
    """Writes one rendered config at a time to a private scratch directory."""

    def __init__(self) -> None:
        # Inside CONFIG_DIR, so the configs' ../../../ paths resolve as before.
        self._tmp = tempfile.TemporaryDirectory(prefix=".render_", dir=CONFIG_DIR)

    def path(self, name: str, text: str) -> Path:
        path = Path(self._tmp.name) / name
        path.write_text(text)
        return path

    def close(self) -> None:
        self._tmp.cleanup()


def invoke(command: str, config: Path) -> None:
    print(f"\n>>> enforceflux {command} {config.name}", flush=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/enforceflux-mpl")
    subprocess.run(["enforceflux", command, "--config", str(config)],
                   cwd=ROOT, env=env, check=True)


def donor_operator(layout: str, n: int, network: str) -> Path | None:
    d = (ROOT / "runs" / f"{run_prefix(DONOR_TAG)}_{layout}_n{n}_{network}_gp"
         / "dispersion")
    return d if (d / "manifest.json").is_file() else None


def materialize_cached_operator(config: Path, *, tag: str, donor: Path) -> None:
    """Reuse H across source statistics while retaining matching truth files."""
    from enforceflux.runs import load_stage_config, open_run_dir

    nature = ROOT / "runs" / f"{run_prefix(tag)}_surface" / "dispersion"
    for name in ("truth_field.nc", "basis_mapping.npz"):
        if not (nature / name).is_file():
            raise FileNotFoundError(f"nature artifact is missing: {nature / name}")
    stage_cfg = load_stage_config(config, expected_stage="dispersion")
    run_dir = open_run_dir(stage="dispersion", run_name=stage_cfg.run_name,
                           outputs_root=stage_cfg.outputs_root, inputs={})
    run_dir.snapshot_config(stage_cfg.snapshot)
    for name in ("truth_field.nc", "basis_mapping.npz"):
        shutil.copy2(nature / name, run_dir.path(name))
    shutil.copy2(donor / "jacobian.npz", run_dir.path("jacobian.npz"))
    run_dir.record_output("truth_field.nc", role="truth_field")
    run_dir.record_output("basis_mapping.npz", role="basis_mapping")
    run_dir.record_output("jacobian.npz", role="jacobian")
    run_dir.add_manifest_field("operator_cache_source", str(donor))
    run_dir.finalize()
    print(f"cached operator: {config.name} <- {donor.parent.name}", flush=True)


def cases(ls, cvs, seeds, shard: tuple[int, int]) -> list[tuple[int, float, int]]:
    """This shard's (L, CV, seed) combinations, donor first.

    Shards split by combination, so no two shards touch the same run directory.
    The donor always lands in shard 0; other shards only read its bLS runs.
    """
    combos = sorted(itertools.product(ls, cvs, seeds),
                    key=lambda c: tag_of(*c) != DONOR_TAG)
    k, n = shard
    return combos[k::n]


def designs():
    return itertools.product(LAYOUTS, N_VALUES, NETWORKS)


def run_bls_donors(renderer: Renderer) -> None:
    for layout, n, network in designs():
        text = render_inversion(*DONOR, layout, n, network, "operator")
        name = f"{DONOR_TAG}_{layout}_n{n}_{network}_operator.yaml"
        if complete(text):
            print(f"skip completed: {name}", flush=True)
            continue
        invoke("dispersion", renderer.path(name, text))


def run_nature(combos, renderer: Renderer) -> None:
    for L, cv, seed in combos:
        text = render_nature(L, cv, seed)
        name = f"les2km_{tag_of(L, cv, seed)}_wind3.yaml"
        if complete(text):
            print(f"skip completed: {name}", flush=True)
            continue
        invoke("dispersion", renderer.path(name, text))


def run_inversions(combos, renderer: Renderer) -> None:
    for L, cv, seed in combos:
        tag = tag_of(L, cv, seed)
        for layout, n, network in designs():
            for stage, command in STAGES:
                text = render_inversion(L, cv, seed, layout, n, network, stage)
                name = f"{tag}_{layout}_n{n}_{network}_{stage}.yaml"
                if complete(text):
                    print(f"skip completed: {name}", flush=True)
                    continue
                config = renderer.path(name, text)
                if stage == "operator":
                    donor = donor_operator(layout, n, network)
                    if donor is None:
                        raise FileNotFoundError(
                            f"no bLS operator for {layout} n{n} {network}; run "
                            "--bls-donors first (and copy the *_gp runs here)")
                    if tag != DONOR_TAG:
                        materialize_cached_operator(config, tag=tag, donor=donor)
                    continue
                invoke(command, config)


def write_configs(combos, out: Path) -> int:
    """Materialize the rendered configs, laid out as the old generator did."""
    (out / "nature_sweep").mkdir(parents=True, exist_ok=True)
    (out / "inversion_sweep").mkdir(parents=True, exist_ok=True)
    count = 0
    for L, cv, seed in combos:
        tag = tag_of(L, cv, seed)
        (out / "nature_sweep" / f"les2km_{tag}_wind3.yaml").write_text(
            render_nature(L, cv, seed))
        count += 1
        for layout, n, network in designs():
            for stage, _ in STAGES:
                (out / "inversion_sweep" / f"{tag}_{layout}_n{n}_{network}_{stage}.yaml"
                 ).write_text(render_inversion(L, cv, seed, layout, n, network, stage))
                count += 1
    return count


def report(combos) -> None:
    nature_done = sum(complete(render_nature(*c)) for c in combos)
    n_designs = len(LAYOUTS) * len(N_VALUES) * len(NETWORKS)
    inv_done = sum(complete(render_inversion(*c, layout, n, network, "analysis"))
                   for c in combos for layout, n, network in designs())
    donors = sum(donor_operator(*d) is not None for d in designs())
    print(f"{donors}/{n_designs} bLS donor operators built")
    print(f"{len(combos)} source fields: {nature_done} done")
    print(f"{len(combos) * n_designs} inversions: {inv_done} done")


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def _seeds(text: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bls-donors", action="store_true", help="build the 16 bLS operators")
    ap.add_argument("--nature", action="store_true", help="evaluate the source fields")
    ap.add_argument("--inversions", action="store_true", help="run the inversion stages")
    ap.add_argument("--count", action="store_true", help="report grid size and completion")
    ap.add_argument("--write", type=Path, metavar="DIR", help="write rendered configs to DIR")
    ap.add_argument("--L", type=_floats, default=L_VALUES_M, help="comma-separated, m")
    ap.add_argument("--cv", type=_floats, default=CV_VALUES, help="comma-separated")
    ap.add_argument("--seeds", type=_seeds, default=SEEDS, help="e.g. 0-7 or 0,3,5-7")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if not (a.bls_donors or a.nature or a.inversions or a.count or a.write):
        ap.error("select --bls-donors, --nature, --inversions, --count, or --write")
    if not 0 <= a.shard < a.nshards:
        ap.error("need 0 <= --shard < --nshards")

    combos = cases(tuple(int(v) for v in a.L), a.cv, a.seeds, (a.shard, a.nshards))
    if a.write:
        print(f"wrote {write_configs(combos, a.write)} configs to {a.write}")
    if a.count:
        report(combos)
    if a.bls_donors or a.nature or a.inversions:
        renderer = Renderer()
        try:
            # The 16 donors are not split by shard; shard 0 builds them all so
            # parallel shards never race on the same run directory.
            if a.bls_donors and a.shard == 0:
                run_bls_donors(renderer)
            if a.nature:
                run_nature(combos, renderer)
            if a.inversions:
                run_inversions(combos, renderer)
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
