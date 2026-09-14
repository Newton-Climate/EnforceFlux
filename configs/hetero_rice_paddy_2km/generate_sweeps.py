#!/usr/bin/env python3
"""Generate the 2 km rice-paddy L, CV, seed, sensor-count and layout sweep.

The 2 km counterpart of configs/hetero_rice_paddy_test/generate_sweeps.py.
Nature cases are evaluated through the 2 km tagged-tracer operator (H.e), not
integrated; see that directory's SWEEPS.md for what that implies. Inversion
configs are built from the 1 km templates in
configs/hetero_rice_paddy_test/inversions/, with the domain, source grid,
sensor geometry and bLS met replaced.

Two sensor layouts, both fences perpendicular to the wind and centred 20 m
beyond the field's projected downwind edge (the 1 km rule):
  f1000  the 1 km sweep's 1000 m fence, unchanged, on the 2 km field
  s2000  a fence scaled with the field to 2000 m
f1000 changes how much of the field the sensors see as well as how well the
field is represented; s2000 holds the covered fraction fixed.

Requires bls_met.json from derive_bls_met.py run on the 2 km donor.
"""
from __future__ import annotations

import copy
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
ROOT = CONFIG_DIR.parents[1]
NATURE_TEMPLATE = CONFIG_DIR / "les2km_l200_cv2p0_wind3.yaml"
INVERSION_TEMPLATE_DIR = ROOT / "configs" / "hetero_rice_paddy_test" / "inversions"
MET_JSON = CONFIG_DIR / "bls_met.json"
NATURE_OUTPUT_DIR = CONFIG_DIR / "nature_sweep"
INVERSION_OUTPUT_DIR = CONFIG_DIR / "inversion_sweep"

# Denser L axis than 1 km: the operator makes each point nearly free, and the
# footprint scale (~90-185 m) sits inside this range, so curvature is visible.
L_VALUES_M = (100, 150, 200, 250, 350, 500)
CV_VALUES = (0.5, 1.0, 2.0)
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
N_VALUES = (1, 2, 3, 4)
NETWORKS = ("op", "point")
LAYOUTS = {"f1000": 1000.0, "s2000": 2000.0}
STAGES = ("operator", "instrument", "flux", "analysis")

PREFIX = "source_heterogeneity_les2km_rice_paddy"
BASE_RUN = f"{PREFIX}_l200_cv2p0_wind3"
DONOR_RUN = f"{BASE_RUN}_surface"
FIELD_HALF_M = 1000.0
FENCE_OFFSET_M = 20.0
# Generated configs sit one directory below this one, three below ROOT.
UP = "../../../"
OPERATOR_NPZ = f"{UP}runs/{DONOR_RUN}_tagged_operator/operator.npz"
DONOR_CASE = f"{UP}runs/{DONOR_RUN}/dispersion/concentration_microhh/microhh_case"
DONOR_TIME_S = 3600


def cv_tag(cv: float) -> str:
    return f"{cv:.1f}".replace(".", "p")


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


# ------------------------------------------------------------------ nature


def nature_config(template: str, L: int, cv: float, seed: int) -> str:
    tag = tag_of(L, cv, seed)
    text = template.replace(BASE_RUN, run_prefix(tag))
    text = re.sub(r"(?m)^(\s*L_m:)\s*[-+0-9.eE]+", rf"\g<1> {float(L):.1f}", text)
    text = re.sub(r"(?m)^(\s*cv:)\s*[-+0-9.eE]+", rf"\g<1> {cv:.1f}", text)
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
    text = text.replace(anchor, anchor
                        + "    restart_from:\n"
                        + f"      case_dir: {DONOR_CASE}\n"
                        + f"      time_s: {DONOR_TIME_S}\n"
                        + f"    operator_npz: {OPERATOR_NPZ} # Evaluate H.e, do not integrate\n")
    return text


# ------------------------------------------------------------------ inversion


def inversion_config(tpl: dict, stage: str, *, L: int, cv: float, seed: int,
                     n: int, network: str, layout: str, met: dict,
                     nature_blob: dict) -> dict:
    c = copy.deepcopy(tpl)
    tag = tag_of(L, cv, seed)
    base = f"{run_prefix(tag)}_{layout}_n{n}_{network}"
    nature_run = f"{run_prefix(tag)}_surface"
    runs = f"{UP}runs"
    recs = receptors(met["toward_bearing_deg"], LAYOUTS[layout], n, network)

    if stage == "operator":
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
    return c


def generate() -> tuple[int, int]:
    if not MET_JSON.is_file():
        raise FileNotFoundError(
            f"{MET_JSON} is missing. Run derive_bls_met.py on the 2 km donor's "
            "microhh_case first; the sensor geometry depends on its wind.")
    met = json.loads(MET_JSON.read_text())
    NATURE_OUTPUT_DIR.mkdir(exist_ok=True)
    INVERSION_OUTPUT_DIR.mkdir(exist_ok=True)
    nature_template = NATURE_TEMPLATE.read_text()
    nature_blob = yaml.safe_load(nature_template)
    templates = {(n, net, st): yaml.safe_load(
        (INVERSION_TEMPLATE_DIR / f"n{n}_{net}_{st}.yaml").read_text())
        for n in N_VALUES for net in NETWORKS for st in STAGES}

    n_nat = n_inv = 0
    for L, cv, seed in itertools.product(L_VALUES_M, CV_VALUES, SEEDS):
        tag = tag_of(L, cv, seed)
        (NATURE_OUTPUT_DIR / f"les2km_{tag}_wind3.yaml").write_text(
            nature_config(nature_template, L, cv, seed))
        n_nat += 1
        for layout, n, net, st in itertools.product(LAYOUTS, N_VALUES, NETWORKS, STAGES):
            c = inversion_config(templates[(n, net, st)], st, L=L, cv=cv, seed=seed,
                                 n=n, network=net, layout=layout, met=met,
                                 nature_blob=nature_blob)
            (INVERSION_OUTPUT_DIR / f"{tag}_{layout}_n{n}_{net}_{st}.yaml").write_text(
                yaml.safe_dump(c, sort_keys=False))
            n_inv += 1
    return n_nat, n_inv


if __name__ == "__main__":
    nn, ni = generate()
    print(f"Generated {nn} nature and {ni} inversion configs")
    sys.exit(0)
