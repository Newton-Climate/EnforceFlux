#!/usr/bin/env python3
"""Run the generated 2 km rice-paddy nature and inversion suites in order.

Same logic as configs/hetero_rice_paddy_test/run_sweeps.py, over the 2 km
names and both sensor layouts. The bLS Jacobian depends only on sensor
geometry and met, so it is built once per (layout, n, network) — 16 builds —
and copied into every other (L, CV, seed).

    python run_sweeps.py --bls-donors        # the 16 bLS builds (needs R/bLSmodelR)
    python run_sweeps.py --nature            # H.e for every field (needs operator.npz)
    python run_sweeps.py --inversions        # instrument/flux/analysis, reusing H

Completed stages (manifest present) are skipped, so any mode can be resumed.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from generate_sweeps import (
    CV_VALUES, L_VALUES_M, LAYOUTS, N_VALUES, NETWORKS, SEEDS, run_prefix, tag_of,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent
DONOR_TAG = tag_of(L_VALUES_M[0], CV_VALUES[0], SEEDS[0])


SHARD = (0, 1)  # (index, count); set from --shard/--nshards


def tags() -> list[str]:
    """This shard's (L, CV, seed) tags, donor first.

    Shards split by tag, so no two shards touch the same run directory. The
    donor tag always lands in shard 0; other shards only read its bLS runs.
    """
    all_tags = [tag_of(L, cv, s) for L, cv, s in
                itertools.product(L_VALUES_M, CV_VALUES, SEEDS)]
    ordered = sorted(all_tags, key=lambda t: t != DONOR_TAG)
    k, n = SHARD
    return ordered[k::n]


def complete(config: Path) -> bool:
    blob = yaml.safe_load(config.read_text())
    return (ROOT / "runs" / blob["run"]["name"] / blob["stage"] / "manifest.json").is_file()


def invoke(command: str, config: Path) -> None:
    print(f"\n>>> enforceflux {command} --config {config}", flush=True)
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


def inv_config(tag: str, layout: str, n: int, network: str, stage: str) -> Path:
    return CONFIG_DIR / "inversion_sweep" / f"{tag}_{layout}_n{n}_{network}_{stage}.yaml"


def run_bls_donors() -> None:
    for layout, n, network in itertools.product(LAYOUTS, N_VALUES, NETWORKS):
        config = inv_config(DONOR_TAG, layout, n, network, "operator")
        if complete(config):
            print(f"skip completed: {config.name}", flush=True)
            continue
        invoke("dispersion", config)


def run_nature() -> None:
    for tag in tags():
        config = CONFIG_DIR / "nature_sweep" / f"les2km_{tag}_wind3.yaml"
        if complete(config):
            print(f"skip completed: {config.name}", flush=True)
            continue
        invoke("dispersion", config)


def run_inversions() -> None:
    stages = (("operator", "dispersion"), ("instrument", "instrument"),
              ("flux", "flux"), ("analysis", "analysis"))
    for tag in tags():
        for layout, n, network in itertools.product(LAYOUTS, N_VALUES, NETWORKS):
            for stage, command in stages:
                config = inv_config(tag, layout, n, network, stage)
                if complete(config):
                    print(f"skip completed: {config.name}", flush=True)
                    continue
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bls-donors", action="store_true")
    ap.add_argument("--nature", action="store_true")
    ap.add_argument("--inversions", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if not 0 <= a.shard < a.nshards:
        ap.error("need 0 <= --shard < --nshards")
    global SHARD
    SHARD = (a.shard, a.nshards)
    if not (a.bls_donors or a.nature or a.inversions):
        ap.error("select --bls-donors, --nature, --inversions, or a combination")
    if a.bls_donors:
        run_bls_donors()
    if a.nature:
        run_nature()
    if a.inversions:
        run_inversions()


if __name__ == "__main__":
    main()
