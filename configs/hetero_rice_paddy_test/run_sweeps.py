#!/usr/bin/env python3
"""Run the generated rice-paddy nature and inversion suites in order."""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import yaml

# Sibling module; importable because a script's own directory leads sys.path.
from generate_sweeps import CV_VALUES, L_VALUES_M, N_VALUES, NETWORKS, SEEDS, cv_tag


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent
# The one combination whose bLS operator is built for real. The Jacobian
# depends on the receptor geometry and the met intervals, and neither varies
# with L, CV, or seed, so every other combination reuses it.
DONOR_TAG = f"l{L_VALUES_M[0]}_cv{cv_tag(CV_VALUES[0])}_s{SEEDS[0]}"
# The same combination as built by the sweep that predates the seed dimension.
PRE_SEED_DONOR_TAG = f"l{L_VALUES_M[0]}_cv{cv_tag(CV_VALUES[0])}"


def tags() -> list[str]:
    """Every (L, CV, seed) combination, donor first."""
    all_tags = [
        f"l{length_m}_cv{cv_tag(cv)}_s{seed}"
        for length_m, cv, seed in itertools.product(L_VALUES_M, CV_VALUES, SEEDS)
    ]
    return sorted(all_tags, key=lambda tag: tag != DONOR_TAG)


def complete(config: Path) -> bool:
    blob = yaml.safe_load(config.read_text())
    manifest = ROOT / "runs" / blob["run"]["name"] / blob["stage"] / "manifest.json"
    return manifest.is_file()


def invoke(command: str, config: Path) -> None:
    print(f"\n>>> enforceflux {command} --config {config}", flush=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/enforceflux-mpl")
    subprocess.run(
        ["enforceflux", command, "--config", str(config)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def donor_operator(n: int, network: str) -> Path | None:
    """The completed bLS operator run this network can copy H from, if any.

    The Jacobian depends on the receptor geometry and the met intervals, and
    neither varies with L, CV, or seed — so the pre-seed sweep's operator is
    still exactly right, and preferring it keeps the seed sweep from rebuilding
    an identical H. Returns None when nothing is built yet, which is the signal
    to run the donor combination for real.
    """
    prefix = "source_heterogeneity_les_rice_paddy"
    for tag in (DONOR_TAG, PRE_SEED_DONOR_TAG):
        donor = ROOT / "runs" / f"{prefix}_{tag}_wind3_n{n}_{network}_gp" / "dispersion"
        if (donor / "manifest.json").is_file():
            return donor
    return None


def materialize_cached_operator(
    config: Path, *, tag: str, donor: Path
) -> None:
    """Reuse H across source statistics while retaining matching truth files."""
    from enforceflux.runs import load_stage_config, open_run_dir

    nature_name = f"source_heterogeneity_les_rice_paddy_{tag}_wind3_surface"
    nature = ROOT / "runs" / nature_name / "dispersion"
    for name in ("truth_field.nc", "basis_mapping.npz"):
        if not (nature / name).is_file():
            raise FileNotFoundError(f"nature artifact is missing: {nature / name}")

    stage_cfg = load_stage_config(config, expected_stage="dispersion")
    run_dir = open_run_dir(
        stage="dispersion",
        run_name=stage_cfg.run_name,
        outputs_root=stage_cfg.outputs_root,
        inputs={},
    )
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


def run_nature() -> None:
    for tag in tags():
        config = CONFIG_DIR / "nature_sweep" / f"les_{tag}_wind3.yaml"
        if complete(config):
            print(f"skip completed: {config.name}", flush=True)
            continue
        invoke("dispersion", config)


def run_inversions() -> None:
    stages = (
        ("operator", "dispersion"),
        ("instrument", "instrument"),
        ("flux", "flux"),
        ("analysis", "analysis"),
    )
    directory = CONFIG_DIR / "inversion_sweep"
    for tag in tags():
        for n in N_VALUES:
            for network in NETWORKS:
                for stage, command in stages:
                    config = directory / f"{tag}_n{n}_{network}_{stage}.yaml"
                    if complete(config):
                        print(f"skip completed: {config.name}", flush=True)
                        continue
                    if stage == "operator":
                        donor = donor_operator(n, network)
                        if donor is not None:
                            materialize_cached_operator(
                                config, tag=tag, donor=donor
                            )
                            continue
                        if tag != DONOR_TAG:
                            raise FileNotFoundError(
                                f"no bLS operator to copy for {config.name}; "
                                f"run the {DONOR_TAG} combination first"
                            )
                    invoke(command, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nature", action="store_true")
    parser.add_argument("--inversions", action="store_true")
    args = parser.parse_args()
    if not args.nature and not args.inversions:
        parser.error("select --nature, --inversions, or both")
    if args.nature:
        run_nature()
    if args.inversions:
        run_inversions()


if __name__ == "__main__":
    main()
