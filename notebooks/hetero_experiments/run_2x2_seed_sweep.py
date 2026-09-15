#!/usr/bin/env python3
"""2x2 over state resolution and observation cadence, across emission seeds.

The design crosses one-cell against nine-cell state with window-averaged
against time-resolved observations, at L=100 m, CV=1.0, n=4 open paths. One
realization cannot separate a real interaction from a lucky draw, so every
cell is run at eight emission seeds.

Both bLS operators are emission-field independent — they depend on receptor
geometry and turbulence, neither of which varies with the seed — so the
hour-long time-resolved build is done once and copied, exactly as
``configs/hetero_rice_paddy_test/sweep.py`` already does for the window-averaged one.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "hetero_rice_paddy_test" / "time_resolved"
SEEDS = range(8)
CASES = ("v_ctl1", "v_tr1", "v_ctl", "v_tr")
# Where the one built time-resolved operator lives.
DONOR_SEED = 0
PREFIX = "source_heterogeneity_les_rice_paddy_l100_cv1p0"


def run_name(seed: int, suffix: str) -> str:
    return f"{PREFIX}_s{seed}_wind3_n4_op{suffix}"


def materialize_time_resolved_operator(seed: int) -> None:
    """Copy the built Jacobian next to this seed's own truth files."""
    from enforceflux.runs import open_run_dir

    target = ROOT / "runs" / run_name(seed, "_tr_gp") / "dispersion"
    if (target / "manifest.json").is_file():
        print(f"skip operator seed {seed}: already materialized", flush=True)
        return

    donor = ROOT / "runs" / run_name(DONOR_SEED, "_tr_gp") / "dispersion"
    if not (donor / "jacobian.npz").is_file():
        raise FileNotFoundError(
            f"time-resolved operator not built yet: {donor/'jacobian.npz'}"
        )
    nature = ROOT / "runs" / f"{PREFIX}_s{seed}_wind3_surface" / "dispersion"
    for name in ("truth_field.nc", "basis_mapping.npz"):
        if not (nature / name).is_file():
            raise FileNotFoundError(f"nature artifact missing: {nature/name}")

    run_dir = open_run_dir(
        stage="dispersion", run_name=run_name(seed, "_tr_gp"),
        outputs_root=ROOT / "runs", inputs={},
    )
    for name, role in (("truth_field.nc", "truth_field"),
                       ("basis_mapping.npz", "basis_mapping")):
        shutil.copy2(nature / name, run_dir.path(name))
        run_dir.record_output(name, role=role)
    shutil.copy2(donor / "jacobian.npz", run_dir.path("jacobian.npz"))
    run_dir.record_output("jacobian.npz", role="jacobian")
    run_dir.add_manifest_field("operator_cache_source", str(donor))
    run_dir.finalize()
    print(f"cached time-resolved operator: seed {seed} <- {donor.parent.name}",
          flush=True)


def write_config(seed: int, case: str) -> Path:
    """Retarget the seed-0 template at this seed's runs."""
    template = CONFIG_DIR / f"l100_cv1p0_s{DONOR_SEED}_n4_op_{case}_flux.yaml"
    text = template.read_text()
    # Every run reference moves to this seed; the observation file is the
    # sweep's own instrument run, which already holds all 45 frames.
    text = text.replace(f"_s{DONOR_SEED}_wind3", f"_s{seed}_wind3")
    text = text.replace("_n4_op_tr/instrument", "_n4_op/instrument")
    out = CONFIG_DIR / f"l100_cv1p0_s{seed}_n4_op_{case}_flux.yaml"
    out.write_text(text)
    return out


def invoke(config: Path) -> None:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/enforceflux-mpl")
    subprocess.run(["enforceflux", "flux", "--config", str(config)],
                   cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args()

    for seed in args.seeds:
        materialize_time_resolved_operator(seed)
        for case in CASES:
            config = write_config(seed, case)
            invoke(config)
            print(f"seed {seed} {case}: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
