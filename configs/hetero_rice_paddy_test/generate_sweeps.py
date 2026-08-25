#!/usr/bin/env python3
"""Generate the rice-paddy L, CV, seed, and sensor-count sweep configs.

Each (L, CV) pair is drawn at several RNG seeds. A seeded nature run is not
integrated: it is evaluated through the tagged-tracer LES operator (see
``notebooks/hetero_experiments/run_les_tagged_operator.py``), which turns an
hours-long LES into a matrix-vector product and makes the seed dimension
affordable. The generated case is otherwise byte-for-byte the case that would
have been integrated, so dropping ``microhh.operator_npz`` recovers the real
LES for any realization worth checking.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent
NATURE_TEMPLATE = CONFIG_DIR / "les_l200_cv2p0_wind3.yaml"
INVERSION_TEMPLATE_DIR = CONFIG_DIR / "inversions"
NATURE_OUTPUT_DIR = CONFIG_DIR / "nature_sweep"
INVERSION_OUTPUT_DIR = CONFIG_DIR / "inversion_sweep"

L_VALUES_M = (100, 250, 500)
CV_VALUES = (0.5, 1.0, 2.0)
# Emission-field realizations per (L, CV). Every seed shares one turbulence
# realization — the operator's — so these sample source-field randomness only.
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
N_VALUES = (1, 2, 3, 4)
NETWORKS = ("op", "point")
STAGES = ("operator", "instrument", "flux", "analysis")

BASE_RUN = "source_heterogeneity_les_rice_paddy_l200_cv2p0_wind3"

# Relative to configs/hetero_rice_paddy_test/nature_sweep/.
OPERATOR_NPZ = "../../../runs/source_heterogeneity_les_tagged_operator/operator.npz"


def cv_tag(cv: float) -> str:
    return f"{cv:.1f}".replace(".", "p")


def parameterized(text: str, *, length_m: int, cv: float, seed: int,
                  run_prefix: str) -> str:
    """Update the parameters and all run references without reformatting YAML."""
    text = text.replace(BASE_RUN, run_prefix)
    text = re.sub(
        r"(?m)^(\s*L_m:)\s*[-+0-9.eE]+(\s*(?:#.*)?)$",
        rf"\g<1> {float(length_m):.1f}\g<2>",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*cv:)\s*[-+0-9.eE]+(\s*(?:#.*)?)$",
        rf"\g<1> {cv:.1f}\g<2>",
        text,
    )
    # Only the source-field seed, whose template value is 0. The bLS particle
    # seed sits at the same indent but holds a distinct nonzero value, so
    # matching on the value keeps this from rewriting the wrong key.
    text = re.sub(
        r"(?m)^(\s*seed:)\s*0(\s*(?:#.*)?)$",
        rf"\g<1> {seed:d}\g<2>",
        text,
    )
    return text


def generate() -> tuple[int, int]:
    NATURE_OUTPUT_DIR.mkdir(exist_ok=True)
    INVERSION_OUTPUT_DIR.mkdir(exist_ok=True)

    nature_template = NATURE_TEMPLATE.read_text()
    inversion_templates = {
        (n, network, stage): (
            INVERSION_TEMPLATE_DIR / f"n{n}_{network}_{stage}.yaml"
        ).read_text()
        for n in N_VALUES
        for network in NETWORKS
        for stage in STAGES
    }

    nature_count = 0
    inversion_count = 0
    for length_m, cv, seed in itertools.product(L_VALUES_M, CV_VALUES, SEEDS):
        tag = f"l{length_m}_cv{cv_tag(cv)}_s{seed}"
        run_prefix = f"source_heterogeneity_les_rice_paddy_{tag}_wind3"

        nature = parameterized(
            nature_template,
            length_m=length_m,
            cv=cv,
            seed=seed,
            run_prefix=run_prefix,
        )
        # Generated configs are one directory deeper than the template.
        nature = nature.replace("meteo_dir: ../../runs/", "meteo_dir: ../../../runs/")
        nature = nature.replace(
            "executable: ../../microhh/", "executable: ../../../microhh/"
        )
        nature = nature.replace(
            "    spinup_seconds: 2700 # Spinup period, s",
            "    spinup_seconds: 0 # Reuse the completed LES turbulent restart",
        )
        nature = nature.replace(
            "    column_sampletime: 0.1 # Column met interval, s",
            "    column_sampletime: 20 # Donor run already contains 10 Hz wind columns",
        )
        nature = nature.replace(
            "        swdump: 1 # Enable dumps",
            "        swdump: 0 # Donor run already contains the wind-field dumps",
        )
        restart = (
            "    restart_from:\n"
            "      case_dir: ../../../runs/source_heterogeneity_les_rice_paddy_"
            "l200_cv2p0_wind3_surface/dispersion/concentration_microhh/microhh_case\n"
            "      time_s: 3600\n"
        )
        nature = nature.replace(
            "    num_workers: 4 # Parallel workers\n",
            "    num_workers: 4 # Parallel workers\n" + restart,
        )
        # Evaluate through the tagged-tracer operator instead of
        # integrating. The rest of the case is unchanged, so deleting this
        # one key runs the realization as a real LES.
        nature = nature.replace(
            "    num_workers: 4 # Parallel workers\n",
            "    num_workers: 4 # Parallel workers\n"
            f"    operator_npz: {OPERATOR_NPZ} # Evaluate H.e, do not integrate\n",
        )
        (NATURE_OUTPUT_DIR / f"les_{tag}_wind3.yaml").write_text(nature)
        nature_count += 1

        for n in N_VALUES:
            for network in NETWORKS:
                for stage in STAGES:
                    inversion = parameterized(
                        inversion_templates[(n, network, stage)],
                        length_m=length_m,
                        cv=cv,
                        seed=seed,
                        run_prefix=run_prefix,
                    )
                    filename = f"{tag}_n{n}_{network}_{stage}.yaml"
                    (INVERSION_OUTPUT_DIR / filename).write_text(inversion)
                    inversion_count += 1

    return nature_count, inversion_count


if __name__ == "__main__":
    nature_count, inversion_count = generate()
    print(f"Generated {nature_count} nature and {inversion_count} inversion configs")
