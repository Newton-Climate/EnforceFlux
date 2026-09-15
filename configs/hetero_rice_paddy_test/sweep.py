#!/usr/bin/env python3
"""Run the rice-paddy L, CV, seed, and sensor-count sweep from code.

The grid lives here, not in thousands of YAML files. Every stage config is
rendered in memory from the two templates (``les_l200_cv2p0_wind3.yaml`` and
``inversions/``), written to a throwaway file for the length of one
``enforceflux`` call, and deleted. Each run directory keeps the resolved config
as ``config.snapshot.yaml``, so nothing is lost by not keeping the files.

    python sweep.py --nature --inversions                      # the full grid below
    python sweep.py --nature --inversions --shard 0 --nshards 4
    python sweep.py --nature --L 150,300 --cv 0.75 --seeds 0-19
    python sweep.py --write /tmp/cfgs --L 250 --cv 1.0 --seeds 3  # inspect or run by hand
    python sweep.py --count                                    # what the grid holds, and what is done

A seeded nature run is not integrated: it is evaluated through the tagged-tracer
LES operator (``notebooks/hetero_experiments/run_les_tagged_operator.py``),
which turns an hours-long LES into a matrix-vector product. The rendered case is
otherwise the case that would have been integrated, so dropping
``microhh.operator_npz`` recovers the real LES for any realization worth checking.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent
NATURE_TEMPLATE = CONFIG_DIR / "les_l200_cv2p0_wind3.yaml"
INVERSION_TEMPLATE_DIR = CONFIG_DIR / "inversions"

# The original 3 x 3 (L = 100/250/500, CV = 0.5/1/2) plus the points between.
# L >= 350 m takes the generator's padded-Cholesky fallback on this 25 x 25,
# 40 m grid (circulant embedding is not PSD there), so seed k maps to a
# different field above that line than below it.
L_VALUES_M = (100, 150, 200, 250, 300, 400, 500)
CV_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
# Emission-field realizations per (L, CV). Every seed shares one turbulence
# realization — the operator's — so these sample source-field randomness only.
SEEDS = tuple(range(20))
N_VALUES = (1, 2, 3, 4)
NETWORKS = ("op", "point")
STAGES = (
    ("operator", "dispersion"),
    ("instrument", "instrument"),
    ("flux", "flux"),
    ("analysis", "analysis"),
)

BASE_RUN = "source_heterogeneity_les_rice_paddy_l200_cv2p0_wind3"
PREFIX = "source_heterogeneity_les_rice_paddy"
# Rendered configs sit one directory below CONFIG_DIR, like the old
# nature_sweep/ and inversion_sweep/, so these relative paths are unchanged.
OPERATOR_NPZ = "../../../runs/source_heterogeneity_les_tagged_operator/operator.npz"

# The one combination whose bLS operator is built for real. The Jacobian
# depends on the receptor geometry and the met intervals, and neither varies
# with L, CV, or seed, so every other combination reuses it.
DONOR_TAG = "l100_cv0p5_s0"
# The same combination as built by the sweep that predates the seed dimension.
PRE_SEED_DONOR_TAG = "l100_cv0p5"


def cv_text(cv: float) -> str:
    """CV as written into configs: one decimal when exact (0.5, 1.0), else in full.

    One decimal keeps the original grid's run names and configs unchanged;
    a fixed one-decimal format would silently turn 0.75 into 0.8.
    """
    return f"{cv:.1f}" if round(cv, 1) == cv else f"{cv:g}"


def cv_tag(cv: float) -> str:
    return cv_text(cv).replace(".", "p")


def tag_of(length_m: int, cv: float, seed: int) -> str:
    return f"l{length_m}_cv{cv_tag(cv)}_s{seed}"


def run_prefix(tag: str) -> str:
    return f"{PREFIX}_{tag}_wind3"


def parameterized(text: str, *, length_m: int, cv: float, seed: int,
                  prefix: str) -> str:
    """Update the parameters and all run references without reformatting YAML."""
    text = text.replace(BASE_RUN, prefix)
    text = re.sub(
        r"(?m)^(\s*L_m:)\s*[-+0-9.eE]+(\s*(?:#.*)?)$",
        rf"\g<1> {float(length_m):.1f}\g<2>",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*cv:)\s*[-+0-9.eE]+(\s*(?:#.*)?)$",
        rf"\g<1> {cv_text(cv)}\g<2>",
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


def _replace(text: str, old: str, new: str) -> str:
    """str.replace that fails if the template no longer contains ``old``."""
    if old not in text:
        raise ValueError(f"nature template no longer contains {old!r}")
    return text.replace(old, new)


def render_nature(length_m: int, cv: float, seed: int) -> str:
    tag = tag_of(length_m, cv, seed)
    text = parameterized(NATURE_TEMPLATE.read_text(), length_m=length_m, cv=cv,
                         seed=seed, prefix=run_prefix(tag))
    # Rendered configs are one directory deeper than the template.
    text = _replace(text, "meteo_dir: ../../runs/", "meteo_dir: ../../../runs/")
    text = _replace(text, "executable: ../../microhh/", "executable: ../../../microhh/")
    text = _replace(
        text,
        "    spinup_seconds: 2700 # Spinup period, s",
        "    spinup_seconds: 0 # Reuse the completed LES turbulent restart",
    )
    text = _replace(
        text,
        "    column_sampletime: 0.1 # Column met interval, s",
        "    column_sampletime: 20 # Donor run already contains 10 Hz wind columns",
    )
    text = _replace(
        text,
        "        swdump: 1 # Enable dumps",
        "        swdump: 0 # Donor run already contains the wind-field dumps",
    )
    restart = (
        "    restart_from:\n"
        "      case_dir: ../../../runs/source_heterogeneity_les_rice_paddy_"
        "l200_cv2p0_wind3_surface/dispersion/concentration_microhh/microhh_case\n"
        "      time_s: 3600\n"
    )
    workers = "    num_workers: 4 # Parallel workers\n"
    text = _replace(text, workers, workers + restart)
    # Evaluate through the tagged-tracer operator instead of integrating. The
    # rest of the case is unchanged, so deleting this one key runs the
    # realization as a real LES.
    return _replace(
        text, workers,
        workers + f"    operator_npz: {OPERATOR_NPZ} # Evaluate H.e, do not integrate\n",
    )


def render_inversion(length_m: int, cv: float, seed: int, n: int, network: str,
                     stage: str) -> str:
    template = INVERSION_TEMPLATE_DIR / f"n{n}_{network}_{stage}.yaml"
    return parameterized(template.read_text(), length_m=length_m, cv=cv, seed=seed,
                         prefix=run_prefix(tag_of(length_m, cv, seed)))


# ------------------------------------------------------------------ running


def complete(text: str) -> bool:
    blob = yaml.safe_load(text)
    manifest = ROOT / "runs" / blob["run"]["name"] / blob["stage"] / "manifest.json"
    return manifest.is_file()


class Renderer:
    """Writes one rendered config at a time to a private scratch directory."""

    def __init__(self) -> None:
        # Inside CONFIG_DIR, so relative paths in the configs resolve as before.
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
    still exactly right. Returns None when nothing is built yet, which is the
    signal to run the donor combination for real.
    """
    for tag in (DONOR_TAG, PRE_SEED_DONOR_TAG):
        donor = ROOT / "runs" / f"{PREFIX}_{tag}_wind3_n{n}_{network}_gp" / "dispersion"
        if (donor / "manifest.json").is_file():
            return donor
    return None


def materialize_cached_operator(config: Path, *, tag: str, donor: Path) -> None:
    """Reuse H across source statistics while retaining matching truth files."""
    from enforceflux.runs import load_stage_config, open_run_dir

    nature = ROOT / "runs" / f"{run_prefix(tag)}_surface" / "dispersion"
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


def cases(ls, cvs, seeds, shard: tuple[int, int]) -> list[tuple[int, float, int]]:
    """This shard's (L, CV, seed) combinations, donor first.

    Shards split by combination, so no two shards touch the same run directory.
    The donor always lands in shard 0; other shards only read its bLS runs.
    """
    combos = sorted(itertools.product(ls, cvs, seeds),
                    key=lambda c: tag_of(*c) != DONOR_TAG)
    k, n = shard
    return combos[k::n]


def run_nature(combos, renderer: Renderer) -> None:
    for length_m, cv, seed in combos:
        text = render_nature(length_m, cv, seed)
        name = f"les_{tag_of(length_m, cv, seed)}_wind3.yaml"
        if complete(text):
            print(f"skip completed: {name}", flush=True)
            continue
        invoke("dispersion", renderer.path(name, text))


def run_inversions(combos, renderer: Renderer) -> None:
    for length_m, cv, seed in combos:
        tag = tag_of(length_m, cv, seed)
        for n, network in itertools.product(N_VALUES, NETWORKS):
            for stage, command in STAGES:
                text = render_inversion(length_m, cv, seed, n, network, stage)
                name = f"{tag}_n{n}_{network}_{stage}.yaml"
                if complete(text):
                    print(f"skip completed: {name}", flush=True)
                    continue
                config = renderer.path(name, text)
                if stage == "operator":
                    donor = donor_operator(n, network)
                    if donor is not None:
                        materialize_cached_operator(config, tag=tag, donor=donor)
                        continue
                    if tag != DONOR_TAG:
                        raise FileNotFoundError(
                            f"no bLS operator to copy for {name}; "
                            f"run the {DONOR_TAG} combination first"
                        )
                invoke(command, config)


def write_configs(combos, out: Path) -> int:
    """Materialize the rendered configs, laid out as the old generator did."""
    (out / "nature_sweep").mkdir(parents=True, exist_ok=True)
    (out / "inversion_sweep").mkdir(parents=True, exist_ok=True)
    count = 0
    for length_m, cv, seed in combos:
        tag = tag_of(length_m, cv, seed)
        (out / "nature_sweep" / f"les_{tag}_wind3.yaml").write_text(
            render_nature(length_m, cv, seed))
        count += 1
        for n, network in itertools.product(N_VALUES, NETWORKS):
            for stage, _ in STAGES:
                (out / "inversion_sweep" / f"{tag}_n{n}_{network}_{stage}.yaml").write_text(
                    render_inversion(length_m, cv, seed, n, network, stage))
                count += 1
    return count


def report(combos) -> None:
    nature_done = sum(complete(render_nature(*c)) for c in combos)
    inv_done = sum(
        complete(render_inversion(*c, n, network, "analysis"))
        for c in combos for n, network in itertools.product(N_VALUES, NETWORKS)
    )
    n_inv = len(combos) * len(N_VALUES) * len(NETWORKS)
    print(f"{len(combos)} source fields: {nature_done} done")
    print(f"{n_inv} inversions: {inv_done} done")


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
    ap.add_argument("--nature", action="store_true", help="evaluate the source fields")
    ap.add_argument("--inversions", action="store_true", help="run the four inversion stages")
    ap.add_argument("--count", action="store_true", help="report grid size and completion")
    ap.add_argument("--write", type=Path, metavar="DIR", help="write rendered configs to DIR")
    ap.add_argument("--L", type=_floats, default=L_VALUES_M, help="comma-separated, m")
    ap.add_argument("--cv", type=_floats, default=CV_VALUES, help="comma-separated")
    ap.add_argument("--seeds", type=_seeds, default=SEEDS, help="e.g. 0-19 or 0,3,5-7")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if not (a.nature or a.inversions or a.count or a.write):
        ap.error("select --nature, --inversions, --count, or --write")
    if not 0 <= a.shard < a.nshards:
        ap.error("need 0 <= --shard < --nshards")

    ls = tuple(int(v) for v in a.L)
    combos = cases(ls, a.cv, a.seeds, (a.shard, a.nshards))
    if a.write:
        print(f"wrote {write_configs(combos, a.write)} configs to {a.write}")
    if a.count:
        report(combos)
    if a.nature or a.inversions:
        renderer = Renderer()
        try:
            if a.nature:
                run_nature(combos, renderer)
            if a.inversions:
                run_inversions(combos, renderer)
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
