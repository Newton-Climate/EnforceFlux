#!/usr/bin/env python3
"""Build the LES tagged-tracer operator and validate it against the nature runs.

The rice-paddy sweep currently pays one LES per (L, cv) pair, and would pay one
per emission seed too. Because the emitted scalar is passive, a single LES that
tags every active surface cell as its own tracer yields the full transport
operator ``H``, after which any emission field — any L, any cv, any seed — is a
matrix-vector product.

Whether that is *usable* turns on the scalar limiters, which are nonlinear (see
:mod:`enforceflux.microhh.tagged_operator`). This script measures that error
directly: it reconstructs each existing nature run from ``H`` and its own
surface boundary condition, and compares against what MicroHH actually
produced.

Stages::

    python run_les_tagged_operator.py build     # write the tagged case
    python run_les_tagged_operator.py run       # run MicroHH (the long step)
    python run_les_tagged_operator.py compare   # H·E vs every nature run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from enforceflux.microhh.tagged_operator import (
    apply_operator,
    build_tagged_case,
    read_ini,
    read_operator,
    run_tagged_case,
    surface_field,
    TaggedCase,
)

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
NATURE_GLOB = "source_heterogeneity_les_rice_paddy_*_wind3_surface"
CASE_SUFFIX = Path("dispersion/concentration_microhh/microhh_case")

# The spinup run every realization warm-starts from, and its restart stamp.
DONOR = RUNS / "source_heterogeneity_les_rice_paddy_l200_cv2p0_wind3_surface" / CASE_SUFFIX
DONOR_TIME_S = 3600
# A restarted nature run supplies the operator's grid, forcing, and timing.
TEMPLATE = RUNS / "source_heterogeneity_les_rice_paddy_l100_cv1p0_wind3_surface" / CASE_SUFFIX

OUT = RUNS / "source_heterogeneity_les_tagged_operator"
CASE_DIR = OUT / "microhh_case"
SPEC_PATH = OUT / "tagged_case.json"
OPERATOR_PATH = OUT / "operator.npz"
REPORT_PATH = OUT / "validation.json"

EXECUTABLE = ROOT / "microhh" / "build" / "microhh"
NUM_WORKERS = 4
# The 2 m measurement plane, as MicroHH stamps it into cross-section filenames.
LEVEL_INDEX = 3
SCALAR = "ch4"


def _save_spec(case: TaggedCase) -> None:
    SPEC_PATH.write_text(json.dumps({
        "case_dir": str(case.case_dir),
        "case_name": case.case_name,
        "tracers": list(case.tracers),
        "cells": case.cells.tolist(),
        "reference_kinematic_flux": case.reference_kinematic_flux,
        "grid": list(case.grid),
        "dtype": case.dtype.str,
        "template": str(TEMPLATE),
        "donor": str(DONOR),
        "donor_time_s": DONOR_TIME_S,
    }, indent=2))


def _load_spec() -> TaggedCase:
    d = json.loads(SPEC_PATH.read_text())
    return TaggedCase(
        case_dir=Path(d["case_dir"]),
        case_name=d["case_name"],
        tracers=tuple(d["tracers"]),
        cells=np.asarray(d["cells"], dtype=int),
        reference_kinematic_flux=float(d["reference_kinematic_flux"]),
        grid=tuple(d["grid"]),
        dtype=np.dtype(d["dtype"]),
    )


def stage_build() -> None:
    case = build_tagged_case(TEMPLATE, CASE_DIR, scalar=SCALAR)
    _save_spec(case)
    print(f"built {case.n_tracers} tracers in {case.case_dir}")
    print(f"reference kinematic flux {case.reference_kinematic_flux:.6e}")


def stage_run() -> None:
    case = _load_spec()
    run_tagged_case(
        case,
        executable=EXECUTABLE,
        num_workers=NUM_WORKERS,
        restart_from=DONOR,
        restart_time_s=DONOR_TIME_S,
    )
    print(f"ran {case.n_tracers}-tracer case in {case.case_dir}")


def _nature_cases() -> list[tuple[str, Path]]:
    out = []
    for d in sorted(RUNS.glob(NATURE_GLOB)):
        case = d / CASE_SUFFIX
        if not case.is_dir():
            continue
        keys = read_ini(sorted(case.glob("*.ini"))[0])
        # The donor starts at t=0; only the restarted realizations share the
        # operator's turbulent state and observation window.
        if int(keys["starttime"]) != DONOR_TIME_S:
            continue
        label = d.name.replace("source_heterogeneity_les_rice_paddy_", "").replace(
            "_wind3_surface", "")
        out.append((label, case))
    return out


def _nature_field(case: Path, times: np.ndarray, grid, dtype) -> np.ndarray:
    itot, jtot, _ = grid
    frames = []
    for t in times:
        path = case / f"{SCALAR}.xy.000.{LEVEL_INDEX:05d}.{int(t):07d}"
        if not path.is_file():
            raise FileNotFoundError(f"nature run is missing {path}")
        frames.append(np.fromfile(path, dtype=dtype).reshape(jtot, itot))
    return np.stack(frames).astype(np.float32)


def _receptor_cells(case: TaggedCase) -> list[tuple[int, int]]:
    """Grid cells holding the case's column receptors.

    Domain-wide agreement is not what the inversion consumes — it samples
    these points, and an operator can conserve mass while being biased here.
    """
    keys = read_ini(case.case_dir / f"{case.case_name}.ini")
    xs = [float(v) for v in keys["coordinates[x]"].split(",")]
    ys = [float(v) for v in keys["coordinates[y]"].split(",")]
    itot, jtot, _ = case.grid
    dx = float(keys["xsize"]) / itot
    dy = float(keys["ysize"]) / jtot
    seen: list[tuple[int, int]] = []
    for x, y in zip(xs, ys):
        cell = (int(y // dy), int(x // dx))
        if cell not in seen:
            seen.append(cell)
    return seen


def check_flow_matches(case: TaggedCase, nature: Path) -> dict:
    """Confirm the operator integrated the same flow as the nature runs.

    The operator is only comparable to them if warm-starting from the shared
    donor really did reproduce their turbulence. It should be bitwise: the
    emitted scalars are passive, and the adaptive timestep reads only the
    velocities and `evisc`, so carrying 196 tracers instead of 2 must not move
    a single flow value. If this ever fails, differences at the receptor are
    flow mismatch and the validation below means nothing.
    """
    from netCDF4 import Dataset

    t_path = case.case_dir / f"{case.case_name}.default.{DONOR_TIME_S:07d}.nc"
    n_path = nature / f"{case.case_name}.default.{DONOR_TIME_S:07d}.nc"
    try:
        probe = Dataset(t_path)
    except OSError as exc:
        # MicroHH holds the statistics file open for the length of the run, so
        # a mid-run `compare` cannot read it. That is not a failed check.
        return {"status": "unreadable", "detail": str(exc)}
    probe.close()

    with Dataset(t_path) as dt_, Dataset(n_path) as dn:
        gt, gn = dt_.groups["default"], dn.groups["default"]
        nt = min(len(dt_["time"][:]), len(dn["time"][:]))
        shared = [
            v for v in gn.variables
            if v in gt.variables and not v.startswith(SCALAR)
        ]
        mismatched = []
        for v in shared:
            a, b = np.asarray(gt[v][:nt]), np.asarray(gn[v][:nt])
            if a.shape != b.shape or not np.array_equal(
                np.nan_to_num(a), np.nan_to_num(b)
            ):
                mismatched.append(v)
        same_time = bool(np.array_equal(dt_["time"][:nt], dn["time"][:nt]))
    return {
        "status": "checked",
        "n_records": int(nt),
        "n_flow_vars": len(shared),
        "n_mismatched": len(mismatched),
        "mismatched": mismatched,
        "time_identical": same_time,
    }


def stage_compare() -> None:
    case = _load_spec()
    times, H = read_operator(case, level_index=LEVEL_INDEX, since_s=DONOR_TIME_S)
    np.savez_compressed(
        OPERATOR_PATH, times_s=times, H=H, cells=case.cells,
        reference_kinematic_flux=case.reference_kinematic_flux,
        tracers=np.array(case.tracers),
        # The plane H lives on. Without it a consumer cannot know which
        # cross-section it is entitled to reconstruct.
        level_index=LEVEL_INDEX,
    )
    print(f"operator H {H.shape} (n_time, jtot, itot, n_tracer) -> {OPERATOR_PATH}")

    print(f"window t={times[0]}..{times[-1]}s ({len(times)} frames)")
    if int(times[0]) == DONOR_TIME_S:
        print(
            "WARNING: the operator has a frame at the restart stamp itself. "
            "MicroHH does not write one, so this is donor output copied in by "
            "the pre-fix _seed_restart."
        )

    receptors = _receptor_cells(case)
    print(f"receptor cells (j, i): {receptors}")

    flow = check_flow_matches(case, _nature_cases()[0][1])
    if flow.get("status") == "unreadable":
        print("flow check: SKIPPED (statistics file still held open by a "
              "running MicroHH; re-run `compare` once the run finishes)")
    elif flow["n_mismatched"] or not flow["time_identical"]:
        print(f"WARNING: flow differs from the nature runs "
              f"({flow['n_mismatched']}/{flow['n_flow_vars']} vars, "
              f"time_identical={flow['time_identical']}): {flow['mismatched'][:5]}. "
              f"Receptor differences below are flow mismatch, not operator error.")
    else:
        print(f"flow check: {flow['n_flow_vars']}/{flow['n_flow_vars']} variables "
              f"bitwise identical to the nature runs over {flow['n_records']} records")

    rows = []
    for label, nature in _nature_cases():
        bot = surface_field(nature, SCALAR, case.grid, case.dtype)
        pred = apply_operator(H, bot, case.cells)
        truth = _nature_field(nature, times, case.grid, case.dtype)

        # Metrics must be mass-weighted. The `[limiter]` positivity clip floors
        # every tracer at ~2.2e-16 independently, so summing n tracers raises
        # the far-field floor n-fold. Against a ~1e-8 signal that is physically
        # nothing, but it makes an unweighted relative error meaningless: the
        # empty cells report ~n*100% while carrying no mass at all.
        scale = float(truth.max())
        resid = pred - truth
        # Cells holding the bulk of the mass, found by descending concentration.
        order = np.argsort(truth.ravel())[::-1]
        cumulative = np.cumsum(truth.ravel()[order])
        keep = order[cumulative <= 0.99 * truth.sum()]
        rel_bulk = np.abs(resid.ravel()[keep]) / truth.ravel()[keep]
        rows.append({
            "case": label,
            "n_time": int(len(times)),
            "truth_max": scale,
            # L1 error over total mass: the headline, immune to empty cells.
            "mass_weighted_rel_err": float(np.abs(resid).sum() / truth.sum()),
            "rmse_rel_to_max": float(np.sqrt(np.mean(resid**2)) / scale),
            "max_abs_err": float(np.abs(resid).max()),
            "mean_bias_rel_to_max": float(resid.mean() / scale),
            # Over the cells carrying the first 99% of mass.
            "bulk_median_rel_err": float(np.median(rel_bulk)),
            "bulk_p95_rel_err": float(np.percentile(rel_bulk, 95)),
            "bulk_n_cells": int(keep.size),
            "total_mass_ratio": float(pred.sum() / truth.sum()),
            # What the inversion actually ingests. Reported as a signed bias
            # too: a systematic offset at the receptor propagates straight
            # into the retrieved flux, where scatter would largely average out.
            "receptor_rel_err": {
                f"j{j}_i{i}": float(
                    np.abs(pred[:, j, i] - truth[:, j, i]).sum() / truth[:, j, i].sum()
                )
                for j, i in receptors
            },
            "receptor_rel_bias": {
                f"j{j}_i{i}": float(
                    (pred[:, j, i] - truth[:, j, i]).sum() / truth[:, j, i].sum()
                )
                for j, i in receptors
            },
            # Superposition error should not grow without bound as the tracers
            # disperse; a rising profile would mean the limiters compound.
            "per_frame_rel_err": [
                float(np.abs(resid[t]).sum() / truth[t].sum())
                if truth[t].sum() > 0 else float("nan")
                for t in range(len(times))
            ],
        })

        # The pre-fix `_seed_restart` left the donor's field at the restart
        # stamp. It sits outside the operator's window, but say so explicitly
        # rather than letting a silent exclusion look like agreement.
        stale = nature / f"{SCALAR}.xy.000.{LEVEL_INDEX:05d}.{DONOR_TIME_S:07d}"
        if stale.is_file() and int(times[0]) > DONOR_TIME_S:
            rows[-1]["excluded_donor_frame_s"] = DONOR_TIME_S

    REPORT_PATH.write_text(json.dumps(rows, indent=2))
    _print_report(rows)


def _print_report(rows: list[dict]) -> None:
    hdr = (f"{'case':14s} {'massw rel':>10s} {'mass ratio':>11s} "
           f"{'recept err':>11s} {'recept bias':>12s}")
    print()
    print("mass-weighted rel err = sum|H.E - truth| / sum(truth); "
          "'bulk' = cells holding the first 99% of mass")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        err = np.mean(list(r['receptor_rel_err'].values()))
        bias = np.mean(list(r['receptor_rel_bias'].values()))
        print(f"{r['case']:14s} {r['mass_weighted_rel_err']:10.3%} "
              f"{r['total_mass_ratio']:11.6f} {err:11.3%} {bias:+12.3%}")
    print()
    print(f"wrote {REPORT_PATH}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("build", "run", "compare"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"build": stage_build, "run": stage_run, "compare": stage_compare}[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
