#!/usr/bin/env python3
"""The fine CV x L mesh behind the phase diagram (``fig1_response_surface``).

The grid is read from ``configs/hetero_rice_paddy_test/sweep.py``, so the table
follows whatever that sweep runs. Every value comes from ``runs/``:

* ``e_signed``: total-flux error, ``(x_opt - Q_true) / Q_true``, as ``build_table.py``
  defines it;
* ``e_pure``: the perfect-transport error of the same field, ``AK dw . phi + (AK - 1)``;
* ``sd_pred_loglinear``: the closed-form ``sqrt(AK^2 dw^T Sigma dw)`` with
  ``Sigma = ln(1 + CV^2) exp(-r/L)`` (``../rep_error_quadratic/quadratic_rep.py``).

    python phase_mesh.py                     # fails unless every mesh run is complete
    python phase_mesh.py --allow-incomplete  # tabulate what exists, e.g. mid-sweep

Writes ``inversions_mesh.csv`` and ``conditions_mesh.csv``. Raises on a failed gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "rep_error_quadratic"))

from quadratic_rep import (  # noqa: E402
    RUNS, covariance, design_weights, grid_from_truth, pure_error, sigma_rep2, spec_for,
)
from scaling_core import condition_metrics  # noqa: E402
from enforceflux.source_fields.lognormal_gp import _circulant_embedding_sample  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sweep", ROOT / "configs/hetero_rice_paddy_test/sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

NETS = {"op": "open_path", "point": "point"}


def sampler(grid, L: float, cv: float) -> str:
    """Which generator path this condition's fields took."""
    spec = spec_for(grid, L, cv)
    z = _circulant_embedding_sample(spec, np.log1p(cv**2), np.random.default_rng(0))
    return "circulant" if z is not None else "cholesky"


def build(allow_incomplete: bool) -> pd.DataFrame:
    rows, missing, designs = [], [], {}
    for L, cv, seed in itertools.product(sweep.L_VALUES_M, sweep.CV_VALUES, sweep.SEEDS):
        prefix = sweep.run_prefix(sweep.tag_of(L, cv, seed))
        truth = RUNS / f"{prefix}_surface" / "dispersion" / "truth_field.nc"
        if not truth.is_file():
            missing.append(truth.parent.parent.name)
            continue
        with xr.open_dataset(truth) as ds:
            F = ds["F_true"].values.copy()
            a = dict(ds.attrs)
        if float(a["L_true_m"]) != L or float(a["cv"]) != cv or int(a["seed"]) != seed:
            raise AssertionError(f"{truth}: attrs {a['L_true_m']}, {a['cv']}, {a['seed']} "
                                 f"do not match the run name ({L}, {cv}, {seed})")
        Q = float(a["Q_true_kg_s"])
        for n, net in itertools.product(sweep.N_VALUES, sweep.NETWORKS):
            run = f"{prefix}_n{n}_{net}"
            mat = RUNS / run / "flux" / "matrices.npz"
            if not mat.is_file():
                missing.append(run)
                continue
            key = (NETS[net], n)
            if key not in designs:
                designs[key] = design_weights(NETS[net], n, run)
            x_opt = float(np.load(mat)["x_opt"][0])
            rows.append({"run": run, "L_m": float(L), "CV": float(cv), "seed": seed,
                         "n": n, "geometry": NETS[net], "Q_true_kg_s": Q,
                         "e_signed": (x_opt - Q) / Q,
                         "e_pure": pure_error(designs[key], F)})
    if missing and not allow_incomplete:
        raise SystemExit(f"{len(missing)} mesh runs incomplete (first: {missing[0]}); "
                         "finish the sweep or pass --allow-incomplete")
    print(f"{len(rows)} inversions tabulated, {len(missing)} runs missing")
    return pd.DataFrame(rows), designs


def main(allow_incomplete: bool) -> int:
    inv, designs = build(allow_incomplete)

    # Gate: the original 8-seed 3 x 3 rows reproduce the published table.
    ref = pd.read_csv(HERE / "inversions.csv")[["run", "e_signed"]]
    m = inv.merge(ref, on="run", suffixes=("", "_ref"))
    d = float(np.abs(m.e_signed - m.e_signed_ref).max())
    if len(m) != len(ref) or d > 1e-12:
        raise AssertionError(f"8-seed gate: {len(m)}/{len(ref)} rows matched, max diff {d:.2e}")
    print(f"gate: {len(m)} published rows reproduced, max diff {d:.1e}")

    cond = condition_metrics(inv, "e_signed")
    pure = condition_metrics(inv, "e_pure")[
        ["geometry", "n", "CV", "L_m", "bias", "sigma_rep", "RMSE"]
    ].rename(columns={"bias": "bias_pure", "sigma_rep": "sigma_rep_pure", "RMSE": "RMSE_pure"})
    cond = cond.merge(pure, on=["geometry", "n", "CV", "L_m"])

    grid, _ = grid_from_truth(next(RUNS.glob(
        "source_heterogeneity_les_rice_paddy_l*_s0_wind3_surface/dispersion/truth_field.nc")))
    pred, path = {}, {}
    for L, cv in itertools.product(sweep.L_VALUES_M, sweep.CV_VALUES):
        cov = covariance(spec_for(grid, L, cv), "loglinear")
        path[(float(L), float(cv))] = sampler(grid, L, cv)
        for key, dw in designs.items():
            pred[key + (float(cv), float(L))] = np.sqrt(sigma_rep2(dw, cov))
    cond["sd_pred_loglinear"] = [pred[(r.geometry, r.n, r.CV, r.L_m)]
                                 for r in cond.itertuples()]
    cond["sampler"] = [path[(r.L_m, r.CV)] for r in cond.itertuples()]

    inv.to_csv(HERE / "inversions_mesh.csv", index=False)
    cond.to_csv(HERE / "conditions_mesh.csv", index=False)
    print(f"wrote inversions_mesh.csv ({len(inv)}) and conditions_mesh.csv ({len(cond)})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-incomplete", action="store_true")
    raise SystemExit(main(ap.parse_args().allow_incomplete))
