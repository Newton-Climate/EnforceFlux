#!/usr/bin/env python3
"""Sampling distribution of the published pooled statistics under the generator.

The study draws every (CV, L) field from ``default_rng(seed)`` with the same
eight seeds, so the nine conditions share one set of underlying normal draws
(common random numbers). Their pooled SDs are therefore not nine independent
estimates, and sampling luck in seeds 0-7 moves every condition together.

This script replays the study design ``R`` times with fresh 8-seed sets — same
generator, same seeding recipe, same designs, same 96-run pooling as the
figures — using the exact perfect-transport map ``e = AK dw.phi + AK - 1``. It
yields the band a pooled statistic should fall in if the quadratic-form model
were exact, and ranks the actual study (seeds 0-7) within it.

Writes pooled_null.csv and pooled_null.npz in this directory only.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from quadratic_rep import (
    RUNS, design_weights, grid_from_truth, phi_of, sample_lognormal_field, spec_for,
)

HERE = Path(__file__).resolve().parent
R = 200
SEED_BASE = 1000  # replicate r uses seeds SEED_BASE + 8r ... + 8r + 7; the study used 0-7
CVS, LS, NS, GEOMS = (0.5, 1.0, 2.0), (100.0, 250.0, 500.0), (1, 2, 3, 4), ("point", "open_path")


def replicate(seeds, grid, designs) -> np.ndarray:
    """E[cv, L, seed, geometry, n] for one 8-seed set."""
    E = np.empty((3, 3, len(seeds), 2, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # L = 500 takes the Cholesky path, as in the study
        for i, cv in enumerate(CVS):
            for j, L in enumerate(LS):
                for k, s in enumerate(seeds):
                    phi = phi_of(sample_lognormal_field(spec_for(grid, L, cv, seed=s),
                                                        np.random.default_rng(s)))
                    for a, g in enumerate(GEOMS):
                        for b, n in enumerate(NS):
                            d = designs[(g, n)]
                            E[i, j, k, a, b] = d.ak * d.dw @ phi + d.ak - 1.0
    return E


def pooled_stats(E: np.ndarray) -> dict[tuple, float]:
    out = {}
    for a, g in enumerate(GEOMS):
        for i, cv in enumerate(CVS):
            x = E[i, :, :, a, :].ravel()
            out[(g, "CV", cv, "sd")] = x.std(ddof=1)
            out[(g, "CV", cv, "mae")] = np.abs(x).mean()
        for j, L in enumerate(LS):
            x = E[:, j, :, a, :].ravel()
            out[(g, "L_m", L, "sd")] = x.std(ddof=1)
            out[(g, "L_m", L, "mae")] = np.abs(x).mean()
    return out


def main(nseeds: int) -> int:
    sfx = "" if nseeds == 8 else f"_s{nseeds}"
    inv = pd.read_csv(HERE.parent / "rep_error_scaling" / "inversions.csv")
    grid, _ = grid_from_truth(RUNS / inv.nature_run.iloc[0] / "dispersion" / "truth_field.nc")
    designs = {k: design_weights(k[0], k[1], g.run.iloc[0]) for k, g in inv.groupby(["geometry", "n"])}

    study = pooled_stats(replicate(range(nseeds), grid, designs))
    # Gate: replaying the study seeds must reproduce the pooled perfect-transport statistics exactly.
    pooled = pd.read_csv(HERE / f"pooled{sfx}.csv")
    for r in pooled.itertuples():
        for stat in ("sd", "mae"):
            got, want = study[(r.geometry, r.pooled_by, r.value, stat)], getattr(r, f"{stat}_emp_pure")
            if abs(got - want) > 1e-10:
                raise AssertionError(f"study replay mismatch {r.geometry} {r.pooled_by}={r.value} {stat}")
    print(f"gate: seeds 0-{nseeds - 1} replay reproduces pooled e_pure statistics exactly")

    reps = []
    for rep in range(R):
        first = SEED_BASE + nseeds * rep  # disjoint from the study seeds and from each other
        reps.append(pooled_stats(replicate(range(first, first + nseeds), grid, designs)))
        if rep % 25 == 0:
            print(f"  replicate {rep}/{R}", flush=True)
    keys = list(study)
    arr = np.array([[r[k] for k in keys] for r in reps])
    rows = []
    for c, (g, by, val, stat) in enumerate(keys):
        v = arr[:, c]
        rows.append({"geometry": g, "pooled_by": by, "value": val, "statistic": stat,
                     "null_q025": np.quantile(v, 0.025), "null_median": np.median(v),
                     "null_q975": np.quantile(v, 0.975), "null_mean": v.mean(),
                     "study_value": study[keys[c]], "n_seeds": nseeds,
                     "study_percentile_in_null": 100 * np.mean(v <= study[keys[c]])})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"pooled_null{sfx}.csv", index=False)
    np.savez_compressed(HERE / f"pooled_null{sfx}.npz", replicates=arr,
                        keys=np.array(["|".join(map(str, k)) for k in keys]))
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=8)
    raise SystemExit(main(ap.parse_args().nseeds))
