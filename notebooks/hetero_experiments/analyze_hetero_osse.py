"""Aggregate the rice-paddy source-heterogeneity OSSE suite into a tidy table.

Beyond the absolute error already stored in each run's analysis/summary.json this
adds three things the summaries do not carry:

  * signed relative error, which separates systematic bias from scatter;
  * an error decomposition that splits the total into the part caused by
    inverting a heterogeneous source with a uniform template ("aggregation")
    and the part caused by the LES-versus-bLS transport mismatch;
  * Rodgers retrieval diagnostics (DFS, chi squared per dof, prior influence).

The decomposition reuses the stored bLS Jacobian, so it needs no new runs. For a
linear inversion of a scalar total flux against a uniform template u,

    Qhat = (G u . y) / (G u . G u)

Substituting the LES observation for y gives the reported estimate; substituting
the bLS prediction of the *true* heterogeneous field, G (F A), gives what the
same inversion would have returned had transport been perfect. The difference
between the two is the transport term.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
PATTERN = re.compile(
    r"source_heterogeneity_les_rice_paddy_l(\d+)_cv(\d+p\d+)_wind3_n(\d)_(op|point)$"
)
Q_TRUE = 0.027778
# The l200/cv2.0 pair is a pilot outside the 3 x 3 design grid.
PILOT_L = 200


# Observation error covariance, exactly as apps/flux_main.py assembles it:
#   Se = instrument noise variance + sigma_repr^2 + (sigma_repr_fraction * flux_scale * |G u|)^2
# Verified to reproduce every stored x_opt_kg_s to six significant figures.
NOISE_VARIANCE = 1.0e-4
SIGMA_REPR = 25000.0
PRIOR_VARIANCE = 1.0  # prior_total_variance, (kg s-1)^2
PRIOR_MEAN = 0.0


def _se(sigma_by_obs: np.ndarray) -> np.ndarray:
    return NOISE_VARIANCE + SIGMA_REPR**2 + sigma_by_obs**2


def _solve(Gu: np.ndarray, y: np.ndarray, Se: np.ndarray) -> tuple[float, float]:
    """Scalar Bayesian linear inversion. Returns (Q_hat, posterior_sigma)."""
    precision = float(Gu / Se @ Gu) + 1.0 / PRIOR_VARIANCE
    q = (float(Gu / Se @ y) + PRIOR_MEAN / PRIOR_VARIANCE) / precision
    return q, precision**-0.5


def _obs_time_mean(instrument_dir: Path) -> np.ndarray:
    """Time-mean observation vector, honouring the validity mask (flux stage uses time_reduce: mean)."""
    with xr.open_dataset(instrument_dir / "obs.nc") as ds:
        y = np.where(ds.valid_mask.values.astype(bool), ds.y_obs.values, np.nan)
        return np.nanmean(y, axis=0)


def _decompose(run: Path, flux: dict) -> dict[str, float]:
    gp = run.parent / f"{run.name}_gp" / "dispersion"
    with xr.open_dataset(gp / "truth_field.nc") as ds:
        emission = (ds.F_true.values * ds.cell_area_m2.values).ravel()  # kg s-1 per cell
        areas = ds.cell_area_m2.values.ravel()
    G = np.load(gp / "jacobian.npz")["G"]

    template = areas / areas.sum()  # uniform template: kg s-1 per cell per unit total flux
    Gu = G @ template
    Se = _se(np.asarray(flux["observation_mode"]["sigma_repr_by_observation"]))

    y_les = _obs_time_mean(run / "instrument")   # LES nature, what the inversion actually saw
    y_bls_true = G @ emission                    # bLS applied to the exact true field

    q_reported, post_sigma = _solve(Gu, y_les, Se)
    q_perfect_transport, _ = _solve(Gu, y_bls_true, Se)

    residual = y_les - Gu * q_reported
    chi2 = float(residual / Se @ residual) + (q_reported - PRIOR_MEAN) ** 2 / PRIOR_VARIANCE
    dof = len(y_les) - 1

    return {
        "Q_hat_recomputed_kg_s": q_reported,
        "posterior_sigma_recomputed_kg_s": post_sigma,
        "bias_total_pct": 100.0 * (q_reported - Q_TRUE) / Q_TRUE,
        "bias_aggregation_pct": 100.0 * (q_perfect_transport - Q_TRUE) / Q_TRUE,
        "bias_transport_pct": 100.0 * (q_reported - q_perfect_transport) / Q_TRUE,
        "chi2": chi2,
        "chi2_per_dof": chi2 / dof if dof > 0 else float("nan"),
        "dof": dof,
    }


def _diagnostics(flux: dict) -> dict[str, float]:
    """Rodgers diagnostics for the scalar total-flux state."""
    prior_var = 1.0  # prior_total_variance, (kg s-1)^2
    post_sigma = float(flux["posterior_sigma_kg_s"][0][0])
    dfs = 1.0 - post_sigma**2 / prior_var
    n_obs = int(flux["n_observations"])
    return {
        "posterior_sigma_kg_s": post_sigma,
        "dfs": dfs,
        "prior_influence": 1.0 - dfs,
        "n_obs": n_obs,
        "dof": n_obs - 1,
    }


def build() -> pd.DataFrame:
    rows = []
    for run in sorted(RUNS.iterdir()):
        m = PATTERN.match(run.name)
        if not m:
            continue
        L = int(m.group(1))
        if L == PILOT_L:
            continue
        summary = run / "analysis" / "summary.json"
        flux_summary = run / "flux" / "summary.json"
        if not (summary.exists() and flux_summary.exists()):
            continue
        analysis = json.loads(summary.read_text())["source_heterogeneity"]
        flux = json.loads(flux_summary.read_text())

        row = {
            "run": run.name,
            "L_m": L,
            "CV": float(m.group(2).replace("p", ".")),
            "n": int(m.group(3)),
            "geometry": "Open path" if m.group(4) == "op" else "Point",
            "abs_error_pct": 100.0 * analysis["E_Q"],
            "Q_hat_kg_s": float(flux["x_opt_kg_s"][0][0]),
            "inverse_crime_flag": analysis["inverse_crime_flag"],
        }
        row.update(_diagnostics(flux))
        row.update(_decompose(run, flux))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["geometry", "CV", "L_m", "n"]).reset_index(drop=True)


def main() -> None:
    df = build()
    out = Path(__file__).parent / "hetero_osse_results.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} with {len(df)} rows")

    bad = (df.Q_hat_recomputed_kg_s - df.Q_hat_kg_s).abs().max()
    assert bad < 1e-9, f"recomputation drifted from stored x_opt by {bad}"
    print(f"recomputation matches stored x_opt to {bad:.2e} kg s-1")

    print("\n-- signed bias by CV (percent) --")
    print(
        df.pivot_table(index="CV", columns="geometry", values="bias_total_pct", aggfunc="mean")
        .round(2)
        .to_csv()
    )
    print("-- error decomposition, mean over all runs (percent) --")
    print(
        df.groupby(["geometry", "CV"])[
            ["bias_total_pct", "bias_aggregation_pct", "bias_transport_pct"]
        ]
        .mean()
        .round(2)
        .to_csv()
    )
    print("-- fraction of runs that underestimate --")
    neg = df.assign(under=df.bias_total_pct < 0).groupby(["geometry", "CV"]).under.mean()
    print((neg * 100).round(0).to_csv())


if __name__ == "__main__":
    main()
