# Representativeness-error scaling analysis

Tests whether the flux-retrieval error in the 576-inversion source-heterogeneity
OSSE follows `sigma_rep ~ CV / sqrt(N_eff)` with `N_eff ~ A / L^2`.

**No simulations are run and nothing outside this directory is written.** Every
script reads `runs/`, `configs/` and `src/` read-only.

Read [`report.md`](report.md) for the findings; the verdict is at the end.

## Rerun

From the repository root, in order:

```bash
cd notebooks/hetero_experiments/rep_error_scaling
python test_scaling_core.py     # ~1 s   validate the fitting code first
python build_table.py           # ~40 s  runs/ -> inversions.csv
python field_stats.py           # ~90 s  N_eff, L_H, error decomposition
python scaling_analysis.py      # ~60 s  metrics, model fits, calibration
python phase_mesh.py            # ~1 min fine CV x L mesh (20 seeds) -> *_mesh.csv, for fig1
python make_figures.py          # ~20 s  figures/
python make_report.py           # ~1 s   report.md
```

Or all at once:

```bash
cd notebooks/hetero_experiments/rep_error_scaling && for s in test_scaling_core build_table field_stats scaling_analysis phase_mesh make_figures make_report; do python $s.py || break; done
```

`build_table.py` raises on any failed integrity gate, so a broken or incomplete
`runs/` tree stops the pipeline rather than propagating quietly.

Requires numpy, scipy, pandas, xarray, matplotlib, statsmodels and PyYAML, plus
an importable `enforceflux` (the scripts add `src/` to `sys.path` themselves).

## Inputs consumed

| Path | Used for |
|---|---|
| `runs/source_heterogeneity_les_rice_paddy_l*_cv*_s[0-7]_wind3_n[1-4]_{op,point}/` | 576 inversions: `flux/summary.json`, `flux/posterior.csv`, `flux/matrices.npz`, `analysis/summary.json`, `instrument/config.snapshot.yaml` |
| `runs/…_s[0-7]_wind3_surface/dispersion/truth_field.nc` | 72 source fields and their design attributes |
| `runs/…_n[1-4]_{op,point}_gp/dispersion/jacobian.npz` | bLS Jacobian, for footprint scale and the error decomposition |
| `notebooks/hetero_experiments/seed_sweep_results.csv` | independent cross-check only — never an input to any result |
| `src/enforceflux/analysis/footprint_scale.py` | `footprint_correlation_length`, reused as-is |
| `src/enforceflux/source_fields/lognormal_gp.py` | `FieldGrid`; also the authority on what `L` and `CV` mean |

The unseeded base runs (no `_s<d>_`) are deliberately excluded: `SWEEPS.md`
records that they carry 46 cross-section frames rather than 45 and must not be
pooled with the seeded ones.

## Outputs produced

| File | Contents |
|---|---|
| `inversions.csv` | 576 rows x 41 columns: design, truth, retrieval, Rodgers diagnostics, provenance |
| `error_decomposition.csv` | 576 rows: retrieval under a perfect transport operator, and the pure-representativeness / transport split |
| `conditions.csv` | 72 conditions (geometry x n x CV x L): bias, MAE, RMSE, sigma_rep with bootstrap and chi-square intervals, all candidate predictors, calibration summaries |
| `field_stats.csv` | 9 CV-L conditions: five N_eff variants, the correlation scale recovered from the realized fields, realized CV |
| `footprint_scales.csv` | 8 designs: footprint correlation length `L_H`, total path length |
| `model_fits.csv` | 7 models x 8 designs: coefficients, exponents, standard errors, R², RMSE, AIC, BIC, leave-one-condition-out error |
| `calibration.csv` | 72 conditions: coverage and log score, reported vs widened by `sigma_rep(H)` |
| `results.json` | bootstrap exponent intervals, pooled footprint test, sensor-number fits |
| `integrity.md` | the 20 integrity gates and their results |
| `figures/fig[1-6]_*.{png,pdf,txt}` | figures; each `.txt` describes the panel in words and repeats the plotted values as CSV |
| `report.md` | methods, fits, model comparison, interpretation, limitations, verdict |

## Files

| Script | Role |
|---|---|
| `scaling_core.py` | condition metrics, bootstrap, the seven candidate models, OLS in log space, leave-one-condition-out CV |
| `test_scaling_core.py` | recovers known exponents from synthetic data; run it before trusting a fit |
| `build_table.py` | assembles `inversions.csv` and enforces the integrity gates |
| `field_stats.py` | effective replication, footprint scales, error decomposition |
| `scaling_analysis.py` | the analysis proper |
| `phase_mesh.py` | the fine CV x L mesh (grid from `configs/hetero_rice_paddy_test/sweep.py`): `inversions_mesh.csv`, `conditions_mesh.csv`, with the closed-form spread; feeds `fig1` only |
| `make_figures.py` | figures and their text descriptions; `fig1` is on the fine mesh, `fig2`-`fig6` on the original 8-seed 3 x 3 |
| `make_report.py` | renders `report.md` from the CSVs — no number in the report is typed by hand |

## Statistical conventions

* The **source realization is the experimental unit**, not the inversion. Model
  fits are at condition level (nine CV-L points per measurement design), and
  every bootstrap resamples the 8 seeds within a condition, never the 576 rows.
* Bias and spread are reported separately and never combined.
* `LOCO` — leave-one-(CV,L)-condition-out RMSE in log space — is the
  generalization metric. With nine design points, in-sample R² is reported but
  not relied on.
* Intervals on `sigma_rep` are percentile bootstrap over 8 realizations, with a
  chi-square interval as a parametric cross-check. They are wide, and are drawn
  on every figure.
