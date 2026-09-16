# Quadratic-form representativeness error: validation against the 576-inversion OSSE

Tests whether `sigma_rep^2 = dw^T Sigma dw` predicts the total-flux retrieval
error of the source-heterogeneity OSSE (L x CV x seed x n x point/open path)
without rerunning LES or the inversion.

**Nothing outside this directory is written.** `runs/`, `src/`,
`../seed_sweep_results.csv` and `../rep_error_scaling/*.csv` are read-only.

## Rerun

```bash
cd notebooks/hetero_experiments/rep_error_quadratic
python run_validation.py     # ~8 min (L = 500 m Monte Carlo takes the Cholesky path)
python shared_seed_null.py   # ~6 min, 200 replays of the 8-seed design
FIGURE_QC_DIR=<figure-build skill>/scripts python make_figures.py   # QC optional
```

Every script raises on a failed gate rather than writing a partial result.

## Method

For the total-only, uniform-template retrieval (verified: the inversion's `G`
equals `G_fine @ (1/N)` exactly, `x_prior = 0`, `Sa = 1`), the perfect-transport
fractional error is exactly linear in the relative field `phi = N F / sum(F)`:

    e = AK * dw . phi + (AK - 1),   dw = w_g - 1/N,   w_g = c / sum(c),   c = (g_tot / Se)^T G_fine

For one observation `w_g = g / sum(g)`, as in the request. For n > 1 or multi-row
open paths, `w_g` is the Se-weighted combination that the weighted least squares
actually applies. `Se` and `G_fine` are constant within each of the 8 designs.
`AK` lies between 0.99988 and 0.99997, so the `(AK - 1)` bias term is below 0.012%.
The gate `dw . phi` reproduces the existing `e_pure_representativeness` in
`../rep_error_scaling/error_decomposition.csv` with a difference of exactly 0.

`Var(e) = AK^2 dw^T Cov(phi) dw` is evaluated three ways:

| variant | `Cov(phi)` | what it leaves out |
|---|---|---|
| `loglinear` | `sigma^2 exp(-r/L)`, `sigma^2 = ln(1 + CV^2)`: **the form as specified** | the nonlinearity of `exp`, and the renormalization to `Q_true` |
| `lognormal` | `expm1(sigma^2 exp(-r/L))`, exact for `exp(Z)` | the renormalization to `Q_true` |
| `mc` | 4000 draws from `sample_lognormal_field` itself | nothing (the generator is exact by construction) |

Covariance, grid and sampler are imported from
`enforceflux.source_fields.lognormal_gp`: `FieldGrid`, `LognormalFieldSpec`,
`sample_lognormal_field`, and the private `_correlation`, which is the generator's
own kernel. None of them is reimplemented.

## Parameterization check (task item 4)

| item | used here | in the generator / runs | match |
|---|---|---|---|
| grid | read from each `truth_field.nc`: 25 x 25, dx = 40 m, origin (-500, -500) | same on all 72 fields (gate) | exact |
| covariance | `_correlation` with `exponential`, `L_true_m` from file | `covariance_model = exponential` in all 72 files | exact |
| log variance | `log1p(CV^2)` | `log1p(cv**2)` | exact |
| realizations | regenerated with `default_rng(seed)`, as the plugin does | stored `F_true` | bitwise (max relative difference 0.0) |
| footprint | `_gp/dispersion/jacobian.npz` `G`, all rows | constant within each design (gate); the unseeded base-run `G` in the contour figure equals the seeded one | exact |

**No parameter mismatch.** The analytic variants differ from the generator only
where the table above says, and the Monte Carlo variant closes that gap. Four
caveats are not parameter mismatches but do affect the comparison:

1. **Common random numbers across conditions.** Seed *k* feeds the same normal
   draws to every (CV, L). At fixed L and seed the log-fields for different CV
   have correlation 1.000, L = 100 vs 250 m about 0.95, and `e_pure` correlates
   0.94-0.99 across CV. The 9 conditions therefore share one set of sampling
   luck. `shared_seed_null.py` replays that design to give the fair band.
2. **L = 500 m takes a different sampler path.** Circulant embedding is not
   positive semi-definite on the 50 x 50 torus at L = 500 m, so the generator
   falls back to padded Cholesky. That applies to the stored fields and to all
   4000 Monte Carlo draws alike (logged in `gates.csv`). The same seed then maps
   to a different field, which is why L = 500 m behaves unlike L = 100 and 250 m below.
3. **Transport error is outside `Sigma`.** The truth concentrations come from the
   LES tagged-tracer operator; the inversion uses bLS. `e_transport` is
   anticorrelated with `e_pure` (r = -0.52 over 576), so total error has a
   *smaller* SD than perfect-transport error for point sensors (median ratio
   0.88; 0.99 for open path). This depends on the single shared LES turbulence
   realization (`SWEEPS.md`). How the 625-cell field maps onto the 196 tagged LES
   cells was not traced.
4. **The 10.4 / 11.8 / 19.0% figures are mean absolute errors, not SDs** (from
   `make_revised_figures.py`). Point-sensor SD by L is 13.6 / 15.2 / 23.9%. Both
   statistics are compared below; the predicted MAE is `sigma * sqrt(2/pi)` for
   the analytic variants and `mean|e|` for the Monte Carlo variant.

## Results

Pooled exactly as in the published figures: point network, 96 inversions per
value of CV (or L), pooled over the other factor, n and seed. A gate checks that
this reproduces the published numbers to within 1e-5. Percentages are
100 x fractional error. "Null 95%" is the band of the perfect-transport statistic
over 200 replays of the shared 8-seed design.

| point sensors | empirical total | empirical perfect-transport | pred. log-linear | pred. lognormal | pred. MC | null median [95%] | study percentile in null |
|---|---|---|---|---|---|---|---|
| SD, CV = 0.5 | 9.9 | 11.4 | 13.7 | 14.4 | 14.1 | 13.6 [9.9, 17.9] | 15 |
| SD, CV = 1.0 | 16.8 | 20.6 | 24.2 | 28.2 | 25.7 | 24.2 [17.1, 37.3] | 19 |
| SD, CV = 2.0 | 26.4 | 32.8 | 36.9 | 53.4 | 41.5 | 37.7 [25.4, 69.2] | 25 |
| SD, L = 100 | 13.6 | 20.2 | 23.8 | 29.4 | 27.2 | 24.2 [15.3, 47.1] | 26 |
| SD, L = 250 | 15.2 | 20.2 | 28.4 | 37.6 | 31.2 | 28.1 [16.4, 53.6] | 12 |
| SD, L = 500 | 23.9 | 25.5 | 27.7 | 39.8 | 29.6 | 25.0 [16.1, 49.1] | 55 |
| MAE, L = 100 | 10.4 | 17.4 | 16.7 | 20.1 | 17.4 | 17.2 [11.8, 26.4] | 51 |
| MAE, L = 250 | 11.8 | 17.7 | 20.0 | 25.4 | 20.0 | 19.5 [13.3, 29.8] | 31 |
| MAE, L = 500 | 19.0 | 19.0 | 19.7 | 26.7 | 19.5 | 18.4 [12.5, 29.1] | 55 |

Open-path rows are in `pooled.csv` and `pooled_null.csv`, and show the same pattern.

Across all 72 conditions (`metrics.csv`; SD with 7 degrees of freedom each):

| target | variant | r (variance) | r (log SD) | Spearman (SD) | RMSE (SD, pp) | median pred/emp SD | pred in emp. chi-square 95% CI | SD of z |
|---|---|---|---|---|---|---|---|---|
| total | log-linear | 0.77 | 0.88 | 0.85 | 9.1 | 1.71 | 75% | 0.75 |
| total | lognormal | 0.84 | 0.90 | 0.88 | 15.8 | 1.92 | 60% | 0.64 |
| total | MC | 0.74 | 0.86 | 0.82 | 11.0 | 1.84 | 71% | 0.72 |
| perfect-transport | log-linear | 0.63 | 0.81 | 0.79 | 9.6 | 1.47 | 78% | 0.83 |
| perfect-transport | lognormal | 0.71 | 0.83 | 0.84 | 15.5 | 1.76 | 69% | 0.70 |
| perfect-transport | MC | 0.61 | 0.80 | 0.78 | 11.2 | 1.64 | 71% | 0.79 |

Model-free check: the empirical perfect-transport SD falls inside the Monte Carlo
95% band for an 8-draw sample SD in 89% of conditions. At L = 100 and 250 m it is
0.48-0.70 of that band's median; at L = 500 m it is 1.02-1.15.

The log-linear form agrees with the generator Monte Carlo to within 0-13% (worst
case CV = 2, L = 100 m, where it is 13% low). The "exact" lognormal form is
1-40% *high* (worst at CV = 2, L = 500 m), because the renormalization to
`Q_true` removes the domain-mean fluctuation that `expm1(sigma^2 R)` keeps. As
the specified approximation, the log-linear form is the best of the two analytic
variants.

## Verdict

* **Ranking: yes.** Across 72 conditions the quadratic form orders the error
  spread well (Spearman 0.85, r = 0.88 on log SD against total error). It
  captures the CV scaling and the point-versus-open-path difference.
* **As the generator's expected spread: yes.** The log-linear `dw^T Sigma dw` is
  within 13% of the exact Monte Carlo, and within about 4% of the null median
  for every pooled point-sensor SD by CV.
* **Reproducing the study's 8-seed numbers: no. It is 1.5-1.7x high.** This is
  mostly not a failure of the approximation. The generator itself (Monte Carlo)
  is 1.64x high against the exact perfect-transport errors. The study's seeds
  0-7 sit at the 12th-26th percentile of the shared-seed null for every pooled
  SD except L = 500 m (which uses a different sampler path, see caveat 2), and
  the anticorrelated transport error shrinks the total spread further.
* **Implication for the published spreads.** The ±10 / 17 / 26% by CV, and the
  flat L = 100-250 m then jump at 500 m, are one low-variance draw of a shared
  seed set. Under the same generator the expected perfect-transport SD at
  CV = 2 is about 38%, with a 95% band of 25-69%. With only 8 seeds shared
  across conditions, the apparent L trend is not separable from seed luck
  (every L band overlaps).

## Update: 20 seeds (2026-09-15)

Seeds 8-19 were added to the sweep (`configs/hetero_rice_paddy_test/sweep.py`,
`SEEDS = range(20)`; `generate_sweeps.py` and its config files were later replaced by it). That is 108 new source fields, evaluated through the existing
tagged-tracer operator with no new LES time, and 864 new inversions, for 1440 in
total. Seeds 0-7 were not rerun. The 20-seed table is built from `runs/`, and its
seed 0-7 rows reproduce `inversions.csv` (difference 1e-16) and
`error_decomposition.csv` (difference 1e-15). Outputs carry the suffix `_s20`; the
8-seed files are unchanged.

```bash
python run_validation.py --nseeds 20 && python shared_seed_null.py --nseeds 20 && python make_figures.py --nseeds 20
```

**The seeds are now typical draws.** PIT Kolmogorov-Smirnov p = 0.08 (8 seeds:
5e-8). The empirical perfect-transport SD is inside the Monte Carlo 95% band for a
20-draw sample SD in 100% of conditions (8 seeds: 89%), at 0.95 of its median
(8 seeds: 0.70).

Across the 72 conditions, 8 vs 20 seeds (`metrics.csv` vs `metrics_s20.csv`):

| target | variant | r (variance) | Spearman | RMSE (SD, pp) | median pred/emp SD | pred in chi-square 95% CI | SD of z |
|---|---|---|---|---|---|---|---|
| perfect-transport | log-linear | 0.63 -> **0.93** | 0.79 -> **0.97** | 9.6 -> **3.5** | 1.47 -> **1.08** | 78% -> **99%** | 0.83 -> **0.93** |
| perfect-transport | MC | 0.61 -> 0.95 | 0.78 -> 0.96 | 11.2 -> 4.1 | 1.64 -> 1.14 | 71% -> 100% | 0.79 -> 0.88 |
| total | log-linear | 0.77 -> **0.95** | 0.85 -> **0.96** | 9.1 -> **6.4** | 1.71 -> **1.30** | 75% -> **82%** | 0.75 -> **0.80** |
| total | lognormal | 0.84 -> 0.97 | 0.88 -> 0.97 | 15.8 -> 14.0 | 1.92 -> 1.51 | 60% -> 42% | 0.64 -> 0.70 |

Pooled point-sensor statistics (96 -> 240 inversions per value):

| point sensors | total, 8 seeds | total, 20 seeds | perfect-transport, 20 seeds | pred. log-linear | null median [95%], 20 seeds |
|---|---|---|---|---|---|
| SD, CV = 0.5 | 9.9 | 10.9 | 13.3 | 13.7 | 13.8 [11.4, 16.8] |
| SD, CV = 1.0 | 16.8 | 17.9 | 23.7 | 24.2 | 25.0 [19.5, 32.9] |
| SD, CV = 2.0 | 26.4 | 26.6 | 38.4 | 36.9 | 39.7 [29.4, 61.8] |
| SD, L = 100 | 13.6 | 15.2 | 27.1 | 23.8 | 25.2 [18.2, 42.5] |
| SD, L = 250 | 15.2 | 20.4 | 29.9 | 28.4 | 29.6 [20.2, 48.5] |
| SD, L = 500 | 23.9 | 22.3 | 24.3 | 27.7 | 27.1 [19.3, 43.3] |
| MAE, L = 100 | 10.4 | 12.3 | 18.0 | 16.7 | 17.2 [13.6, 22.3] |
| MAE, L = 250 | 11.8 | 15.5 | 20.1 | 20.0 | 19.7 [15.7, 27.2] |
| MAE, L = 500 | 19.0 | 17.0 | 18.4 | 19.7 | 18.9 [14.9, 26.3] |

The study now sits at the 16th-64th percentile of its own shared-seed null.

### What changed

* **The approximation holds for representativeness.** With 20 seeds, the
  first-order `dw^T Sigma dw` predicts the perfect-transport error spread to
  within about 8% (median), condition by condition, with 99% of predictions inside
  the sampling interval. The 8-seed overprediction was seed luck, as the
  shared-seed null said.
* **The remaining gap is transport, not representativeness.** Against total
  error the log-linear form is still 1.30x high. `e_transport` is anticorrelated
  with `e_pure` (r = -0.62 over 1440; -0.52 over 576), so the LES-versus-bLS
  mismatch cancels part of the representativeness error. This comes from a single
  turbulence realization and is not in `Sigma`.
* **The published L story changes.** The jump to L = 500 m came from the 8 seeds.
  At 20 seeds, point-sensor MAE is 12.3 / 15.5 / 17.0% (was 10.4 / 11.8 / 19.0%),
  and SD is 15.2 / 20.4 / 22.3% (was 13.6 / 15.2 / 23.9%): a gradual rise, not a
  step. The CV story holds: total SD is 10.9 / 17.9 / 26.6%.
* **More seeds narrow the null only slowly.** The 95% band on pooled point SD at
  CV = 2 goes from [25, 69]% (8 seeds) to [29, 62]% (20 seeds). The seeds are
  shared across conditions and the lognormal tails are heavy, so the pooled
  numbers still carry about ±30% relative sampling uncertainty at CV = 2.

## Files

| file | contents |
|---|---|
| `quadratic_rep.py` | `design_weights`, `covariance`, `sigma_rep2`, `pure_error`, `monte_carlo_phi` |
| `run_validation.py` | gates, predictions, `conditions.csv` (72), `inversions_z.csv` (576), `pooled.csv`, `metrics.csv`, `kernels.npz`, `mc_errors.npz`, `gates.csv` |
| `shared_seed_null.py` | `pooled_null.csv` / `.npz`: 200 replays of the shared 8-seed design |
| `make_figures.py` | `figures/fig1-4`, each `.png`, `.pdf` and `.txt` (alt text + plotted values as CSV) |

Figures: `fig1_pooled_comparison` (published pooling plus null band),
`fig2_scatter_{loglinear,mc}` (72 conditions, predicted vs empirical variance,
by CV and by L, total and perfect-transport), `fig3_case_level` (QQ of z, PIT of
stored fields, SD of z per condition), `fig4_mismatch_kernels` (dw for point and
open path at n = 1 and 4, with 50/90% cumulative `w_g` contours).
