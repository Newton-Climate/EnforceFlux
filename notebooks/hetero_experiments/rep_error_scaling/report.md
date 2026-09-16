# Does representativeness error follow a scaling law?

Analysis of the 576-inversion source-heterogeneity OSSE (`configs/hetero_rice_paddy_test`). No new simulations were run.

## 1. Summary

The proposed scaling `sigma_rep ~ CV/sqrt(N_eff)` with `N_eff = A/L^2` is **not supported**. Imposing its exponents explains 0.52 of the variance in log sigma_rep (mean over the eight measurement designs) — less than using CV alone (0.60). Its leave-one-condition-out error is 0.43 in log space, versus 0.20 once the exponents are freed.

What *is* supported is the **separable form**: a product of an amplitude term and a length term does collapse all nine CV–L conditions, at every measurement design, with the exponents both far from the first-order prediction of 1:

```
sigma_rep  =  a · CV^alpha · (L/sqrt(A))^beta
           alpha = 0.81 ± 0.02   (range 0.78–0.83 across 8 designs)
           beta  = 0.49 ± 0.09   (range 0.31–0.54)
```

Substituting the log-field amplitude `sigma_log = sqrt(ln(1+CV^2))` for CV — the quantity the generator actually controls — moves the amplitude exponent to 1.14 ± 0.03, i.e. consistent with 1. The physically cleanest empirical statement is therefore

```
sigma_rep  ≈  a · sqrt(ln(1 + CV^2)) · (L/sqrt(A))^(1/2)
     a = 0.33 open path,  0.44 point (n = 1)
```

Three further results, each of which stands independently of the scaling question:

1. **The source total does not vary.** The generator renormalizes every realization to `Q_true` exactly, so the realized domain-integrated flux has identically zero variance across seeds (`field_stats.csv: source_total_sd_kg_s = 0`). sigma_rep here is not sampling error on the source total; it is the footprint sampling a pattern whose total is already fixed.
2. **Sensor replication and path subdivision are different things.** Adding log n to the fit gives an exponent of -0.489 (95% CI -0.584 to -0.393) for point sensors — consistent with 1/sqrt(n) — and -0.038 (CI -0.141 to 0.065) for open paths, consistent with zero. The open-path design holds total path length at 1000 m, so n = 4 subdivides the same ground that n = 1 already covers.
3. **H predicts where the reported uncertainty stops being conservative.** `R = |error|/sigma_reported` correlates with log H at r = 0.85, and 68% coverage falls monotonically from 1.00 at the lowest H to 0.58 at H = 1.00 — below nominal.

## 2. Experiment and data provenance

### How the 576 is constructed

```
3 correlation lengths L  (100, 250, 500 m)
x 3 emission contrasts CV  (0.5, 1.0, 2.0)
x 8 source-field seeds  (0-7)
x 4 instrument counts n  (1, 2, 3, 4)
x 2 measurement geometries  (open path, matched point)
= 576 inversions
```

Each row of `inversions.csv` is one run directory under `runs/`. Design metadata is read from files, never parsed from run names: L, CV, seed, the covariance model and `Q_true` come from the nature run's `truth_field.nc` global attributes; instrument count, technology and path length from the instrument stage's `config.snapshot.yaml`; the retrieval and its diagnostics from `flux/summary.json`, `flux/posterior.csv` and `flux/matrices.npz`. Run names are carried as provenance and checked against the recorded values as one of the integrity gates.

### The source fields

`src/enforceflux/source_fields/lognormal_gp.py` samples a Gaussian field `Z` with **exponential** correlation `rho(r) = exp(-r/L)` and variance `sigma^2 = log1p(CV^2)`, forms `F = exp(Z - sigma^2/2)`, then rescales so `sum(F·area) == Q_true` exactly. So:

* **L is the e-folding correlation length of the log field**, not a Gaussian kernel width and not an integral scale.
* CV is exactly the marginal coefficient of variation of the emitted field.
* The domain is 25 x 25 cells at 40 m = 1000 x 1000 m, `A = 1e6 m^2`.
* Every field carries the same total flux by construction.

### Conditioning that limits interpretation

Per `configs/hetero_rice_paddy_test/SWEEPS.md`, the seeded nature cases are evaluated as `H·e` through one tagged-tracer LES operator warm-started from a single restart. **All 576 inversions share one turbulence realization.** The seed dimension samples emission-field randomness only, so sigma_rep here is source representativeness *conditional on one eddy field*, not total atmospheric uncertainty. The operator itself reproduces the nine real LES runs to about 4–5% mass-weighted (`runs/source_heterogeneity_les_tagged_operator/validation.json`), which is a floor under any error reported here.

The bLS Jacobian used in the inversion was verified in this analysis to be bit-identical across all 72 (L, CV, seed) combinations at each of the eight designs, confirming the operator-reuse claim.

### Integrity checks

| check | result | detail |
|---|---|---|
| row count is 576 | PASS | got 576 |
| no duplicate design cells | PASS | 0 duplicates |
| CV levels | PASS | [0.5, 1, 2] |
| L levels | PASS | [100, 250, 500] |
| seeds | PASS | [0, 1, 2, 3, 4, 5, 6, 7] |
| n levels | PASS | [1, 2, 3, 4] |
| geometries | PASS | [open_path, point] |
| full 3x3x8x4x2 crossing, one run each | PASS | 576 cells, min 1, max 1 |
| all inversions converged | PASS | 0 not converged |
| Q_true identical across runs | PASS | range 0.000e+00 kg/s |
| sum(F_true*area) reproduces Q_true | PASS | max abs diff 6.939e-18 kg/s |
| abs(e_signed) matches analysis-stage E_Q | PASS | max abs diff 3.053e-16 |
| e_signed matches seed_sweep_results.csv | PASS | 576 matched rows, max abs diff 4.999e-07 |
| run names agree with recorded metadata | PASS |  |
| no inverse crime flagged | PASS | 0 flagged |
| single-state total-flux retrievals | PASS |  |
| n_obs equals sensor count | PASS |  |
| state units kg s-1 | PASS | [kg s-1] |
| covariance model is exponential everywhere | PASS | [exponential] |
| source domain is 1e6 m2 | PASS | [1e+06] |

Assembled 576 rows from `runs/source_heterogeneity_les_rice_paddy_l*_cv*_s[0-7]_wind3_n[1-4]_*`.

Design cells: L [100, 250, 500] m x CV [0.5, 1, 2] x seed 0-7 x n [1, 2, 3, 4] x geometry (open path, point) = 576 inversions.

## 3. Metrics

Conditions are the 72 cells of (geometry x n x CV x L), each holding the 8 source realizations. Bias and spread are kept separate throughout; nothing folds one into the other.

Bias is **not** negligible. It is distinguishable from zero at p < 0.05 in 10 of 72 conditions (8 realizations each, so the test has little power), and `|bias|/sigma_rep` has median 0.40 and maximum 1.58. More telling is its structure: pooled over all designs the mean bias is -2.0% at L = 100 m, -3.8% at L = 250 m and 9.0% at L = 500 m — a sign flip, not noise. Spread is used as the primary y variable because the scaling question is about realization-to-realization variability, but RMSE is reported alongside and the two diverge where bias is large.

Uncertainty on each sigma_rep is a 95% percentile bootstrap over the 8 realizations, with a chi-square interval as a parametric cross-check. BCa was not used: with 8 points its acceleration term is noisier than the correction it applies. The intervals are wide — typically a factor of two — and every figure shows them.

### Primary subset: point sensors, n = 1

| CV | L (m) | bias (%) | p | MAE (%) | RMSE (%) | sigma_rep (%) | 95% boot |
|---|---|---|---|---|---|---|---|
| 0.5 | 100 | +1.7 | 0.60 | 6.5 | 8.2 | 8.6 | 3.5–10.7 |
| 0.5 | 250 | +0.0 | 0.99 | 7.2 | 9.1 | 9.7 | 4.3–12.0 |
| 0.5 | 500 | +10.4 | 0.12 | 14.4 | 18.5 | 16.4 | 8.4–20.6 |
| 1 | 100 | -2.7 | 0.64 | 13.2 | 15.0 | 15.7 | 5.9–19.6 |
| 1 | 250 | -6.2 | 0.35 | 15.7 | 17.5 | 17.5 | 7.5–21.6 |
| 1 | 500 | +12.2 | 0.29 | 24.4 | 31.0 | 30.4 | 14.1–38.3 |
| 2 | 100 | -8.9 | 0.38 | 23.2 | 26.6 | 26.7 | 10.2–34.8 |
| 2 | 250 | -14.1 | 0.19 | 26.3 | 29.5 | 27.7 | 13.3–35.0 |
| 2 | 500 | +14.1 | 0.44 | 38.3 | 47.2 | 48.2 | 23.3–59.8 |

![Figure 1](figures/fig1_response_surface.png)

## 4. Effective replication, and why A/L^2 is the wrong estimate

`N_eff = A/L^2` is not defensible for this field. Three corrections, in increasing order of how much they matter:

**(a) Integral correlation area.** For `rho(r) = exp(-r/L)` the integral correlation area is `int rho dA = 2·pi·L^2`, so `N_eff = A/(2·pi·L^2)` — 6.3x smaller than `A/L^2`. This is a constant factor and moves only the coefficient, not the collapse.

**(b) The bounded domain.** `A/L^2` diverges from the true effective sample size once L approaches the domain. Using the textbook estimator `N_eff = 1/mean_ij rho(r_ij)` over the actual 25x25 grid:

| L (m) | A/L^2 | A/(2πL²) | N_eff (bounded, log field) | N_eff (bounded, lognormal, CV=2) | L̂ recovered from the fields (m) |
|---|---|---|---|---|---|
| 100 | 100 | 15.92 | 20.71 | 39.10 | 90 |
| 250 | 16 | 2.55 | 5.17 | 9.22 | 147 |
| 500 | 4 | 0.64 | 2.52 | 4.04 | 179 |

The last column is the e-folding scale recovered by fitting `exp(-r/L)` to the empirical autocorrelation of `log F_true` pooled over the eight realizations. It recovers L = 100 m well (90 m) but saturates at 179 m for the nominal L = 500 m. A 1 km domain cannot hold a 500 m correlation length, so the L = 500 m condition is a finite-domain corner, not a clean third point on an L axis. This is the single largest reason the fitted beta lands near 0.5 rather than 1.

**(c) The lognormal transformation.** The emitted field is `exp(Z)`, whose correlation is `rho_F(r) = expm1(sigma^2·rho_Z(r))/CV^2`. This makes `N_eff` depend on CV as well as L (right-hand columns above), which is a genuine alternative rather than a relabelling of L.

Substituting the corrected `N_eff` into the proposed one-parameter form (model M5) raises the mean R^2 from 0.52 to 0.83 and lowers LOCO from 0.43 to 0.26. So a defensible N_eff rescues a large part of the proposed form — but not all of it, and not to the level of a free-exponent fit.

## 5. Model comparison

All models are fitted as `log(sigma_rep) = log a + offset + sum_k p_k log x_k` on the nine CV–L conditions of a single measurement design, so AIC and BIC are comparable and differ only in parameter count. OLS rather than weighted: the sampling SE of `log s` is `1/sqrt(2(m-1))` for every condition here, so weights would be uniform. `LOCO` is the leave-one-(CV,L)-condition-out RMSE in log space — with nine design points it is the only generalization number worth quoting.

Means over the eight measurement designs:

| model | form | R² (log) | LOCO | BIC | wins on LOCO |
|---|---|---|---|---|---|
| M0 | a·CV | 0.60 | 0.41 | 9.4 | 0 / 8 |
| M1 | a·L/√A | -0.08 | 0.67 | 18.3 | 0 / 8 |
| M2 | a·CV·L/√A | 0.52 | 0.43 | 10.4 | 0 / 8 |
| M3 | a·CV^α·(L/√A)^β | 0.94 | 0.20 | -4.3 | 6 / 8 |
| M4 | a·σ_log^α·(L/√A)^β | 0.94 | 0.20 | -4.4 | 2 / 8 |
| M5 | a·CV/√N_eff,lognormal | 0.83 | 0.26 | 1.2 | 0 / 8 |
| M6 | a·σ_log/√N_eff,lognormal | 0.76 | 0.32 | 4.9 | 0 / 8 |

Bootstrapping over source realizations (500 draws, resampling the 8 seeds within each condition and refitting), point sensors at n = 1:

* CV exponent: median 0.78, 95% CI [0.46, 1.06] — **1 is not excluded**
* L/sqrt(A) exponent: median 0.38, 95% CI [0.14, 0.70] — **1 is excluded**

So the amplitude exponent is compatible with the first-order prediction and the length exponent is not. The single-condition CIs on alpha are wide enough that its apparent sublinearity in CV would not on its own justify a claim; what makes it credible is that it is reproduced at all eight designs (0.78–0.83) and that the lognormal generator predicts exactly this substitution in advance.

![Figure 2](figures/fig2_collapse.png)

![Figure 3](figures/fig3_model_comparison.png)

## 6. Footprint scale (exploratory)

The footprint correlation length `L_H` was computed with the existing `enforceflux.analysis.footprint_scale.footprint_correlation_length` applied to the bLS Jacobian. Because that Jacobian is design-only, there are exactly eight values:

| geometry | n | L_H (m) | total path length (m) |
|---|---|---|---|
| open_path | 1 | 185 | 1000 |
| open_path | 2 | 169 | 1000 |
| open_path | 3 | 158 | 1000 |
| open_path | 4 | 150 | 1000 |
| point | 1 | 92 | 0 |
| point | 2 | 115 | 0 |
| point | 3 | 111 | 0 |
| point | 4 | 91 | 0 |

`Pi = L/L_H` does not beat `L/sqrt(A)`. Pooled across all designs with a single intercept, Pi gives RMSE 0.241 against 0.252 for L/sqrt(A) — a marginal improvement, not a decisive one. With per-design intercepts the two are identical by construction, since `log Pi = log L - log L_H` and `L_H` is constant within a design. A saturating length `L_eff = sqrt(L^2 + c·L_H^2)` optimizes to c = 0.00, i.e. no saturation term is preferred. L_H spans only 91–185 m across the eight designs, which is too narrow a range to identify a footprint dependence. **This is a null result about the experiment's leverage, not evidence that footprint scale is irrelevant.**

## 7. Decomposition: how much is source, how much is transport

The retrieval is an unconstrained weighted least square (verified to reproduce `x_opt` to 3e-17 kg/s over all 576 runs). Replacing the LES-generated observations with the bLS forward model of the same true field therefore gives, in closed form, the retrieval that a perfect transport operator would have produced:

```
Q_pred = [ (g/Se)·(J e_true) + x_prior/Sa ] / [ (g/Se)·g + 1/Sa ]
```

`e_pure = (Q_pred - Q_true)/Q_true` is pure source representativeness; the remainder is bLS-versus-LES operator mismatch. Mean run-to-run SD of each component:

| geometry | sigma_rep total | pure representativeness | transport |
|---|---|---|---|
| open_path | 13.2% | 13.5% | 3.8% |
| point | 15.7% | 18.3% | 11.9% |

Source representativeness dominates for both geometries. Point sensors carry roughly three times the transport component that open paths do, which is the expected consequence of a 1000 m path integral averaging over eddy structure that a point does not. Note that the components are not independent — for point sensors the total is smaller than the pure term, so the two partially cancel.

## 8. Sensor number and geometry

| geometry | log-n exponent | 95% CI | 1/√n in CI | 0 in CI |
|---|---|---|---|---|
| open_path | -0.038 | -0.141 to +0.065 | no | yes |
| point | -0.489 | -0.584 to -0.393 | yes | no |

Point sensors follow 1/sqrt(n); open paths do not improve at all. This is a fact about the design, not about open paths in general: total open-path length is held at 1000 m, so n = 1 is one 1000 m path and n = 4 is four 250 m paths covering the same ground. **Any manuscript claim that open paths do not benefit from replication must state this constraint.**

Geometry changes the coefficient, not the functional form. At n = 1 the open-path sigma_rep is on average 0.59x the point value, and the fitted alpha and beta intervals overlap at every n (Figure 5). Spatial integration acts like a larger coefficient on the same law, not a different law. The data do **not** support restating this as increased effective spatial replication of the *source*: the source total is fixed by construction, so there is no source-total sampling variance for a measurement to average down.

![Figure 4](figures/fig4_sensor_number.png)

![Figure 5](figures/fig5_geometry.png)

## 9. Uncertainty calibration

| H | R = mean(\|e\|/σ) | 68% coverage, reported | 68%, widened | 95%, reported | log score, reported | log score, widened |
|---|---|---|---|---|---|---|
| 0.050 | 0.18 | 1.00 | 1.00 | 1.00 | 3.95 | 3.94 |
| 0.100 | 0.30 | 1.00 | 1.00 | 1.00 | 3.92 | 3.88 |
| 0.125 | 0.21 | 1.00 | 1.00 | 1.00 | 3.95 | 3.91 |
| 0.200 | 0.53 | 0.88 | 0.89 | 1.00 | 3.78 | 3.71 |
| 0.250 | 0.37 | 0.97 | 0.98 | 1.00 | 3.87 | 3.80 |
| 0.500 | 0.61 | 0.81 | 0.90 | 0.98 | 3.68 | 3.56 |
| 1.000 | 0.97 | 0.58 | 0.83 | 0.89 | 3.23 | 3.25 |

The reported posterior sigma is 28% of Q_true on average and is set almost entirely by the prescribed representation-error budget in the flux config (`sigma_repr_fraction: 0.3`), so it varies with the measurement design and not at all with the source field. The consequence is visible in Figure 6C: reported sigma takes eight values, while sigma_rep varies by a factor of ten at fixed design.

At low H this makes the intervals far too conservative (coverage 1.00 against a nominal 0.68). At the highest H it becomes genuinely under-calibrated (0.58). Adding an empirical `sigma_total^2 = sigma_reported^2 + sigma_rep(H)^2` term, with sigma_rep(H) taken from a leave-one-condition-out fit so it is never calibrated on the condition it scores, restores the top end to 0.83 and improves the log score there (3.23 to 3.25), but degrades the mean log score across all H (3.77 to 3.72) by over-widening the already-conservative low-H conditions.

So H predicts *where* the formal uncertainty fails, but the additive correction is not a net improvement in this OSSE, because the prescribed sigma_repr already over-covers the bulk. This is a candidate parameterization derived from one simulated experiment, not a general uncertainty correction.

![Figure 6](figures/fig6_calibration.png)

## 10. Limitations

* **Eight source realizations per condition.** Every sigma_rep has a 95% interval spanning roughly a factor of two. Nine design points support two free exponents and no more.
* **One turbulence realization for all 576 inversions.** These are source-representativeness statistics conditional on a single eddy field, one wind direction and one 45-minute window. Nothing here bounds atmospheric variability.
* **A 4–5% operator floor.** The tagged-tracer operator's own error sets a floor under the reported spreads, non-negligible for the smallest ones (the CV = 0.5, L = 100 m open-path conditions).
* **L = 500 m is a finite-domain corner.** The realized correlation scale saturates at 179 m. Any L exponent fitted over 100–500 m in a 1 km domain is contaminated by this.
* **Conserved open-path length** confounds 'open path' with 'more spatial integration' at n = 1 and with 'subdivision' at n > 1.
* **The prescribed sigma_repr** means the calibration result speaks to this inversion configuration, not to bLS inversions in general.
* **Only three L and three CV values**, on a grid too coarse to distinguish a power law from a saturating function.

## 11. Answers to the decision criteria

1. **Does CV/sqrt(N_eff) collapse better than CV or L alone?** No. With `N_eff = A/L^2` it is worse than CV alone (mean R^2 0.52 vs 0.60; LOCO 0.43 vs 0.41). With a bounded-domain lognormal N_eff it beats both (R^2 0.83), but still loses to the free-exponent fit at every design.
2. **Are the fitted exponents consistent with CV·L/sqrt(A)?** Partly. alpha = 0.81 ± 0.02 on CV, and the bootstrap CI does not exclude 1; on sigma_log it is 1.14 ± 0.03, i.e. linear. beta = 0.49 ± 0.09 on L/sqrt(A), and the bootstrap CI [0.14, 0.70] excludes 1. The amplitude half of the proposed law holds; the length half does not.
3. **Is A/L² a reasonable N_eff?** No. It ignores the factor 2π from the integral correlation area, diverges from the true effective sample size on a bounded domain (by 5–6x here), and ignores the lognormal transformation, which makes N_eff CV-dependent.
4. **Does the scaling hold across sensor counts?** Yes for the exponents. alpha spans 0.78–0.83 and beta 0.31–0.54 across all eight designs. The coefficient shifts with n for point sensors (as 1/sqrt(n)) and not at all for open paths.
5. **Does open-path sampling change the coefficient or the form?** The coefficient. Exponent CIs overlap at every n (Figure 5B, 5C).
6. **Can H predict under-calibration?** Yes, monotonically: R rises from 0.18 to 0.97 and 68% coverage falls from 1.00 to 0.58 across the six distinct H values. The additive correction fixes the tail but over-widens the bulk.
7. **How strong is the evidence?** See the verdict.

## 12. Recommended manuscript language

> Representativeness error scales as the product of an emission-amplitude term and a source-length term, but not with the exponents a first-order independent-patch argument predicts. Across all eight measurement designs the run-to-run spread is well described by `sigma_rep = a · sqrt(ln(1+CV²)) · (L/sqrt(A))^beta` with beta = 0.49 ± 0.09, not 1. The amplitude dependence is linear in the standard deviation of the underlying log-normal field rather than in its coefficient of variation, as the field construction implies. The length dependence is substantially weaker than `A/L²` replication counting predicts, because on a bounded domain the effective number of independent source regions saturates: at L = 500 m in a 1 km domain the realized correlation scale is only 179 m and the domain is worth fewer than three independent samples.
>
> We therefore present `H = CV/sqrt(N_eff)` as an organizing variable for interpreting when network design is limited by source heterogeneity, not as a predictive scaling law. Its main practical value in this experiment is diagnostic: `H` orders the conditions in which formal posterior uncertainty ceases to bound actual flux error.

Avoid claiming: that `sigma_rep ∝ CV·L`; that spatial integration is equivalent to additional source replication; that the sensor-count results generalize beyond a network on one upwind line; or that any of this is separable from the single turbulence realization it was measured under.

## SCALING VERDICT

```
Proposed scaling: PARTIALLY SUPPORTED.
  The separable amplitude x length form is supported at all 8 designs.
  The specific exponents alpha = beta = 1 are not: beta is near 1/2 and
  its bootstrap CI excludes 1. As written, sigma_rep = a*CV*L/sqrt(A)
  predicts worse than CV alone.

Best empirical relationship:
  sigma_rep  =  0.33 * sqrt(ln(1+CV^2)) * (L/sqrt(A))^0.54    [open path]
  sigma_rep  =  0.44 * sqrt(ln(1+CV^2)) * (L/sqrt(A))^0.38    [point, n=1]
  with the point coefficient falling as 1/sqrt(n) and the open-path
  coefficient flat in n (conserved 1000 m total path length).

Main evidence:
  R^2 = 0.93-0.99 in log space and leave-one-condition-out RMSE
  0.08-0.24, versus 0.43 for the proposed form. Exponents
  reproduce across all 8 independent measurement designs (alpha
  0.78-0.83, beta 0.31-0.54), which is the strongest single
  argument that the relationship is real and not a nine-point fit.

Main failure mode:
  The length exponent. A/L^2 replication counting assumes an unbounded
  domain; the 1 km domain saturates. The recovered correlation scale is
  90 m, 147 m and 179 m for nominal L = 100, 250 and 500 m.
  Secondarily, CV is the wrong amplitude variable for a lognormal field;
  sqrt(ln(1+CV^2)) is, and substituting it makes the amplitude exponent 1.

Implication for the manuscript:
  Present H as an empirical organizing relationship (b), not a predictive
  scaling law (a). It is stronger than a conceptual interpretation (c):
  the separable form and its exponents replicate across designs and
  cross-validate. But three L values in a domain only 2 L wide at the top
  end, under one eddy field, cannot establish a law. The most defensible
  framing is diagnostic: H orders the conditions under which reported
  posterior uncertainty stops bounding actual flux error.

Highest-value next simulation:
  A larger source domain at fixed L. The single most confounded result
  here is the length exponent, and it is confounded by A/L^2 -> 4 at
  L = 500 m. Rerunning L = 100-500 m in a 3 km x 3 km source domain would
  separate a genuine beta ~ 1/2 from a finite-domain artifact, and it
  needs no new turbulence: the tagged-tracer operator already makes any
  emission field a matrix-vector product. Second priority is 3-4
  turbulence realizations at one (CV, L), to establish how much of the
  coefficient a is eddy-field specific.
```
