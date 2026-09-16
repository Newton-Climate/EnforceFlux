# Integrity checks — 576-inversion table

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
