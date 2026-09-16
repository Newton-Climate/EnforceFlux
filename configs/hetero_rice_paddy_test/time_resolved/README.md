# Time-resolved, 9-cell inversion

The sweep inverts a single total flux (`total_only: true`) from one
window-averaged observation per instrument. That caps the information content:
degrees of freedom for signal cannot exceed the number of observations, so four
sensors give at most four, and a nine-cell state would be mostly prior.

These configs invert the **time series** on the 3x3 basis instead.

## What changes, and why each piece is needed

**One bLS footprint per turbulence window** (`interval_reduce: none`). The
operator gains a row per (interval, instrument), ordered interval-major —
row index `t * n_instruments + i`, the layout the flux stage indexes.

**Turbulence recomputed per window** (`intervals_from_nature` with
`window_s: 60`). This is the part that makes the extra rows worth having. Over
the 29 windows the donor LES gives u* from 0.247 to 0.403 m/s, Obukhov length
from -69 to -16 m, and wind direction from 1.4 to 61.0 degrees. Rows built from
one window-averaged interval would all share a footprint and add no rank; rows
built from their own turbulence see the source area from different directions.
Note also that a footprint at the mean wind direction is not the mean of
footprints across directions — the previous single-interval operator was an
approximation in that sense too, not merely a coarser one.

**A restricted window** (`start_s: 3660`, `end_s: 5400`, and
`time_index_range: [0, 29]` on the flux side). bLS needs the SGS-inclusive
wall-model `ustar`/`obuk`, and only the donor run writes them, at 10 Hz, and
only out to 5400 s. The restarted runs' own columns cover the full window but
at 20 s and without those diagnostics, which would silently fall back to
resolved-covariance u* — biased low near the LES wall. So the honest window is
the 29 minutes where proper met exists, not the full 45. Extending it means
re-running the donor LES with fast column output to 6300 s.

**The nine-cell state** (`total_only: false`), which is the point of the
exercise.

## Caveat to check in the output, not assume away

`sigma_repr` was calibrated for window-averaged observations. An instantaneous
cross-section fluctuates about the bLS ensemble mean by much more than its
time average does, so the per-observation error here is larger — and correlated
in time. Read chi squared per dof: substantially above 1 means the observation
error is under-specified at this cadence and the posterior sigma is too small
by roughly its square root. That diagnostic, not the projected DFS, is what
says whether the extra rows are real information.

## Running

```bash
enforceflux dispersion --config configs/hetero_rice_paddy_test/time_resolved/l100_cv1p0_s0_n4_op_tr_operator.yaml
enforceflux instrument --config configs/hetero_rice_paddy_test/time_resolved/l100_cv1p0_s0_n4_op_tr_instrument.yaml
enforceflux flux --config configs/hetero_rice_paddy_test/time_resolved/l100_cv1p0_s0_n4_op_tr_flux.yaml
enforceflux analysis --config configs/hetero_rice_paddy_test/time_resolved/l100_cv1p0_s0_n4_op_tr_analysis.yaml
```

The operator stage builds 29 bLS windows single-threaded and is much slower
than the sweep's cached-operator path; raise `model_params.ncores` if the
run's RNG stream does not need to match the sweep.
