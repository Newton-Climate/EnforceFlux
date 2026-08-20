# `flux` — solve the Bayesian inverse problem

**What it does.** Combines a transport model (either a concentration field or
a pre-built Jacobian from `dispersion`) with an observation stream (from
`instrument`, or receptors sampled directly from a field) and solves for the
posterior emission rates by **optimal estimation**:

```
x̂ = x_a + S_a Gᵀ (G S_a Gᵀ + S_e)⁻¹ (y − G x_a)
```

where `x_a` is your prior emissions, `S_a` its covariance, `S_e` the observation
noise covariance, `G` the transport Jacobian, and `y` the observations.

The state `x` is a **time series of emission rates per source** — one value per
source per simulation timestep. Internally the sustained unit-rate release from
`dispersion` is treated as a step response and `G` is the block-Toeplitz
convolution of the first-differenced (impulse) kernel. That is the reason you
can invert a *time-varying* leak from a *steady-source* simulation without
rerunning the model.

## Pipeline position

```
dispersion  →  (instrument)  →  [ flux ]  →  analysis
```

`flux` always needs a `dispersion` RunDir to build `G`. It optionally consumes
an `instrument` (or `obs`) RunDir when you want to use realistic sensor noise
instead of hand-specified receptors.

## Run it

```bash
enforceflux flux --config configs/main/flux.yaml
```

Prints the shapes and convergence status, then writes summary JSON, matrices
NPZ, and a posterior CSV to `runs/<run.name>/flux/`.

## Config walkthrough

Full annotated template:
[`configs/main/flux.yaml`](../../configs/main/flux.yaml).

### `stage` / `run` / `inputs`

```yaml
stage: flux
run:
  name: my_experiment
inputs:
  dispersion: ../../runs/my_experiment/dispersion    # always required
  obs:        ../../runs/my_experiment/instrument    # for instrument_netcdf mode
```

### `flux.input.mode`

The two ways to build observations:

- **`simulation_receptors`** — Sample the dispersion concentration field at the
  `receptors:` you list in this YAML. You supply the observed values and their
  noise sigma directly. Fast and self-contained; good for pure math OSSEs.
- **`instrument_netcdf`** — Read `y_obs` and per-observation noise variance
  from an upstream `instrument` (or `obs`) RunDir. Includes the real forward
  operator and heteroscedastic noise model. **Preferred for realistic OSSEs.**
  The `receptors:` block is ignored.

There is also a **`prebuilt_operator`** mode used by the source-heterogeneity
experiments — you rarely wire this by hand; it activates automatically when the
upstream dispersion run produced a Jacobian and a basis mapping.

### `flux.receptors` (only for `simulation_receptors`)

```yaml
receptors:
  - id: rec_north
    lon: -121.75
    lat:  39.156
    observed: [1.2, 1.3, 1.5, 1.6]      # time series; length == n_timesteps
    sigma:    0.2                        # scalar OR matching-length list
```

### `flux.observations.default_sigma`

Fallback observation noise (`ppm` or `ng m⁻³`, matching your field units) used
when the instrument stream does not carry its own. In pure `instrument_netcdf`
runs, the per-sample noise variance from the operator overrides this.

### `flux.inversion` — the solver

```yaml
inversion:
  method: linear             # linear | nonlinear | nonnegative
  prior_flux_kg_s: 0.0
  prior_variance:  1.0e-8
  prior_sigma_fraction: 0.5
```

- **`method: linear`** — closed-form OE. Correct for a passive tracer, which is
  linear in flux. This is the right choice almost always.
- **`method: nonlinear`** — iterative Levenberg-Marquardt around `F(x) = G x`.
  Same answer as `linear` for a linear forward, just slower. Use only when
  experimenting with a non-linear forward operator.
- **`method: nonnegative`** — bounded-variable inversion that forbids negative
  fluxes. Useful when the linear answer keeps posting small negative values you
  want to suppress.

### Prior emissions and covariance

- **`prior_flux_kg_s`** — scalar (applied to every source and window), length
  `n_sources`, or a nested `n_sources × n_windows` list.
- **Prior covariance precedence** (first found wins):
  1. `prior_covariance_diag` — per-source variances, length `n_sources` or
     `n_sources × n_windows`.
  2. `prior_variance` — scalar; same for every source.
  3. `prior_sigma_fraction × |x_a|` — degenerates to zero when the prior
     mean is zero, which pins the posterior to the prior. Prefer setting an
     explicit `prior_variance` when your prior mean is zero.

### `flux.inversion.background`

Optional nuisance state that absorbs a background offset in the observations:

- `constant` — one scalar background per run.
- `gradient` — a plane `β₀ + β_x·lon + β_y·lat` (requires
  `simulation_receptors` mode).

Set `model: none` (or omit) to disable.

## What lands in `runs/<run.name>/flux/`

| File | Role | What it is |
|------|------|------------|
| `summary.json` | `summary` | Prior and posterior fluxes with `1σ`, source names, convergence, upstream RunDir pointers, background diagnostics. Human-readable. |
| `matrices.npz` | `matrices` | `G`, `y_obs`, `Se_diag`, `x_prior`, `Sa_diag`, `x_opt`, `y_prior`, `y_opt`, posterior covariance `Sx`, averaging kernel `A`. Numpy-loadable. |
| `posterior.csv` | `posterior` | One row per `(source, flux_window)` with prior, posterior, and posterior sigma in kg s⁻¹. |
| `manifest.json` | — | Machine-readable output list. |
| `config.snapshot.yaml` | — | Verbatim YAML the run consumed. |

## Reading the answer

Load `matrices.npz` in Python:

```python
import numpy as np
d = np.load("runs/my_experiment/flux/matrices.npz")
x_opt   = d["x_opt"]           # posterior emission rates
Sx      = d["Sx"]              # posterior covariance
A       = d["averaging_kernel"]
sigma   = np.sqrt(np.diag(Sx))
```

Feed the same file into the `analysis` stage for DFS, per-cell uncertainty
reduction, sensor ablation, and plots — see [analysis.md](analysis.md).

## Common gotchas

- **The posterior equals the prior.** Your prior variance is too small
  relative to observation noise: no information is flowing in. Increase
  `prior_variance`.
- **The posterior is huge and noisy.** Your `G` is nearly rank-deficient
  (sensors don't constrain some sources). Look at `averaging_kernel` diagonals
  in `analysis` — anything ≪ 1 is being pulled to the prior.
- **`instrument_netcdf` mode complains about missing `inputs.obs`.** Add an
  `obs:` line under `inputs:` pointing at the instrument RunDir.
- **You expected time-varying posteriors but got constants.** Check that your
  dispersion `end - start` and output step generate more than one timestep,
  and that the observation stream has matching length.
- **Nonnegative + background nuisance.** Not supported — the nonnegative solver
  ignores the background block. Use `linear` if you need both.
