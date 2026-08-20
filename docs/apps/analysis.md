# `analysis` — information-content diagnostics and plots

**What it does.** Answers the question at the heart of every OSSE: *how well
can this sensor network actually see the sources?* It reads a completed run
and computes, per source cell:

- **DFS** (degrees of freedom for signal, `Tr[A]`) — an integer-ish score of
  how many independent pieces of information the observations carry.
- **Averaging kernel `A`** — how much of each posterior estimate comes from
  the data versus from the prior. Diagonal near 1 = data-driven; near 0 =
  prior-driven.
- **Posterior uncertainty reduction** — `1 − √(Sₓᵢᵢ / Sₐᵢᵢ)` per cell.
- **Sensor ablation** — how much DFS you lose (or gain) by removing (or
  adding) each sensor. Cheap ranking of which sensors carry their weight.

It can also render plots: footprints, DFS spatial maps, posterior sigma
maps, sensor rankings, and a wind rose from a met NetCDF.

## Pipeline position

```
dispersion  →  instrument  →  flux  →  [ analysis ]
```

`analysis` accepts either:

- an **`instrument`** RunDir — preferred; you get the full information-content
  suite because the noise model `R` is available; or
- a **`dispersion`** RunDir — fallback; only the sim-only diagnostics
  (footprint statistics, wind rose) since there is no `R`.

At least one of the two is required.

## Run it

```bash
enforceflux analysis --config configs/main/analysis.yaml
```

Prints a summary and writes `summary.json` plus plots under
`runs/<run.name>/analysis/`.

## Config walkthrough

Full annotated template:
[`configs/main/analysis.yaml`](../../configs/main/analysis.yaml).

### `stage` / `run` / `inputs`

```yaml
stage: analysis
run:
  name: my_experiment
inputs:
  instrument: ../../runs/my_experiment/instrument   # preferred
  # dispersion: ../../runs/my_experiment/dispersion # fallback
```

If both are provided, `instrument` wins.

### `analysis.input.kind`

- `auto` (default) — inspect the input NetCDF's variables and decide. If it
  has `y_obs` / `valid_mask`, treat as instrument; otherwise as simulation.
- `simulation` — force sim-only diagnostics.
- `instrument` — force instrument diagnostics (fails if the file lacks the
  needed variables).

### `analysis.analysis.prior_variance`

Information content is measured **relative to the prior**. This scalar sets
what "informative" means: a small prior variance means the data has to work
hard to move the posterior; a large one makes the data look very informative.
Use `prior_covariance_diag: [...]` for per-source values.

### `analysis.visualization`

```yaml
visualization:
  enabled: true
  time_index: 0
  level_index: 0
  release_index: 0
  wind_rose:
    enabled: false
    # netcdf: <path to a met file with u10 / v10 vars>
    u_var: u10
    v_var: v10
    n_dir_bins: 16
    calm_threshold: 0.2
```

The three `*_index` fields pick which slice of the concentration field to
render on the 2-D maps. Time-averaged diagnostics (DFS, averaging kernel, all
posterior statistics) are computed across the full record regardless.

**About the wind rose.** For any fixed sensor network, the distribution of
wind directions over your OSSE window is very often the single best predictor
of which sources you can actually see. If you have an ERA5 (or other) met
NetCDF with 10-m winds, enable the wind rose and point at it — the resulting
plot is the fastest way to sanity-check a puzzling DFS map.

## What lands in `runs/<run.name>/analysis/`

| File | Role | What it is |
|------|------|------------|
| `summary.json` | `summary` | DFS total, per-sensor DFS, averaging kernel diagonal, posterior variance and uncertainty reduction per source, sensor ablation ranking, eigenspectrum of the dual-space Fisher matrix, upstream RunDir pointers. |
| `plots/*.png` | `plot_<stem>` | Rendered figures when `visualization.enabled: true`. |
| `manifest.json` | — | Machine-readable output list. |
| `config.snapshot.yaml` | — | Verbatim YAML the run consumed. |

Typical fields in `summary.json`:

| Field | Meaning |
|-------|---------|
| `dfs_total` | Total degrees of freedom for signal. Roughly, how many source cells your network sees independently. |
| `dfs_per_sensor` | Contribution of each sensor to `dfs_total`. |
| `averaging_kernel` | Diagonal of `A`; 1 = fully data-driven, 0 = fully prior-driven. |
| `posterior_variance` | Diagonal of `Sₓ` per source. |
| `uncertainty_reduction` | Per-source fractional reduction from the prior. |
| `eigenvalues` | Eigenvalues of the `m×m` dual-space Fisher matrix. |

## Common gotchas

- **Everything reads as zero-information.** Your prior variance is too large
  (or too small — small variance can look uninformative in some conventions).
  Try `prior_variance: 1.0` first, then scale.
- **The DFS map is patchy in odd places.** Usually a wind-direction artefact
  — turn on the wind rose and see if the "seen" sources line up with the
  wind sectors.
- **The wind-rose plot errors out.** The NetCDF you pointed at doesn't have
  `u10` / `v10`; set `u_var` and `v_var` to your file's actual variable names.
- **You wanted full diagnostics but got sim-only.** Add
  `inputs.instrument:` — a dispersion-only run cannot produce averaging
  kernels because there is no observation noise model to invert against.
