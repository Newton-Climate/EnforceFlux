# `osse-sweep` — run many OSSEs across a parameter grid

**What it does.** Executes the full `dispersion → flux → analysis` pipeline
across a Cartesian grid of parameters, reusing intermediate results wherever
possible, and writes one **row per (parameter combination, replicate)** to a
Parquet file. Built for the source-heterogeneity experiments (paper M4
sweeps) but generic enough for any repeated OSSE.

## Pipeline position

`osse-sweep` **wraps** the per-stage apps. Each sweep point calls the same
`dispersion`, `flux`, and `analysis` code paths you would call from the
command line — the sweep driver just handles the loop, the cache, and the
tabulation.

## Run it

```bash
enforceflux osse-sweep --config configs/source_heterogeneity_e1_aermod/pilot_sweep.yaml
```

Prints the run name and the output Parquet path when finished.

## Config walkthrough

Example: [`configs/source_heterogeneity_e1_aermod/pilot_sweep.yaml`](../../configs/source_heterogeneity_e1_aermod/pilot_sweep.yaml).

### `stage` / `run` / `inputs`

```yaml
stage: osse_sweep
run:
  name: source_heterogeneity_e1_aermod_pilot
inputs: {}      # sweep is a source stage; per-point inputs come from `sweep.base`
```

### `sweep.base` — the per-point templates

```yaml
sweep:
  base:
    dispersion: dispersion.yaml
    flux:       flux.yaml
    analysis:   analysis.yaml
```

Paths are relative to the sweep YAML's directory. Every sweep point starts by
loading these three templates, then overlays the parameter values for the
current grid cell.

### `sweep.grid` — what to vary

```yaml
grid:
  dispersion.sources.config.covariance.L_m: [200, 500, 1000]
  dispersion.sources.config.cv:             [0.25, 1.0, 1.5]
  dispersion.sources.config.seed:           {range: [0, 3]}
  instrument.network: {kind: nested_space_filling, ns: [2, 4, 8, 16], seeds: [0, 1]}
  flux.inversion.prior_covariance.L_B_m:    [800.0]
```

Each key is a **dotted path** into the merged config; the value is a list of
values to try (or a shorthand like `{range: [a, b]}` for integer sweeps, or a
plugin-defined structured spec like `nested_space_filling`).

The number of runs is the Cartesian product of every axis. Keep this in mind:
`3 × 3 × 3 × (4×2) × 1 = 216` runs above.

### `sweep.cache.reuse_H_across`

```yaml
cache:
  reuse_H_across:
    - dispersion.sources.config.seed
    - dispersion.sources.config.cv
    - dispersion.sources.config.covariance.L_m
```

The transport Jacobian `H` depends only on receptors, meteorology, and the
source *geometry* — not on the emission rate realisation `cv`, the field seed,
or the correlation length `L_m` inside those. Listing those axes as
"reuse-across" tells the sweep driver it can compute `H` once and reapply it
across all combinations that differ only in those keys — often a 10–100×
speedup.

### `sweep.parallel.workers`

Number of parallel worker processes. `1` (serial) is portable and works in
locked-down environments; increase when you have cores to spare and enough
memory for `workers × per-run peak`.

### `sweep.output.parquet`

Path (relative to the config) for the resulting DataFrame. One row per sweep
point contains the input axes, the run's flux and analysis summaries flattened
into columns, and metadata about which `H` cache slot it used.

## What lands where

- The Parquet file at `sweep.output.parquet`.
- Per-point RunDirs under `runs/<sweep.run.name>/<pointhash>/…` (one folder
  per stage per point).

Load the Parquet for analysis:

```python
import pandas as pd
df = pd.read_parquet("runs/.../sweep.parquet")
df.groupby("dispersion.sources.config.covariance.L_m")["analysis.dfs_total"].mean()
```

## Common gotchas

- **The sweep is enormous.** Check the product of your axis lengths before
  starting. Add axes gradually.
- **`reuse_H_across` is wrong.** If any of the listed axes actually affects
  the Jacobian, the sweep will silently use a stale `H`. Only list axes that
  vary the emissions or the inversion, not the transport geometry.
- **Some rows are missing in the Parquet.** A point that raises inside the
  underlying stage is logged and skipped; look for `error` columns in the
  Parquet or under the corresponding RunDir.
- **Runs go serial even with `workers > 1`.** The sweep tries to keep one
  transport-cache slot per worker; if your platform doesn't fork cleanly, it
  falls back to serial. Check the driver's startup log.
