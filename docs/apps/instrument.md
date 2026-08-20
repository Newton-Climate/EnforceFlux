# `instrument` — sample the concentration field with virtual sensors

**What it does.** Reads a concentration field produced by `dispersion` and
plays back what a real sensor would have measured at each time step: applies
the sensor's forward operator (a point sample, a line integral for open-path,
a footprint-weighted average for eddy covariance, etc.) and adds a realistic
heteroscedastic noise draw. The result is an observation file that looks
exactly like a real measurement stream, so the downstream `flux` stage cannot
tell whether it is looking at truth-derived or field-collected data.

## Pipeline position

```
dispersion  →  [ instrument ]  →  flux  →  analysis
```

`instrument` requires an upstream `dispersion` RunDir that ran in
`simulation` mode (i.e. produced `concentration.nc`).

## Run it

```bash
enforceflux instrument --config configs/main/instrument.yaml
```

The stage prints one line per configured sensor and writes an observation
NetCDF plus a human-readable CSV under `runs/<run.name>/instrument/`.

## Config walkthrough

Full annotated template:
[`configs/main/instrument.yaml`](../../configs/main/instrument.yaml).

### `stage` / `run` / `inputs`

```yaml
stage: instrument
run:
  name: my_experiment
inputs:
  dispersion: ../../runs/single_source_aermod/dispersion
```

`inputs.dispersion` **must** point at a completed dispersion run — either an
absolute path or one relative to the config file.

### `instrument.level_index` / `release_index`

Which slice of the concentration field to sample.

- `level_index: 0` — surface layer (almost always what you want for
  ground-based sensors).
- `release_index: 0` — the first / only release. Change only when your
  dispersion output has multiple per-source releases stored separately.

`variable_name:` is auto-detected (`concentration`, then `ch4_mixing_ratio`,
then FLEXPART's `spec001_mr`); set it only to override.

### `instrument.operator.random_seed`

Fixed seed for reproducible OSSEs. Change it to draw an independent
realisation of the same experiment.

### `instrument.instruments` — the sensor list

Each entry describes one deployed sensor.

```yaml
- id: OP_north            # any string; must be unique
  tech_id: OP             # which technology (see table below)
  mode: good              # good | challenging | bad
  lon: -121.75            # geographic position (matches the field's axes)
  lat:  39.156
  z:    3.0               # sensor height (m)
  response_scale: 1.0     # multiplicative gain on the operator output
  # OP-specific geometry:
  path_length_m: 200.0
  path_bearing_deg: 0.0
  # EC/CH-style footprint:
  footprint_sigma_m: 100.0
  footprint_wind_dir_deg: 270.0
```

### Supported technologies

`tech_id` picks the sensor technology; `mode` picks its operating condition.

| `tech_id` | Operator | Observable | Modes |
|-----------|----------|------------|-------|
| `OP` | line integral | ppm concentration along a path | `good` |
| `EC` | footprint-weighted flux | nmol m⁻² s⁻¹ | `good` |
| `CH` | point flux (chamber) | nmol m⁻² s⁻¹ | `good` |
| `AIR` | aircraft column | ppb | `good` |
| `MSAT` | satellite column | ppb | `good` |
| `LP_ESN` | multi-path inversion | kg hr⁻¹ | `good` / `challenging` / `bad` |
| `IM_LS` | plume imaging | kg hr⁻¹ | `good` / `challenging` / `bad` |
| `BP_GML` | active LiDAR path integral | kg hr⁻¹ | `good` / `challenging` / `bad` |

Only `LP_ESN`, `IM_LS`, and `BP_GML` define all three modes. Everything else is
`good`-only.

### Geometry conventions

- **Positions are lon/lat** — matching the axes of the dispersion NetCDF.
  You do not have to convert to metres yourself.
- For **open-path (`OP`) sensors**, `path_length_m` and `path_bearing_deg`
  describe the beam. Bearing is degrees clockwise from north (0° = due
  north beam, 90° = due east beam).
- For **eddy-covariance / chamber footprints**, `footprint_sigma_m` is a
  Gaussian half-width and `footprint_wind_dir_deg` is the wind direction the
  footprint is aligned against.

When the dispersion output is wind-aligned (as it is for LES runs), the driver
projects every deployed sensor — including OP beam directions — into the
LES's rotated frame so the sampled path is physically the same line the real
sensor would trace.

## What lands in `runs/<run.name>/instrument/`

| File | Role | What it is |
|------|------|------------|
| `obs.nc` | `obs` | Canonical NetCDF with `time × instrument` variables: `sampled_concentration`, `y_clean` (pre-noise), `y_obs` (with noise), `valid_mask`, `noise_variance`, plus each instrument's id and lon/lat. This is what `flux` reads. |
| `obs.csv` | `obs_csv` | Flat one-row-per-`(time, sensor)` CSV — easiest to open in Excel or pandas for sanity checks. |
| `manifest.json` | — | Machine-readable output list. |
| `config.snapshot.yaml` | — | Verbatim YAML the run consumed. |

## Wiring it into `flux`

Point `flux.inputs.obs` at the instrument RunDir and set
`flux.input.mode: instrument_netcdf` — see [flux.md](flux.md).

## Common gotchas

- **"Instrument outside the concentration grid"** — the sensor's lon/lat
  falls outside the domain your dispersion run computed. Widen the domain in
  `dispersion` or move the sensor.
- **All `y_obs` are zero** — the plume did not reach the sensors. Check
  wind direction and sensor placement; open the concentration NetCDF (e.g. in
  Panoply) to see where the plume went.
- **`valid_mask` is mostly `False`** — the noise model's detection limit
  masked out the value. Move the sensor closer, increase emissions, or drop
  to a lower-noise `mode`.
- **You want a different noise realisation** — bump `operator.random_seed`.
