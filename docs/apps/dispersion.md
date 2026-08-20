# `dispersion` — run a transport model

**What it does.** Runs an atmospheric transport model — AERMOD, FLEXPART, or
MicroHH — from a single YAML config, and writes the answer in one canonical
shape regardless of which model you chose. It has two modes:

- `simulation` — a **forward run**, producing a concentration field
  `concentration(time, y, x)` NetCDF. This is what a downwind sensor would
  see if the sources emitted at their specified rates.
- `operator` — an **inversion Jacobian** `G` (observations × sources), which
  tells you how much each source contributes to each receptor per unit of
  emission. `flux` consumes this directly.

## Pipeline position

```
met (optional)  →  [ dispersion ]  →  instrument  →  flux  →  analysis
```

`dispersion` is a **source stage** — it does not need a previous stage. When
you want realistic weather it can consume a `met` RunDir, but the starter
templates ship with an inline synthetic diurnal cycle so they work standalone.

## Which model should I pick?

| Model | Runtime | Domain | Physics | When to use |
|-------|---------|--------|---------|-------------|
| **aermod** *(default)* | seconds | 100 m – few km | Steady-state Gaussian plume, similarity-scaled turbulence, plume rise. **Differentiable in JAX** — gets you sensitivities of concentration to every input essentially for free. | Sub-kilometre single-source or small-cluster problems; rapid OSSE sweeps; any time you need `dG/dinput`. |
| **flexpart** | minutes | regional (10 – 500 km) | Lagrangian particles driven by ERA5 winds. Handles wind shear, mesoscale flow. | Multi-hour or multi-day cases, regional networks, mountain terrain. Needs the compiled binary and GRIB weather. |
| **microhh** | tens of minutes to hours | 100 m – few km | Large-eddy simulation — resolves turbulence explicitly. | Ground truth for the other two; scintillation or open-path studies where the turbulent field itself matters. Needs the compiled binary. |

`aermod` is the default and needs neither an external binary nor downloaded
weather. Start there.

## Run it

```bash
enforceflux dispersion --config configs/main/dispersion.yaml
enforceflux dispersion --config configs/main/dispersion.yaml --model flexpart
enforceflux dispersion --config configs/main/dispersion.yaml --mode operator
enforceflux dispersion --config configs/main/dispersion.yaml --dry-run
```

CLI overrides:

- `--model {aermod|flexpart|microhh}` — pick the backend without editing YAML.
- `--mode {simulation|operator}` — pick the answer shape.
- `--dry-run` — write each model's input files without executing. Also the
  only path when a Fortran binary is not built.
- `--print-jacobian` — in operator mode, dump the full `G` matrix.

## Config walkthrough

The starter template
[`configs/main/dispersion.yaml`](../../configs/main/dispersion.yaml) is
extensively commented. The important ideas:

### One config, any model

**Everything above the model blocks is shared and authoritative.** Switching
`dispersion.transport.model` runs the *identical scenario* through a different
backend — same weather, same sources, same receptors, same output grid.

Each model block only carries what is unique to it: a compiled binary path, a
particle count, an LES box. A model block that repeats a shared key
(`sources`, `met`, `domain`, `receptors`, `output`) is rejected by the loader —
that would let two backends silently disagree about the scenario.

### `dispersion.transport`

```yaml
transport:
  model: aermod           # aermod | flexpart | microhh
  mode:  simulation       # simulation | operator
  start: "2020-03-31T00:00:00"
  end:   "2020-03-31T06:00:00"
```

`start`/`end` bound both the meteorology window and the run period.

### `dispersion.met`

Where the weather comes from. Two styles:

**Inline records** (what the starter template uses):

```yaml
met:
  records:
    - {time: "2020-03-31T00:00", wind_speed_m_s: 2.5, wind_direction_deg: 250.0,
       temperature_k: 285.0, mixing_height_m: 300.0,
       friction_velocity_m_s: 0.20, sensible_heat_flux_w_m2: -20.0}
    - ...
```

Only `wind_speed_m_s` and `wind_direction_deg` are required. Everything else has
sensible defaults. `wind_direction_deg` is the **meteorological** convention:
the direction the wind comes *from*.

FLEXPART **cannot** consume inline records — its Fortran binary reads GRIB
directly. For a FLEXPART run, use `met.era5` (see below).

**Real ERA5 from a `met` run** (see [met.md](met.md)):

```yaml
met:
  era5:
    meteo_dir:      ../../runs/sacramento_valley_2020_july/met
    available_file: ../../runs/sacramento_valley_2020_july/met/AVAILABLE
    longitude: -121.75    # profile location for AERMOD/MicroHH
    latitude:   39.15
    surface_roughness_m: 0.15
```

The AERMOD and MicroHH plugins convert this to their own forcing shape via the
shared meteorology adapter — see the README section on `MetSeries`.

### `dispersion.domain`

```yaml
domain:
  origin_lon: -121.75    # the ONLY lon/lat in this file
  origin_lat:  39.15
  x_min: -4300.0         # extent in metres east/north of the origin
  x_max:  4300.0
  y_min: -4450.0
  y_max:  4450.0
  spacing_m: 100.0
  heights_m: [100.0]     # FLEXPART output level tops (m AGL)
```

**Geometry is a contract.** `origin_lon` / `origin_lat` are the only geographic
coordinates in the file. All sources, receptors, and domain bounds are in
**metres** east/north of that origin. AERMOD and MicroHH consume the metres
directly; the FLEXPART path handles a small local projection on the way out.

### `dispersion.sources`

```yaml
sources:
  - id: leak
    x_m: 0.0
    y_m: 0.0
    alt_m: 5.0
    emission_rate_kg_s: 2.7777778e-2   # 100 kg/hr
    prior_mean_kg_s: 1.4e-2            # used only in operator mode
    prior_std_kg_s:  1.4e-2
```

Each source has a metre-frame position (relative to the domain origin), a
release height, and an emission rate. For inversion (operator mode), also give
a prior mean and standard deviation.

### `dispersion.receptors`

Optional in `simulation` mode; **required** in `operator` mode.

```yaml
receptors:
  - {id: tower_n, x_m:    0.0, y_m:  650.0, alt_m: 3.0}
  - {id: tower_e, x_m:  600.0, y_m:    0.0, alt_m: 3.0}
```

### Model-specific blocks

Only the block matching `transport.model` is read.

```yaml
aermod:
  reduce: stack              # stack | mean — see below
  default_stack: {height_m: 5.0}
  receptor_path_samples: 1

flexpart:
  executable: ../../flexpart/src/FLEXPART
  options_dir: ../../flexpart/options
  n_particles: 100000
  species_number: 24         # 24 = CH4
  output_step_seconds: 3600
  sync_seconds: 900

microhh:
  executable: ../../microhh/build/microhh
  num_workers: 4
  grid: {itot: 192, jtot: 96, ktot: 64, xsize: 3840.0, ysize: 1920.0, zsize: 2048.0}
  spinup_seconds: 1800
  runtime_seconds: 3600
  sampletime: 60
  met_reduce: daytime_mean
```

**About `aermod.reduce`.** AERMOD solves every meteorological hour separately.
`reduce` controls what happens to the resulting hour axis:

- `stack` (default) — every *(hour, receptor)* pair is its own observation
  row. This is where the information content of a varying wind lives, and is
  what you almost always want for OSSEs.
- `mean` — collapse to a period-mean observation. Only correct when your real
  observation genuinely averages the whole window.

## Using real ERA5

1. Run [`met`](met.md) to download and index ERA5 for your window/box.
2. Replace the inline `met.records:` with `met.era5:` pointing at the RunDir
   the met stage created.

For FLEXPART, that is the only supported met source. For AERMOD and MicroHH,
the adapter derives everything the model needs (surface stress, boundary-layer
height, wind profile) from the same ERA5 files — see the meteorology-adapter
section of the top-level README for what conversions happen and why
`surface_roughness_m` is a parameter you supply rather than one ERA5 provides.

## What lands in `runs/<run.name>/dispersion/`

| Mode | File | Role | What it is |
|------|------|------|------------|
| `simulation` | `concentration.nc` | `concentration_field` | Canonical `concentration(time, y, x)` in ng m⁻³. Same layout for every backend. |
| `operator` | `jacobian.npz` | `jacobian` | `G` matrix (obs × sources), row/column labels, and units. |
| both | `manifest.json` | — | Machine-readable output list. |
| both | `config.snapshot.yaml` | — | The YAML this run consumed. |

Each backend's native output (FLEXPART's six-dimensional grid, MicroHH's binary
cross-sections) is kept alongside `concentration.nc` for debugging.

## Common gotchas

- **"metric key set on a lat/lon field"** or vice versa — you gave one of
  `origin_lon/origin_lat/x_min/…` where the other kind was expected. The
  loader intentionally rejects mixed geometries.
- **FLEXPART with `--model flexpart` and inline records** — not supported.
  Either switch to `met.era5:` or use the bundled test met from the FLEXPART
  submodule (see the single-source FLEXPART example config).
- **FLEXPART fails on macOS occasionally** — see the "FLEXPART macOS
  flakiness" note in the top-level project memory; run single-threaded and
  validate output time coverage before reusing.
- **The concentration field is empty everywhere** — usually the wind is blowing
  the plume out of your domain. Check `wind_direction_deg` (it is the direction
  the wind comes *from*) and your `x_min/x_max/y_min/y_max` bounds.
- **Operator mode complains about missing receptors** — receptors are optional
  in simulation mode but mandatory in operator mode. Add them.
