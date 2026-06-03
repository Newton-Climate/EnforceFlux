# EnforceFlux

EnforceFlux is a plugin-driven OSSE (Observation System Simulation Experiment) framework for evaluating methane monitoring systems against legal enforceability metrics. It wraps the FLEXPART Lagrangian particle dispersion model to simulate CH4 transport and provides a Bayesian inversion pipeline for evaluating how well a given instrument network can attribute emissions.

**Pipeline stages:**

1. Source process model — flux, location, timing
2. Atmospheric transport — Gaussian baseline or FLEXPART Lagrangian
3. Forward operator (G matrix) — receptor concentrations per unit source
4. Instrument operator — noise model and averaging
5. Virtual observations (OSSE)
6. Flux inversion — Bayesian linear inversion
7. Evaluation metrics — Fisher information, posterior covariance, averaging kernel

---

## Contents

- [Installation](#installation)
- [Compiling FLEXPART](#compiling-flexpart)
- [ERA5 meteorological forcing](#era5-meteorological-forcing)
- [Methane Transport Simulation](#methane-transport-simulation)
  - [Quick start](#quick-start-simulation)
  - [YAML config reference](#yaml-config-reference)
- [OSSE Pipeline](#osse-pipeline)
- [Plugin system](#plugin-system)
- [Repo layout](#repo-layout)
- [Testing](#testing)

---

## Installation

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

This installs the package in editable mode and registers all built-in plugins. Run the Gaussian-transport quickstart to verify the Python stack:

```bash
python -m enforceflux.cli --config examples/quickstart_config.json
```

---

## Compiling FLEXPART

FLEXPART must be compiled from source before any transport simulation can run. The binary lives at `flexpart/src/FLEXPART` after a successful build.

### System dependencies

**macOS (Homebrew):**

```bash
brew install gcc eccodes netcdf netcdf-fortran
```

**Linux (apt):**

```bash
sudo apt-get install gfortran libeccodes-dev libnetcdf-dev libnetcdff-dev
```

### Build environment

On macOS the compiler needs help finding Homebrew headers and libraries. Set these before building (add them to your shell profile to make them permanent):

```bash
export CPATH="/opt/homebrew/include:$(brew --prefix eccodes)/include:$(brew --prefix netcdf)/include:$(brew --prefix netcdf-fortran)/include"
export LIBRARY_PATH="/opt/homebrew/lib:$(brew --prefix eccodes)/lib:$(brew --prefix netcdf)/lib:$(brew --prefix netcdf-fortran)/lib"
```

On Linux these paths are typically handled automatically by the package manager.

### Build

```python
from enforceflux.flexpart import FlexpartCompiler

compiler = FlexpartCompiler()
result = compiler.build(jobs=4)   # parallel make
print(result.executable)          # → …/flexpart/src/FLEXPART
```

Or from the command line:

```bash
python - <<'PY'
from enforceflux.flexpart import FlexpartCompiler
FlexpartCompiler().build(jobs=4)
PY
```

`FlexpartCompiler` automatically patches the upstream makefile to remove x86-specific flags (`-mcmodel=large`, `-march=native`) that do not compile cleanly on Apple Silicon, and pre-populates `gitversion.txt` so the GNU `sed -i` rule is not needed.

### Runtime environment

When running FLEXPART with OpenMP, set these before launching:

```bash
ulimit -s unlimited
export OMP_PLACES=cores
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=4
```

If you are driving FLEXPART from the Python plugin, pass these in the transport config `env` field instead of exporting them globally.

### Verify the binary

```bash
pytest -q                              # full suite
pytest -q -m flexpart_integration     # binary smoke test only
```

The smoke test compiles if needed and checks that the binary prints `Welcome to FLEXPART`.

---

## Methane Transport Simulation

`FlexpartSimulation` is a standalone forward-simulation wrapper. It accepts a YAML config describing the domain, emission sources, and run settings; writes all FLEXPART input files; executes the model; and converts the gridded concentration output to a clean NetCDF file.

It supports two source types:

- **Point sources** — single-location releases (landfills, well pads, dairies). Emission rate is specified in kg s⁻¹ and total mass is integrated over the release duration.
- **Diffuse sources** — area emissions (rice paddies, wetlands, agricultural zones). Emission flux is specified in kg m⁻² s⁻¹. The bounding box is subdivided into a regular lat/lon grid of FLEXPART release cells; each cell's mass is computed using the proper cosine-latitude spherical area.

### Quick start (simulation)

```python
from enforceflux.flexpart import FlexpartSimulation

sim = FlexpartSimulation.from_yaml("examples/simulation_config.yaml")

# Inspect generated FLEXPART input files without running:
run_dir = sim.prepare()

# Full run → returns path to output NetCDF:
output_nc = sim.run()
```

Or construct programmatically:

```python
from datetime import datetime, timezone
from enforceflux.flexpart import (
    FlexpartSimulation, SimulationConfig, PointSource, DiffuseSource,
)

t0 = datetime(2020, 6, 15, 6, tzinfo=timezone.utc)
t1 = datetime(2020, 6, 15, 18, tzinfo=timezone.utc)

cfg = SimulationConfig(
    executable="flexpart/src/FLEXPART",
    options_dir="flexpart/options",
    available_file="flexpart/AVAILABLE",
    meteo_dir="/data/era5/20200615",
    run_dir="runs/simulation",
    start=t0,
    end=t1,
    output_step_s=3600,
    domain_lon_min=-123.0,
    domain_lat_min=37.0,
    domain_lon_max=-119.0,
    domain_lat_max=40.0,
    domain_dx=0.1,
    domain_dy=0.1,
    heights_m=[100.0, 500.0, 1000.0],
    sources=[
        PointSource(
            id="landfill_A",
            lon=-121.5,
            lat=38.6,
            alt_m=5.0,
            emission_rate_kg_s=5e-4,
            start=t0,
            end=t1,
        ),
        DiffuseSource(
            id="rice_cv",
            lon_min=-122.0, lon_max=-121.0,
            lat_min=38.5,   lat_max=39.5,
            alt_m=2.0,
            emission_flux_kg_m2_s=1.4e-9,
            start=t0,
            end=t1,
            cell_size_deg=0.1,
        ),
    ],
    output_path="outputs/simulation_ch4.nc",
)

output_nc = FlexpartSimulation(cfg).run()
```

---

### YAML config reference

A fully-annotated example lives at [`examples/simulation_config.yaml`](examples/simulation_config.yaml). Every key is documented below.

---

#### `flexpart` — FLEXPART binary and directories

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `executable` | path | — | Path to the compiled `FLEXPART` binary. Required. |
| `options_dir` | path | — | Directory containing FLEXPART input templates (`COMMAND`, `RELEASES`, `OUTGRID`, `SPECIES/`, etc.). The contents are copied into the run directory before each run. Required. |
| `available_file` | path | — | Path to the `AVAILABLE` file listing meteorological input files. Required. |
| `meteo_dir` | path | — | Directory containing meteorological input files (ERA5 or ECMWF GRIB format). Required. |
| `run_dir` | path | `runs/simulation` | Working directory for the run. **Recreated from scratch on every call to `run()` or `prepare()`.** |

Relative paths are resolved against the directory containing the YAML file.

---

#### `simulation` — timing and integration settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `start` | ISO-8601 string | — | Simulation start time in UTC, e.g. `"2020-06-15T06:00:00"`. Required. |
| `end` | ISO-8601 string | — | Simulation end time in UTC. Must be after `start`. Required. |
| `output_step_seconds` | int | `3600` | Interval (s) at which gridded concentrations are written to the output file. Also used for output averaging and receptor sampling intervals. |
| `sync_seconds` | int | `900` | FLEXPART internal synchronization interval (s). Controls particle advection sub-step. Must divide `output_step_seconds` evenly. |

---

#### `domain` — output grid

Defines the geographic extent and resolution of the gridded concentration output. Does not need to match the meteorological grid.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lon_min` | float | — | Western boundary of the output grid (degrees, −180 to 180). Required. |
| `lat_min` | float | — | Southern boundary of the output grid (degrees, −90 to 90). Required. |
| `lon_max` | float | — | Eastern boundary of the output grid. Required. |
| `lat_max` | float | — | Northern boundary of the output grid. Required. |
| `dx` | float | `0.1` | Longitude grid spacing (degrees). Coarser = faster, less memory. |
| `dy` | float | `0.1` | Latitude grid spacing (degrees). |
| `heights_m` | list of float | `[100.0, 500.0, 1000.0]` | Upper boundaries (m above ground) of the vertical output layers. The number of layers affects memory use and output file size. |

The number of grid cells in each direction is `round((lon_max - lon_min) / dx)` and `round((lat_max - lat_min) / dy)`.

---

#### `species` — tracer species

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `CH4` | Tracer name. Must correspond to a `SPECIES_<name>` file in `options/SPECIES/`. Built-in options include `CH4`, `CO2`, `CO`, `N2O`, `O3`, `SO2`. |

---

#### `sources` — emission sources

A YAML list. Each entry has a required `type` field (`point` or `diffuse`). Sources with no `start`/`end` inherit the simulation window.

##### Point source

```yaml
- type: point
  id: my_source          # unique string identifier
  lon: -121.50           # degrees, WGS-84
  lat:  38.60
  alt_m: 5.0             # release height above ground (m)
  emission_rate_kg_s: 2.0e-4   # total emission rate (kg s⁻¹)
  n_particles: 10000     # number of Lagrangian particles released
  start: "2020-06-15T06:00:00"  # optional; inherits simulation start
  end:   "2020-06-15T18:00:00"  # optional; inherits simulation end
```

Total mass released = `emission_rate_kg_s × (end − start)` in seconds.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `id` | string | — | Unique source identifier. Appears in output NetCDF attributes. |
| `lon` | float | — | Release longitude (degrees). |
| `lat` | float | — | Release latitude (degrees). |
| `alt_m` | float | `10.0` | Release altitude above ground (m). |
| `emission_rate_kg_s` | float | — | Continuous emission rate (kg s⁻¹). Required. |
| `n_particles` | int | `10000` | Number of Lagrangian particles. More particles → lower Monte Carlo noise but slower run and more memory. |
| `start` | ISO-8601 | simulation start | Release start time. |
| `end` | ISO-8601 | simulation end | Release end time. |

##### Diffuse source

```yaml
- type: diffuse
  id: rice_paddies_cv
  lon_min: -122.0
  lon_max: -121.0
  lat_min:  38.5
  lat_max:  39.5
  alt_m: 2.0
  emission_flux_kg_m2_s: 1.4e-9   # kg m⁻² s⁻¹
  cell_size_deg: 0.1               # discretization cell size (degrees)
  n_particles_per_cell: 500
  start: "2020-06-15T06:00:00"
  end:   "2020-06-15T18:00:00"
```

The bounding box is subdivided into `cell_size_deg × cell_size_deg` cells. Each cell becomes one FLEXPART `&RELEASE` block. Per-cell mass = `flux × cell_area_m² × duration_s`, where cell area is computed using a cosine-latitude spherical correction.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `id` | string | — | Unique source identifier. |
| `lon_min`, `lon_max` | float | — | Longitude bounds of the emission area (degrees). |
| `lat_min`, `lat_max` | float | — | Latitude bounds of the emission area (degrees). |
| `alt_m` | float | `2.0` | Release height above ground (m). For surface fluxes use 1–5 m. |
| `emission_flux_kg_m2_s` | float | — | Emission flux per unit area (kg m⁻² s⁻¹). Required. Typical rice paddy value: 1–2 × 10⁻⁹ kg m⁻² s⁻¹ (~90–170 mg m⁻² day⁻¹). |
| `cell_size_deg` | float | `0.1` | Cell discretization in degrees. Smaller cells give higher spatial resolution but many more FLEXPART releases (memory/cost scales quadratically). |
| `n_particles_per_cell` | int | `1000` | Particles per cell. Reduce if memory is tight; increase for lower noise. |
| `start` | ISO-8601 | simulation start | Emission start time. |
| `end` | ISO-8601 | simulation end | Emission end time. |

**Number of FLEXPART releases** from a diffuse source = `ceil((lon_max − lon_min) / cell_size_deg) × ceil((lat_max − lat_min) / cell_size_deg)`. A 1° × 1° region at 0.1° cell size produces 100 releases.

---

#### `output` — output file settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `path` | path | `outputs/simulation.nc` | Path for the output NetCDF file. Parent directories are created automatically. |
| `compress` | bool | `true` | Apply zlib compression (level 4) to all variables in the output file. Typically reduces file size 3–5×. |
| `per_source` | bool | `false` | If `true`, FLEXPART writes a separate concentration field for each individual release (`IOUTPUTFOREACHRELEASE=1`). The output file will have a release dimension. Useful for source attribution but increases file size proportionally to the number of releases. |

---

### Output NetCDF structure

The output file is a CF-1.8 compliant NetCDF4 file derived from FLEXPART's `grid_time_*.nc` output. Variable names are translated:

| FLEXPART variable | Output variable | Units | Description |
|-------------------|----------------|-------|-------------|
| `spec001` | `ch4_concentration` | ng m⁻³ | CH4 mass concentration |
| `spec001_mr` | `ch4_mixing_ratio` | ng kg⁻¹ | CH4 mass mixing ratio |

Global attributes include `simulation_start`, `simulation_end`, `source_ids`, `n_point_sources`, and `n_diffuse_sources`.

Dimensions follow FLEXPART conventions: `(time, height, latitude, longitude)` for gridded output, with an additional `releases` dimension when `per_source: true`.

---

## ERA5 meteorological forcing

FLEXPART is a **driven** model — it needs meteorological wind, temperature, and humidity fields to advect particles. ERA5 reanalysis from the Copernicus Climate Data Store (CDS) is the standard input.

### One-time setup

**1. Install optional dependencies:**

```bash
pip install -e '.[meteo]'     # adds cdsapi + eccodes to the venv
```

The system `eccodes` library must also be installed (see [Compiling FLEXPART](#compiling-flexpart) — `brew install eccodes` / `apt install libeccodes-dev` covers this).

**2. Create a CDS account and API key:**

Register for free at <https://cds.climate.copernicus.eu/>. After logging in, go to your profile page and copy your **UID** and **API key**. Create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```

### Download ERA5

```python
from enforceflux.meteo import ERA5Downloader

dl = ERA5Downloader(
    output_dir="inputs/meteo",
    timestep_hours=3,      # 3-hourly is standard for FLEXPART
)

result = dl.download(
    start="2020-06-15T00:00",
    end="2020-06-15T18:00",
    bbox=(-124, 36, -118, 41),   # (lon_min, lat_min, lon_max, lat_max)
)

print(result.available_file)     # inputs/meteo/AVAILABLE
print(result.n_timesteps)        # 8  (00–21 UTC at 3-hour intervals + 1 pad each side)
```

Point the simulation config at the downloaded data:

```yaml
flexpart:
  meteo_dir: inputs/meteo
  available_file: inputs/meteo/AVAILABLE
```

### What gets downloaded

The downloader makes three CDS requests per calendar day in the requested period:

| Request | CDS dataset | Variables |
|---------|-------------|-----------|
| Pressure levels | `reanalysis-era5-pressure-levels` | u, v, T, q, cloud liquid/ice water, cloud fraction — at 16 standard levels from 1000 to 10 hPa |
| Single levels (analysis) | `reanalysis-era5-single-levels` | 10 m winds, 2 m T/Td, MSLP, surface pressure, boundary-layer height, SST, total cloud cover |
| Single levels (fluxes) | same dataset | large-scale and convective precipitation, surface sensible/latent heat flux, surface momentum flux (accumulated) |

A fourth request downloads time-invariant fields (land-sea mask, orography) on the first run; these are reused for all subsequent downloads to the same `output_dir`.

After downloading, the daily GRIB files are split by timestep and merged (pressure-level + single-level + static) into one GRIB file per timestep named `EA{YYYYMMDD}{HH}`. The `AVAILABLE` index file lists all timestep files in the format FLEXPART expects.

### Caching and re-runs

Downloads are skipped if the output file already exists. Re-running `dl.download()` with the same `output_dir` and time range is safe and fast — only missing files are fetched. The `AVAILABLE` file is always regenerated to reflect whatever files are present.

### Domain and resolution

```python
dl = ERA5Downloader(
    output_dir="inputs/meteo",
    timestep_hours=1,             # 1-hourly for high-resolution runs
    pressure_levels=[             # custom levels (subset for faster downloads)
        "1000", "850", "700", "500", "300", "200", "100",
    ],
)
```

The `bbox` parameter clips ERA5 to a regional domain, which substantially reduces file size and download time. A 10° × 10° domain at 3-hourly, 16-level resolution for one day is roughly 200–400 MB.

---

## OSSE Pipeline

The full OSSE pipeline (inversion + evaluation) is driven by a JSON config and accessed via the CLI or the Python API.

```bash
python -m enforceflux.cli --config examples/quickstart_config.json
```

```python
from enforceflux.osse import run_osse
from enforceflux.config import load_config

project = load_config("examples/quickstart_config.json")
result = run_osse(project)
print(result.metrics)
```

### JSON config format

```json
{
  "random_seed": 7,
  "domain": {
    "x_min": 0,     "x_max": 1000,
    "y_min": 0,     "y_max": 1000,
    "grid_spacing": 250,
    "crs": "EPSG:32610",
    "crs_wgs84": "EPSG:4326"
  },
  "components": {
    "source":     { "plugin": "enforceflux.source.static",        "config": { "sources": [...] } },
    "instrument": { "plugin": "enforceflux.instrument.static",    "config": { "instruments": [...] } },
    "transport":  { "plugin": "enforceflux.transport.gaussian",   "config": { "sigma": 200.0, "wind": [50.0, 0.0] } },
    "inversion":  { "plugin": "enforceflux.inversion.bayesian",   "config": { "r_cond": 1e-6 } }
  }
}
```

To use FLEXPART as the transport model, swap the transport component:

```json
"transport": {
  "plugin": "enforceflux.transport.flexpart",
  "config": {
    "executable": "flexpart/src/FLEXPART",
    "base_run_dir": "runs/flexpart",
    "options_dir": "flexpart/options",
    "available_file": "flexpart/AVAILABLE",
    "meteo_dir": "/path/to/meteo",
    "unit_emission_rate": 1.0,
    "dry_run": false
  }
}
```

`dry_run: true` writes all input files and skips execution — useful for verifying the generated RELEASES and RECEPTORS files.

---

## Plugin system

Plugins are discovered via entry points. Four extension points exist:

| Entry point group | Interface | Built-in plugins |
|-------------------|-----------|-----------------|
| `enforceflux.source` | `ISourceModel` | `static` |
| `enforceflux.instrument` | `IInstrumentModel` | `static` |
| `enforceflux.transport` | `ITransportModel` | `gaussian`, `flexpart` |
| `enforceflux.inversion` | `IInversionEngine` | `bayesian` |

Register a custom plugin in your package's `pyproject.toml`:

```toml
[project.entry-points."enforceflux.transport"]
my_transport = "my_pkg.transport:MyTransportModel"
```

Then reference it in a config as `"plugin": "enforceflux.transport.my_transport"`.

---

## Repo layout

```
src/enforceflux/
    __init__.py
    cli.py                  # entry-point CLI
    config.py               # config dataclasses + JSON loader
    osse.py                 # pipeline orchestration
    metrics.py              # evaluation metrics
    core/                   # abstract base classes
    models/                 # Source, Instrument dataclasses
    flexpart/
        __init__.py
        build.py            # FlexpartCompiler — patches makefile, runs make
        runner.py           # FlexpartRunner — OSSE G-matrix runs (one run per source)
        wrapper.py          # FlexpartWrapper — build + run facade
        simulation.py       # FlexpartSimulation — YAML-driven forward simulation
    meteo/
        __init__.py
        era5.py             # ERA5Downloader — CDS fetch, GRIB split, AVAILABLE writer
    plugins/                # built-in plugin implementations
    retrieval/              # inversion engine
    utils/                  # plugin registry

flexpart/                   # upstream FLEXPART model source (submodule)
    src/                    # Fortran source + compiled FLEXPART binary
    options/                # default FLEXPART input templates
    options/SPECIES/        # tracer species definitions

examples/
    quickstart_config.json  # toy OSSE with Gaussian transport
    simulation_config.yaml  # forward CH4 simulation with FLEXPART

data/                       # California GHG inventory data + analysis scripts
docs/
    flexpart_build.md       # extended build notes
tests/
```

---

## Testing

```bash
# Fast tests — no compiled binary required
pytest -q

# Include integration test (compiles FLEXPART, runs binary)
pytest -q -m flexpart_integration
```

The integration test is marked `flexpart_integration` and is skipped by default. It:

1. Compiles FLEXPART with `FlexpartCompiler` if the binary is absent.
2. Runs the binary with a minimal pathnames file.
3. Asserts the welcome message appears in stdout/stderr.
