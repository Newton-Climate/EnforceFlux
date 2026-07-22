# EnforceFlux

EnforceFlux is a modular OSSE (Observing System Simulation Experiment) framework for evaluating methane monitoring systems. It combines ERA5 meteorological reanalysis, Lagrangian particle dispersion (FLEXPART) or AERMOD-style plume transport, a Bayesian inversion pipeline, and satellite data tools to answer: *how well can a proposed sensor network attribute emissions to specific sources?*

![System architecture](archetecture_diagram.png)

---

## Contents

- [Installation](#installation)
- [Capabilities](#capabilities)
- [Apps](#apps)
- [Compiling FLEXPART](#compiling-flexpart)
- [ERA5 meteorological forcing](#era5-meteorological-forcing)
- [Running a transport model](#running-a-transport-model)
- [Meteorology adapter](#meteorology-adapter)
- [AERMOD transport](#aermod-transport)
- [Methane Transport Simulation](#methane-transport-simulation)
  - [Quick start](#quick-start-simulation)
  - [YAML config reference](#yaml-config-reference)
- [OSSE Pipeline](#osse-pipeline)
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

This installs the package in editable mode and registers all built-in plugins.

**Optional extras:**

```bash
pip install -e '.[meteo]'     # ERA5 download — cdsapi, eccodes
pip install -e '.[analysis]'  # plotting + geospatial figures — matplotlib, scipy,
                              #   xarray, pandas, rasterio, cartopy, shapely
pip install -e '.[dev]'       # tooling — pytest, pytest-cov, ruff, mypy
pip install -e '.[all]'       # everything above
```

**Or use the Makefile** (creates `.venv`, installs all extras, then clones and
compiles FLEXPART):

```bash
make env && source .venv/bin/activate
make install        # pip install -e ".[all]"  +  FLEXPART build
make test           # pytest (skips flexpart_integration by default)
make figures        # regenerate the single-source OSSE figures
```

**System dependencies** (required for FLEXPART and ERA5):

```bash
# macOS
brew install gcc eccodes netcdf netcdf-fortran

# Linux
sudo apt-get install gfortran libeccodes-dev libnetcdf-dev libnetcdff-dev
```

---

## Capabilities

| Capability | Description |
|---|---|
| **Transport — FLEXPART** | Lagrangian backward-mode footprint computation using FLEXPART 11. Releases particles from sensor locations; footprint encodes emission sensitivity per source cell. Best for regional (10–500 km) domains with WRF or ERA5 met. |
| **Transport — AERMOD** | AERMOD-style steady-state plume dispersion (similarity-scaled turbulence, Briggs plume rise, bi-Gaussian convective boundary layer) implemented in JAX. Builds the full G Jacobian in milliseconds and is differentiable with respect to emissions, geometry, and meteorology. Ideal for sub-km single-source scenarios and rapid OSSE sweeps. |
| **ERA5 downloader** | Downloads ECMWF ERA5 reanalysis (pressure-level + single-level + fluxes) via CDS API and reformats to FLEXPART-ready GRIB files with an `AVAILABLE` index. |
| **Instrument operator** | Models open-path (OP), point-sensor (PT), and remote-sensing (RS) instruments. Applies per-technology noise (Sₑ) and generates simulated observations ŷ = G·x + ε. |
| **Bayesian inversion** | Computes optimal posterior x̂ = μₐ + Sₐ Gᵀ (Sₑ + G Sₐ Gᵀ)⁻¹ (y − G μₐ). |
| **Information content analysis** | Woodbury-accelerated (O(m²·n)) computation of posterior covariance Sₓ, averaging kernel A, DFS = Tr[A], and dual-space eigenspectrum. Scales to 10,000+ source cells. |
| **Sensor ablation study** | Incremental and leave-one-out sensor ranking by DFS contribution. |
| **Sacramento Valley OSSE** | Multi-source, multi-instrument OSSE over Central Valley using April and July 2020 meteorology. Benchmarks 3-sensor OP networks against 1–5 sensor configurations. |
| **Bottom-up inventory analysis** | Loads EPA GHGI gridded CH4 and USDA CDL 30 m land cover; computes per-cell rice emission factors; compares EPA GHGI vs. CARB vs. IPCC reference ranges. |
| **TROPOMI analysis** | Grids Sentinel-5P XCH4 retrievals; computes seasonal anomaly maps (rice season minus fallow); performs valley-scale enhancement detection. |

---

## Apps

The `apps/` directory contains end-to-end pipeline scripts driven by YAML configs. Run any app from the repo root with `python apps/<script>.py --config apps/<config>.yaml`.

### `met_main.py` — ERA5 downloader
Downloads ERA5 for a specified date range and geographic bounding box. Outputs FLEXPART-ready GRIB files and an `AVAILABLE` index.

```bash
python apps/met_main.py --config apps/met_main.yaml
```

### `transport_main.py` — Any transport model, one config

```bash
python apps/transport_main.py --config apps/transport_main.yaml --model aermod
```

Runs AERMOD, FLEXPART, or MicroHH from the shared schema described in
[Running a transport model](#running-a-transport-model), writing a canonical
`concentration(time, y, x)` NetCDF regardless of backend.

### `simulation_main.py` — Forward FLEXPART simulation
Runs a forward CH4 transport simulation: point and/or diffuse sources → gridded concentration field → NetCDF. Useful for visualizing how methane from a given source pattern spreads over a domain.

```bash
python apps/simulation_main.py --config apps/simulation_main.yaml
```

### `flux_main.py` — Flux inversion
Runs the full inversion pipeline: loads a pre-computed G matrix + prior emissions → Bayesian posterior → flux estimates with uncertainty. Outputs posterior flux maps and uncertainty reduction statistics.

```bash
python apps/flux_main.py --config apps/flux_main.yaml
```

### `analysis_main.py` — Information content analysis
Takes an existing transport Jacobian G and sensor configuration and computes DFS, averaging kernel, posterior covariance, and sensor ablation rankings.

```bash
python apps/analysis_main.py --config apps/analysis_main.yaml
```

### `instrument_main.py` — Instrument OSSE
Single-source instrument sensitivity experiment. Builds an analytical Gaussian-plume G for a configurable sensor network, runs the full information content analysis, and generates diagnostic figures (footprints, DFS spatial map, posterior uncertainty, sensor ablation).

```bash
python apps/instrument_main.py --config apps/instrument_main.yaml
```

### Examples

Standalone demo scripts live in `examples/`:

| Script | Description |
|---|---|
| `single_source_instrument_demo.py` | 3-sensor open-path network, 500 m from a point source. Gaussian plume G, Woodbury ICA. |
| `sacramento_valley_2020.py` | Multi-source Sacramento Valley OSSE; April vs. July met comparison. |
| `gaussian_plume_single_source_demo.py` | Minimal FLEXPART plume forward simulation with 1 source. |
| `aermod_single_source_demo.py` | AERMOD Jacobian, concentration field, and meteorological sensitivities via autodiff. |
| `era5_driven_models.py` | One ERA5 window converted into AERMOD, MicroHH, and FLEXPART forcing. |
| `osse_25kg_leak_demo.py` | OSSE for a 25 kg hr⁻¹ leak detection scenario. |

---

## Compiling FLEXPART

FLEXPART must be compiled from source before any Lagrangian transport simulation can run. The binary lives at `flexpart/src/FLEXPART` after a successful build.

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

On macOS set these before building (add to your shell profile to make them permanent):

```bash
export CPATH="/opt/homebrew/include:$(brew --prefix eccodes)/include:$(brew --prefix netcdf)/include:$(brew --prefix netcdf-fortran)/include"
export LIBRARY_PATH="/opt/homebrew/lib:$(brew --prefix eccodes)/lib:$(brew --prefix netcdf)/lib:$(brew --prefix netcdf-fortran)/lib"
```

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

```bash
ulimit -s unlimited
export OMP_PLACES=cores
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=4
```

Pass these in the transport config `env` field when driving FLEXPART from Python.

### Verify

```bash
pytest -q                              # full suite
pytest -q -m flexpart_integration     # binary smoke test only
```

---

## ERA5 meteorological forcing

FLEXPART is a **driven** model — it needs meteorological wind, temperature, and humidity fields to advect particles. ERA5 reanalysis from the Copernicus Climate Data Store (CDS) is the standard input.

### One-time setup

```bash
pip install -e '.[meteo]'     # adds cdsapi + eccodes to the venv
```

Register at <https://cds.climate.copernicus.eu/> and create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```

### Download ERA5

```python
from enforceflux.meteo import ERA5Downloader

dl = ERA5Downloader(output_dir="inputs/meteo", timestep_hours=3)

result = dl.download(
    start="2020-06-15T00:00",
    end="2020-06-15T18:00",
    bbox=(-124, 36, -118, 41),   # (lon_min, lat_min, lon_max, lat_max)
)

print(result.available_file)     # inputs/meteo/AVAILABLE
```

Point the simulation config at the downloaded data:

```yaml
flexpart:
  meteo_dir: inputs/meteo
  available_file: inputs/meteo/AVAILABLE
```

### What gets downloaded

Three CDS requests per calendar day:

| Request | CDS dataset | Variables |
|---------|-------------|-----------|
| Pressure levels | `reanalysis-era5-pressure-levels` | u, v, T, q, cloud water/ice, cloud fraction — 16 levels 1000–10 hPa |
| Single levels (analysis) | `reanalysis-era5-single-levels` | 10 m winds, 2 m T/Td, MSLP, surface pressure, BLH, SST, cloud cover |
| Single levels (fluxes) | same dataset | precipitation, surface heat fluxes, momentum flux (accumulated) |

A fourth request downloads time-invariant fields (land-sea mask, orography) on the first run; reused thereafter. Downloads are skipped when the output file already exists.

---

## Running a transport model

Every transport model runs from one YAML, and the model is a single line in it:

```bash
python apps/transport_main.py --config apps/transport_main.yaml
python apps/transport_main.py --config apps/transport_main.yaml --model flexpart
python apps/transport_main.py --config apps/transport_main.yaml --mode operator
```

Everything above the model blocks is **shared and authoritative** — meteorology,
sources, receptors, domain, output — so switching `transport.model` runs the
identical scenario through a different model. Each model then gets a block for
what has no counterpart elsewhere (a compiled binary, a particle count, an LES
box). A model block may not restate a shared key; the loader rejects that,
because silently divergent scenarios are the failure mode this design exists to
prevent.

```yaml
transport:
  model: aermod          # aermod | flexpart | microhh
  mode: simulation       # simulation (field) | operator (Jacobian)

met:
  era5: {meteo_dir: ../runs/.../meteo_april_week, longitude: -121.75, latitude: 39.15}

sources:
  - {id: leak, lon: -121.75, lat: 39.15, alt_m: 5.0, emission_rate_kg_s: 2.7777778e-2}

output:
  path: ../runs/transport_main/transport_main.nc

flexpart:
  executable: ../flexpart/src/FLEXPART
  n_particles: 100000
microhh:
  executable: ../microhh/build/microhh
  grid: {itot: 192, jtot: 96, ktot: 64}
```

### The output contract

Both modes return the same `TransportRunResult` regardless of model:

| Mode | Filled | Shape |
|---|---|---|
| `simulation` | `field`, `output_path` | canonical `concentration(time, y, x)` in ng m⁻³ |
| `operator` | `g`, `row_labels`, `column_labels` | observation × source, ng m⁻³ / (kg s⁻¹) |

Simulation output is normalised into one NetCDF layout for all three backends —
FLEXPART's six-dimensional `(nageclass, pointspec, time, height, lat, lon)` grid
and MicroHH's raw binary cross-sections both become the same
`concentration(time, y, x)` file with metric axes and 2-D `longitude`/`latitude`.
Normalisation selects one age class and level and sums releases, so each
backend's native output is kept alongside it.

Geometry is given in longitude/latitude; AERMOD's metric frame is handled by a
local azimuthal-equidistant projection centred on the domain.

**Time is never averaged away.** AERMOD solves every meteorological hour
separately, and `aermod.reduce` controls only what happens to the resulting hour
axis: `stack` (the default here) makes each *(hour, receptor)* pair its own
observation row, which is where the information content of a varying wind lives.
`mean` gives the Jacobian of a period-mean observation, and is only correct when
the observation genuinely averages the whole window.

### What each model still needs

FLEXPART and MicroHH need their compiled binaries; `--dry-run` generates their
input files without executing, which is also what happens when the binary is
absent. The generated native config is written into the run directory
(`flexpart_generated.yaml`, `microhh_generated.yaml`) rather than passed in
memory, so a run that dies inside Fortran leaves behind exactly the input it was
given.

FLEXPART cannot be driven by inline `met.records` at all — it reads ERA5 GRIB
itself, so it requires `met.era5`.

---

## Meteorology adapter

Every transport model wants its forcing in a different shape. `MetSeries` is the
single canonical representation, read once from ERA5 and converted per model, so
a run's meteorology is specified in exactly one place:

```
ERA5 GRIB ──► MetSeries ──┬──► AERMOD    SurfaceMet         (per hour)
                          ├──► MicroHH   Forcing            (one steady column)
                          └──► FLEXPART  FlexpartMetSource  (GRIB paths)
```

```python
from enforceflux.meteo import met_series_from_era5, to_aermod, to_microhh_forcing, to_flexpart

series = met_series_from_era5(
    "runs/sacramento_valley_2020/meteo_april_week", -121.75, 39.15,
    start="2020-03-31", end="2020-04-01", surface_roughness_m=0.15,
)
print(series.summary())          # hourly U, direction, T, zi, u*, H, L

aermod_met  = to_aermod(series)                          # list[SurfaceMet]
les_forcing = to_microhh_forcing(series, reduce="daytime_mean")
flexpart    = to_flexpart(series).require_coverage()     # validated GRIB pointer
```

The ERA5 fields used are `10u`/`10v`, `2t`, `sp`, `blh`, `sshf`, and optionally
`ewss`/`nsss` — the same files that drive FLEXPART, so no second met source is
introduced. Two ERA5 conventions are normalised on read: accumulated fluxes are
divided by their own `stepRange` window, and the heat-flux sign is flipped to
positive-upward.

The AERMOD plugin can read ERA5 directly, in place of an inline `met` block:

```json
"transport_operator": {
    "plugin": "enforceflux.transport_operator.aermod",
    "config": {
        "era5": {
            "meteo_dir": "runs/sacramento_valley_2020/meteo_april_week",
            "longitude": -121.75, "latitude": 39.15,
            "start": "2020-03-31T00:00", "end": "2020-04-01T00:00",
            "surface_roughness_m": 0.15
        }
    }
}
```

Three things worth knowing:

- **Roughness is yours, not ERA5's.** ERA5 has no `z0` usable at dispersion
  scales, so `surface_roughness_m` is a site parameter you supply. It feeds both
  the log-law `u*` and the models downstream.
- **`u*` defaults to the log law, not ERA5 stress.** ERA5's surface stress
  includes subgrid *orographic form drag*, which over complex terrain makes `u*`
  ~1.5–2× larger than the local shear that actually mixes a plume. Pass
  `friction_velocity="stress"` over flat, homogeneous terrain.
- **Collapsing a long series to one LES forcing is refused by default.** A
  vector mean over veering wind is arithmetically correct but physically absurd
  — ten days of Sacramento spring met average to 0.4 m/s despite no calm hour.
  `to_microhh_forcing` checks `MetSeries.directional_consistency` and tells you
  to narrow the window; a single day here scores 0.94 and gives 2.9 m/s.

---

## AERMOD transport

The default near-field transport model. AERMOD's dispersion formulation
(similarity-scaled turbulence, Briggs plume rise, a bi-Gaussian convective
boundary layer, a reflected stable layer) reimplemented in JAX, so it is fast,
vectorized, and **differentiable with respect to every input** — emission rates,
source and stack geometry, and meteorology alike. It replaces the earlier toy
Gaussian plume operator.

It needs no external binary and no meteorological files: the boundary layer is
specified directly, either as a Pasquill-Gifford stability class or as measured
similarity parameters.

```python
from enforceflux.aermod import AermodConfig, AermodModel, Receptor, SurfaceMet, StackParameters

config = AermodConfig(
    met=[SurfaceMet(wind_speed_m_s=3.0, wind_direction_deg=270.0, stability_class="D")],
    receptors=[Receptor(id="tower", x=500.0, y=0.0, z=3.0)],
    default_stack=StackParameters(height_m=10.0),
    concentration_units="ug_m3_per_g_s",
)
model = AermodModel(config)

G = model.jacobian(sources)                    # inversion Jacobian (receptor × source)
field = model.grid_field(sources)              # forward concentration field
d_met = model.sensitivity_to_met(sources)      # ∂C/∂(u*, zi, 1/L, w*, z0, ...)
```

Both registry plugins are thin wrappers over that API:

```json
"transport_operator": {
    "plugin": "enforceflux.transport_operator.aermod",
    "config": {
        "met": [{"wind_speed_m_s": 3.0, "wind_direction_deg": 270.0,
                 "stability_class": "D", "mixing_height_m": 800.0}],
        "default_stack": {"height_m": 10.0},
        "concentration_units": "ug_m3_per_g_s"
    }
}
```

Multiple `met` entries are treated as independent hours and collapsed by
`reduce` (`"mean"`, `"max"`, or `"none"` to keep the hour axis).
`enforceflux.transport_simulation.aermod` takes the same config plus a `grid`
and writes a `(time, y, x)` concentration NetCDF.

Coordinates are Cartesian metres, so a projected CRS (UTM et al.) is required.

**Scope:** this is a research reimplementation, not EPA regulatory AERMOD. The
penetrated-plume source, terrain (AERMAP), building downwash (PRIME), area and
volume source integration, and deposition are not modelled — see
`src/enforceflux/aermod/dispersion.py` for the full list. Do not use it for
regulatory demonstrations.

---

## Methane Transport Simulation

`FlexpartSimulation` is a standalone forward-simulation wrapper. It accepts a YAML config, writes all FLEXPART input files, executes the model, and converts gridded output to a clean NetCDF file.

Two source types are supported:

- **Point sources** — single-location releases (landfills, well pads, dairies). Rate in kg s⁻¹.
- **Diffuse sources** — area emissions (rice paddies, wetlands). Flux in kg m⁻² s⁻¹. The bounding box is subdivided into a lat/lon grid; per-cell mass uses cosine-latitude spherical area correction.

### Quick start (simulation)

```python
from enforceflux.flexpart import FlexpartSimulation

sim = FlexpartSimulation.from_yaml("examples/simulation_config.yaml")
output_nc = sim.run()   # returns path to output NetCDF
```

### YAML config reference

A fully-annotated example lives at [`examples/simulation_config.yaml`](examples/simulation_config.yaml).

#### `flexpart` — binary and directories

| Key | Type | Description |
|-----|------|-------------|
| `executable` | path | Path to the compiled `FLEXPART` binary. Required. |
| `options_dir` | path | Directory containing FLEXPART input templates. Required. |
| `available_file` | path | Path to the `AVAILABLE` meteorological index. Required. |
| `meteo_dir` | path | Directory containing ERA5 GRIB files. Required. |
| `run_dir` | path | Working directory. Recreated on every `run()`. Default: `runs/simulation`. |

#### `simulation` — timing

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `start` | ISO-8601 | — | UTC simulation start. |
| `end` | ISO-8601 | — | UTC simulation end. |
| `output_step_seconds` | int | `3600` | Output write interval (s). |
| `sync_seconds` | int | `900` | FLEXPART internal sub-step (must divide `output_step_seconds`). |

#### `domain` — output grid

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lon_min/max` | float | — | Western/eastern boundary (degrees). |
| `lat_min/max` | float | — | Southern/northern boundary (degrees). |
| `dx` / `dy` | float | `0.1` | Longitude/latitude grid spacing (degrees). |
| `heights_m` | list | `[100, 500, 1000]` | Vertical layer upper boundaries (m AGL). |

#### Sources

```yaml
# Point source
- type: point
  id: landfill_A
  lon: -121.50
  lat:  38.60
  alt_m: 5.0
  emission_rate_kg_s: 2.0e-4
  n_particles: 10000

# Diffuse source
- type: diffuse
  id: rice_paddies_cv
  lon_min: -122.0
  lon_max: -121.0
  lat_min:  38.5
  lat_max:  39.5
  alt_m: 2.0
  emission_flux_kg_m2_s: 1.4e-9
  cell_size_deg: 0.1
  n_particles_per_cell: 500
```

---

## OSSE Pipeline

The information content analysis is the core of the OSSE:

```python
from enforceflux.analysis.information_core import analyze_information_content_spatial

result = analyze_information_content_spatial(G, Se, Sa, obs_groups, source_names)
print(f"DFS = {result['dfs_total']:.2f}")
print(f"Posterior uncertainty reduction: {result['uncertainty_reduction_pct']:.1f}%")
```

Key outputs:

| Field | Shape | Description |
|-------|-------|-------------|
| `dfs_total` | scalar | Total degrees of freedom for signal = Tr[A] |
| `dfs_per_sensor` | (m,) | DFS contribution of each sensor |
| `averaging_kernel` | (n,) | Diagonal of A (1D when spatial) |
| `posterior_variance` | (n,) | Diagonal of Sₓ |
| `uncertainty_reduction` | (n,) | 1 − √(Sₓᵢᵢ / Sₐᵢᵢ) per source cell |
| `eigenvalues` | (m,) | Eigenvalues of the m×m dual-space Fisher matrix |

---

## Repo layout

```
src/enforceflux/
    flexpart/
        simulation.py       # YAML-driven forward simulation
        build.py            # FlexpartCompiler — patches makefile, runs make
        runner.py           # FlexpartRunner — backward-mode G-matrix runs
    transport/
        run_config.py       # The shared, model-agnostic run schema
        translate.py        # Shared config → each model's native config
        canonical.py        # Canonical concentration(time, y, x) output
        runner.py           # run_transport — dispatch + normalise
    meteo/
        era5.py             # ERA5Downloader — CDS fetch + AVAILABLE writer
        record.py           # MetRecord / MetSeries — the canonical met format
        era5_profile.py     # ERA5 GRIB → MetSeries at a point
        adapters.py         # MetSeries → AERMOD / MicroHH / FLEXPART forcing
    analysis/
        information_core.py # Woodbury ICA: DFS, averaging kernel, posterior Σ
        instrument.py       # Instrument operator and Instrument dataclass
    aermod/
        dispersion.py       # Differentiable AERMOD plume kernel (JAX)
        meteorology.py      # Similarity-theory boundary-layer parameters
        model.py            # AermodModel — Jacobians and concentration fields

apps/
    met_main.py             # ERA5 download pipeline
    transport_main.py       # Any transport model from one shared config
    simulation_main.py      # Forward FLEXPART simulation
    flux_main.py            # Flux inversion
    analysis_main.py        # Information content analysis
    instrument_main.py      # Instrument OSSE

examples/
    single_source_instrument_demo.py
    sacramento_valley_2020.py
    gaussian_plume_single_source_demo.py
    aermod_single_source_demo.py
    era5_driven_models.py
    osse_25kg_leak_demo.py

data/
    bottomup/               # EPA GHGI gridded CH4 NetCDF
    tropomi_files/          # Sentinel-5P XCH4 CSV retrievals
    presentation_figures/   # make_figures.py — publication-quality figures

flexpart/                   # FLEXPART 11 source (Fortran submodule)
    src/                    # compiled binary lives here
    options/                # FLEXPART input templates
    options/SPECIES/        # CH4, CO2, CO, N2O, O3, SO2 species files

docs/
    flexpart_build.md       # extended build notes for Apple Silicon
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
