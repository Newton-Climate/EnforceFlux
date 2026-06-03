# EnforceFlux

EnforceFlux is a plugin-driven OSSE (Observation System Simulation Experiment) framework for evaluating methane monitoring systems against legal enforceability metrics. It mirrors the architecture diagram and proposal:

1. **Source process model** (flux, location, timing)
2. **Atmospheric transport / turbulence** (Gaussian baseline + FLEXPART plugin)
3. **Physical concentration field** (implicit via Green's functions)
4. **Measurement / forward operator** (G matrix from transport + instrument geometry)
5. **Instrument operator** (noise + averaging)
6. **Virtual observations (OSSE)**
7. **Retrieval & flux inversion** (Bayesian linear inversion)
8. **Evaluation & information metrics** (Fisher information, null space, posterior covariance, averaging kernel)

## Quick start

```bash
python -m pip install -e .
python -m enforceflux.cli --config examples/quickstart_config.json
```

The quickstart runs a toy OSSE with a Gaussian transport approximation and prints key metrics.

Entry-point plugins are registered when the package is installed (editable install is fine).

## Plugin registry

Plugins are discovered via entry points:

- `enforceflux.source`
- `enforceflux.instrument`
- `enforceflux.transport`
- `enforceflux.inversion`

Built-ins are registered in `pyproject.toml`. You can add new plugins by exposing entry points:

```toml
[project.entry-points."enforceflux.transport"]
my_transport = "my_pkg.transport:MyTransport"
```

## FLEXPART integration

The `flexpart/` folder contains the upstream FLEXPART model. The FLEXPART transport plugin runs the compiled binary externally and reads NetCDF receptor outputs to build the forward operator `G`.

Build instructions are in `docs/flexpart_build.md`.

You can also drive the vendored model directly from Python:

```python
from enforceflux.config import DomainConfig
from enforceflux.flexpart import FlexpartWrapper

domain = DomainConfig(
    x_min=0,
    x_max=1000,
    y_min=0,
    y_max=1000,
    grid_spacing=250,
    crs="EPSG:32610",
)

wrapper = FlexpartWrapper(
    domain=domain,
    config={
        "build_if_missing": True,
        "build_jobs": 2,
        "base_run_dir": "runs/flexpart",
        "options_dir": "flexpart/options",
        "available_file": "flexpart/AVAILABLE",
        "meteo_dir": "/path/to/meteo",
    },
)

wrapper.compile()
# wrapper.run(sources, instruments) once meteorology is available
```

Minimal FLEXPART transport config:

```json
{
  "plugin": "enforceflux.transport.flexpart",
  "config": {
    "build_if_missing": true,
    "build_jobs": 2,
    "executable": "flexpart/src/FLEXPART",
    "base_run_dir": "runs/flexpart",
    "options_dir": "flexpart/options",
    "pathnames_template": "flexpart/pathnames",
    "available_file": "flexpart/AVAILABLE",
    "meteo_dir": "/path/to/meteo",
    "output_dir": "output",
    "unit_emission_rate": 1.0,
    "dry_run": false
  }
}
```

## Config format

New-style configs specify plugin choices per component (legacy configs still work):

```json
{
  "random_seed": 7,
  "domain": {
    "x_min": 0,
    "x_max": 1000,
    "y_min": 0,
    "y_max": 1000,
    "grid_spacing": 250,
    "crs": "EPSG:32610",
    "crs_wgs84": "EPSG:4326"
  },
  "components": {
    "source": {
      "plugin": "enforceflux.source.static",
      "config": { "sources": [ ... ] }
    },
    "instrument": {
      "plugin": "enforceflux.instrument.static",
      "config": { "instruments": [ ... ] }
    },
    "transport": {
      "plugin": "enforceflux.transport.gaussian",
      "config": { "sigma": 200.0, "wind": [50.0, 0.0] }
    },
    "inversion": {
      "plugin": "enforceflux.inversion.bayesian",
      "config": { "r_cond": 1e-6 }
    }
  }
}
```

## Repo layout

- `src/enforceflux/` core OSSE pipeline + plugins
- `examples/` runnable demo configs
- `flexpart/` upstream transport model
- `docs/` developer notes

## Next steps

Decide which instruments and source-types you want modeled first, then we can plug in realistic noise/averaging and connect to FLEXPART output.
