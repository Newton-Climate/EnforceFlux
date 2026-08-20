# EnforceFlux apps

EnforceFlux is built as a **pipeline of small command-line apps**. Each app does
one job — download weather, run a dispersion model, sample sensors, invert for
fluxes, and so on — and writes its outputs into a shared `runs/` folder so the
next app can pick them up.

If you are new here, read the apps in this order — it mirrors the flow of a real
OSSE (Observing System Simulation Experiment):

| # | App | What it does | Doc |
|---|-----|--------------|-----|
| 1 | **met** | Download ERA5 weather over your region and time window. | [met.md](met.md) |
| 2 | **dispersion** | Run a transport model (AERMOD, FLEXPART, or MicroHH) to get either a concentration field or a Jacobian `G`. | [dispersion.md](dispersion.md) |
| 3 | **instrument** | Sample the concentration field with virtual sensors and add realistic noise. | [instrument.md](instrument.md) |
| 4 | **flux** | Solve the Bayesian inverse problem for emissions. | [flux.md](flux.md) |
| 5 | **analysis** | Compute information-content diagnostics (DFS, averaging kernel, uncertainty reduction) and make plots. | [analysis.md](analysis.md) |
|   | **obs** | *(planned)* Ingest real observations in the same format as `instrument`. | [obs.md](obs.md) |
|   | **osse** | Legacy one-shot driver that does everything from a single JSON. | [osse.md](osse.md) |
|   | **osse-sweep** | Run many OSSEs across a parameter grid. | [osse-sweep.md](osse-sweep.md) |

## The shared pattern

Every app is called the same way:

```bash
enforceflux <app> --config <path/to/your.yaml>
```

Every config starts with the same three-line header telling the driver which
stage this is, what to call the run, and where its upstream data lives:

```yaml
stage: dispersion          # which app this config is for
run:
  name: my_experiment      # groups outputs under runs/my_experiment/dispersion/
inputs:                    # points at previous stages' output folders
  dispersion: ../../runs/my_experiment/dispersion   # (only when this stage needs one)
```

Everything else in the YAML is stage-specific and lives in a block named after
the stage (`met:`, `dispersion:`, `instrument:`, `flux:`, `analysis:`).

## Where things get written

All outputs land under `runs/<run.name>/<stage>/`. Each such folder contains:

- The stage's data files (e.g. `concentration.nc`, `obs.nc`, `summary.json`).
- A `manifest.json` listing every output file with a **role** (like
  `concentration_field`, `obs`, `summary`).
- A `config.snapshot.yaml` — an exact copy of the YAML the run consumed.

Downstream apps read by role, not by filename, so you can wire two stages
together just by pointing `inputs:` at the previous stage's folder.

## Starter templates

Templates that work out of the box live in
[`configs/main/`](../../configs/main/). Copy the relevant one, edit the
`run.name` and any physical parameters, then run.

```bash
enforceflux dispersion --config configs/main/dispersion.yaml
enforceflux instrument --config configs/main/instrument.yaml
enforceflux flux       --config configs/main/flux.yaml
enforceflux analysis   --config configs/main/analysis.yaml
```

The four commands above form a minimal end-to-end OSSE — no ERA5 download, no
FLEXPART build required — using an inline synthetic weather record and the
default (differentiable) AERMOD backend.
