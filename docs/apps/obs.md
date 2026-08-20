# `obs` — real-observation ingress *(planned)*

**Status.** Reserved subcommand — the parser is wired but the payload code
lands with the run-artifact contract (D2b). Running it today prints a stub
message and exits with code 2.

**What it will do.** Take a user-supplied observation file (CSV, NetCDF, or
whatever standard we settle on for your data source) and normalise it into the
same `obs` NetCDF schema `instrument` produces, so `flux` can consume real
observations without knowing whether they came from a simulator or an
instrument in the field.

## Pipeline position

```
              [ obs ]  →  flux  →  analysis
dispersion  ─────────────────↑
```

## Once wired

```bash
enforceflux obs --config configs/<your_experiment>/obs.yaml
```

The output RunDir will look identical to an `instrument` RunDir (`obs.nc`,
`obs.csv`, `manifest.json`), and you point `flux.inputs.obs:` at it exactly
the same way — see [flux.md](flux.md#stage--run--inputs).

## What you can do today

Until this app ships, the accepted workarounds are:

1. Convert your real observations to the `instrument` `obs.nc` schema by hand
   (variables: `time`, `instrument_id`, `instrument_lon`, `instrument_lat`,
   `y_obs`, `valid_mask`, `noise_variance`). Any downstream `flux` run will
   read it via the existing `obs` role.
2. If your observations already exist as a well-formed NetCDF, place them
   under a manually created `runs/<name>/obs/` folder with a
   `manifest.json` that maps `obs → obs.nc`, and point `flux.inputs.obs:` at
   the folder.
