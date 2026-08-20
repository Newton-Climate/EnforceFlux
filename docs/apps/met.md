# `met` — download ERA5 meteorology

**What it does.** Downloads ECMWF ERA5 reanalysis (winds, temperature, humidity,
boundary-layer height, surface fluxes) from the Copernicus Climate Data Store
(CDS) for a date range and lat/lon box, and writes them as FLEXPART-ready GRIB
files plus an `AVAILABLE` index. Downstream stages consume this folder by
pointing at the run directory it creates.

**When you need it.**

- Running **FLEXPART** — it strictly requires ERA5 GRIB on model levels.
- Running **AERMOD or MicroHH** with realistic weather instead of the inline
  synthetic diurnal cycle in the starter configs.
- Computing wind roses in the `analysis` stage from real data.

You can skip this stage entirely for quick tests: the AERMOD and MicroHH
templates in [`configs/main/`](../../configs/main/) ship with a small inline
`met.records` block so they run standalone.

## Pipeline position

```
[ met ]  →  dispersion  →  instrument  →  flux  →  analysis
```

The `met` stage is a **source stage** — it reads no upstream stage. It produces
a folder that FLEXPART or the MicroHH ERA5 adapter can read directly.

## One-time setup

```bash
pip install -e '.[meteo]'      # cdsapi + eccodes
```

Register at <https://cds.climate.copernicus.eu/> and put your key in
`~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```

## Run it

```bash
enforceflux met --config configs/sacramento_valley_2020/met.yaml
```

Add `--force` to re-download even when the existing `AVAILABLE` file already
covers your window (useful after changing the bounding box or resolution — the
coverage check only inspects the time axis).

The stage prints its progress, then writes everything to
`runs/<run.name>/met/`. Expect the download to be **slow** (CDS queues each
request) and the output to be **large** — a week of Sacramento Valley on the
0.25° model-level grid is a few GB.

## Config walkthrough

A fully annotated example is
[`configs/sacramento_valley_2020/met.yaml`](../../configs/sacramento_valley_2020/met.yaml).
Only the fields you actually need to touch are listed below.

### `run.name`

Names the output folder: everything lands under
`runs/<run.name>/met/`. Pick something you'll recognise — e.g.
`sacramento_valley_2020_july`.

### `met.era5.start` / `met.era5.end`

ISO-8601 UTC strings that bound the download.

> **Pad the window.** FLEXPART interpolates between met steps and aborts if a
> simulation reaches past the last `AVAILABLE` entry. Add at least one extra
> timestep on each side of your simulation window.

### `met.era5.timestep_hours`

Time resolution in hours. Options:

- `1` — ERA5 native rate. Big downloads; sharpest near-field plumes.
- `3` — common compromise. Interpolation across the 3 h gap noticeably blurs
  plume direction when the wind shifts.

### `met.era5.vertical_mode`

- `model_levels` — ECMWF hybrid sigma-pressure levels. **Required for
  FLEXPART.** Use this unless you have a specific reason not to.
- `pressure_levels` — standard pressure surfaces. FLEXPART cannot use these.

### `met.era5.bbox`

Your geographic box. Either:

```yaml
bbox:
  lon_min: -122.5
  lat_min:  38.6
  lon_max: -121.0
  lat_max:  39.8
```

or `[lon_min, lat_min, lon_max, lat_max]`.

> **Pad the box.** Particles that leave the met domain vanish. A tight box
> silently truncates plumes on your downwind edge. Add ~1° of buffer around
> everything you care about.

### `met.era5.model_level_grid_deg`

Horizontal spacing of the download. **ERA5's native resolution is ~0.25°
(~28 km)**. Requesting finer only interpolates and adds no real information —
that is why FLEXPART cannot resolve a sub-kilometre plume even with a fine
output grid.

### `met.credentials.cdsapirc`

Optional path to an alternative `.cdsapirc`. Omit to use `~/.cdsapirc`.

### `met.skip_if_available_covers_window`

If `true`, the stage exits without downloading when the current `AVAILABLE`
already covers `start → end`. The check only looks at the time axis, so switch
this off when you change the box or the vertical mode.

## What lands in `runs/<run.name>/met/`

| File | Role | What it is |
|------|------|------------|
| `AVAILABLE` | `available` | FLEXPART's time index. Downstream configs point their `available_file:` here. |
| `EA<yyyymmddhh>` | `grib_EA…` | One GRIB per timestep. |
| `EA_static.grib` | `static` | Time-invariant fields (land–sea mask, orography). |
| `manifest.json` | — | Machine-readable list of every output above with its role. |
| `config.snapshot.yaml` | — | Verbatim copy of the YAML this run consumed. |

## Wiring it into a downstream run

Point FLEXPART at the folder:

```yaml
dispersion:
  flexpart:
    meteo_dir:      ../../runs/sacramento_valley_2020_july/met
    available_file: ../../runs/sacramento_valley_2020_july/met/AVAILABLE
```

Or point the AERMOD ERA5 adapter at it — see
[dispersion.md](dispersion.md#using-real-era5).

## Common gotchas

- **`cdsapi` errors on start-up** usually mean `~/.cdsapirc` is missing or has
  the wrong URL — make sure it matches the current CDS endpoint above.
- **The download hangs at "queued"** — CDS queues can take hours. Leave it
  running; you can watch the CDS website for progress.
- **You changed the bbox but nothing re-downloaded** — set
  `skip_if_available_covers_window: false` or add `--force`; the coverage check
  only inspects the time axis.
- **FLEXPART fails immediately with a met error** — check that your simulation
  window is inside the ERA5 window and that `AVAILABLE` has the expected number
  of lines.
