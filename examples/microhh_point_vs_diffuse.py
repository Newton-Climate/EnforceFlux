"""Point leak vs diffuse rice paddy, in one convective hour of Sacramento met.

Runs the SAME LES twice, changing only the emission geometry:

    leak   one point source, 100 kg/hr, sigma 25 m
    paddy  300 x 300 m area, 1.8e-5 kg/s, tiled 6x6 at 50 m spacing, sigma 40 m

Because MicroHH fixes rndseed=2, perturbs only `th`, and treats ch4 as a
passive scalar (swthermo=dry, no buoyancy feedback), both runs integrate a
bit-identical velocity field. Every difference in the scalar is therefore
source geometry and nothing else. ``--check-flow`` verifies that claim against
the two runs' turbulence statistics instead of assuming it.

Outputs, in runs/sacramento_convective_compare/:
    leak/ch4_4d.nc, paddy/ch4_4d.nc   (time, z, y, x) scalar fields
    comparison.nc                     receptor series, raw and normalised
    flow_check.txt                    turbulence-identity residuals

Usage:
    python examples/microhh_point_vs_diffuse.py --dry-run
    python examples/microhh_point_vs_diffuse.py
    python examples/microhh_point_vs_diffuse.py --variants paddy
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml

from microhh_sacramento_diurnal import dumps_to_netcdf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "examples" / "sacramento_convective_compare.yaml"

# The paddy geometry lives here, not in the YAML, so side/flux/tiling stay in
# one place and the 36 source entries are never hand-maintained.
PADDY_SIDE_M = 300.0
PADDY_FLUX_KG_M2_S = 2.0e-10     # April pre-season (sacval_april_2020.yaml)
PADDY_TILES = 6                  # 6x6 => 50 m spacing
PADDY_SIGMA_M = 40.0             # 28.3 m std vs 50 m spacing => 0.36% ripple
LEAK_SIGMA_M = 25.0

CONVECTIVE_HOUR_UTC = 18


def paddy_sources(side: float, flux: float, tiles: int):
    """Tile a square area source with `tiles` x `tiles` point sources.

    The transport schema has no area-source type (RunSource is a point), so a
    diffuse source has to be built from points. Each carries flux * its share
    of the area; with sigma ~ 0.57 * spacing the blobs sum to a uniform patch.
    """
    from enforceflux.transport.run_config import RunSource

    spacing = side / tiles
    per = flux * side * side / tiles**2
    centres = [-side / 2 + spacing / 2 + i * spacing for i in range(tiles)]
    return [
        RunSource(id=f"paddy_{i}{j}", x_m=x, y_m=y,
                  emission_rate_kg_s=per, altitude_m=2.0)
        for j, y in enumerate(centres)
        for i, x in enumerate(centres)
    ], spacing, per


def select_record(series, hour: int):
    """The met record at `hour` UTC, which must be convective."""
    for r in series:
        if r.time.hour == hour:
            if r.sensible_heat_flux_w_m2 <= 0:
                raise ValueError(
                    f"{r.time:%H:%M}Z has H={r.sensible_heat_flux_w_m2:+.1f} W/m2, "
                    "which is not a convective condition."
                )
            return r
    raise ValueError(f"No met record at {hour:02d}:00 UTC in "
                     f"{series.start:%Y-%m-%d %H:%M}..{series.end:%H:%M}")


def variant_run(run, name: str):
    """The base run config with this variant's sources and blob width."""
    micro = run.model_options.get("microhh", {})
    if name == "leak":
        sources, sigma = list(run.sources), LEAK_SIGMA_M
    elif micro.get("surface_flux_sources"):
        # The config already describes the paddy as a 2-D surface-flux BC, which
        # supersedes the blob tiling below — do not also tile it into volumetric
        # sources or the field would emit twice.
        sources = list(run.sources)
        sigma = float(micro.get("source_sigma_m", PADDY_SIGMA_M))
    else:
        sources, _, _ = paddy_sources(PADDY_SIDE_M, PADDY_FLUX_KG_M2_S, PADDY_TILES)
        sigma = PADDY_SIGMA_M

    options = {m: dict(o) for m, o in run.model_options.items()}
    options["microhh"]["source_sigma_m"] = sigma
    return dataclasses.replace(run, sources=sources, model_options=options), sigma


def check_flow_identity(cfgs: dict, out_path: Path) -> str:
    """Verify both runs really did integrate the same turbulence.

    The claim is that a passive scalar cannot change the flow, so the velocity
    statistics must agree to round-off. If they do not, the comparison is
    confounded and the scalar differences cannot be read as geometry alone.
    """
    import xarray as xr

    def open_stats(cfg):
        path = cfg.case_dir / f"{cfg.case_name}.default.0000000.nc"
        # MicroHH nests its profiles in a "default" group; older builds are flat.
        try:
            return xr.open_dataset(path, group="default")
        except (OSError, KeyError):
            return xr.open_dataset(path)

    stats = {name: open_stats(cfg) for name, cfg in cfgs.items()}

    lines = ["Turbulence-identity check (leak vs paddy)", ""]
    worst = 0.0
    for field in ("u", "v", "w", "th"):
        candidates = [v for v in (f"{field}_2", field) if v in stats["leak"]]
        if not candidates:
            continue
        var = candidates[0]
        a = np.asarray(stats["leak"][var].values, dtype=float)
        b = np.asarray(stats["paddy"][var].values, dtype=float)
        scale = np.nanmax(np.abs(a)) or 1.0
        diff = np.nanmax(np.abs(a - b)) / scale
        worst = max(worst, diff)
        lines.append(f"  {var:<6} max |leak-paddy| / scale = {diff:.3e}")

    verdict = ("IDENTICAL — scalar differences are source geometry alone"
               if worst < 1e-10 else
               "DIVERGED — the two runs did NOT share a flow field; "
               "differences below are confounded by turbulence noise")
    lines += ["", f"worst residual: {worst:.3e}", f"verdict: {verdict}"]
    text = "\n".join(lines)
    out_path.write_text(text + "\n")
    return text


def compare_receptors(cfgs: dict, record, out_path: Path):
    """Receptor series for both variants, raw and emission-normalised."""
    import xarray as xr

    from enforceflux.microhh import read_receptor_series

    data, totals = {}, {}
    for name, cfg in cfgs.items():
        rs = read_receptor_series(cfg)
        keep = rs.times_s >= cfg.spinup_s
        data[name] = (rs, keep)
        totals[name] = sum(s.emission_rate_kg_s for s in cfg.sources)

    rs0, keep0 = data["leak"]
    # datetime64 rather than tz-aware datetimes: xarray cannot serialise the
    # latter. The met is UTC throughout, so dropping tzinfo loses nothing.
    stamps = np.array(
        [(record.time + timedelta(seconds=float(t - cfgs["leak"].spinup_s))).replace(tzinfo=None)
         for t in rs0.times_s[keep0]],
        dtype="datetime64[ns]",
    )

    ds = xr.Dataset(coords={"time": stamps, "receptor": list(rs0.receptor_ids)})
    for name, (rs, keep) in data.items():
        ds[name] = (("time", "receptor"), rs.values[keep, :])
        ds[name].attrs.update(units="kg kg-1",
                              total_emission_kg_s=totals[name])
        # Normalised: concentration per unit emission. The scalar is passive
        # and the source term linear, so this is the only fair comparison of
        # DISPERSION between sources that differ ~1500x in strength.
        ds[f"{name}_norm"] = ds[name] / totals[name]
        ds[f"{name}_norm"].attrs.update(units="kg kg-1 per kg s-1",
                                        long_name=f"{name} concentration per unit emission")

    ds.attrs.update(
        condition=f"{record.time:%Y-%m-%d %H:%M}Z convective",
        sensible_heat_flux_w_m2=float(record.sensible_heat_flux_w_m2),
        wind_speed_m_s=float(record.wind_speed_m_s),
        wind_direction_deg=float(record.wind_direction_deg),
        boundary_layer_m=float(record.mixing_height_m),
        note=("leak and paddy share a bit-identical flow field (fixed rndseed, "
              "passive scalar); differences are emission geometry only."),
    )
    ds.to_netcdf(out_path)
    return ds, totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hour", type=int, default=CONVECTIVE_HOUR_UTC)
    parser.add_argument("--variants", type=str, default="leak,paddy")
    args = parser.parse_args()

    from enforceflux.microhh import MicroHHRunner, load_microhh_config
    from enforceflux.transport import TransportRunConfig, translate

    run = TransportRunConfig.from_file(args.config)
    series = translate.build_met_series(run)
    record = select_record(series, args.hour)
    root = run.output.path.parent
    root.mkdir(parents=True, exist_ok=True)

    from enforceflux.meteo.record import MetSeries
    one = MetSeries(records=(record,), longitude=series.longitude,
                    latitude=series.latitude, provenance=series.provenance)

    w_star = (9.81 / record.potential_temperature_k
              * record.kinematic_heat_flux_k_m_s * record.mixing_height_m) ** (1 / 3)
    print(f"Condition : {record.time:%Y-%m-%d %H:%M}Z  H={record.sensible_heat_flux_w_m2:+.0f} W/m² "
          f"U={record.wind_speed_m_s:.2f} m/s dir={record.wind_direction_deg:.0f}° "
          f"BL={record.mixing_height_m:.0f} m  w*={w_star:.2f} m/s")
    print(f"Output    : {root}\n")

    cfgs = {}
    for name in args.variants.split(","):
        v_run, sigma = variant_run(run, name)
        total = sum(s.emission_rate_kg_s for s in v_run.sources)
        var_dir = root / name

        cfg_path = translate.write_microhh_config(v_run, one, var_dir)
        blob = yaml.safe_load(cfg_path.read_text())
        blob["simulation"]["start"] = record.time.strftime("%Y-%m-%dT%H:%M:%S")
        cfg_path.write_text(yaml.safe_dump(blob, sort_keys=False))
        cfg = load_microhh_config(cfg_path)
        cfgs[name] = cfg

        print(f"[{name}] {len(v_run.sources)} source(s), sigma={sigma:.0f} m, "
              f"total={total:.4e} kg/s ({total*3600:.4f} kg/hr)")

        result = MicroHHRunner(cfg).run(dry_run=args.dry_run)
        if args.dry_run:
            print(f"     prepared {result.ini_path.name} (npx,npy)={cfg.decomposition}\n")
            continue

        field = dumps_to_netcdf(cfg, var_dir)
        print(f"     done → {field.relative_to(root)} "
              f"({field.stat().st_size/1e9:.2f} GB)\n")

    if args.dry_run:
        return 0

    if len(cfgs) == 2:
        print(check_flow_identity(cfgs, root / "flow_check.txt"), "\n")
        ds, totals = compare_receptors(cfgs, record, root / "comparison.nc")
        print("Time-mean normalised concentration (kg kg-1 per kg s-1):")
        for rid in ds.receptor.values:
            lk = float(ds["leak_norm"].sel(receptor=rid).mean())
            pd = float(ds["paddy_norm"].sel(receptor=rid).mean())
            ratio = pd / lk if lk else float("nan")
            print(f"  {str(rid):<12} leak={lk:.4e}  paddy={pd:.4e}  paddy/leak={ratio:6.2f}")
        print(f"\ncomparison → {root/'comparison.nc'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
