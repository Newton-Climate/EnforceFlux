"""Drive a full diurnal cycle of MicroHH LES over the Sacramento Valley.

MicroHH takes one steady forcing per run, so a real day cannot be a single
integration (see examples/sacramento_diurnal_24h.yaml for why). This driver
slices 2020-03-31 into the eight 3-hourly ERA5 records and runs one LES
segment per record, each in its own wind-aligned box.

Per segment it produces:
  * ``ch4_4d.nc``      — (time, z, y, x) scalar field from MicroHH's [dump]
  * column / cross files — the native instrument and slice output

and at the end stitches every segment's receptor columns into a single
``instruments.nc`` on absolute UTC time.

Run with::

    python examples/microhh_sacramento_diurnal.py
    python examples/microhh_sacramento_diurnal.py --dry-run   # write inputs only
    python examples/microhh_sacramento_diurnal.py --segments 0,1

Segments already carrying a ``.done`` marker are skipped, so an interrupted
run resumes where it stopped.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "examples" / "sacramento_diurnal_24h.yaml"
SEGMENT_HOURS = 3


def build_segments(run, series):
    """One (index, record) per ERA5 record that starts a 3 h window in range."""
    end = run.end or series.end
    return [(i, r) for i, r in enumerate(series) if r.time + timedelta(hours=SEGMENT_HOURS) <= end]


def write_segment_config(run, series, record, seg_dir):
    """Generate the segment's case YAML, forced by this one record."""
    from enforceflux.meteo.record import MetSeries
    from enforceflux.transport import translate

    one = MetSeries(
        records=(record,),
        longitude=series.longitude,
        latitude=series.latitude,
        provenance={**series.provenance, "segment_time": record.time.isoformat()},
    )
    # met_reduce: mean over a single record is the identity, so the segment is
    # forced by this hour exactly — not by any average over the day.
    path = translate.write_microhh_config(run, one, seg_dir)

    blob = yaml.safe_load(path.read_text())
    # write_microhh_config stamps transport.start on every segment; each
    # segment actually begins at its own record's time.
    blob["simulation"]["start"] = record.time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(yaml.safe_dump(blob, sort_keys=False))
    return path


def read_grid(case_dir: Path, itot: int, jtot: int, ktot: int):
    """Read MicroHH's ``grid.0000000``: x, xh, y, yh, z, zh written back-to-back.

    Precision is inferred from the file size, exactly as microhh_tools does.
    """
    path = case_dir / "grid.0000000"
    word = round(path.stat().st_size / (2 * itot + 2 * jtot + 2 * ktot))
    dtype = np.float64 if word == 8 else np.float32
    raw = np.fromfile(path, dtype=dtype)

    dims, offset = {}, 0
    for name, n in (("x", itot), ("xh", itot), ("y", jtot),
                    ("yh", jtot), ("z", ktot), ("zh", ktot)):
        dims[name] = raw[offset:offset + n]
        offset += n
    return dims, dtype


def dumps_to_netcdf(cfg, seg_dir):
    """Convert MicroHH's raw 3-D [dump] binaries to a (time,z,y,x) NetCDF.

    Deliberately does NOT shell out to microhh/python/3d_to_nc.py: that script
    builds a multiprocessing.Pool at module scope with no __main__ guard, which
    works under fork (Linux) but recurses and dies under spawn (macOS default).
    The on-disk format is simple enough to read directly, and doing so lets the
    box geometry travel with the field.

    A dump file is itot*jtot*ktot values in the build's precision, k slowest.
    """
    import xarray as xr

    g = cfg.grid
    dims, dtype = read_grid(cfg.case_dir, g.itot, g.jtot, g.ktot)
    expected = g.itot * g.jtot * g.ktot

    snapshots, times = [], []
    for path in sorted(cfg.case_dir.glob(f"{cfg.scalar_name}.0*")):
        stamp = path.suffix.lstrip(".")
        if len(stamp) != 7 or not stamp.isdigit():
            continue
        field = np.fromfile(path, dtype=dtype)
        if field.size != expected:
            raise ValueError(
                f"{path.name} holds {field.size} values, expected {expected} "
                f"for a {g.itot}x{g.jtot}x{g.ktot} grid — precision or layout mismatch."
            )
        snapshots.append(field.reshape(g.ktot, g.jtot, g.itot))
        times.append(int(stamp))

    if not snapshots:
        raise FileNotFoundError(
            f"No 3-D dumps in {cfg.case_dir}. Enable them with "
            "extra_ini: {dump: {swdump: 1, dumplist: <scalar>, sampletime: N}}."
        )

    order = np.argsort(times)
    ds = xr.Dataset(
        {cfg.scalar_name: (("time", "z", "y", "x"),
                           np.stack([snapshots[i] for i in order]))},
        coords={"time": np.asarray(times, dtype=float)[order],
                "z": dims["z"], "y": dims["y"], "x": dims["x"]},
    )
    ds["time"].attrs.update(units="s", long_name="seconds since simulation start")
    for axis in ("x", "y", "z"):
        ds[axis].attrs.update(units="m")
    ds[cfg.scalar_name].attrs.update(units="kg kg-1", long_name="scalar mixing ratio")
    # The box is wind-aligned, so the field is meaningless without its bearing.
    ds.attrs.update(
        x_bearing_deg=cfg.x_bearing_deg,
        source_x0_m=cfg.source_x0, source_y0_m=cfg.source_y0,
        origin_lon=cfg.origin_lon, origin_lat=cfg.origin_lat,
        spinup_s=cfg.spinup_s,
        note="+x points downwind along x_bearing_deg; coordinates are box-local metres",
    )

    target = seg_dir / "ch4_4d.nc"
    encoding = {cfg.scalar_name: {"zlib": True, "complevel": 4}}
    ds.to_netcdf(target, encoding=encoding)

    # The raw binaries are the bulk of the disk cost and are now redundant.
    for path in cfg.case_dir.glob(f"{cfg.scalar_name}.0*"):
        if len(path.suffix.lstrip(".")) == 7 and path.suffix.lstrip(".").isdigit():
            path.unlink()
    return target


def stitch_instruments(cfg_by_segment, out_path):
    """Concatenate every segment's receptor columns onto absolute UTC time.

    Each segment's clock restarts at 0, and its first ``spinup_s`` is smooth
    non-physical field, so retained samples are mapped as
    ``record_time + (t - spinup)`` — which makes the segments tile the day
    contiguously rather than overlap.
    """
    import xarray as xr

    from enforceflux.microhh import read_receptor_series

    frames = []
    for record, cfg in cfg_by_segment:
        rs = read_receptor_series(cfg)
        keep = rs.times_s >= cfg.spinup_s
        stamps = [record.time + timedelta(seconds=float(t - cfg.spinup_s))
                  for t in rs.times_s[keep]]
        frames.append(xr.Dataset(
            {"concentration": (("time", "receptor"), rs.values[keep, :])},
            coords={"time": stamps, "receptor": list(rs.receptor_ids)},
        ))

    combined = xr.concat(frames, dim="time").sortby("time")
    combined["concentration"].attrs.update(
        units="kg kg-1",   # MicroHH's raw scalar mixing ratio, not canonicalised
        long_name="near-surface CH4 from a MicroHH diurnal LES sequence",
        note=("Each 3 h segment is a separate LES forced by its own ERA5 record; "
              "turbulence is re-spun at every segment boundary, so eddy-scale "
              "continuity across boundaries is absent by construction."),
    )
    combined.to_netcdf(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="write each segment's inputs without integrating")
    parser.add_argument("--segments", type=str, default=None,
                        help="comma-separated segment indices to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="re-run segments that are already complete")
    args = parser.parse_args()

    from enforceflux.microhh import MicroHHRunner, load_microhh_config
    from enforceflux.transport import TransportRunConfig, translate

    run = TransportRunConfig.from_file(args.config)
    series = translate.build_met_series(run)
    root = run.output.path.parent
    root.mkdir(parents=True, exist_ok=True)

    segments = build_segments(run, series)
    if args.segments:
        wanted = {int(s) for s in args.segments.split(",")}
        segments = [s for s in segments if s[0] in wanted]

    print(f"Sacramento diurnal LES — {len(segments)} segments of {SEGMENT_HOURS} h")
    print(f"met: {len(series)} records, {series.start} → {series.end}")
    print(f"out: {root}\n")

    completed = []
    for index, record in segments:
        seg_dir = root / f"seg{index:02d}_{record.time:%H%M}Z"
        marker = seg_dir / ".done"

        cfg_path = write_segment_config(run, series, record, seg_dir)
        cfg = load_microhh_config(cfg_path)

        heat = record.sensible_heat_flux_w_m2
        regime = "convective" if heat > 10 else ("stable" if heat < -5 else "near-neutral")
        print(f"[{index}] {record.time:%Y-%m-%d %H:%M}Z  {regime:<13} "
              f"U={record.wind_speed_m_s:.2f} m/s  dir={record.wind_direction_deg:.0f}°  "
              f"H={heat:+.0f} W/m²  BL={record.mixing_height_m:.0f} m")

        if marker.exists() and not args.force:
            print("     already complete — skipping\n")
            completed.append((record, cfg))
            continue

        result = MicroHHRunner(cfg).run(dry_run=args.dry_run)
        if args.dry_run:
            print(f"     prepared {result.ini_path.name} "
                  f"(npx,npy)={cfg.decomposition}\n")
            continue

        field = dumps_to_netcdf(cfg, seg_dir)
        size_gb = field.stat().st_size / 1e9
        marker.write_text(record.time.isoformat())
        completed.append((record, cfg))
        print(f"     done → {field.name} ({size_gb:.2f} GB)\n")

    if args.dry_run or not completed:
        return 0

    out = stitch_instruments(completed, root / "instruments.nc")
    print(f"instrument timeseries → {out}")
    print(f"3-D fields            → {root}/seg*/ch4_4d.nc (one box per segment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
