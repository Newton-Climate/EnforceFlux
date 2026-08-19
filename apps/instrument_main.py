#!/usr/bin/env python3
"""Instrument-operator stage — sample a concentration field with real sensors.

Reads the ``concentration_field`` produced by the dispersion stage, applies
each instrument's forward operator + heteroscedastic noise model, and emits
an ``obs`` artifact (NetCDF + CSV) that the flux stage consumes.

Roles:
  in : dispersion → concentration_field (concentration.nc)
  out: instrument.obs        → obs.nc     (canonical: time × instrument)
       instrument.obs_csv    → obs.csv    (flat, one row per (time, sensor))

Usage:
    enforceflux instrument --config configs/instrument/main.yaml
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enforceflux.instrument import InstrumentOperator
from enforceflux.instrument.models import Instrument


def _find_var(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _parse_instruments(items: list[dict]) -> list[Instrument]:
    instruments: list[Instrument] = []
    for item in items:
        instruments.append(
            Instrument(
                id=str(item["id"]),
                tech_id=str(item["tech_id"]),
                mode=str(item.get("mode", "good")),
                x=float(item["lon"]),
                y=float(item["lat"]),
                z=float(item.get("z", 0.0)),
                path_length_m=float(item.get("path_length_m", 200.0)),
                path_bearing_deg=float(item.get("path_bearing_deg", 0.0)),
                footprint_sigma_m=float(item.get("footprint_sigma_m", 100.0)),
                footprint_wind_dir_deg=float(item.get("footprint_wind_dir_deg", 270.0)),
            )
        )
    if not instruments:
        raise ValueError("No instruments configured. Add at least one instrument in YAML.")
    return instruments


def _extract_2d_field(var, time_index: int, level_index: int, release_index: int) -> np.ndarray:
    key = {
        "time": time_index, "times": time_index,
        "height": level_index, "level": level_index, "lev": level_index,
        "releases": release_index, "release": release_index, "pointspec": release_index,
        "nageclass": 0,
    }
    idx: list[object] = []
    for dim in var.dimensions:
        d = dim.lower()
        # Canonical dims: y, x. FLEXPART: latitude, longitude.
        if d in ("latitude", "lat", "ylat", "y", "longitude", "lon", "xlon", "x"):
            idx.append(slice(None))
        elif d in key:
            idx.append(key[d])
        else:
            idx.append(0)
    arr = np.asarray(var[tuple(idx)])
    return np.asarray(np.squeeze(arr), dtype=float)


def _sample_nearest(field_2d, lons, lats, lon, lat) -> float:
    """Nearest-grid sample. Handles 1-D or 2-D lat/lon coordinates.

    Canonical dispersion output ships 2-D lon/lat fields (y, x); native
    FLEXPART output ships 1-D coordinate axes.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    if lons.ndim == 2 and lats.ndim == 2 and lons.shape == lats.shape == field_2d.shape:
        d2 = (lons - lon) ** 2 + (lats - lat) ** 2
        flat = int(np.argmin(d2))
        iy, ix = np.unravel_index(flat, field_2d.shape)
        return float(field_2d[iy, ix])
    if lons.ndim == 1 and lats.ndim == 1:
        iy = int(np.argmin(np.abs(lats - lat)))
        ix = int(np.argmin(np.abs(lons - lon)))
        if field_2d.shape == (len(lats), len(lons)):
            return float(field_2d[iy, ix])
        if field_2d.shape == (len(lons), len(lats)):
            return float(field_2d[ix, iy])
    raise ValueError(
        f"Unable to map concentration field to lat/lon axes. "
        f"Field shape={field_2d.shape}, lat shape={lats.shape}, lon shape={lons.shape}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply instrument operator to dispersion output; emit obs.nc + obs.csv"
    )
    parser.add_argument("--config", required=True, help="Path to instrument-stage YAML")
    return parser


def main() -> None:
    from netCDF4 import Dataset
    from enforceflux.runs import load_stage_config, open_run_dir, read_upstream

    args = build_parser().parse_args()

    stage_cfg = load_stage_config(args.config, expected_stage="instrument")
    block = stage_cfg.block

    if "dispersion" not in stage_cfg.inputs:
        raise ValueError(
            f"{stage_cfg.yaml_path}: `inputs.dispersion:` is required — "
            f"point it at the RunDir of a completed dispersion stage."
        )
    dispersion_up = read_upstream(stage_cfg.inputs["dispersion"])
    sim_nc = dispersion_up.file("concentration_field")

    var_name_cfg = block.get("variable_name")
    level_index = int(block.get("level_index", 0))
    release_index = int(block.get("release_index", 0))

    instruments_cfg = block.get("instruments") or []
    instruments = _parse_instruments(instruments_cfg)

    op_cfg = block.get("operator") or {}
    seed = int(op_cfg.get("random_seed", 42))
    op = InstrumentOperator(instruments, rng=np.random.default_rng(seed))

    response_scale = np.array(
        [float(item.get("response_scale", 1.0)) for item in instruments_cfg],
        dtype=float,
    )

    run_dir = open_run_dir(
        stage="instrument",
        run_name=stage_cfg.run_name,
        outputs_root=stage_cfg.outputs_root,
        inputs={k: str(v) for k, v in stage_cfg.inputs.items()},
    )
    run_dir.snapshot_config(stage_cfg.snapshot)
    out_nc = run_dir.path("obs.nc")
    out_csv = run_dir.path("obs.csv")

    with Dataset(sim_nc) as src:
        # Canonical NetCDF ships `concentration`; FLEXPART native output uses
        # `ch4_mixing_ratio` or `spec001_mr`. Try the config-provided name
        # first, then the standard fallbacks.
        default_candidates = (
            "concentration", "ch4_mixing_ratio", "ch4_concentration",
            "spec001_mr", "spec001",
        )
        var_candidates = (
            (str(var_name_cfg), *default_candidates) if var_name_cfg else default_candidates
        )
        vname = _find_var(src, tuple(var_candidates))
        if vname is None:
            raise KeyError(
                f"No concentration variable found in {sim_nc}. "
                "Provide `variable_name:` in the instrument YAML."
            )

        lon_name = _find_var(src, ("longitude", "lon", "xlon"))
        lat_name = _find_var(src, ("latitude", "lat", "ylat"))
        time_name = _find_var(src, ("time", "Times"))
        if lon_name is None or lat_name is None:
            raise KeyError("Upstream NetCDF has no longitude / latitude coordinate variables")

        lons = np.asarray(src.variables[lon_name][:], dtype=float)
        lats = np.asarray(src.variables[lat_name][:], dtype=float)
        conc_var = src.variables[vname]

        time_size = 1
        if "time" in [d.lower() for d in conc_var.dimensions]:
            tdim = [d for d in conc_var.dimensions if d.lower() == "time"][0]
            time_size = len(src.dimensions[tdim])

        sampled = np.zeros((time_size, len(instruments)), dtype=float)
        y_clean = np.zeros_like(sampled)
        y_obs = np.zeros_like(sampled)
        valid_mask = np.zeros_like(sampled, dtype=bool)
        noise_var = np.zeros_like(sampled)

        for ti in range(time_size):
            field = _extract_2d_field(conc_var, ti, level_index, release_index)
            row = np.array(
                [_sample_nearest(field, lons, lats, inst.x, inst.y) for inst in instruments],
                dtype=float,
            )
            sampled[ti, :] = row
            g_t = (row * response_scale).reshape(len(instruments), 1)
            result = op.simulate_observations(g_t, np.array([1.0], dtype=float))
            y_clean[ti, :] = result.y_clean
            y_obs[ti, :] = result.y_obs
            valid_mask[ti, :] = result.valid_mask
            noise_var[ti, :] = np.diag(result.R)

        time_labels: list[str] = []
        if time_name and time_name in src.variables:
            tvar = src.variables[time_name]
            tvals = np.asarray(tvar[:]).reshape(-1)
            if len(tvals) == time_size:
                try:
                    from netCDF4 import num2date
                    t_units = getattr(tvar, "units")
                    t_cal = getattr(tvar, "calendar", "standard")
                    dts = num2date(tvals, units=t_units, calendar=t_cal)
                    dt_iter = np.asarray(dts, dtype=object).reshape(-1)
                    time_labels = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in dt_iter]
                except Exception:
                    time_labels = [_as_text(v) for v in tvals]

    with Dataset(out_nc, "w", format="NETCDF4") as dst:
        dst.createDimension("time", sampled.shape[0])
        dst.createDimension("instrument", sampled.shape[1])

        tvar = dst.createVariable("time", "i4", ("time",))
        tvar[:] = np.arange(sampled.shape[0], dtype=np.int32)

        iid = dst.createVariable("instrument_id", str, ("instrument",))
        iid[:] = np.array([inst.id for inst in instruments], dtype=object)
        ilon = dst.createVariable("instrument_lon", "f8", ("instrument",))
        ilat = dst.createVariable("instrument_lat", "f8", ("instrument",))
        ilon[:] = np.array([inst.x for inst in instruments], dtype=float)
        ilat[:] = np.array([inst.y for inst in instruments], dtype=float)

        v_sample = dst.createVariable("sampled_concentration", "f8", ("time", "instrument"), zlib=True)
        v_clean = dst.createVariable("y_clean", "f8", ("time", "instrument"), zlib=True)
        v_obs = dst.createVariable("y_obs", "f8", ("time", "instrument"), zlib=True)
        v_valid = dst.createVariable("valid_mask", "i1", ("time", "instrument"), zlib=True)
        v_nvar = dst.createVariable("noise_variance", "f8", ("time", "instrument"), zlib=True)

        v_sample[:] = sampled
        v_clean[:] = y_clean
        v_obs[:] = y_obs
        v_valid[:] = valid_mask.astype(np.int8)
        v_nvar[:] = noise_var

        v_sample.units = "same_as_input_field"
        v_clean.units = "instrument_native_or_scaled"
        v_obs.units = "instrument_native_or_scaled"
        v_nvar.units = "(instrument_units)^2"
        dst.title = "EnforceFlux instrument stage — obs"
        dst.source_dispersion = str(dispersion_up.root)
        dst.source_concentration_field = str(sim_nc)
        dst.concentration_variable = str(vname)
        dst.random_seed = seed
        dst.level_index = level_index
        dst.release_index = release_index

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_index", "time_label", "instrument_id", "tech_id", "mode",
            "lon", "lat", "sampled_concentration", "y_clean", "y_obs",
            "valid", "noise_std",
        ])
        for ti in range(sampled.shape[0]):
            for ii, inst in enumerate(instruments):
                writer.writerow([
                    ti,
                    time_labels[ti] if ti < len(time_labels) else "",
                    inst.id, inst.tech_id, inst.mode,
                    inst.x, inst.y,
                    sampled[ti, ii], y_clean[ti, ii], y_obs[ti, ii],
                    bool(valid_mask[ti, ii]),
                    float(np.sqrt(noise_var[ti, ii])) if np.isfinite(noise_var[ti, ii]) else np.nan,
                ])

    run_dir.record_output("obs.nc", role="obs")
    run_dir.record_output("obs.csv", role="obs_csv")
    contract = run_dir.finalize()

    print("EnforceFlux instrument")
    print(f"Config     : {stage_cfg.yaml_path}")
    print(f"Run name   : {stage_cfg.run_name}")
    print(f"Run dir    : {run_dir.root}")
    print(f"Upstream   : {dispersion_up.root}")
    print(f"Input NC   : {sim_nc}")
    print(f"Instruments: {len(instruments)}")
    print(f"Obs NC     : {out_nc}")
    print(f"Obs CSV    : {out_csv}")
    print(f"Manifest   : {contract['manifest']}")


if __name__ == "__main__":
    main()
