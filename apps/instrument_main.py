#!/usr/bin/env python3
"""Apply EnforceFlux instrument operator to simulation NetCDF output.

Usage:
    python apps/instrument_main.py --config apps/instrument_main.yaml
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enforceflux.instrument import InstrumentOperator
from enforceflux.instrument.models import Instrument


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    return yaml


def _find_var(ds, candidates: tuple[str, ...]) -> str | None:
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
        "time": time_index,
        "times": time_index,
        "height": level_index,
        "level": level_index,
        "lev": level_index,
        "releases": release_index,
        "release": release_index,
        "pointspec": release_index,
        "nageclass": 0,
    }
    idx: list[object] = []
    for dim in var.dimensions:
        d = dim.lower()
        if d in ("latitude", "lat", "ylat", "longitude", "lon", "xlon"):
            idx.append(slice(None))
        elif d in key:
            idx.append(key[d])
        else:
            idx.append(0)

    arr = np.asarray(var[tuple(idx)])
    return np.asarray(np.squeeze(arr), dtype=float)


def _sample_nearest(field_2d: np.ndarray, lons: np.ndarray, lats: np.ndarray, lon: float, lat: float) -> float:
    iy = int(np.argmin(np.abs(lats - lat)))
    ix = int(np.argmin(np.abs(lons - lon)))

    if field_2d.shape == (len(lats), len(lons)):
        return float(field_2d[iy, ix])
    if field_2d.shape == (len(lons), len(lats)):
        return float(field_2d[ix, iy])

    raise ValueError(
        "Unable to map concentration field to lat/lon axes. "
        f"Field shape={field_2d.shape}, lat={len(lats)}, lon={len(lons)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply instrument operator to simulation NetCDF and write reproducible outputs"
    )
    parser.add_argument("--config", required=True, help="Path to instrument YAML config")
    return parser


def main() -> None:
    from netCDF4 import Dataset

    parser = build_parser()
    args = parser.parse_args()

    yaml = _require_yaml()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text()) or {}

    sim_nc = Path(cfg.get("input", {}).get("simulation_netcdf", "")).expanduser().resolve()
    if not sim_nc.exists():
        raise FileNotFoundError(f"Simulation NetCDF not found: {sim_nc}")

    var_name_cfg = cfg.get("input", {}).get("variable_name")
    level_index = int(cfg.get("input", {}).get("level_index", 0))
    release_index = int(cfg.get("input", {}).get("release_index", 0))

    instruments_cfg = cfg.get("instruments", [])
    instruments = _parse_instruments(instruments_cfg)

    seed = int(cfg.get("operator", {}).get("random_seed", 42))
    op = InstrumentOperator(instruments, rng=np.random.default_rng(seed))

    out_nc = Path(cfg.get("output", {}).get("instrument_netcdf", "outputs/instrument_output.nc"))
    out_csv = Path(cfg.get("output", {}).get("timeseries_csv", "outputs/instrument_observations.csv"))
    out_nc = out_nc.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    response_scale = np.array(
        [float(item.get("response_scale", 1.0)) for item in instruments_cfg],
        dtype=float,
    )

    with Dataset(sim_nc) as src:
        var_candidates = (
            str(var_name_cfg),
            "ch4_mixing_ratio",
            "ch4_concentration",
            "spec001_mr",
            "spec001",
        ) if var_name_cfg else (
            "ch4_mixing_ratio",
            "ch4_concentration",
            "spec001_mr",
            "spec001",
        )
        vname = _find_var(src, tuple(var_candidates))
        if vname is None:
            raise KeyError(
                "No concentration variable found. Provide input.variable_name in YAML."
            )

        lon_name = _find_var(src, ("longitude", "lon", "xlon"))
        lat_name = _find_var(src, ("latitude", "lat", "ylat"))
        time_name = _find_var(src, ("time", "Times"))

        if lon_name is None or lat_name is None:
            raise KeyError("NetCDF must contain longitude and latitude coordinate variables")

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
                [
                    _sample_nearest(field, lons, lats, inst.x, inst.y)
                    for inst in instruments
                ],
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

        dst.title = "EnforceFlux instrument operator output"
        dst.source_simulation = str(sim_nc)
        dst.concentration_variable = str(vname)
        dst.random_seed = seed
        dst.level_index = level_index
        dst.release_index = release_index
        dst.note = (
            "y_clean and y_obs are generated with InstrumentOperator from nearest-grid sampled "
            "simulation concentrations and optional response_scale factors."
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_index",
                "time_label",
                "instrument_id",
                "tech_id",
                "mode",
                "lon",
                "lat",
                "sampled_concentration",
                "y_clean",
                "y_obs",
                "valid",
                "noise_std",
            ]
        )
        for ti in range(sampled.shape[0]):
            for ii, inst in enumerate(instruments):
                writer.writerow(
                    [
                        ti,
                        time_labels[ti] if ti < len(time_labels) else "",
                        inst.id,
                        inst.tech_id,
                        inst.mode,
                        inst.x,
                        inst.y,
                        sampled[ti, ii],
                        y_clean[ti, ii],
                        y_obs[ti, ii],
                        bool(valid_mask[ti, ii]),
                        float(np.sqrt(noise_var[ti, ii])) if np.isfinite(noise_var[ti, ii]) else np.nan,
                    ]
                )

    print("EnforceFlux instrument_main")
    print(f"Config       : {config_path}")
    print(f"Input NC     : {sim_nc}")
    print(f"Instruments  : {len(instruments)}")
    print(f"Output NC    : {out_nc}")
    print(f"Output CSV   : {out_csv}")


if __name__ == "__main__":
    main()
