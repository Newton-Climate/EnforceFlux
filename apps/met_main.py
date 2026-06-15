#!/usr/bin/env python3
"""Download ERA5 meteorology for FLEXPART from a YAML config.

Usage:
    python apps/met_main.py --config apps/met_main.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from enforceflux.meteo.era5 import ERA5Downloader, available_covers_window


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    return yaml


def _resolve_path(path_like: str | Path, *, base: Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _parse_bbox(raw) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None

    if isinstance(raw, dict):
        try:
            return (
                float(raw["lon_min"]),
                float(raw["lat_min"]),
                float(raw["lon_max"]),
                float(raw["lat_max"]),
            )
        except KeyError as exc:
            raise ValueError(
                "era5.bbox must include lon_min, lat_min, lon_max, lat_max"
            ) from exc

    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(float(v) for v in raw)

    raise ValueError(
        "era5.bbox must be either a mapping with lon/lat keys or a 4-item list"
    )


def _configure_credentials(cfg: dict, *, yaml_dir: Path) -> None:
    creds_cfg = cfg.get("credentials", {}) or {}
    cdsapirc = creds_cfg.get("cdsapirc")
    if not cdsapirc:
        return

    rc_path = _resolve_path(cdsapirc, base=yaml_dir)
    if not rc_path.exists():
        raise FileNotFoundError(f"Configured credentials.cdsapirc does not exist: {rc_path}")
    os.environ["CDSAPI_RC"] = str(rc_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ERA5 meteorology and write FLEXPART AVAILABLE"
    )
    parser.add_argument("--config", required=True, help="Path to met_main YAML config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore skip-if-covered setting and always execute download",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    yaml = _require_yaml()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text()) or {}
    yaml_dir = config_path.parent

    meteo_cfg = cfg.get("meteo", {}) or {}
    era5_cfg = cfg.get("era5", {}) or {}
    run_cfg = cfg.get("run", {}) or {}

    output_dir_raw = meteo_cfg.get("output_dir")
    if not output_dir_raw:
        raise ValueError("meteo.output_dir is required")
    output_dir = _resolve_path(output_dir_raw, base=yaml_dir)

    start = era5_cfg.get("start")
    end = era5_cfg.get("end")
    if not start or not end:
        raise ValueError("era5.start and era5.end are required")

    timestep_hours = int(era5_cfg.get("timestep_hours", 3))
    vertical_mode = str(era5_cfg.get("vertical_mode", "pressure_levels"))
    if vertical_mode not in {"pressure_levels", "model_levels"}:
        raise ValueError("era5.vertical_mode must be 'pressure_levels' or 'model_levels'")

    pressure_levels_raw = era5_cfg.get("pressure_levels")
    pressure_levels = [str(v) for v in pressure_levels_raw] if pressure_levels_raw else None

    model_grid = era5_cfg.get("model_level_grid_deg", 0.25)
    model_level_grid_deg = None if model_grid is None else float(model_grid)
    model_level_allow_global_fallback = bool(
        era5_cfg.get("model_level_allow_global_fallback", False)
    )
    cleanup_raw_daily_grib = bool(era5_cfg.get("cleanup_raw_daily_grib", False))

    bbox = _parse_bbox(era5_cfg.get("bbox"))

    available_filename = str(meteo_cfg.get("available_filename", "AVAILABLE"))
    preferred_available = output_dir / available_filename

    _configure_credentials(cfg, yaml_dir=yaml_dir)

    skip_if_covered = bool(run_cfg.get("skip_if_available_covers_window", False))
    check_timestep_hours = int(run_cfg.get("check_timestep_hours", timestep_hours))

    if skip_if_covered and not args.force and preferred_available.exists():
        covered = available_covers_window(
            preferred_available,
            start,
            end,
            timestep_hours=check_timestep_hours,
        )
        if covered:
            print("EnforceFlux met_main")
            print(f"Config      : {config_path}")
            print(f"Output dir  : {output_dir}")
            print(f"AVAILABLE   : {preferred_available}")
            print("Skipping download: AVAILABLE already covers requested window.")
            return

    downloader = ERA5Downloader(
        output_dir=output_dir,
        timestep_hours=timestep_hours,
        pressure_levels=pressure_levels,
        vertical_mode=vertical_mode,
        model_level_grid_deg=model_level_grid_deg,
        model_level_allow_global_fallback=model_level_allow_global_fallback,
        cleanup_raw_daily_grib=cleanup_raw_daily_grib,
    )

    print("EnforceFlux met_main")
    print(f"Config      : {config_path}")
    print(f"Window      : {start} -> {end}")
    print(f"Output dir  : {output_dir}")
    print(f"Mode        : {vertical_mode}")
    if bbox is None:
        print("BBox        : global")
    else:
        print(
            "BBox        : "
            f"lon[{bbox[0]}, {bbox[2]}], lat[{bbox[1]}, {bbox[3]}]"
        )

    result = downloader.download(start=start, end=end, bbox=bbox)

    if preferred_available != result.available_file:
        preferred_available.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.available_file, preferred_available)
        available_path = preferred_available
    else:
        available_path = result.available_file

    print(f"Downloaded  : {result.n_timesteps} timestep files")
    print(f"AVAILABLE   : {available_path}")
    if result.files:
        print(f"First file  : {result.files[0]}")
        print(f"Last file   : {result.files[-1]}")


if __name__ == "__main__":
    main()
