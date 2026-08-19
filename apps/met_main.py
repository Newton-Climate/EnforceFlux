#!/usr/bin/env python3
"""Meteorology stage — download ERA5 for FLEXPART.

Source stage: consumes no upstream RunDir. Writes the FLEXPART-ready GRIB
timestep files and the AVAILABLE index under ``runs/<run.name>/met/`` so
downstream stages can point their ``meteo_dir`` / ``available_file`` at a
canonical RunDir path.

Roles:
  out: met.available   → AVAILABLE          — FLEXPART meteorology index
       met.grib_<name> → EA*                — one entry per timestep GRIB
       met.static      → EA_static.grib     — static fields (LSM, orography)

Usage:
    enforceflux met --config configs/sacramento_valley_2020/met.yaml
    enforceflux met --config configs/sacramento_valley_2020/met.yaml --force
"""
import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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
                "met.era5.bbox must include lon_min, lat_min, lon_max, lat_max"
            ) from exc

    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(float(v) for v in raw)

    raise ValueError(
        "met.era5.bbox must be either a mapping with lon/lat keys or a 4-item list"
    )


def _configure_credentials(block: dict, *, yaml_dir: Path) -> None:
    creds_cfg = block.get("credentials") or {}
    cdsapirc = creds_cfg.get("cdsapirc")
    if not cdsapirc:
        return

    rc_path = _resolve_path(cdsapirc, base=yaml_dir)
    if not rc_path.exists():
        raise FileNotFoundError(
            f"Configured met.credentials.cdsapirc does not exist: {rc_path}"
        )
    os.environ["CDSAPI_RC"] = str(rc_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ERA5 meteorology and write FLEXPART AVAILABLE"
    )
    parser.add_argument("--config", required=True, help="Path to met-stage YAML")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore skip-if-covered setting and always execute download",
    )
    return parser


def main() -> None:
    from enforceflux.meteo.era5 import ERA5Downloader, available_covers_window
    from enforceflux.runs import load_stage_config, open_run_dir

    args = build_parser().parse_args()

    stage_cfg = load_stage_config(args.config, expected_stage="met")
    block = stage_cfg.block
    era5_cfg = block.get("era5") or {}

    start = era5_cfg.get("start")
    end = era5_cfg.get("end")
    if not start or not end:
        raise ValueError("met.era5.start and met.era5.end are required")

    timestep_hours = int(era5_cfg.get("timestep_hours", 3))
    vertical_mode = str(era5_cfg.get("vertical_mode", "pressure_levels"))
    if vertical_mode not in {"pressure_levels", "model_levels"}:
        raise ValueError("met.era5.vertical_mode must be 'pressure_levels' or 'model_levels'")

    pressure_levels_raw = era5_cfg.get("pressure_levels")
    pressure_levels = [str(v) for v in pressure_levels_raw] if pressure_levels_raw else None

    model_grid = era5_cfg.get("model_level_grid_deg", 0.25)
    model_level_grid_deg = None if model_grid is None else float(model_grid)
    model_level_allow_global_fallback = bool(
        era5_cfg.get("model_level_allow_global_fallback", False)
    )
    cleanup_raw_daily_grib = bool(era5_cfg.get("cleanup_raw_daily_grib", False))

    bbox = _parse_bbox(era5_cfg.get("bbox"))

    available_filename = str(block.get("available_filename", "AVAILABLE"))
    skip_if_covered = bool(block.get("skip_if_available_covers_window", False))
    check_timestep_hours = int(block.get("check_timestep_hours", timestep_hours))

    _configure_credentials(block, yaml_dir=stage_cfg.yaml_dir)

    run_dir = open_run_dir(
        stage="met",
        run_name=stage_cfg.run_name,
        outputs_root=stage_cfg.outputs_root,
        inputs={k: str(v) for k, v in stage_cfg.inputs.items()},
    )
    run_dir.snapshot_config(stage_cfg.snapshot)

    output_dir = run_dir.root
    preferred_available = output_dir / available_filename

    print("EnforceFlux met")
    print(f"Config      : {stage_cfg.yaml_path}")
    print(f"Run name    : {stage_cfg.run_name}")
    print(f"Run dir     : {run_dir.root}")
    print(f"Window      : {start} -> {end}")
    print(f"Mode        : {vertical_mode}")
    if bbox is None:
        print("BBox        : global")
    else:
        print(
            "BBox        : "
            f"lon[{bbox[0]}, {bbox[2]}], lat[{bbox[1]}, {bbox[3]}]"
        )

    if skip_if_covered and not args.force and preferred_available.exists():
        covered = available_covers_window(
            preferred_available,
            start,
            end,
            timestep_hours=check_timestep_hours,
        )
        if covered:
            print("Skipping download: AVAILABLE already covers requested window.")
            _record_run_dir_files(run_dir, output_dir, preferred_available)
            contract = run_dir.finalize()
            print(f"AVAILABLE   : {preferred_available}")
            print(f"Manifest    : {contract['manifest']}")
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

    result = downloader.download(start=start, end=end, bbox=bbox)

    # ERA5Downloader always writes AVAILABLE at output_dir/AVAILABLE. If the
    # caller renamed it, rename in place so the manifest role points at the
    # requested filename.
    if result.available_file != preferred_available:
        preferred_available.parent.mkdir(parents=True, exist_ok=True)
        result.available_file.rename(preferred_available)
    available_path = preferred_available

    _record_run_dir_files(run_dir, output_dir, available_path)
    contract = run_dir.finalize()

    print(f"Downloaded  : {result.n_timesteps} timestep files")
    print(f"AVAILABLE   : {available_path}")
    if result.files:
        print(f"First file  : {result.files[0]}")
        print(f"Last file   : {result.files[-1]}")
    print(f"Manifest    : {contract['manifest']}")


def _record_run_dir_files(run_dir, output_dir: Path, available_path: Path) -> None:
    """Register AVAILABLE + every EA* GRIB with the RunDir manifest."""
    run_dir.record_output(available_path.name, role="available")
    for p in sorted(output_dir.glob("EA*")):
        if not p.is_file():
            continue
        if p.name == "EA_static.grib":
            role = "static"
        else:
            role = f"grib_{p.name}"
        run_dir.record_output(p.name, role=role)


if __name__ == "__main__":
    main()
