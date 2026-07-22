# pyright: reportMissingTypeStubs=false

"""Single-source FLEXPART plume demo.

This example runs a 100 kg/hr single-point-source case with FLEXPART,
then renders the concentration evolution as a GIF and saves a final-frame
preview so you can validate that the model output looks physically sensible.

Run from the repo root:
    python examples/gaussian_plume_single_source_demo.py
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from enforceflux.analysis import (  # type: ignore[import-untyped]  # noqa: E402
    create_simulation_movie_from_netcdf,
    load_simulation_netcdf,
    plot_simulation_heatmap_from_netcdf,
)
from enforceflux.core.base import ITransportSimulation  # noqa: E402
from enforceflux.utils.plugin_registry import get_plugin  # noqa: E402

CONFIG_YAML = Path(__file__).parent / "gaussian_plume_single_source_sacramento.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "gaussian_plume_single_source_sacramento"
DEFAULT_OUTPUT_NC = DEFAULT_OUTPUT_DIR / "flexpart_single_source.nc"
DEFAULT_GIF = DEFAULT_OUTPUT_DIR / "flexpart_single_source.gif"
DEFAULT_PREVIEW = DEFAULT_OUTPUT_DIR / "flexpart_single_source_final_frame.png"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-source FLEXPART simulation and render a realistic plume GIF."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the NetCDF, GIF, and preview image.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=4,
        help="Frames per second for the GIF.",
    )
    return parser


def _run_flexpart(output_dir: Path, output_nc: Path) -> Path:
    if output_nc.exists():
        output_nc.unlink()

    print("Running FLEXPART for the single-source case …")
    simulation = get_plugin(
        "enforceflux.transport_simulation", "flexpart", ITransportSimulation
    )()
    written = simulation.simulate(
        [],
        None,
        {
            "sim_config": str(CONFIG_YAML),
            "run_dir": str(output_dir / "flexpart_run"),
            "output_path": str(output_nc),
        },
    ).output_path
    print(f"Wrote NetCDF: {_display_path(written)}")
    return written


def _surface_frame_summary(concentration, dim_names=None) -> tuple[int, int, float]:
    try:
        import numpy as np

        arr = np.asarray(concentration, dtype=float)
        if arr.ndim == 6:
            surface = arr[0, 0, :, 0, :, :]
        elif arr.ndim == 5:
            surface = arr[0, :, 0, :, :]
        elif arr.ndim == 4:
            surface = arr[:, 0, :, :]
        elif arr.ndim == 3:
            surface = arr
        else:
            raise ValueError(f"Unsupported concentration shape: {arr.shape}")

        peak = float(np.nanmax(surface[-1]))
        return int(surface.shape[0]), int(surface.shape[-1] * surface.shape[-2]), peak
    except Exception:
        return 0, 0, float("nan")


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_nc = output_dir / DEFAULT_OUTPUT_NC.name
    gif_path = output_dir / DEFAULT_GIF.name
    preview_path = output_dir / DEFAULT_PREVIEW.name

    print("=" * 72)
    print("Single-source FLEXPART plume demo")
    print("=" * 72)
    print("Source   : sacramento_point_100kghr")
    print("Model    : FLEXPART")
    print("Goal     : validate the plume evolution and animation pipeline")
    print(f"Output   : {_display_path(output_dir)}")
    print("Mode     : always re-run and overwrite NetCDF output")

    output_nc = _run_flexpart(output_dir, output_nc)

    data = load_simulation_netcdf(output_nc)
    concentration = data["concentration"]
    dimensions = data["dimensions"]
    time_labels = data.get("time_labels") or []

    print(f"NetCDF   : variable={data['variable_name']} shape={concentration.shape}")
    if dimensions:
        print(f"Dimensions: {dimensions}")
    if time_labels:
        print(f"Frames   : {len(time_labels)} time slices")

    create_simulation_movie_from_netcdf(
        output_nc,
        output_path=gif_path,
        level_index=0,
        release_index=0,
        fps=args.gif_fps,
        use_log_scale=True,
        title_prefix="Single-source FLEXPART plume",
    )
    print(f"Wrote GIF: {_display_path(gif_path)}")

    plot_simulation_heatmap_from_netcdf(
        output_nc,
        time_index=-1,
        level_index=0,
        release_index=0,
        use_log_scale=True,
        title="Single-source FLEXPART plume (final frame)",
    )[0].savefig(preview_path, dpi=150, bbox_inches="tight")
    print(f"Wrote preview: {_display_path(preview_path)}")

    n_time, n_cells, peak = _surface_frame_summary(concentration, dimensions)
    if n_time:
        print(f"Validation: {n_time} time steps, {n_cells} surface cells, final-frame peak={peak:.3g}")

    print("Done.")


if __name__ == "__main__":
    main()