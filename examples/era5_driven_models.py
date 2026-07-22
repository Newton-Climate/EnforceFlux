"""One ERA5 window, three transport models — the canonical met adapter.

Reads ERA5 GRIB once into a :class:`~enforceflux.meteo.record.MetSeries`, then
converts that single object into each model's native forcing:

* AERMOD   — per-hour ``SurfaceMet``, used here to build a Jacobian;
* MicroHH  — one steady idealised ``Forcing`` block for an LES case;
* FLEXPART — a validated pointer back at the GRIB files it reads itself.

Run from the repo root (needs ERA5 GRIB, e.g. from ``apps/met_main.py``)::

    python examples/era5_driven_models.py
    python examples/era5_driven_models.py --meteo-dir runs/... --date 2020-03-31
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from enforceflux.aermod import (  # noqa: E402
    AermodConfig,
    AermodModel,
    Receptor,
    StackParameters,
)
from enforceflux.meteo import (  # noqa: E402
    met_series_from_era5,
    microhh_box_bearing,
    to_aermod,
    to_flexpart,
    to_microhh_forcing,
)
from enforceflux.models.source import Source  # noqa: E402

DEFAULT_METEO = REPO_ROOT / "runs" / "sacramento_valley_2020" / "meteo_april_week"
SACRAMENTO_LON, SACRAMENTO_LAT = -121.75, 39.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meteo-dir", type=Path, default=DEFAULT_METEO)
    parser.add_argument("--longitude", type=float, default=SACRAMENTO_LON)
    parser.add_argument("--latitude", type=float, default=SACRAMENTO_LAT)
    parser.add_argument("--date", default="2020-03-31", help="UTC day to read (YYYY-MM-DD).")
    parser.add_argument(
        "--roughness",
        type=float,
        default=0.15,
        help="Local aerodynamic roughness [m] — ERA5 has none usable at this scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.meteo_dir.is_dir():
        sys.exit(
            f"ERA5 meteorology not found at {args.meteo_dir}.\n"
            "Download it first:  python apps/met_main.py --config apps/met_main.yaml"
        )

    start = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    series = met_series_from_era5(
        args.meteo_dir,
        args.longitude,
        args.latitude,
        start=start,
        end=start + timedelta(days=1),
        surface_roughness_m=args.roughness,
    )

    print("=" * 78)
    print("CANONICAL MET (read once, from ERA5)")
    print("=" * 78)
    print(series.summary())
    print(
        f"\n  grid point sampled: "
        f"({series.provenance['grid_lat']:.2f}, {series.provenance['grid_lon']:.2f})"
        f"   u* from: {series.provenance['friction_velocity']}"
        f"   directional consistency: {series.directional_consistency:.2f}"
    )

    # ── AERMOD ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("AERMOD — per-hour SurfaceMet (lossless: u*, H, zi passed through)")
    print("=" * 78)
    aermod_met = to_aermod(series)
    print(f"  {len(aermod_met)} hours; first: L={aermod_met[0].monin_obukhov_length_m:.1f} m, "
          f"u*={aermod_met[0].friction_velocity_m_s:.3f} m/s, "
          f"zi={aermod_met[0].mixing_height_m:.0f} m")

    source = Source(
        id="leak", kind="point", x=0.0, y=0.0,
        flux_true=100.0, flux_prior_mean=50.0, flux_prior_std=50.0,  # kg/hr
    )
    receptors = [
        Receptor(id="n_300m", x=0.0, y=300.0, z=2.0),
        Receptor(id="e_300m", x=300.0, y=0.0, z=2.0),
        Receptor(id="s_300m", x=0.0, y=-300.0, z=2.0),
        Receptor(id="w_300m", x=-300.0, y=0.0, z=2.0),
    ]
    model = AermodModel(
        AermodConfig(
            met=aermod_met,
            receptors=tuple(receptors),
            default_stack=StackParameters(height_m=5.0),
            emission_scale_to_kg_s=1.0 / 3600.0,
            concentration_units="ppb_ch4_per_kg_s",
        )
    )
    g = model.jacobian([source])
    # Steady ERA5 wind means only the downwind receptors see anything at all;
    # exact zeros upwind are the model working, not a bug.
    print("\n  Jacobian [ppb per kg/hr], mean over the day:")
    for receptor, row in zip(receptors, g):
        marker = "" if row[0] > 0.0 else "   (upwind all day)"
        print(f"    {receptor.id:>8s}  {row[0]:12.4g}{marker}")

    # ── MicroHH ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("MicroHH — one steady Forcing (LES takes a single idealised column)")
    print("=" * 78)
    try:
        forcing = to_microhh_forcing(series, reduce="daytime_mean")
    except ValueError as exc:
        print(f"  refused: {exc}")
    else:
        bearing = microhh_box_bearing(series.daytime())
        print(f"  u_geo={forcing.u_geo:.2f} m/s  v_geo={forcing.v_geo:.2f} m/s  "
              f"z0m={forcing.z0m}  zi={forcing.boundary_layer_height_m:.0f} m")
        print(f"  thl_surface={forcing.thl_surface_K:.2f} K  "
              f"w'th'={forcing.surface_heat_flux_K_m_s:.4f} K m/s")
        print(f"  set the case's x_bearing_deg to {bearing:.1f} so +x points downwind")

    # ── FLEXPART ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FLEXPART — GRIB pointer (it reads ERA5 itself; scalars cannot drive it)")
    print("=" * 78)
    flexpart = to_flexpart(series)
    for key, value in flexpart.as_config().items():
        print(f"  {key}: {value}")
    print(f"  covers {series.start:%Y-%m-%d %H:%M} → {series.end:%Y-%m-%d %H:%M}: "
          f"{flexpart.covers_window}")


if __name__ == "__main__":
    main()
