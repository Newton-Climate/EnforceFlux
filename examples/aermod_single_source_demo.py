"""AERMOD-style transport: Jacobian, concentration field, and autodiff sensitivities.

A single 100 kg/hr point source with three downwind open-path sensors, run
across a diurnal cycle of stability classes. Demonstrates the three things the
model is for:

1. ``jacobian``            — the G matrix the inversion consumes;
2. ``grid_field``          — a forward concentration field (optionally NetCDF);
3. ``sensitivity_to_met``  — ∂concentration/∂meteorology, free via JAX.

Run from the repo root::

    python examples/aermod_single_source_demo.py
    python examples/aermod_single_source_demo.py --netcdf runs/aermod_demo.nc
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from enforceflux.aermod import (  # noqa: E402
    AermodConfig,
    AermodModel,
    Receptor,
    ReceptorGrid,
    StackParameters,
    SurfaceMet,
    write_grid_netcdf,
)
from enforceflux.models.source import Source  # noqa: E402

KG_PER_HR_TO_KG_PER_S = 1.0 / 3600.0

# A diurnal sequence: stable night, convective midday, neutral evening.
DIURNAL_MET = [
    ("06:00", "F", 1.5, 250.0, 150.0),
    ("09:00", "C", 2.5, 288.0, 600.0),
    ("12:00", "A", 3.0, 295.0, 1500.0),
    ("15:00", "B", 4.0, 297.0, 1400.0),
    ("18:00", "D", 3.5, 291.0, 700.0),
    ("21:00", "E", 2.0, 283.0, 200.0),
]


def build_config(receptors, grid=None) -> AermodConfig:
    met = [
        SurfaceMet(
            wind_speed_m_s=speed,
            wind_direction_deg=270.0,  # wind from the west; plume travels east
            temperature_k=temperature,
            stability_class=stability,
            mixing_height_m=mixing_height,
            surface_roughness_m=0.15,  # open farmland
            timestamp=f"2020-07-01T{hour}:00",
        )
        for hour, stability, speed, temperature, mixing_height in DIURNAL_MET
    ]
    return AermodConfig(
        met=met,
        receptors=tuple(receptors),
        grid=grid,
        default_stack=StackParameters(height_m=5.0),
        emission_scale_to_kg_s=KG_PER_HR_TO_KG_PER_S,  # fluxes are given in kg/hr
        concentration_units="ppb_ch4_per_kg_s",
        reduce="mean",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--netcdf",
        type=Path,
        default=None,
        help="Write the concentration field to this NetCDF path.",
    )
    args = parser.parse_args()

    source = Source(
        id="leak",
        kind="point",
        x=0.0,
        y=0.0,
        flux_true=100.0,  # kg/hr
        flux_prior_mean=50.0,
        flux_prior_std=50.0,
    )
    receptors = [
        Receptor(id="op_200m", x=200.0, y=0.0, z=2.0),
        Receptor(id="op_500m", x=500.0, y=0.0, z=2.0),
        Receptor(id="op_500m_off", x=500.0, y=150.0, z=2.0),
    ]
    grid = ReceptorGrid(
        x_min=-200.0, x_max=1500.0, y_min=-600.0, y_max=600.0, spacing_m=25.0, height_m=2.0
    )

    model = AermodModel(build_config(receptors, grid))

    # 1. Jacobian — mean over the diurnal cycle, ppb per kg/hr.
    g = model.jacobian([source])
    print("Jacobian G [ppb per kg/hr], diurnal mean:")
    for receptor, row in zip(receptors, g):
        print(f"  {receptor.id:>12s}  {row[0]:10.3f}")

    # 2. Per-hour concentrations at the true emission rate.
    print("\nConcentration [ppb] at 100 kg/hr, by hour:")
    hourly = model.concentrations([source])
    header = "  ".join(f"{r.id:>12s}" for r in receptors)
    print(f"  {'hour':>5s}  {'class':>5s}  {header}")
    for (hour, stability, *_), row in zip(DIURNAL_MET, hourly):
        values = "  ".join(f"{v:12.2f}" for v in row)
        print(f"  {hour:>5s}  {stability:>5s}  {values}")

    # 3. Autodiff: how the total signal responds to the boundary layer.
    sensitivity = model.sensitivity_to_met([source])
    print("\n∂(total concentration)/∂(met parameter), by hour:")
    print(f"  {'hour':>5s}  {'∂/∂u*':>12s}  {'∂/∂zi':>12s}  {'∂/∂(1/L)':>12s}  {'∂/∂w*':>12s}")
    for (hour, *_), du, dzi, dl, dw in zip(
        DIURNAL_MET,
        np.asarray(sensitivity.u_star),
        np.asarray(sensitivity.mixing_height),
        np.asarray(sensitivity.inv_obukhov_length),
        np.asarray(sensitivity.w_star),
    ):
        print(f"  {hour:>5s}  {du:12.4g}  {dzi:12.4g}  {dl:12.4g}  {dw:12.4g}")

    # 4. Concentration field.
    field = model.grid_field([source])
    peak = np.unravel_index(np.argmax(field.values), field.values.shape)
    print(
        f"\nField {field.values.shape} (time, y, x); peak {field.values[peak]:.1f} ppb "
        f"at hour {DIURNAL_MET[peak[0]][0]}, "
        f"x={field.x[peak[2]]:.0f} m, y={field.y[peak[1]]:.0f} m"
    )

    if args.netcdf:
        path = write_grid_netcdf(
            field,
            args.netcdf,
            timestamps=[m.timestamp for m in model.config.met],
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
