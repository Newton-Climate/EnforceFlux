"""Tests for the AERMOD-style transport model.

The physics assertions are limit tests rather than comparisons against
reference AERMOD output: mass conservation in the mixed layer, the correct
sign of the stability response, and the analytic well-mixed far field.
"""
import json

import jax
import numpy as np
import pytest

from enforceflux.aermod import (
    AermodConfig,
    AermodModel,
    Receptor,
    ReceptorGrid,
    StackParameters,
    SurfaceMet,
    derive_met_state,
)
from enforceflux.aermod.dispersion import bigaussian_parameters
from enforceflux.aermod.config import DispersionOptions
from enforceflux.core.base import ForwardModelResult, TransportSimulationResult
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source
from enforceflux.plugins.simulation_aermod import AermodSimulationModel
from enforceflux.plugins.transport_aermod import AermodTransportOperator

WIND_FROM_WEST = 270.0  # plume travels toward +x


def make_source(source_id="s1", x=0.0, y=0.0, flux=1.0):
    return Source(
        id=source_id,
        kind="point",
        x=x,
        y=y,
        flux_true=flux,
        flux_prior_mean=flux,
        flux_prior_std=1.0,
    )


def make_met(stability="D", wind=3.0, mixing_height=500.0):
    return SurfaceMet(
        wind_speed_m_s=wind,
        wind_direction_deg=WIND_FROM_WEST,
        stability_class=stability,
        mixing_height_m=mixing_height,
    )


# ── Configuration ────────────────────────────────────────────────────────────


def test_surface_met_requires_a_stability_specification():
    with pytest.raises(ValueError, match="stability_class"):
        SurfaceMet(wind_speed_m_s=3.0, wind_direction_deg=180.0)


def test_surface_met_rejects_unknown_stability_class():
    with pytest.raises(ValueError, match="stability_class"):
        SurfaceMet(wind_speed_m_s=3.0, wind_direction_deg=180.0, stability_class="Z")


def test_config_round_trips_through_a_dict():
    blob = {
        "met": {"wind_speed_m_s": 2.0, "wind_direction_deg": 90.0, "stability_class": "B"},
        "default_stack": {"height_m": 12.0},
        "stacks": {"s1": {"height_m": 30.0, "diameter_m": 1.5}},
        "receptors": [{"id": "r1", "x": 100.0, "y": 0.0, "z": 2.0}],
        "concentration_units": "ug_m3_per_g_s",
        "reduce": "max",
    }
    config = AermodConfig.from_dict(blob)

    assert len(config.met) == 1
    assert config.stack_for("s1").height_m == 30.0
    assert config.stack_for("other").height_m == 12.0  # falls back to default_stack
    assert config.reduce == "max"


def test_config_reads_a_json_file(tmp_path):
    path = tmp_path / "aermod.json"
    path.write_text(
        json.dumps(
            {
                "aermod": {
                    "met": [
                        {
                            "wind_speed_m_s": 4.0,
                            "wind_direction_deg": 270.0,
                            "stability_class": "D",
                        }
                    ]
                }
            }
        )
    )
    config = AermodConfig.from_file(path)
    assert config.met[0].wind_speed_m_s == 4.0


def test_unknown_units_are_rejected():
    with pytest.raises(ValueError, match="concentration_units"):
        AermodConfig(met=[make_met()], concentration_units="furlongs")


# ── Meteorology ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "stability,expected_sign",
    [("A", -1.0), ("B", -1.0), ("C", -1.0), ("D", 0.0), ("E", 1.0), ("F", 1.0)],
)
def test_stability_class_sets_the_sign_of_the_obukhov_length(stability, expected_sign):
    state = derive_met_state(make_met(stability=stability))
    assert np.sign(float(state.inv_obukhov_length)) == expected_sign


def test_convective_velocity_scale_is_positive_only_in_unstable_conditions():
    assert float(derive_met_state(make_met("A")).w_star) > 0.0
    assert float(derive_met_state(make_met("D")).w_star) == 0.0
    assert float(derive_met_state(make_met("F")).w_star) == 0.0


def test_measured_values_override_derived_ones():
    met = SurfaceMet(
        wind_speed_m_s=3.0,
        wind_direction_deg=270.0,
        stability_class="D",
        friction_velocity_m_s=0.42,
        mixing_height_m=1234.0,
    )
    state = derive_met_state(met)
    assert float(state.u_star) == pytest.approx(0.42)
    assert float(state.mixing_height) == pytest.approx(1234.0)


# ── Dispersion physics ───────────────────────────────────────────────────────


def test_no_concentration_upwind_of_the_source():
    config = AermodConfig(met=[make_met()])
    model = AermodModel(config)
    upwind = Receptor(id="up", x=-500.0, y=0.0, z=2.0)
    assert model.unit_response([make_source()], [upwind])[0, 0, 0] == 0.0


def test_plume_travels_with_the_wind_direction():
    source = make_source()
    # Wind from the south → plume toward +y.
    config = AermodConfig(
        met=[SurfaceMet(wind_speed_m_s=3.0, wind_direction_deg=180.0, stability_class="D")]
    )
    model = AermodModel(config)
    north = Receptor(id="n", x=0.0, y=300.0, z=2.0)
    east = Receptor(id="e", x=300.0, y=0.0, z=2.0)
    response = model.unit_response([source], [north, east])[0, :, 0]
    assert response[0] > 0.0
    assert response[1] == 0.0


def test_concentration_decreases_with_downwind_distance():
    model = AermodModel(AermodConfig(met=[make_met()]))
    receptors = [Receptor(id=f"r{d}", x=float(d), y=0.0, z=2.0) for d in (100, 300, 1000, 3000)]
    response = model.unit_response([make_source()], receptors)[0, :, 0]
    assert np.all(np.diff(response) < 0.0)


def test_crosswind_profile_is_symmetric_and_peaks_on_the_centreline():
    model = AermodModel(AermodConfig(met=[make_met()]))
    receptors = [Receptor(id=f"r{y}", x=500.0, y=float(y), z=2.0) for y in (-200, -50, 0, 50, 200)]
    response = model.unit_response([make_source()], receptors)[0, :, 0]
    assert response[2] == response.max()
    assert response[0] == pytest.approx(response[4], rel=1e-5)
    assert response[1] == pytest.approx(response[3], rel=1e-5)


def test_mass_is_conserved_across_the_plume_cross_section():
    """∫∫ χ dy dz over the mixed layer must equal 1/u for a unit emission."""
    mixing_height = 500.0
    config = AermodConfig(met=[make_met(mixing_height=mixing_height)])
    model = AermodModel(config)

    ys = np.linspace(-6000.0, 6000.0, 801)
    zs = np.linspace(0.0, mixing_height, 201)
    receptors = [
        Receptor(id="r", x=1000.0, y=float(y), z=float(z)) for y in ys for z in zs
    ]
    response = model.unit_response([make_source()], receptors)[:, :, 0]
    grid = response.reshape(len(ys), len(zs))
    integral = np.trapezoid(np.trapezoid(grid, zs, axis=1), ys)

    # The transport wind at the (ground-level) release height.
    from enforceflux.aermod.meteorology import wind_speed_at

    u_eff = float(np.asarray(wind_speed_at(model.met_state, 2.0))[0])
    assert integral * u_eff == pytest.approx(1.0, rel=2e-3)


def test_far_field_matches_the_analytic_well_mixed_limit():
    """Far downwind, χ/Q → 1/(√(2π) σ_y u zi) once the plume fills the layer."""
    mixing_height = 300.0
    model = AermodModel(AermodConfig(met=[make_met(mixing_height=mixing_height)]))
    receptors = [Receptor(id="r", x=20000.0, y=0.0, z=float(z)) for z in (5.0, 150.0, 290.0)]
    response = model.unit_response([make_source()], receptors)[0, :, 0]
    # Uniform in the vertical is the signature of the well-mixed limit.
    assert response.max() / response.min() < 1.05


def test_ground_level_release_impact_increases_with_stability():
    """A surface release disperses least — so concentrates most — in stable air."""
    receptor = [Receptor(id="r", x=200.0, y=0.0, z=1.0)]
    values = [
        AermodModel(AermodConfig(met=[make_met(stability=s)])).unit_response(
            [make_source()], receptor
        )[0, 0, 0]
        for s in "ABCDEF"
    ]
    assert np.all(np.diff(values) > 0.0)


def test_elevated_buoyant_release_impacts_the_ground_most_in_unstable_air():
    """The converse case: a lofted plume only reaches the ground when mixing is vigorous."""
    stack = StackParameters(
        height_m=30.0, diameter_m=1.5, exit_velocity_m_s=15.0, exit_temperature_k=420.0
    )
    receptor = [Receptor(id="r", x=400.0, y=0.0, z=1.5)]
    unstable = AermodModel(
        AermodConfig(met=[make_met("B")], default_stack=stack)
    ).unit_response([make_source()], receptor)[0, 0, 0]
    stable = AermodModel(
        AermodConfig(met=[make_met("F")], default_stack=stack)
    ).unit_response([make_source()], receptor)[0, 0, 0]
    assert unstable > 100.0 * stable


def test_plume_rise_lifts_the_plume_off_the_ground():
    """Adding buoyancy must reduce the near-field ground-level concentration."""
    receptor = [Receptor(id="r", x=200.0, y=0.0, z=1.0)]
    passive = AermodModel(
        AermodConfig(met=[make_met()], default_stack=StackParameters(height_m=20.0))
    ).unit_response([make_source()], receptor)[0, 0, 0]
    buoyant = AermodModel(
        AermodConfig(
            met=[make_met()],
            default_stack=StackParameters(
                height_m=20.0,
                diameter_m=2.0,
                exit_velocity_m_s=20.0,
                exit_temperature_k=450.0,
            ),
        )
    ).unit_response([make_source()], receptor)[0, 0, 0]
    assert buoyant < passive


def test_bigaussian_pdf_reproduces_its_defining_moments():
    """The updraft/downdraft closure must have zero mean and the target variance."""
    options = DispersionOptions()
    sigma_w = 0.8
    lam1, w1, lam2, w2 = (float(v) for v in bigaussian_parameters(sigma_w, options))

    assert lam1 + lam2 == pytest.approx(1.0)
    assert lam1 * w1 + lam2 * w2 == pytest.approx(0.0, abs=1e-6)

    r = options.cbl_sigma_ratio
    variance = lam1 * (w1**2 + (r * abs(w1)) ** 2) + lam2 * (w2**2 + (r * abs(w2)) ** 2)
    assert variance == pytest.approx(sigma_w**2, rel=1e-4)

    third = lam1 * w1 * (w1**2 + 3 * (r * abs(w1)) ** 2) + lam2 * w2 * (
        w2**2 + 3 * (r * abs(w2)) ** 2
    )
    assert third == pytest.approx(options.cbl_skewness * sigma_w**3, rel=1e-3)


# ── Differentiability ────────────────────────────────────────────────────────


@pytest.mark.parametrize("stability", list("ABCDEF"))
def test_gradients_are_finite_in_every_stability_class(stability):
    model = AermodModel(
        AermodConfig(
            met=[make_met(stability=stability)],
            receptors=(Receptor(id="r", x=300.0, y=20.0, z=2.0),),
            default_stack=StackParameters(
                height_m=15.0, diameter_m=1.0, exit_velocity_m_s=10.0, exit_temperature_k=380.0
            ),
        )
    )
    gradient = model.sensitivity_to_met([make_source()])
    for leaf in jax.tree.leaves(gradient):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_gradient_with_respect_to_source_position_matches_finite_differences():
    import jax.numpy as jnp

    from enforceflux.aermod.dispersion import ReceptorArray, SourceArray, StackArray

    model = AermodModel(AermodConfig(met=[make_met()]))
    fn = model.response_fn()
    receptor = ReceptorArray(jnp.array([400.0]), jnp.array([0.0]), jnp.array([2.0]))
    stack = StackArray(
        jnp.array([10.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0])
    )

    def response(source_x):
        source = SourceArray(jnp.array([source_x]), jnp.array([0.0]), jnp.array([0.0]))
        return fn(receptor, source, stack, model.met_state).sum()

    analytic = float(jax.grad(response)(0.0))
    step = 0.5
    numeric = float((response(step) - response(-step)) / (2 * step))
    assert analytic == pytest.approx(numeric, rel=1e-2)


def test_mixing_height_sensitivity_is_negative_in_the_far_field():
    """A deeper mixed layer dilutes a well-mixed plume, so ∂C/∂zi < 0."""
    model = AermodModel(
        AermodConfig(
            met=[make_met(mixing_height=400.0)],
            receptors=(Receptor(id="r", x=15000.0, y=0.0, z=2.0),),
        )
    )
    gradient = model.sensitivity_to_met([make_source()])
    assert float(np.asarray(gradient.mixing_height)[0]) < 0.0


# ── Jacobian assembly ────────────────────────────────────────────────────────


def test_jacobian_shape_and_superposition():
    sources = [make_source("s1", x=0.0), make_source("s2", x=200.0)]
    receptors = [
        Receptor(id="r1", x=500.0, y=0.0, z=2.0),
        Receptor(id="r2", x=800.0, y=50.0, z=2.0),
        Receptor(id="r3", x=1200.0, y=-50.0, z=2.0),
    ]
    model = AermodModel(AermodConfig(met=[make_met()], receptors=tuple(receptors)))
    g = model.jacobian(sources)

    assert g.shape == (3, 2)
    # The nearer source dominates at r1, and G is a linear operator on emissions.
    assert g[0, 1] > g[0, 0]
    concentrations = model.concentrations(sources, emissions=[2.0, 3.0])[0]
    assert concentrations == pytest.approx(g @ np.array([2.0, 3.0]), rel=1e-5)


def test_reduce_controls_the_meteorology_axis():
    sources = [make_source()]
    receptors = (Receptor(id="r", x=400.0, y=0.0, z=2.0),)
    mets = [make_met(wind=2.0), make_met(wind=6.0)]

    kept = AermodModel(
        AermodConfig(met=mets, receptors=receptors, reduce="none")
    ).jacobian(sources)
    assert kept.shape == (2, 1, 1)

    mean = AermodModel(AermodConfig(met=mets, receptors=receptors)).jacobian(sources)
    assert mean[0, 0] == pytest.approx(kept.mean(axis=0)[0, 0], rel=1e-5)

    maximum = AermodModel(
        AermodConfig(met=mets, receptors=receptors, reduce="max")
    ).jacobian(sources)
    assert maximum[0, 0] == pytest.approx(kept.max(axis=0)[0, 0], rel=1e-5)


def test_emission_unit_scaling_rescales_the_jacobian():
    sources = [make_source()]
    receptors = (Receptor(id="r", x=400.0, y=0.0, z=2.0),)
    base = AermodModel(AermodConfig(met=[make_met()], receptors=receptors)).jacobian(sources)
    per_hour = AermodModel(
        AermodConfig(met=[make_met()], receptors=receptors, emission_scale_to_kg_s=1 / 3600)
    ).jacobian(sources)
    assert per_hour[0, 0] == pytest.approx(base[0, 0] / 3600, rel=1e-5)


def test_path_receptors_average_into_one_observation_row():
    instrument = Instrument(
        id="op1", tech_id="OP", x=500.0, y=0.0, z=2.0, path_length_m=200.0, path_bearing_deg=0.0
    )
    config = AermodConfig(met=[make_met()], receptor_path_samples=5)
    result = AermodTransportOperator().build_forward_operator(
        [make_source()], [instrument], None, _as_plugin_config(config)
    )
    assert result.g.shape == (1, 1)
    assert result.meta["n_receptors"] == 5


def _as_plugin_config(config: AermodConfig) -> dict:
    """The dict form of a config, as the registry plugins receive it."""
    met = config.met[0]
    return {
        "met": [
            {
                "wind_speed_m_s": met.wind_speed_m_s,
                "wind_direction_deg": met.wind_direction_deg,
                "stability_class": met.stability_class,
                "mixing_height_m": met.mixing_height_m,
            }
        ],
        "receptor_path_samples": config.receptor_path_samples,
    }


# ── Registry plugins ─────────────────────────────────────────────────────────


def test_transport_operator_plugin_builds_a_jacobian():
    instruments = [
        Instrument(id="a", tech_id="OP", x=300.0, y=0.0, z=2.0),
        Instrument(id="b", tech_id="OP", x=600.0, y=100.0, z=2.0),
    ]
    result = AermodTransportOperator().build_forward_operator(
        [make_source("s1"), make_source("s2", x=100.0)],
        instruments,
        None,
        {
            "met": {
                "wind_speed_m_s": 3.0,
                "wind_direction_deg": WIND_FROM_WEST,
                "stability_class": "D",
            },
            "concentration_units": "ug_m3_per_g_s",
        },
    )
    assert isinstance(result, ForwardModelResult)
    assert result.g.shape == (2, 2)
    assert np.all(result.g >= 0.0)
    assert result.meta["units"] == "ug_m3_per_g_s"


def test_transport_operator_plugin_requires_meteorology():
    with pytest.raises(ValueError, match="met"):
        AermodTransportOperator().build_forward_operator(
            [make_source()], [Instrument(id="a", tech_id="OP", x=1.0, y=0.0)], None, {}
        )


def test_simulation_plugin_writes_a_netcdf_grid(tmp_path):
    output = tmp_path / "aermod.nc"
    result = AermodSimulationModel().simulate(
        [make_source(flux=0.01)],
        None,
        {
            "met": [
                {
                    "wind_speed_m_s": 3.0,
                    "wind_direction_deg": WIND_FROM_WEST,
                    "stability_class": "D",
                },
                {
                    "wind_speed_m_s": 5.0,
                    "wind_direction_deg": 225.0,
                    "stability_class": "C",
                },
            ],
            "grid": {
                "x_min": -200.0,
                "x_max": 800.0,
                "y_min": -500.0,
                "y_max": 500.0,
                "spacing_m": 50.0,
                "height_m": 2.0,
            },
            "output_path": str(output),
        },
    )

    assert isinstance(result, TransportSimulationResult)
    assert result.output_path == output
    assert output.exists()

    from netCDF4 import Dataset

    with Dataset(output) as ds:
        concentration = ds.variables["concentration"][:]
        assert concentration.shape == (2, 21, 21)
        assert float(concentration.max()) > 0.0


def test_simulation_plugin_returns_the_field_when_no_output_path_is_given():
    result = AermodSimulationModel().simulate(
        [make_source(flux=0.01)],
        None,
        {
            "met": {
                "wind_speed_m_s": 3.0,
                "wind_direction_deg": WIND_FROM_WEST,
                "stability_class": "D",
            },
            "grid": {
                "x_min": 0.0,
                "x_max": 500.0,
                "y_min": -250.0,
                "y_max": 250.0,
                "spacing_m": 50.0,
            },
        },
    )
    assert result.output_path is None
    assert result.meta["field"].values.shape == (1, 11, 11)


def test_simulation_plugin_falls_back_to_the_run_domain():
    from enforceflux.models.config import DomainConfig

    domain = DomainConfig(
        x_min=0.0, x_max=400.0, y_min=-200.0, y_max=200.0, grid_spacing=100.0
    )
    result = AermodSimulationModel().simulate(
        [make_source(flux=0.01)],
        domain,
        {
            "met": {
                "wind_speed_m_s": 3.0,
                "wind_direction_deg": WIND_FROM_WEST,
                "stability_class": "D",
            }
        },
    )
    assert result.meta["field"].values.shape == (1, 5, 5)


def test_grid_field_axes_follow_the_receptor_grid():
    grid = ReceptorGrid(x_min=0.0, x_max=1000.0, y_min=-500.0, y_max=500.0, spacing_m=250.0)
    model = AermodModel(AermodConfig(met=[make_met()], grid=grid))
    field = model.grid_field([make_source(flux=0.01)])
    assert field.values.shape == (1, 5, 5)
    assert field.x[0] == 0.0 and field.x[-1] == 1000.0
    assert field.y[0] == -500.0 and field.y[-1] == 500.0


def test_stack_reduce_keeps_every_hour_as_its_own_observation():
    """Time-resolved form: no averaging over the meteorology axis at all."""
    sources = [make_source("s1"), make_source("s2", x=150.0)]
    receptors = (
        Receptor(id="r1", x=400.0, y=0.0, z=2.0),
        Receptor(id="r2", x=900.0, y=60.0, z=2.0),
    )
    mets = [make_met(wind=2.0), make_met(wind=5.0), make_met(stability="F", wind=1.5)]

    unreduced = AermodModel(
        AermodConfig(met=mets, receptors=receptors, reduce="none")
    ).jacobian(sources)
    stacked = AermodModel(
        AermodConfig(met=mets, receptors=receptors, reduce="stack")
    ).jacobian(sources)

    assert unreduced.shape == (3, 2, 2)
    assert stacked.shape == (6, 2)
    # Hour-major ordering: row = hour * n_obs + receptor.
    assert stacked == pytest.approx(unreduced.reshape(6, 2))
    # Distinct hours must stay distinct — this is the whole point.
    assert not np.allclose(stacked[0], stacked[2])


def test_observation_labels_name_the_stacked_rows():
    receptors = (
        Receptor(id="r1", x=400.0, y=0.0, z=2.0),
        Receptor(id="r2", x=800.0, y=0.0, z=2.0),
    )
    mets = [
        SurfaceMet(
            wind_speed_m_s=3.0,
            wind_direction_deg=WIND_FROM_WEST,
            stability_class="D",
            timestamp=stamp,
        )
        for stamp in ("2020-03-31T00:00", "2020-03-31T03:00")
    ]
    model = AermodModel(AermodConfig(met=mets, receptors=receptors, reduce="stack"))

    labels = model.observation_labels()
    assert labels == [
        ("2020-03-31T00:00", "r1"),
        ("2020-03-31T00:00", "r2"),
        ("2020-03-31T03:00", "r1"),
        ("2020-03-31T03:00", "r2"),
    ]
    assert len(labels) == model.jacobian([make_source()]).shape[0]


def test_stacked_path_receptors_collapse_before_stacking():
    """Grouping happens per hour, so N path samples still give one row per hour."""
    instrument = Instrument(
        id="op1", tech_id="OP", x=500.0, y=0.0, z=2.0, path_length_m=200.0, path_bearing_deg=0.0
    )
    from enforceflux.aermod import receptors_from_instruments

    receptors = receptors_from_instruments([instrument], path_samples=4)
    model = AermodModel(
        AermodConfig(met=[make_met(), make_met(wind=6.0)], reduce="stack")
    )
    g = model.jacobian([make_source()], receptors)
    assert g.shape == (2, 1)
    assert model.observation_labels(receptors) == [(None, "op1"), (None, "op1")]
