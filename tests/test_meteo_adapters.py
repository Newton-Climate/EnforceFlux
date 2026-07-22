"""Tests for the canonical met record and the per-model adapters.

The adapter tests build :class:`MetRecord`s directly so they run anywhere; the
ERA5 reader tests need real GRIB files and skip when they are absent.
"""
import importlib.util
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from enforceflux.meteo.adapters import (
    microhh_box_bearing,
    to_aermod,
    to_flexpart,
    to_microhh_forcing,
)
from enforceflux.meteo.record import AIR_CP, AIR_DENSITY, MetRecord, MetSeries

ERA5_DIR = Path(__file__).resolve().parents[1] / "runs" / "sacramento_valley_2020" / "meteo_april_week"
SACRAMENTO = (-121.75, 39.15)

requires_era5 = pytest.mark.skipif(
    not ERA5_DIR.is_dir() or importlib.util.find_spec("eccodes") is None,
    reason="ERA5 GRIB test data not present or eccodes not installed",
)


def make_record(hour=0, wind=3.0, direction=270.0, heat_flux=100.0, **kwargs):
    defaults = dict(
        time=datetime(2020, 3, 30, tzinfo=timezone.utc) + timedelta(hours=hour),
        wind_speed_m_s=wind,
        wind_direction_deg=direction,
        temperature_k=290.0,
        mixing_height_m=800.0,
        friction_velocity_m_s=0.3,
        sensible_heat_flux_w_m2=heat_flux,
        surface_roughness_m=0.1,
        surface_pressure_pa=100000.0,
    )
    defaults.update(kwargs)
    return MetRecord(**defaults)


# ── Canonical record ─────────────────────────────────────────────────────────


def test_record_rejects_unphysical_values():
    with pytest.raises(ValueError, match="mixing_height_m"):
        make_record(mixing_height_m=0.0)
    with pytest.raises(ValueError, match="friction_velocity_m_s"):
        make_record(friction_velocity_m_s=0.0)


def test_obukhov_length_sign_follows_the_heat_flux():
    assert make_record(heat_flux=150.0).obukhov_length_m < 0.0  # convective
    assert make_record(heat_flux=-20.0).obukhov_length_m > 0.0  # stable
    assert math.isinf(make_record(heat_flux=0.0).obukhov_length_m)  # neutral


def test_kinematic_heat_flux_conversion():
    record = make_record(heat_flux=120.6)
    assert record.kinematic_heat_flux_k_m_s == pytest.approx(120.6 / (AIR_DENSITY * AIR_CP))


def test_wind_direction_conventions():
    record = make_record(direction=270.0)  # from the west
    assert record.wind_toward_deg == 90.0  # blows toward the east
    east, north = record.wind_components_m_s()
    assert east == pytest.approx(3.0)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_potential_temperature_uses_surface_pressure():
    at_reference = make_record(surface_pressure_pa=100000.0)
    assert at_reference.potential_temperature_k == pytest.approx(at_reference.temperature_k)
    aloft = make_record(surface_pressure_pa=90000.0)
    assert aloft.potential_temperature_k > aloft.temperature_k


def test_series_sorts_records_and_exposes_its_span():
    series = MetSeries(
        records=(make_record(hour=6), make_record(hour=0), make_record(hour=3)),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    assert [r.time.hour for r in series] == [0, 3, 6]
    assert series.start.hour == 0 and series.end.hour == 6


def test_series_window_and_daytime_selection():
    series = MetSeries(
        records=(
            make_record(hour=0, heat_flux=-20.0),
            make_record(hour=3, heat_flux=150.0),
            make_record(hour=6, heat_flux=180.0),
        ),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    base = datetime(2020, 3, 30, tzinfo=timezone.utc)
    assert len(series.window(base, base + timedelta(hours=3))) == 2
    assert len(series.daytime()) == 2
    with pytest.raises(ValueError, match="No met records"):
        series.window(base + timedelta(days=5), base + timedelta(days=6))


def test_series_mean_averages_wind_as_a_vector():
    """Opposing winds must cancel, not average to their scalar speed."""
    series = MetSeries(
        records=(make_record(hour=0, direction=270.0), make_record(hour=3, direction=90.0)),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    assert series.mean().wind_speed_m_s == pytest.approx(0.0, abs=1e-6)
    assert series.directional_consistency == pytest.approx(0.0, abs=1e-6)


def test_directional_consistency_is_one_for_a_steady_wind():
    series = MetSeries(
        records=(make_record(hour=0), make_record(hour=3), make_record(hour=6)),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    assert series.directional_consistency == pytest.approx(1.0)


# ── AERMOD adapter ───────────────────────────────────────────────────────────


def test_to_aermod_passes_measured_parameters_through():
    record = make_record(wind=4.0, direction=225.0, heat_flux=150.0)
    met = to_aermod(record)[0]

    assert met.wind_speed_m_s == 4.0
    assert met.wind_direction_deg == 225.0
    assert met.friction_velocity_m_s == record.friction_velocity_m_s
    assert met.sensible_heat_flux_w_m2 == record.sensible_heat_flux_w_m2
    assert met.mixing_height_m == record.mixing_height_m
    # Measured similarity parameters mean no Pasquill class is needed.
    assert met.stability_class is None
    assert met.monin_obukhov_length_m < 0.0


def test_to_aermod_handles_the_neutral_singularity():
    """L is infinite at neutral; AERMOD needs a finite stand-in."""
    met = to_aermod(make_record(heat_flux=0.0))[0]
    assert math.isfinite(met.monin_obukhov_length_m)
    assert abs(met.monin_obukhov_length_m) >= 1.0e6


def test_to_aermod_output_drives_a_model():
    from enforceflux.aermod import AermodConfig, AermodModel, Receptor
    from enforceflux.models.source import Source

    series = MetSeries(
        records=(make_record(hour=0), make_record(hour=3, heat_flux=-20.0)),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    config = AermodConfig(
        met=to_aermod(series),
        receptors=(Receptor(id="r", x=400.0, y=0.0, z=2.0),),
    )
    source = Source(
        id="s", kind="point", x=0.0, y=0.0, flux_true=1.0, flux_prior_mean=1.0, flux_prior_std=1.0
    )
    g = AermodModel(config).jacobian([source])
    assert g.shape == (1, 1)
    assert g[0, 0] > 0.0


def test_to_aermod_preserves_the_hour_axis():
    series = MetSeries(
        records=tuple(make_record(hour=h) for h in (0, 3, 6)),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    met = to_aermod(series)
    assert len(met) == 3
    assert [m.timestamp for m in met] == [r.time.isoformat() for r in series]


# ── MicroHH adapter ──────────────────────────────────────────────────────────


def test_to_microhh_forcing_maps_units_and_roughness():
    record = make_record(heat_flux=120.6, surface_roughness_m=0.2)
    forcing = to_microhh_forcing(record)

    assert forcing.surface_heat_flux_K_m_s == pytest.approx(record.kinematic_heat_flux_k_m_s)
    assert forcing.z0m == 0.2
    assert forcing.z0h == pytest.approx(0.02)  # z0m/10 convention
    assert forcing.boundary_layer_height_m == record.mixing_height_m
    assert forcing.thl_surface_K == pytest.approx(record.potential_temperature_k)


def test_microhh_box_aligned_with_the_wind_has_no_cross_component():
    record = make_record(wind=3.0, direction=270.0)
    forcing = to_microhh_forcing(record)
    assert forcing.u_geo == pytest.approx(3.0)
    assert forcing.v_geo == pytest.approx(0.0, abs=1e-9)
    assert microhh_box_bearing(record) == pytest.approx(90.0)


def test_microhh_wind_rotates_into_a_given_box_bearing():
    """A box 90° off the wind sees the flow entirely as a cross component."""
    record = make_record(wind=3.0, direction=270.0)  # blows toward 90°
    forcing = to_microhh_forcing(record, x_bearing_deg=0.0)  # box +x points north
    assert forcing.u_geo == pytest.approx(0.0, abs=1e-6)
    assert abs(forcing.v_geo) == pytest.approx(3.0)


def test_microhh_refuses_to_collapse_a_veering_series():
    series = MetSeries(
        records=(make_record(hour=0, direction=270.0), make_record(hour=3, direction=90.0)),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    with pytest.raises(ValueError, match="too variable"):
        to_microhh_forcing(series)
    # The guard is a default, not a law.
    forcing = to_microhh_forcing(series, min_directional_consistency=0.0)
    assert forcing.u_geo == pytest.approx(0.0, abs=1e-6)


def test_microhh_accepts_a_steady_series():
    series = MetSeries(
        records=tuple(make_record(hour=h, direction=270.0 + h) for h in (0, 3, 6)),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
    )
    assert to_microhh_forcing(series).u_geo > 2.9


# ── FLEXPART adapter ─────────────────────────────────────────────────────────


def test_to_flexpart_refuses_a_series_without_era5_provenance():
    series = MetSeries(
        records=(make_record(),), longitude=SACRAMENTO[0], latitude=SACRAMENTO[1]
    )
    with pytest.raises(ValueError, match="reads the GRIB files directly"):
        to_flexpart(series)


def test_flexpart_met_source_reports_missing_coverage(tmp_path):
    series = MetSeries(
        records=(make_record(),),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
        provenance={"meteo_dir": str(tmp_path)},  # no AVAILABLE file
    )
    source = to_flexpart(series)
    assert source.covers_window is False
    with pytest.raises(ValueError, match="does not cover"):
        source.require_coverage()


def test_flexpart_met_source_yields_yaml_keys(tmp_path):
    series = MetSeries(
        records=(make_record(),),
        longitude=SACRAMENTO[0],
        latitude=SACRAMENTO[1],
        provenance={"meteo_dir": str(tmp_path)},
    )
    config = to_flexpart(series).as_config()
    assert set(config) == {"meteo_dir", "available_file"}
    assert config["available_file"].endswith("AVAILABLE")


# ── ERA5 reader (needs GRIB data) ────────────────────────────────────────────


@requires_era5
def test_era5_reader_produces_a_physical_diurnal_cycle():
    from enforceflux.meteo import met_series_from_era5

    # Two days is enough to see the cycle; reading the whole directory means
    # opening ~1000 GRIB messages per file.
    start = datetime(2020, 3, 30, tzinfo=timezone.utc)
    series = met_series_from_era5(
        ERA5_DIR, *SACRAMENTO, start=start, end=start + timedelta(days=2),
        surface_roughness_m=0.15,
    )
    assert len(series) > 8
    assert series.provenance["source"] == "era5"

    for record in series:
        assert 0.0 < record.wind_speed_m_s < 60.0
        assert 200.0 < record.temperature_k < 340.0
        assert record.mixing_height_m >= 50.0
        assert 0.0 <= record.wind_direction_deg < 360.0

    # Convective hours must have the deeper boundary layer.
    convective = [r for r in series if r.is_convective]
    stable = [r for r in series if not r.is_convective]
    assert convective and stable
    mean_convective = sum(r.mixing_height_m for r in convective) / len(convective)
    mean_stable = sum(r.mixing_height_m for r in stable) / len(stable)
    assert mean_convective > 2.0 * mean_stable


@requires_era5
def test_era5_window_selection_and_stress_option():
    from enforceflux.meteo import met_series_from_era5

    start = datetime(2020, 3, 31, tzinfo=timezone.utc)
    end = datetime(2020, 4, 1, tzinfo=timezone.utc)
    series = met_series_from_era5(ERA5_DIR, *SACRAMENTO, start=start, end=end)
    assert all(start <= r.time <= end for r in series)

    stress = met_series_from_era5(
        ERA5_DIR, *SACRAMENTO, start=start, end=end, friction_velocity="stress"
    )
    # ERA5 stress includes orographic form drag, so u* exceeds the log-law value.
    assert stress[0].friction_velocity_m_s > series[0].friction_velocity_m_s


@requires_era5
def test_era5_series_feeds_every_model():
    from enforceflux.meteo import met_series_from_era5

    start = datetime(2020, 3, 31, tzinfo=timezone.utc)
    day = met_series_from_era5(
        ERA5_DIR, *SACRAMENTO, start=start, end=start + timedelta(days=1)
    )

    assert len(to_aermod(day)) == len(day)
    assert to_microhh_forcing(day, reduce="daytime_mean").u_geo > 0.0
    assert to_flexpart(day).meteo_dir == ERA5_DIR


@requires_era5
def test_era5_rejects_an_out_of_range_window():
    from enforceflux.meteo import met_series_from_era5

    with pytest.raises(ValueError, match="No ERA5 records"):
        met_series_from_era5(
            ERA5_DIR,
            *SACRAMENTO,
            start=datetime(1999, 1, 1, tzinfo=timezone.utc),
            end=datetime(1999, 1, 2, tzinfo=timezone.utc),
        )


def test_era5_reader_rejects_a_missing_directory():
    from enforceflux.meteo import met_series_from_era5

    with pytest.raises(FileNotFoundError):
        met_series_from_era5("/nonexistent/meteo", *SACRAMENTO)
