"""Tests for the model-agnostic transport run layer.

Most tests use inline meteorology so they run without ERA5 data; the ones that
involve FLEXPART need real GRIB (it cannot be driven by scalars) and skip when
it is absent. AERMOD executes fully; the binary-backed models are exercised as
far as they can go without a compiled executable — which is exactly the
contract that matters for them: the native config generated from the shared
YAML must load with that model's own loader.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from enforceflux.transport import (
    CanonicalField,
    DomainProjection,
    TransportRunConfig,
    read_canonical,
    run_transport,
    write_canonical,
)
from enforceflux.transport import canonical, translate

SOURCE_LON, SOURCE_LAT = -121.75, 39.15

ERA5_DIR = (
    Path(__file__).resolve().parents[1] / "runs" / "sacramento_valley_2020" / "meteo_april_week"
)
requires_era5 = pytest.mark.skipif(
    not ERA5_DIR.is_dir(), reason="ERA5 GRIB test data not present"
)
# FLEXPART can only be driven by GRIB, so the cross-model tests use ERA5.
ERA5_MET = {
    "era5": {
        "meteo_dir": str(ERA5_DIR),
        "surface_roughness_m": 0.15,
    }
}


def base_config(**overrides) -> dict:
    """A minimal shared config: one source, three receptors, inline met."""
    config = {
        "transport": {
            "model": "aermod",
            "mode": "simulation",
            "start": "2020-03-31T00:00:00",
            "end": "2020-03-31T06:00:00",
        },
        "met": {
            "records": [
                {
                    "time": "2020-03-31T00:00",
                    "wind_speed_m_s": 3.0,
                    "wind_direction_deg": 270.0,  # from the west → plume toward +x
                    "mixing_height_m": 800.0,
                    "friction_velocity_m_s": 0.3,
                    "sensible_heat_flux_w_m2": 120.0,
                },
                {
                    "time": "2020-03-31T03:00",
                    "wind_speed_m_s": 2.0,
                    "wind_direction_deg": 270.0,
                    "mixing_height_m": 200.0,
                    "friction_velocity_m_s": 0.15,
                    "sensible_heat_flux_w_m2": -20.0,
                },
            ]
        },
        "domain": {
            "origin_lon": SOURCE_LON,
            "origin_lat": SOURCE_LAT,
            "x_min": -2600.0,
            "x_max": 2600.0,
            "y_min": -2200.0,
            "y_max": 2200.0,
            "spacing_m": 200.0,
            "receptor_height_m": 2.0,
        },
        "sources": [
            {
                "id": "leak",
                "x_m": 0.0,
                "y_m": 0.0,
                "alt_m": 5.0,
                "emission_rate_kg_s": 0.0278,
            }
        ],
        "receptors": [
            {"id": "east", "x_m": 500.0, "y_m": 0.0, "alt_m": 2.0},
            {"id": "west", "x_m": -500.0, "y_m": 0.0, "alt_m": 2.0},
        ],
        "output": {"path": "out.nc"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def load(tmp_path, **overrides) -> TransportRunConfig:
    blob = base_config(**overrides)
    blob["output"] = {"path": str(tmp_path / "out.nc")}
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(blob))
    return TransportRunConfig.from_file(path)


# ── Shared schema ────────────────────────────────────────────────────────────


def test_config_loads_the_shared_sections(tmp_path):
    run = load(tmp_path)
    assert run.model == "aermod"
    assert run.mode == "simulation"
    assert len(run.sources) == 1 and len(run.receptors) == 2
    assert run.sources[0].emission_rate_kg_s == 0.0278
    assert run.start is not None and run.end is not None


def test_unknown_model_and_mode_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="transport.model"):
        load(tmp_path, transport={"model": "calpuff"})
    with pytest.raises(ValueError, match="transport.mode"):
        load(tmp_path, transport={"mode": "sideways"})


def test_model_block_may_not_restate_a_shared_key(tmp_path):
    """The whole point of one file is that models cannot diverge."""
    with pytest.raises(ValueError, match="redefines shared key"):
        load(tmp_path, aermod={"sources": [{"id": "other", "x_m": 0.0, "y_m": 0.0}]})


def test_operator_mode_requires_receptors(tmp_path):
    with pytest.raises(ValueError, match="receptors"):
        load(tmp_path, transport={"mode": "operator"}, receptors=[])


def test_model_specific_blocks_are_isolated(tmp_path):
    run = load(
        tmp_path,
        aermod={"reduce": "mean"},
        flexpart={"executable": "/bin/false", "n_particles": 5},
    )
    assert run.option("reduce") == "mean"
    # The active model sees only its own block.
    import dataclasses

    as_flexpart = dataclasses.replace(run, model="flexpart")
    assert as_flexpart.option("n_particles") == 5
    assert as_flexpart.option("reduce") is None


# ── Projection ───────────────────────────────────────────────────────────────


def test_projection_round_trips_and_centres_on_the_domain():
    projection = DomainProjection(SOURCE_LON, SOURCE_LAT)
    x, y = projection.to_xy(SOURCE_LON, SOURCE_LAT)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)

    lon, lat = projection.to_lonlat(1000.0, 2000.0)
    back_x, back_y = projection.to_xy(lon, lat)
    assert back_x == pytest.approx(1000.0, abs=1e-3)
    assert back_y == pytest.approx(2000.0, abs=1e-3)


def test_sources_project_into_metres(tmp_path):
    run = load(tmp_path)
    sources = translate.projected_sources(run)
    assert len(sources) == 1
    # This source sits at the origin, so its Cartesian position is (0, 0).
    assert sources[0].x == pytest.approx(0.0, abs=1.0)
    assert sources[0].y == pytest.approx(0.0, abs=1.0)
    assert sources[0].z == 5.0
    assert sources[0].flux_true == 0.0278

    # A receptor declared 500 m east stays 500 m east.
    receptors = translate.projected_receptors(run)
    east = next(r for r in receptors if r["id"] == "east")
    assert 400.0 < east["x"] < 600.0
    assert east["y"] == pytest.approx(0.0, abs=5.0)


# ── Canonical output ─────────────────────────────────────────────────────────


def test_canonical_field_validates_its_axes():
    with pytest.raises(ValueError, match="Axis mismatch"):
        CanonicalField(
            x=np.arange(3.0), y=np.arange(4.0), values=np.zeros((2, 3, 3))
        )
    with pytest.raises(ValueError, match=r"\(time, y, x\)"):
        CanonicalField(x=np.arange(3.0), y=np.arange(3.0), values=np.zeros((3, 3)))


def test_canonical_netcdf_round_trips(tmp_path):
    field = CanonicalField(
        x=np.linspace(0.0, 400.0, 5),
        y=np.linspace(-200.0, 200.0, 3),
        values=np.random.default_rng(0).random((2, 3, 5)),
        timestamps=("2020-03-31T00:00", "2020-03-31T03:00"),
        meta={"model": "test"},
    )
    path = write_canonical(field, tmp_path / "canonical.nc")
    restored = read_canonical(path)

    assert restored.values.shape == field.values.shape
    assert restored.values == pytest.approx(field.values, rel=1e-6)
    assert restored.timestamps == field.timestamps
    assert restored.units == canonical.CANONICAL_UNITS
    assert restored.meta["model"] == "test"


def test_flexpart_canonicaliser_collapses_the_extra_axes(tmp_path):
    """A synthetic FLEXPART-shaped file must reduce to (time, y, x)."""
    from netCDF4 import Dataset

    path = tmp_path / "flexpart.nc"
    n_time, n_height, n_lat, n_lon = 3, 2, 5, 4
    with Dataset(path, "w") as ds:
        for name, size in (
            ("nageclass", 1), ("pointspec", 2), ("time", n_time),
            ("height", n_height), ("latitude", n_lat), ("longitude", n_lon),
        ):
            ds.createDimension(name, size)
        lon = ds.createVariable("longitude", "f8", ("longitude",))
        lon[:] = np.linspace(-121.8, -121.7, n_lon)
        lat = ds.createVariable("latitude", "f8", ("latitude",))
        lat[:] = np.linspace(39.1, 39.2, n_lat)
        var = ds.createVariable(
            "ch4_mixing_ratio", "f4",
            ("nageclass", "pointspec", "time", "height", "latitude", "longitude"),
        )
        var.units = "ng m-3"
        data = np.ones((1, 2, n_time, n_height, n_lat, n_lon))
        data[:, :, :, 1, :, :] = 99.0  # upper level must not be selected
        var[:] = data

    field = canonical.from_flexpart_netcdf(path, projection=DomainProjection(-121.75, 39.15))
    assert field.values.shape == (n_time, n_lat, n_lon)
    # Two releases summed at the surface level → 2.0, not the 99.0 aloft.
    assert field.values == pytest.approx(2.0)
    assert field.longitude.shape == (n_lat, n_lon)
    assert field.meta["model"] == "flexpart"


def test_flexpart_canonicaliser_refuses_unexpected_units(tmp_path):
    from netCDF4 import Dataset

    path = tmp_path / "wrong_units.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("latitude", 2)
        ds.createDimension("longitude", 2)
        ds.createVariable("longitude", "f8", ("longitude",))[:] = [0.0, 1.0]
        ds.createVariable("latitude", "f8", ("latitude",))[:] = [0.0, 1.0]
        var = ds.createVariable("ch4_mixing_ratio", "f4", ("time", "latitude", "longitude"))
        var.units = "ppb"
        var[:] = 1.0

    with pytest.raises(ValueError, match="ng m-3"):
        canonical.from_flexpart_netcdf(path)


def test_aermod_canonicaliser_refuses_wrong_units():
    from enforceflux.aermod.model import GridField

    field = GridField(
        x=np.arange(3.0), y=np.arange(3.0), z=2.0,
        values=np.zeros((1, 3, 3)), units="ppb_ch4_per_kg_s", meta={},
    )
    with pytest.raises(ValueError, match="ng m-3"):
        canonical.from_aermod(field)


# ── End-to-end runs ──────────────────────────────────────────────────────────


def test_aermod_simulation_writes_a_canonical_netcdf(tmp_path):
    run = load(tmp_path)
    result = run_transport(run)

    assert result.model == "aermod"
    assert result.mode == "simulation"
    assert result.units == canonical.CANONICAL_UNITS
    assert result.output_path is not None and result.output_path.exists()

    field = read_canonical(result.output_path)
    # One slice per met record, and geographic coordinates attached.
    assert field.values.shape[0] == 2
    assert field.longitude is not None and field.latitude is not None
    assert field.values.max() > 0.0
    assert len(field.timestamps) == 2


def test_aermod_operator_stacks_every_hour(tmp_path):
    run = load(tmp_path, transport={"mode": "operator"}, aermod={"reduce": "stack"})
    result = run_transport(run)

    assert result.g.shape == (4, 1)  # 2 hours × 2 receptors
    assert result.column_labels == ("leak",)
    assert result.row_labels[0][1] == "east"
    assert result.units == "ng m-3 / (kg s-1)"
    # Wind is from the west in both hours: the east receptor sees the plume, the
    # west receptor is upwind and sees nothing.
    east_rows = [g for label, g in zip(result.row_labels, result.g) if label[1] == "east"]
    west_rows = [g for label, g in zip(result.row_labels, result.g) if label[1] == "west"]
    assert all(row[0] > 0.0 for row in east_rows)
    assert all(row[0] == 0.0 for row in west_rows)


def test_operator_mean_and_stack_agree_on_average(tmp_path):
    stacked = run_transport(
        load(tmp_path, transport={"mode": "operator"}, aermod={"reduce": "stack"})
    ).g
    averaged = run_transport(
        load(tmp_path, transport={"mode": "operator"}, aermod={"reduce": "mean"})
    ).g
    assert averaged.shape == (2, 1)
    # Row order is hour-major, so 'east' rows are 0 and 2.
    assert averaged[0, 0] == pytest.approx(np.mean(stacked[[0, 2], 0]), rel=1e-5)


@requires_era5
def test_flexpart_generates_a_config_its_own_loader_accepts(tmp_path):
    """The contract for a binary-backed model: the generated native config is
    valid input for that model's own loader, and faithfully carries the shared
    scenario. (Executing it additionally needs the compiled binary and the
    FLEXPART options templates, which is a separate concern.)"""
    from enforceflux.flexpart.sim_config import load_simulation_config

    run = load(
        tmp_path,
        transport={"model": "flexpart"},
        met=ERA5_MET,
        flexpart={
            "executable": str(tmp_path / "FLEXPART"),
            "options_dir": str(tmp_path / "options"),
            "n_particles": 1234,
        },
    )
    series = translate.build_met_series(run)
    generated = translate.write_flexpart_config(run, series, tmp_path / "run")

    native = load_simulation_config(generated)
    assert len(native.sources) == 1
    assert native.sources[0].emission_rate_kg_s == 0.0278
    assert native.sources[0].n_particles == 1234
    # The FLEXPART grid is projected out of the Cartesian domain, so assert it
    # matches the derived bounds rather than a literal — that IS the contract.
    assert native.domain_lon_min == pytest.approx(run.domain.lon_min)
    assert native.domain_lat_max == pytest.approx(run.domain.lat_max)
    # ...and that the derived box really does bracket the declared origin.
    assert native.domain_lon_min < SOURCE_LON < native.domain_lon_max
    assert native.start.isoformat().startswith("2020-03-31")
    # FLEXPART is pointed back at the GRIB the canonical series came from.
    assert native.meteo_dir == ERA5_DIR


def test_microhh_generates_a_config_its_own_loader_accepts(tmp_path):
    from enforceflux.microhh.sim_config import load_microhh_config

    run = load(
        tmp_path,
        transport={"model": "microhh"},
        microhh={
            "executable": str(tmp_path / "microhh"),
            "grid": {"itot": 64, "jtot": 32, "ktot": 16},
            "met_reduce": "mean",
        },
    )
    series = translate.build_met_series(run)
    generated = translate.write_microhh_config(run, series, tmp_path / "run")

    native = load_microhh_config(generated)
    assert native.grid.itot == 64 and native.grid.jtot == 32
    assert len(native.sources) == 1
    assert native.sources[0].emission_rate_kg_s == 0.0278
    assert [r.id for r in native.receptors] == ["east", "west"]
    # Forcing came from the same canonical met the other models used: wind from
    # the west means the box points east.
    assert native.x_bearing_deg == pytest.approx(90.0, abs=1.0)
    assert native.forcing.u_geo > 0.0


def test_binary_models_report_missing_settings_clearly(tmp_path):
    run = load(tmp_path, transport={"model": "flexpart"})
    with pytest.raises(ValueError, match="executable"):
        run_transport(run, dry_run=True)


def test_flexpart_rejects_inline_met_with_an_actionable_message(tmp_path):
    """FLEXPART reads GRIB itself, so scalar met genuinely cannot drive it."""
    run = load(
        tmp_path,
        transport={"model": "flexpart"},
        flexpart={
            "executable": str(tmp_path / "FLEXPART"),
            "options_dir": str(tmp_path / "options"),
        },
    )
    with pytest.raises(ValueError, match="met.era5"):
        run_transport(run, dry_run=True)


@requires_era5
def test_every_model_accepts_the_identical_shared_config(tmp_path):
    """Switching transport.model must be the only change needed.

    Each model is taken as far as it goes without its binary: AERMOD runs, and
    the other two produce a native config their own loader accepts.
    """
    from enforceflux.flexpart.sim_config import load_simulation_config
    from enforceflux.microhh.sim_config import load_microhh_config

    shared = dict(
        met=ERA5_MET,
        aermod={"reduce": "mean"},
        flexpart={
            "executable": str(tmp_path / "FLEXPART"),
            "options_dir": str(tmp_path / "options"),
        },
        microhh={"executable": str(tmp_path / "microhh"), "met_reduce": "mean"},
    )

    seen_met = []
    for model in ("aermod", "flexpart", "microhh"):
        run = load(tmp_path, transport={"model": model}, **shared)
        series = translate.build_met_series(run)
        seen_met.append(len(series))

        if model == "aermod":
            result = run_transport(run)
            assert result.output_path.exists()
            assert result.units == canonical.CANONICAL_UNITS
            assert result.field.values.shape[0] == len(series)
        elif model == "flexpart":
            native = load_simulation_config(
                translate.write_flexpart_config(run, series, tmp_path / model)
            )
            assert native.sources[0].emission_rate_kg_s == 0.0278
        else:
            native = load_microhh_config(
                translate.write_microhh_config(run, series, tmp_path / model)
            )
            assert native.sources[0].emission_rate_kg_s == 0.0278

    # The identical meteorology reached all three.
    assert len(set(seen_met)) == 1
