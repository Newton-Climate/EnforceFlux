"""
Unit and integration tests for FlexpartBackwardRunner.

Unit tests (no binary required):
    - to_jacobian() conversion math and shape invariance.
    - Dry-run file structure: COMMAND, RELEASES, OUTGRID, pathnames.
    - COMMAND has LDIRECT=-1 and IOUTPUTFOREACHRELEASE=1.
    - RELEASES instantaneous release at obs_time with MASS=1.
    - G-matrix shape matches (n_instruments, n_sources).

Integration tests (marked flexpart_integration — require FLEXPART binary + met data):
    - Backward run produces grid_time_*.nc footprint file.
    - Footprint values are non-negative.
    - Footprint is non-zero at the source location.
    - Reciprocity: G_bwd × 1e12 / V_cell ≈ G_fwd  (within 25 %).
      Validates that backward-mode footprint equals the forward
      source-receptor sensitivity for the same source-receptor pair.
"""
import dataclasses
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from enforceflux.models.config import DomainConfig
from enforceflux.flexpart import (
    FOOTPRINT_TO_JACOBIAN,
    FlexpartBackwardRunner,
    FlexpartSimulation,
    load_simulation_config,
)
from enforceflux.flexpart.simulation import PointSource, SimulationConfig
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).resolve().parents[1]
_BINARY    = REPO_ROOT / "flexpart" / "src" / "FLEXPART"
_OPTIONS   = REPO_ROOT / "flexpart" / "tests" / "default_options"
_AVAILABLE = REPO_ROOT / "flexpart" / "tests" / "default_winds" / "AVAILABLE"
_METEO     = REPO_ROOT / "flexpart" / "tests" / "testdata"

_SIM_START = datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_SIM_END   = datetime(2009, 1, 1, 3, 0, 0, tzinfo=timezone.utc)

# Source and receptor placed at the same location for maximum footprint overlap.
# (WGS-84 domain: x = lon, y = lat)
_SOURCE_LON, _SOURCE_LAT = 7.5, 51.5
_INST_LON,   _INST_LAT   = 7.5, 51.5   # co-located for reciprocity check

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_base_config(tmp_path: Path, name: str = "base", ldirect: int = 1,
                      **overrides) -> SimulationConfig:
    defaults = dict(
        executable=_BINARY.resolve(),
        options_dir=_OPTIONS.resolve(),
        available_file=_AVAILABLE.resolve(),
        meteo_dir=_METEO.resolve(),
        run_dir=(tmp_path / name).resolve(),
        start=_SIM_START,
        end=_SIM_END,
        output_step_s=3600,
        domain_lon_min=-25.0, domain_lat_min=10.0,
        domain_lon_max=60.0,  domain_lat_max=75.0,
        domain_dx=1.0, domain_dy=1.0,
        heights_m=[100.0, 500.0, 1000.0, 50000.0],
        sources=[],
        output_path=(tmp_path / f"{name}_out.nc").resolve(),
        species_name="CH4", species_number=24,
        nxshift=0,
        ldirect=ldirect,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _wgs84_domain() -> DomainConfig:
    """WGS-84 domain: x = longitude, y = latitude (no projection)."""
    return DomainConfig(
        x_min=-25.0, x_max=60.0,
        y_min=10.0,  y_max=75.0,
        grid_spacing=1.0,
        crs="EPSG:4326",
        crs_wgs84="EPSG:4326",
    )


def _instrument(lon: float = _INST_LON, lat: float = _INST_LAT) -> Instrument:
    return Instrument(id="obs", tech_id="LP_ESN", mode="good", x=lon, y=lat, z=10.0)


def _source(lon: float = _SOURCE_LON, lat: float = _SOURCE_LAT,
            flux: float = 5e-3) -> Source:
    return Source(id="src", kind="point", x=lon, y=lat,
                  flux_true=flux, flux_prior_mean=flux * 0.6, flux_prior_std=flux * 0.4)


def _grid_cell_area_m2(lat_deg: float, dx_deg: float = 1.0, dy_deg: float = 1.0) -> float:
    """Approximate horizontal area of a 1° × 1° grid cell at given latitude."""
    R = 6_371_000.0
    return (math.radians(dx_deg) * R * math.cos(math.radians(lat_deg))) * \
           (math.radians(dy_deg) * R)


# ─── Unit tests: to_jacobian ─────────────────────────────────────────────────

class TestToJacobian:
    def test_scalar_conversion(self):
        """G_raw [s] × 1e12 / V_cell [m³] = G_phys [ng/m³/(kg/s)]."""
        g_raw = np.array([[4062.0]])
        area  = np.array([7.7e9])       # ~1°×1° cell at 51.5°N in m²
        height = 100.0                   # mixing height m

        G = FlexpartBackwardRunner.to_jacobian(g_raw, area, mixing_height_m=height)

        V_cell = area[0] * height
        expected = 4062.0 * 1e12 / V_cell
        assert abs(G[0, 0] - expected) / expected < 1e-9

    def test_shape_preserved(self):
        """Output shape matches input (m, n)."""
        g_raw = np.ones((3, 5))
        areas = np.ones(5) * 1e10
        G = FlexpartBackwardRunner.to_jacobian(g_raw, areas)
        assert G.shape == (3, 5)

    def test_zero_raw_gives_zero_physical(self):
        g_raw = np.zeros((2, 3))
        areas = np.ones(3) * 1e10
        G = FlexpartBackwardRunner.to_jacobian(g_raw, areas)
        assert np.all(G == 0)

    def test_linearity_in_raw(self):
        """Doubling g_raw doubles G_phys."""
        areas = np.array([1e10, 2e10])
        g1 = np.array([[1.0, 2.0]])
        g2 = g1 * 2
        G1 = FlexpartBackwardRunner.to_jacobian(g1, areas)
        G2 = FlexpartBackwardRunner.to_jacobian(g2, areas)
        np.testing.assert_allclose(G2, G1 * 2)

    def test_footprint_to_jacobian_constant_is_1e12(self):
        assert FOOTPRINT_TO_JACOBIAN == 1e12


# ─── Unit tests: dry-run file structure ──────────────────────────────────────

class TestBackwardDryRun:
    def _make_runner(self, tmp_path: Path) -> FlexpartBackwardRunner:
        base_cfg = _make_base_config(tmp_path)
        return FlexpartBackwardRunner(
            base_config=base_cfg,
            domain=_wgs84_domain(),
            config={
                "base_run_dir": str((tmp_path / "bwd_runs").resolve()),
                "n_particles": 1000,
                "dry_run": True,
            },
        )

    def test_dry_run_creates_run_directory(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.run([_instrument()], [_source()])
        assert (tmp_path / "bwd_runs" / "receptor_obs").is_dir()

    def test_dry_run_creates_pathnames(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.run([_instrument()], [_source()])
        assert (tmp_path / "bwd_runs" / "receptor_obs" / "pathnames").exists()

    def test_dry_run_creates_command(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.run([_instrument()], [_source()])
        assert (tmp_path / "bwd_runs" / "receptor_obs" / "options" / "COMMAND").exists()

    def test_dry_run_creates_outgrid(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.run([_instrument()], [_source()])
        assert (tmp_path / "bwd_runs" / "receptor_obs" / "options" / "OUTGRID").exists()

    def test_dry_run_creates_releases(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.run([_instrument()], [_source()])
        assert (tmp_path / "bwd_runs" / "receptor_obs" / "options" / "RELEASES").exists()

    def test_dry_run_g_matrix_shape(self, tmp_path):
        """Dry run still returns a correctly shaped zero G matrix."""
        runner = self._make_runner(tmp_path)
        instruments = [_instrument(), _instrument(lon=8.5)]
        sources     = [_source(), _source(lon=6.5), _source(lon=5.5)]
        result = runner.run(instruments, sources)
        assert result.g.shape == (2, 3)


# ─── Unit tests: COMMAND file content ────────────────────────────────────────

class TestBackwardCommandContent:
    def _run_and_read_command(self, tmp_path: Path) -> str:
        base_cfg = _make_base_config(tmp_path)
        runner = FlexpartBackwardRunner(
            base_config=base_cfg,
            domain=_wgs84_domain(),
            config={"base_run_dir": str((tmp_path / "runs").resolve()),
                    "dry_run": True},
        )
        runner.run([_instrument()], [_source()])
        return (tmp_path / "runs" / "receptor_obs" / "options" / "COMMAND").read_text()

    def test_command_ldirect_is_minus_one(self, tmp_path):
        text = self._run_and_read_command(tmp_path)
        assert "LDIRECT=               -1," in text

    def test_command_ioutputforeachrelease_is_one(self, tmp_path):
        text = self._run_and_read_command(tmp_path)
        assert "IOUTPUTFOREACHRELEASE= 1," in text

    def test_command_ibdate_matches_config_start(self, tmp_path):
        text = self._run_and_read_command(tmp_path)
        assert "IBDATE=         20090101," in text

    def test_command_iedate_matches_config_end(self, tmp_path):
        text = self._run_and_read_command(tmp_path)
        assert "IEDATE=         20090101," in text


# ─── Unit tests: RELEASES file content ───────────────────────────────────────

class TestBackwardReleasesContent:
    def _run_and_read_releases(self, tmp_path: Path,
                               lon: float = _INST_LON,
                               lat: float = _INST_LAT) -> str:
        base_cfg = _make_base_config(tmp_path)
        runner = FlexpartBackwardRunner(
            base_config=base_cfg,
            domain=_wgs84_domain(),
            config={"base_run_dir": str((tmp_path / "runs").resolve()),
                    "dry_run": True},
        )
        runner.run([_instrument(lon=lon, lat=lat)], [_source()])
        return (tmp_path / "runs" / "receptor_obs" / "options" / "RELEASES").read_text()

    def test_releases_has_single_block(self, tmp_path):
        text = self._run_and_read_releases(tmp_path)
        # Count "&RELEASE\n" to exclude the "&RELEASES_CTRL" header prefix
        assert text.count("&RELEASE\n") == 1

    def test_releases_idate1_equals_idate2(self, tmp_path):
        """Backward release is instantaneous at obs_time (IDATE1 == IDATE2)."""
        text = self._run_and_read_releases(tmp_path)
        dates = re.findall(r"IDATE[12]\s*=\s*(\d+)", text)
        times = re.findall(r"ITIME[12]\s*=\s*(\d+)", text)
        assert len(dates) == 2 and dates[0] == dates[1], "IDATE1 ≠ IDATE2"
        assert len(times) == 2 and times[0] == times[1], "ITIME1 ≠ ITIME2"

    def test_releases_date_matches_obs_time(self, tmp_path):
        """Backward release is at cfg.end (the observation time)."""
        text = self._run_and_read_releases(tmp_path)
        dates = re.findall(r"IDATE[12]\s*=\s*(\d+)", text)
        assert dates[0] == "20090101"
        times = re.findall(r"ITIME[12]\s*=\s*(\d+)", text)
        assert times[0] == "030000"    # 03:00:00 = cfg.end

    def test_releases_mass_is_one_kg(self, tmp_path):
        """Unit-mass release normalises the footprint output."""
        text = self._run_and_read_releases(tmp_path)
        m = re.search(r"MASS\s*=\s*([\d.E+\-]+)", text)
        assert m, "MASS not found in backward RELEASES"
        assert abs(float(m.group(1)) - 1.0) < 1e-6

    def test_releases_instrument_location_written(self, tmp_path):
        text = self._run_and_read_releases(tmp_path, lon=7.5, lat=51.5)
        assert "7.500000" in text
        assert "51.500000" in text

    def test_releases_species_number_written(self, tmp_path):
        text = self._run_and_read_releases(tmp_path)
        assert "SPECNUM_REL=          24," in text


# ─── Unit tests: registry dispatch via FlexpartTransportOperator ─────────────

class TestRegistryBackwardDispatch:
    """The registry-facing plugin routes mode='backward' to FlexpartBackwardRunner."""

    def _write_sim_config_yaml(self, tmp_path: Path) -> Path:
        """Write a minimal SimulationConfig YAML for backward dry runs."""
        yaml_path = tmp_path / "sim_config.yaml"
        yaml_path.write_text(
            "flexpart:\n"
            f"  executable: {(_BINARY).resolve()}\n"
            f"  options_dir: {_OPTIONS.resolve()}\n"
            f"  available_file: {_AVAILABLE.resolve()}\n"
            f"  meteo_dir: {_METEO.resolve()}\n"
            f"  run_dir: {(tmp_path / 'sim_run').resolve()}\n"
            "simulation:\n"
            "  start: '2009-01-01T00:00:00'\n"
            "  end: '2009-01-01T03:00:00'\n"
            "  nxshift: 0\n"
            "domain:\n"
            "  lon_min: -25.0\n"
            "  lon_max: 60.0\n"
            "  lat_min: 10.0\n"
            "  lat_max: 75.0\n"
            "  dx: 1.0\n"
            "  dy: 1.0\n"
            "  heights_m: [100.0, 500.0, 1000.0, 50000.0]\n"
        )
        return yaml_path

    def _plugin(self):
        from enforceflux.utils.plugin_registry import get_plugin
        from enforceflux.core.base import ITransportOperator

        return get_plugin(
            "enforceflux.transport_operator", "flexpart", ITransportOperator
        )()

    def test_backward_dispatch_returns_correct_shape(self, tmp_path):
        plugin = self._plugin()
        result = plugin.build_forward_operator(
            [_source(), _source(lon=6.5)],
            [_instrument()],
            _wgs84_domain(),
            {
                "mode": "backward",
                "sim_config": str(self._write_sim_config_yaml(tmp_path)),
                "base_run_dir": str((tmp_path / "bwd_runs").resolve()),
                "dry_run": True,
            },
        )
        assert result.g.shape == (1, 2)
        assert result.meta["mode"] == "backward"

    def test_backward_unit_conversion_applied(self, tmp_path):
        """Supplying source_areas_m2 converts the raw footprint to physical units."""
        plugin = self._plugin()
        # dry_run yields a zero G, so check that meta reports the physical units
        # and mixing height (the conversion path executed without raising).
        result = plugin.build_forward_operator(
            [_source()],
            [_instrument()],
            _wgs84_domain(),
            {
                "mode": "backward",
                "sim_config": str(self._write_sim_config_yaml(tmp_path)),
                "base_run_dir": str((tmp_path / "bwd_runs").resolve()),
                "dry_run": True,
                "source_areas_m2": [_grid_cell_area_m2(_SOURCE_LAT)],
                "mixing_height_m": 100.0,
            },
        )
        assert result.meta["units"] == "ng m-3 / (kg s-1)"
        assert result.meta["mixing_height_m"] == 100.0

    def test_backward_requires_sim_config(self, tmp_path):
        plugin = self._plugin()
        with pytest.raises(ValueError, match="sim_config"):
            plugin.build_forward_operator(
                [_source()], [_instrument()], _wgs84_domain(), {"mode": "backward"}
            )

    def test_unknown_mode_raises(self, tmp_path):
        plugin = self._plugin()
        with pytest.raises(ValueError, match="Unknown FLEXPART transport mode"):
            plugin.build_forward_operator(
                [_source()], [_instrument()], _wgs84_domain(), {"mode": "sideways"}
            )


# ─── Integration tests ────────────────────────────────────────────────────────

@pytest.mark.flexpart_integration
class TestBackwardIntegration:
    """Requires FLEXPART binary and bundled test met data."""

    def _make_runner(self, tmp_path: Path, **cfg_overrides) -> FlexpartBackwardRunner:
        base_cfg = _make_base_config(tmp_path)
        config = {
            "base_run_dir": str((tmp_path / "bwd").resolve()),
            "n_particles": 5000,
            "cache": False,
            "surface_only": True,
        }
        config.update(cfg_overrides)
        return FlexpartBackwardRunner(
            base_config=base_cfg,
            domain=_wgs84_domain(),
            config=config,
        )

    # ── Footprint file tests ──────────────────────────────────────────────────

    def test_backward_produces_footprint_netcdf(self, tmp_path):
        runner = self._make_runner(tmp_path)
        runner.run([_instrument()], [_source()])
        footprint_files = list(
            (tmp_path / "bwd" / "receptor_obs" / "output").glob("grid_time_*.nc")
        )
        assert footprint_files, "No grid_time_*.nc found in backward output directory"

    def test_backward_g_shape_single(self, tmp_path):
        result = self._make_runner(tmp_path).run([_instrument()], [_source()])
        assert result.g.shape == (1, 1)

    def test_backward_g_shape_multi(self, tmp_path):
        instruments = [_instrument(_INST_LON, _INST_LAT),
                       _instrument(_INST_LON + 1.0, _INST_LAT)]
        sources     = [_source(), _source(_SOURCE_LON + 2.0, _SOURCE_LAT)]
        result = self._make_runner(tmp_path).run(instruments, sources)
        assert result.g.shape == (2, 2)

    # ── Footprint value tests ─────────────────────────────────────────────────

    def test_backward_footprint_nonnegative(self, tmp_path):
        result = self._make_runner(tmp_path).run([_instrument()], [_source()])
        assert np.all(result.g >= 0), f"Negative footprint values: {result.g.min()}"

    def test_backward_footprint_nonzero_at_source(self, tmp_path):
        """
        When instrument and source are co-located, particles released from the
        receptor stay in the source grid cell long enough to produce non-zero footprint.
        """
        result = self._make_runner(tmp_path).run([_instrument()], [_source()])
        assert result.g[0, 0] > 0, (
            "Footprint at source location is zero — backward run may have failed"
        )

    def test_backward_footprint_larger_at_near_source_than_far(self, tmp_path):
        """
        A source co-located with the receptor should have a higher footprint than
        a source far away (10° displaced), reflecting physical locality.
        """
        near_src = _source(_SOURCE_LON,        _SOURCE_LAT)
        far_src  = _source(_SOURCE_LON + 10.0, _SOURCE_LAT)
        result = self._make_runner(tmp_path).run([_instrument()], [near_src, far_src])
        assert result.g[0, 0] >= result.g[0, 1], (
            f"Near footprint {result.g[0,0]:.3g} < far footprint {result.g[0,1]:.3g}"
        )

    # ── Reciprocity test ──────────────────────────────────────────────────────

    def test_reciprocity_point_source(self, tmp_path):
        """
        Source-receptor reciprocity: G_fwd ≈ G_bwd.

        Forward path:
            Run FlexpartSimulation (forward, LDIRECT=+1) with Q = 5e-3 kg/s.
            Read gridded concentration at the instrument grid cell.
            G_fwd = c [ng/m³] / Q [kg/s]

        Backward path:
            Run FlexpartBackwardRunner (LDIRECT=-1) from the instrument location.
            Read footprint at the source grid cell: fp [s].
            G_bwd = fp × 1e12 / V_cell [ng/m³/(kg/s)]

        Tolerance: 25 % (particle noise with 5 000 particles; would be <5 % at 200 k).

        NOTE: directory names are kept short to avoid FLEXPART's internal 256-char
        path buffer limit.
        """
        Q_kg_s = 5e-3   # source emission rate

        # ── Forward run  (short name: "fr") ───────────────────────────────────
        fwd_src = PointSource(
            id="fwd_src",
            lon=_SOURCE_LON, lat=_SOURCE_LAT, alt_m=5.0,
            emission_rate_kg_s=Q_kg_s,
            start=_SIM_START, end=_SIM_END,
            n_particles=5000,
        )
        fwd_cfg = _make_base_config(tmp_path, "fr", sources=[fwd_src])
        fwd_nc  = FlexpartSimulation(fwd_cfg).run()

        from netCDF4 import Dataset
        with Dataset(fwd_nc) as ds:
            c_field = np.asarray(ds.variables["ch4_mixing_ratio"][:])
            lons = np.asarray(ds.variables["longitude"][:])
            lats = np.asarray(ds.variables["latitude"][:])

        # Concentration at the instrument grid cell, last time step, surface level
        ilat = int(np.argmin(np.abs(lats - _INST_LAT)))
        ilon = int(np.argmin(np.abs(lons - _INST_LON)))
        c_surface = c_field[0, 0, -1, 0, ilat, ilon]   # ng/m³

        G_fwd = float(c_surface) / Q_kg_s               # ng/m³ / (kg/s)

        # ── Backward run  (short name: "br") ──────────────────────────────────
        bwd_runner = FlexpartBackwardRunner(
            base_config=_make_base_config(tmp_path, "br"),
            domain=_wgs84_domain(),
            config={
                "base_run_dir": str((tmp_path / "bk").resolve()),
                "n_particles": 5000,
                "cache": False,
                "surface_only": True,
            },
        )
        bwd_result = bwd_runner.run([_instrument()], [_source(flux=Q_kg_s)])

        fp_raw = float(bwd_result.g[0, 0])              # [s]

        # Convert to physical Jacobian units
        area_m2  = np.array([_grid_cell_area_m2(_SOURCE_LAT)])
        G_bwd = float(FlexpartBackwardRunner.to_jacobian(
            bwd_result.g, area_m2, mixing_height_m=100.0
        )[0, 0])

        # ── Reciprocity check ─────────────────────────────────────────────────
        if G_fwd == 0:
            pytest.skip(
                f"Forward concentration at ({_INST_LON}°, {_INST_LAT}°) is zero "
                "— particle statistics too low to check reciprocity."
            )
        if G_bwd == 0:
            pytest.skip(
                f"Backward footprint at ({_SOURCE_LON}°, {_SOURCE_LAT}°) is zero "
                "— particle statistics too low to check reciprocity."
            )

        ratio = G_fwd / G_bwd
        assert 0.75 <= ratio <= 1.33, (
            f"Reciprocity violated: G_fwd/G_bwd = {ratio:.3f}  (expected 1.0 ± 25 %)\n"
            f"  G_fwd  = {G_fwd:.4g}  ng/m³/(kg/s)  [from forward concentration]\n"
            f"  G_bwd  = {G_bwd:.4g}  ng/m³/(kg/s)  [from backward footprint × 1e12/V]\n"
            f"  fp_raw = {fp_raw:.4g}  s              [raw spec001_mr]"
        )
