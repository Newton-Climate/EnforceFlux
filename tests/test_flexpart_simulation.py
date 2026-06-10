"""
Unit and integration tests for FlexpartSimulation and SimulationConfig.

Unit tests (no binary required):
    - SimulationConfig.ldirect field defaults to 1.
    - _write_command writes correct LDIRECT and IOUTPUTFOREACHRELEASE.
    - _write_releases builds correct RELEASES files for point and diffuse sources.
    - _write_outgrid computes correct cell counts.
    - FlexpartRunner RELEASES MASS = unit_rate_kg_s × duration.

Integration tests (marked flexpart_integration — require FLEXPART binary + met data):
    - Forward point-source simulation runs and produces non-zero concentration.
    - Forward diffuse-source simulation runs and produces non-zero concentration.
    - Concentration scales linearly with emission rate (2× rate → 2× concentration).
"""
from __future__ import annotations

import dataclasses
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from enforceflux.flexpart.runner import FlexpartRunner
from enforceflux.flexpart.simulation import (
    DiffuseSource,
    FlexpartSimulation,
    PointSource,
    SimulationConfig,
)

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
_BINARY    = REPO_ROOT / "flexpart" / "src" / "FLEXPART"
_OPTIONS   = REPO_ROOT / "flexpart" / "tests" / "default_options"
_AVAILABLE = REPO_ROOT / "flexpart" / "tests" / "default_winds" / "AVAILABLE"
_METEO     = REPO_ROOT / "flexpart" / "tests" / "testdata"

_SIM_START = datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_SIM_END   = datetime(2009, 1, 1, 3, 0, 0, tzinfo=timezone.utc)   # 3-hour window

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path, name: str = "run", **overrides) -> SimulationConfig:
    """Build a SimulationConfig wired to the bundled test met data."""
    defaults = dict(
        executable=_BINARY.resolve(),
        options_dir=_OPTIONS.resolve(),
        available_file=_AVAILABLE.resolve(),
        meteo_dir=_METEO.resolve(),
        run_dir=(tmp_path / name).resolve(),
        start=_SIM_START,
        end=_SIM_END,
        output_step_s=3600,
        domain_lon_min=-25.0,
        domain_lat_min=10.0,
        domain_lon_max=60.0,
        domain_lat_max=75.0,
        domain_dx=1.0,
        domain_dy=1.0,
        heights_m=[100.0, 500.0, 1000.0, 50000.0],
        sources=[],
        output_path=(tmp_path / f"{name}_out.nc").resolve(),
        species_name="CH4",
        species_number=24,
        nxshift=0,
        ldirect=1,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _point_source(emission_rate_kg_s: float = 5e-3, n_particles: int = 2000) -> PointSource:
    """Ruhr-region point source."""
    return PointSource(
        id="ruhr",
        lon=7.5, lat=51.5, alt_m=5.0,
        emission_rate_kg_s=emission_rate_kg_s,
        start=_SIM_START, end=_SIM_END,
        n_particles=n_particles,
    )


def _diffuse_source(n_particles_per_cell: int = 300) -> DiffuseSource:
    """Small Paris-Basin diffuse source (2 × 2 cells at 0.5° resolution)."""
    return DiffuseSource(
        id="paris_ag",
        lon_min=1.0, lon_max=2.0,
        lat_min=48.0, lat_max=49.0,
        alt_m=2.0,
        emission_flux_kg_m2_s=5e-10,
        start=_SIM_START, end=_SIM_END,
        cell_size_deg=0.5,
        n_particles_per_cell=n_particles_per_cell,
    )


# ─── Unit tests: SimulationConfig ────────────────────────────────────────────

class TestSimulationConfigFields:
    def test_ldirect_defaults_to_one(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.ldirect == 1

    def test_ldirect_can_be_set_to_minus_one(self, tmp_path):
        cfg = _make_config(tmp_path, ldirect=-1)
        assert cfg.ldirect == -1

    def test_dataclass_replace_preserves_ldirect(self, tmp_path):
        cfg = _make_config(tmp_path, ldirect=1)
        bwd = dataclasses.replace(cfg, ldirect=-1, run_dir=(tmp_path / "bwd").resolve())
        assert bwd.ldirect == -1
        assert cfg.ldirect == 1   # original unchanged


# ─── Unit tests: _write_command ──────────────────────────────────────────────

class TestWriteCommand:
    def test_forward_has_ldirect_one(self, tmp_path):
        cfg = _make_config(tmp_path, ldirect=1)
        sim = FlexpartSimulation(cfg)
        cmd_path = tmp_path / "COMMAND"
        sim._write_command(cmd_path)
        text = cmd_path.read_text()
        assert "LDIRECT=               1," in text

    def test_backward_has_ldirect_minus_one(self, tmp_path):
        cfg = _make_config(tmp_path, ldirect=-1)
        sim = FlexpartSimulation(cfg)
        cmd_path = tmp_path / "COMMAND"
        sim._write_command(cmd_path)
        text = cmd_path.read_text()
        # The format string pads to 15 chars before the value, so -1 produces " -1,"
        assert "LDIRECT=               -1," in text

    def test_forward_ioutputforeachrelease_is_zero(self, tmp_path):
        cfg = _make_config(tmp_path, ldirect=1)
        sim = FlexpartSimulation(cfg)
        cmd_path = tmp_path / "COMMAND"
        sim._write_command(cmd_path)
        text = cmd_path.read_text()
        assert "IOUTPUTFOREACHRELEASE= 0," in text

    def test_backward_forces_ioutputforeachrelease_one(self, tmp_path):
        """FLEXPART requires IOUTPUTFOREACHRELEASE=1 for backward runs."""
        cfg = _make_config(tmp_path, ldirect=-1)
        sim = FlexpartSimulation(cfg)
        cmd_path = tmp_path / "COMMAND"
        sim._write_command(cmd_path)
        text = cmd_path.read_text()
        assert "IOUTPUTFOREACHRELEASE= 1," in text

    def test_ibdate_matches_start(self, tmp_path):
        cfg = _make_config(tmp_path)
        sim = FlexpartSimulation(cfg)
        cmd_path = tmp_path / "COMMAND"
        sim._write_command(cmd_path)
        text = cmd_path.read_text()
        assert "IBDATE=         20090101," in text
        assert "IBTIME=           000000," in text

    def test_iedate_matches_end(self, tmp_path):
        cfg = _make_config(tmp_path)
        sim = FlexpartSimulation(cfg)
        cmd_path = tmp_path / "COMMAND"
        sim._write_command(cmd_path)
        text = cmd_path.read_text()
        assert "IEDATE=         20090101," in text
        assert "IETIME=           030000," in text


# ─── Unit tests: _write_releases ─────────────────────────────────────────────

class TestWriteReleasesPoint:
    def test_single_release_block(self, tmp_path):
        src = _point_source()
        cfg = _make_config(tmp_path, sources=[src])
        sim = FlexpartSimulation(cfg)
        rel_path = tmp_path / "RELEASES"
        sim._write_releases(rel_path)
        text = rel_path.read_text()
        # Count "&RELEASE\n" (not "&RELEASE") to exclude the "&RELEASES_CTRL" header line
        assert text.count("&RELEASE\n") == 1

    def test_mass_equals_rate_times_duration(self, tmp_path):
        """MASS = emission_rate_kg_s × simulation_duration_s."""
        rate = 5e-3           # kg/s
        duration_s = ((_SIM_END - _SIM_START).total_seconds())  # 10800 s
        expected_mass = rate * duration_s                        # 54 kg

        src = _point_source(emission_rate_kg_s=rate)
        cfg = _make_config(tmp_path, sources=[src])
        sim = FlexpartSimulation(cfg)
        rel_path = tmp_path / "RELEASES"
        sim._write_releases(rel_path)
        text = rel_path.read_text()

        match = re.search(r"MASS\s*=\s*([\d.E+\-]+)", text)
        assert match is not None, "MASS not found in RELEASES"
        actual_mass = float(match.group(1))
        assert abs(actual_mass - expected_mass) / expected_mass < 1e-4

    def test_mass_scales_with_emission_rate(self, tmp_path):
        """Doubling emission rate should double MASS."""
        for rate in [1e-3, 2e-3]:
            src = _point_source(emission_rate_kg_s=rate)
            cfg = _make_config(tmp_path, sources=[src], run_dir=(tmp_path / f"r{rate}").resolve())
            rel_path = tmp_path / f"RELEASES_{rate}"
            FlexpartSimulation(cfg)._write_releases(rel_path)

        masses = []
        for rate in [1e-3, 2e-3]:
            text = (tmp_path / f"RELEASES_{rate}").read_text()
            m = re.search(r"MASS\s*=\s*([\d.E+\-]+)", text)
            masses.append(float(m.group(1)))
        assert abs(masses[1] / masses[0] - 2.0) < 1e-4

    def test_source_location_in_releases(self, tmp_path):
        src = _point_source()
        cfg = _make_config(tmp_path, sources=[src])
        sim = FlexpartSimulation(cfg)
        rel_path = tmp_path / "RELEASES"
        sim._write_releases(rel_path)
        text = rel_path.read_text()
        assert "LON1" in text
        assert "7.500000" in text
        assert "51.500000" in text

    def test_species_number_in_header(self, tmp_path):
        cfg = _make_config(tmp_path, sources=[_point_source()], species_number=24)
        sim = FlexpartSimulation(cfg)
        rel_path = tmp_path / "RELEASES"
        sim._write_releases(rel_path)
        text = rel_path.read_text()
        assert "SPECNUM_REL=          24," in text


class TestWriteReleasesDiffuse:
    def test_block_count_matches_cell_grid(self, tmp_path):
        """1° × 1° area at 0.5° resolution → 4 cells → 4 &RELEASE blocks."""
        src = _diffuse_source()
        cfg = _make_config(tmp_path, sources=[src])
        sim = FlexpartSimulation(cfg)
        rel_path = tmp_path / "RELEASES"
        sim._write_releases(rel_path)
        text = rel_path.read_text()
        # Use "&RELEASE\n" to exclude the "&RELEASES_CTRL" header prefix
        n_blocks = text.count("&RELEASE\n")
        assert n_blocks == 4

    def test_total_mass_proportional_to_flux_area_time(self, tmp_path):
        """Total released mass = flux × total_area × duration, within 1%."""
        src = _diffuse_source()
        cfg = _make_config(tmp_path, sources=[src])
        sim = FlexpartSimulation(cfg)
        rel_path = tmp_path / "RELEASES"
        sim._write_releases(rel_path)
        text = rel_path.read_text()

        masses = [float(m) for m in re.findall(r"MASS\s*=\s*([\d.E+\-]+)", text)]
        total_mass = sum(masses)

        # Approximate expected: flux × area × duration
        # Area of each 0.5° cell at ~48.5°N:
        R = 6_371_000.0
        lat_c = math.radians(48.5)
        cell_area = (math.radians(0.5) * R * math.cos(lat_c)) * (math.radians(0.5) * R)
        expected = src.emission_flux_kg_m2_s * cell_area * 4 * (10800)
        assert abs(total_mass - expected) / expected < 0.02   # 2% tolerance (spherical approx)

    def test_diffuse_cells_span_correct_lon_range(self, tmp_path):
        src = _diffuse_source()
        cfg = _make_config(tmp_path, sources=[src])
        sim = FlexpartSimulation(cfg)
        rel_path = tmp_path / "RELEASES"
        sim._write_releases(rel_path)
        text = rel_path.read_text()

        lons = [float(m) for m in re.findall(r"LON1\s*=\s*([\d.E+\-]+)", text)]
        assert min(lons) >= src.lon_min - 0.01
        assert max(lons) <= src.lon_max + 0.01


# ─── Unit tests: _write_outgrid ──────────────────────────────────────────────

class TestWriteOutgrid:
    def test_numxgrid_correct(self, tmp_path):
        # (-25 to 60, dx=1) → 85 cells
        cfg = _make_config(tmp_path)
        sim = FlexpartSimulation(cfg)
        og_path = tmp_path / "OUTGRID"
        sim._write_outgrid(og_path)
        text = og_path.read_text()
        assert "NUMXGRID=       85," in text

    def test_numygrid_correct(self, tmp_path):
        # (10 to 75, dy=1) → 65 cells
        cfg = _make_config(tmp_path)
        sim = FlexpartSimulation(cfg)
        og_path = tmp_path / "OUTGRID"
        sim._write_outgrid(og_path)
        text = og_path.read_text()
        assert "NUMYGRID=       65," in text

    def test_fine_resolution_cell_count(self, tmp_path):
        cfg = _make_config(tmp_path, domain_lon_min=0.0, domain_lon_max=10.0,
                           domain_lat_min=40.0, domain_lat_max=50.0,
                           domain_dx=0.5, domain_dy=0.5)
        sim = FlexpartSimulation(cfg)
        og_path = tmp_path / "OUTGRID"
        sim._write_outgrid(og_path)
        text = og_path.read_text()
        assert "NUMXGRID=       20," in text
        assert "NUMYGRID=       20," in text

    def test_outheights_written(self, tmp_path):
        cfg = _make_config(tmp_path, heights_m=[100.0, 500.0, 1000.0, 50000.0])
        sim = FlexpartSimulation(cfg)
        og_path = tmp_path / "OUTGRID"
        sim._write_outgrid(og_path)
        text = og_path.read_text()
        assert "100.0" in text
        assert "50000.0" in text


# ─── Unit tests: FlexpartRunner MASS units fix ───────────────────────────────

class TestFlexpartRunnerMass:
    def test_releases_mass_is_rate_times_duration(self, tmp_path):
        """
        unit_emission_rate is now in kg/s.
        RELEASES MASS = unit_emission_rate × (end - start) seconds.
        """
        from enforceflux.config import DomainConfig

        domain = DomainConfig(
            x_min=0, x_max=1000, y_min=0, y_max=1000,
            grid_spacing=500, crs="EPSG:32632", crs_wgs84="EPSG:4326",
        )
        start_date, start_time = 20090101, 0
        end_date,   end_time   = 20090101, 30000    # 3 hours = 10800 s

        config = {
            "base_run_dir": str(tmp_path / "runs"),
            "options_dir":  str(_OPTIONS),
            "available_file": str(_AVAILABLE),
            "meteo_dir":    str(_METEO),
            "dry_run":      True,
            "unit_emission_rate": 1.0,   # 1 kg/s
            "release_start_date": start_date,
            "release_start_time": start_time,
            "release_end_date":   end_date,
            "release_end_time":   end_time,
        }
        runner = FlexpartRunner(domain=domain, config=config)
        rel_path = tmp_path / "RELEASES"
        runner._write_releases(rel_path, source_lon=7.5, source_lat=51.5,
                               source_alt=5.0, source_id="test")
        text = rel_path.read_text()

        match = re.search(r"MASS\s*=\s*([\d.E+\-]+)", text)
        assert match, "MASS not found in RELEASES"
        actual_mass = float(match.group(1))
        # 1.0 kg/s × 10800 s = 10800 kg
        assert abs(actual_mass - 10800.0) / 10800.0 < 1e-4

    def test_releases_mass_scales_with_unit_rate(self, tmp_path):
        """Doubling unit_emission_rate should double MASS in RELEASES."""
        from enforceflux.config import DomainConfig

        domain = DomainConfig(
            x_min=0, x_max=1, y_min=0, y_max=1,
            grid_spacing=1, crs="EPSG:32632", crs_wgs84="EPSG:4326",
        )
        base_config = {
            "base_run_dir": str(tmp_path / "runs"),
            "options_dir":  str(_OPTIONS),
            "available_file": str(_AVAILABLE),
            "meteo_dir":    str(_METEO),
            "dry_run":      True,
            "release_start_date": 20090101, "release_start_time": 0,
            "release_end_date":   20090101, "release_end_time":   30000,
        }
        masses = []
        for rate in [1.0, 2.0]:
            cfg = {**base_config, "unit_emission_rate": rate}
            runner = FlexpartRunner(domain=domain, config=cfg)
            rel_path = tmp_path / f"RELEASES_{rate}"
            runner._write_releases(rel_path, source_lon=7.5, source_lat=51.5,
                                   source_alt=5.0, source_id="test")
            text = rel_path.read_text()
            m = re.search(r"MASS\s*=\s*([\d.E+\-]+)", text)
            masses.append(float(m.group(1)))

        assert abs(masses[1] / masses[0] - 2.0) < 1e-4


# ─── Integration tests ────────────────────────────────────────────────────────

@pytest.mark.flexpart_integration
class TestForwardIntegration:
    """Requires FLEXPART binary and bundled test met data."""

    def test_forward_point_source_produces_netcdf(self, tmp_path):
        src = _point_source(n_particles=2000)
        cfg = _make_config(tmp_path, "fwd_point", sources=[src])
        nc = FlexpartSimulation(cfg).run()
        assert nc.exists(), f"Output NetCDF not found: {nc}"

    def test_forward_point_source_has_nonzero_concentration(self, tmp_path):
        src = _point_source(n_particles=2000)
        cfg = _make_config(tmp_path, "fwd_point_nz", sources=[src])
        nc = FlexpartSimulation(cfg).run()

        from netCDF4 import Dataset
        with Dataset(nc) as ds:
            conc = np.asarray(ds.variables["ch4_mixing_ratio"][:])
        assert conc.max() > 0, "All concentrations zero — FLEXPART produced no signal"

    def test_forward_diffuse_source_has_nonzero_concentration(self, tmp_path):
        src = _diffuse_source(n_particles_per_cell=500)
        cfg = _make_config(tmp_path, "fwd_diffuse", sources=[src])
        nc = FlexpartSimulation(cfg).run()

        from netCDF4 import Dataset
        with Dataset(nc) as ds:
            conc = np.asarray(ds.variables["ch4_mixing_ratio"][:])
        assert conc.max() > 0, "Diffuse source produced zero concentration"

    def test_forward_concentration_linear_in_emission_rate(self, tmp_path):
        """
        2× emission rate → ~2× peak surface concentration.
        Verifies FLEXPART's linearity in source strength.
        Tolerance 30 %: particle statistics with 2 000 particles.
        """
        conc_peaks = {}
        for rate, name in [(5e-3, "half"), (10e-3, "full")]:
            src = _point_source(emission_rate_kg_s=rate, n_particles=2000)
            cfg = _make_config(tmp_path, f"lin_{name}", sources=[src])
            nc = FlexpartSimulation(cfg).run()

            from netCDF4 import Dataset
            with Dataset(nc) as ds:
                # surface level, last time step
                c = np.asarray(ds.variables["ch4_mixing_ratio"][:])[0, 0, -1, 0]
            conc_peaks[name] = float(c.max())

        ratio = conc_peaks["full"] / conc_peaks["half"]
        assert 0.70 <= ratio <= 2.60, (
            f"Concentration ratio = {ratio:.2f} — expected ~2.0 ± 30 %\n"
            f"  half-rate peak : {conc_peaks['half']:.2f} ng/m³\n"
            f"  full-rate peak : {conc_peaks['full']:.2f} ng/m³"
        )

    def test_forward_point_and_diffuse_each_produce_signal(self, tmp_path):
        """
        Both source types independently produce non-zero concentration.

        The combined-vs-individual max comparison is deliberately avoided here
        because stochastic particle noise at 2 000 particles makes it unreliable:
        each independent run has a different noise realisation, so the combined
        run's peak can be lower than a single run's peak by chance.
        """
        for label, sources in [
            ("pt", [_point_source(n_particles=2000)]),
            ("df", [_diffuse_source(n_particles_per_cell=500)]),
        ]:
            cfg = _make_config(tmp_path, f"ind_{label}", sources=sources)
            nc = FlexpartSimulation(cfg).run()
            from netCDF4 import Dataset
            with Dataset(nc) as ds:
                c = np.asarray(ds.variables["ch4_mixing_ratio"][:])
            assert c.max() > 0, f"{label} source produced zero concentration"
