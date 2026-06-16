"""Unit tests for the FLEXPART transport *simulation* registry plugin.

These exercise the simulation plugin via dry runs (no FLEXPART binary): the
plugin loads a SimulationConfig YAML, optionally overrides the integration
direction, and prepares FLEXPART inputs without executing.
"""
from pathlib import Path

import pytest

from enforceflux.core.base import ITransportSimulation, TransportSimulationResult
from enforceflux.utils.plugin_registry import get_plugin

REPO_ROOT = Path(__file__).resolve().parents[1]
_BINARY = REPO_ROOT / "flexpart" / "src" / "FLEXPART"
_OPTIONS = REPO_ROOT / "flexpart" / "tests" / "default_options"
_AVAILABLE = REPO_ROOT / "flexpart" / "tests" / "default_winds" / "AVAILABLE"
_METEO = REPO_ROOT / "flexpart" / "tests" / "testdata"


def _write_sim_config_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "sim_config.yaml"
    yaml_path.write_text(
        "flexpart:\n"
        f"  executable: {_BINARY.resolve()}\n"
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
        "sources:\n"
        "  - id: src\n"
        "    type: point\n"
        "    lon: 7.5\n"
        "    lat: 51.5\n"
        "    emission_rate_kg_s: 5.0e-3\n"
        f"output:\n"
        f"  path: {(tmp_path / 'out.nc').resolve()}\n"
    )
    return yaml_path


def _plugin() -> ITransportSimulation:
    return get_plugin(
        "enforceflux.transport_simulation", "flexpart", ITransportSimulation
    )()


def test_registry_resolves_simulation_plugin():
    plugin = _plugin()
    assert isinstance(plugin, ITransportSimulation)


def test_dry_run_prepares_without_output(tmp_path):
    plugin = _plugin()
    result = plugin.simulate(
        [], None, {"sim_config": str(_write_sim_config_yaml(tmp_path)), "dry_run": True}
    )
    assert isinstance(result, TransportSimulationResult)
    assert result.output_path is None
    assert result.meta["prepared"] is True
    assert result.meta["backend"] == "flexpart"
    # prepare() writes the standard FLEXPART input files
    run_dir = Path(result.meta["run_dir"])
    assert (run_dir / "options" / "COMMAND").exists()
    assert (run_dir / "options" / "RELEASES").exists()
    assert (run_dir / "pathnames").exists()


def test_forward_is_default_direction(tmp_path):
    plugin = _plugin()
    result = plugin.simulate(
        [], None, {"sim_config": str(_write_sim_config_yaml(tmp_path)), "dry_run": True}
    )
    assert result.meta["ldirect"] == 1
    command = (Path(result.meta["run_dir"]) / "options" / "COMMAND").read_text()
    assert "LDIRECT=               1," in command


def test_ldirect_override_to_backward(tmp_path):
    plugin = _plugin()
    result = plugin.simulate(
        [],
        None,
        {
            "sim_config": str(_write_sim_config_yaml(tmp_path)),
            "dry_run": True,
            "ldirect": -1,
        },
    )
    assert result.meta["ldirect"] == -1
    command = (Path(result.meta["run_dir"]) / "options" / "COMMAND").read_text()
    assert "LDIRECT=               -1," in command


def test_output_path_and_run_dir_overrides(tmp_path):
    plugin = _plugin()
    custom_run = tmp_path / "custom_run"
    custom_out = tmp_path / "custom_out.nc"
    result = plugin.simulate(
        [],
        None,
        {
            "sim_config": str(_write_sim_config_yaml(tmp_path)),
            "dry_run": True,
            "run_dir": str(custom_run),
            "output_path": str(custom_out),
        },
    )
    # Override takes effect: prepared inputs land in the overridden run dir.
    assert Path(result.meta["run_dir"]) == custom_run.resolve()
    assert (custom_run / "options" / "COMMAND").exists()


def test_missing_sim_config_raises(tmp_path):
    plugin = _plugin()
    with pytest.raises(ValueError, match="sim_config"):
        plugin.simulate([], None, {"dry_run": True})
