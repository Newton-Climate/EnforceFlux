"""Tests for enforceflux.runs.stage_config.load_stage_config."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from enforceflux.runs import load_stage_config


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_load_stage_config_happy_path(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "d.yaml",
        {
            "stage": "dispersion",
            "run": {"name": "smoke", "outputs_root": "runs/"},
            "dispersion": {"transport": {"model": "aermod"}},
        },
    )
    cfg = load_stage_config(p, expected_stage="dispersion")
    assert cfg.stage == "dispersion"
    assert cfg.run_name == "smoke"
    assert cfg.outputs_root == (tmp_path / "runs").resolve()
    assert cfg.inputs == {}
    assert cfg.block == {"transport": {"model": "aermod"}}
    assert cfg.yaml_path == p.resolve()
    assert cfg.yaml_dir == p.resolve().parent


def test_wrong_stage_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "d.yaml", {"stage": "dispersion", "run": {"name": "s"}, "dispersion": {}})
    with pytest.raises(ValueError, match="stage='dispersion' but this driver expects stage='flux'"):
        load_stage_config(p, expected_stage="flux")


def test_missing_stage_key(tmp_path: Path) -> None:
    p = _write(tmp_path, "d.yaml", {"run": {"name": "s"}, "dispersion": {}})
    with pytest.raises(ValueError, match="missing top-level `stage:` key"):
        load_stage_config(p, expected_stage="dispersion")


def test_missing_run_name(tmp_path: Path) -> None:
    p = _write(tmp_path, "d.yaml", {"stage": "dispersion", "run": {}, "dispersion": {}})
    with pytest.raises(ValueError, match="`run.name` is required"):
        load_stage_config(p, expected_stage="dispersion")


def test_missing_stage_block(tmp_path: Path) -> None:
    p = _write(tmp_path, "d.yaml", {"stage": "dispersion", "run": {"name": "s"}})
    with pytest.raises(ValueError, match="missing `dispersion:` block"):
        load_stage_config(p, expected_stage="dispersion")


def test_inputs_resolved_relative_to_yaml_dir(tmp_path: Path) -> None:
    up = tmp_path / "up" / "dispersion"
    up.mkdir(parents=True)
    p = _write(
        tmp_path,
        "d.yaml",
        {
            "stage": "flux",
            "run": {"name": "s"},
            "inputs": {"dispersion": "up/dispersion"},
            "flux": {},
        },
    )
    cfg = load_stage_config(p, expected_stage="flux")
    assert cfg.inputs["dispersion"] == up.resolve()


def test_absolute_input_preserved(tmp_path: Path) -> None:
    up = tmp_path / "elsewhere"
    up.mkdir()
    p = _write(
        tmp_path,
        "d.yaml",
        {
            "stage": "flux",
            "run": {"name": "s"},
            "inputs": {"dispersion": str(up)},
            "flux": {},
        },
    )
    cfg = load_stage_config(p, expected_stage="flux")
    assert cfg.inputs["dispersion"] == up.resolve()


def test_outputs_root_defaults_to_none(tmp_path: Path) -> None:
    p = _write(tmp_path, "d.yaml", {"stage": "dispersion", "run": {"name": "s"}, "dispersion": {}})
    cfg = load_stage_config(p, expected_stage="dispersion")
    assert cfg.outputs_root is None
