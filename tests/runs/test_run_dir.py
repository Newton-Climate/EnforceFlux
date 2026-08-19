"""Contract tests for enforceflux.runs (manifest round-trip + role lookup)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from enforceflux.runs import (
    OUTPUT_CONTRACT_VERSION,
    open_run_dir,
    read_upstream,
)


def _cfg() -> dict:
    return {
        "stage": "dispersion",
        "run": {"name": "smoke", "outputs_root": "runs/"},
        "dispersion": {"model": "aermod", "sources": [{"name": "src1"}]},
    }


def test_finalize_writes_manifest_and_config_snapshot(tmp_path: Path) -> None:
    rd = open_run_dir(stage="dispersion", run_name="smoke", outputs_root=tmp_path)
    rd.snapshot_config(_cfg())

    conc = rd.path("concentration.nc")
    conc.write_bytes(b"NC\x00\x00fake")
    rd.record_output("concentration.nc", role="concentration_field")

    hcol = rd.path("unit_response.nc")
    hcol.write_bytes(b"NC\x00\x00fake-H")
    rd.record_output("unit_response.nc", role="H_columns")

    contract = rd.finalize()

    assert contract["output_dir"].endswith("smoke/dispersion")
    assert Path(contract["manifest"]).is_file()
    assert set(contract["files"]) == {"concentration_field", "H_columns"}

    manifest = json.loads(Path(contract["manifest"]).read_text())
    assert manifest["contract_version"] == OUTPUT_CONTRACT_VERSION
    assert manifest["stage"] == "dispersion"
    assert manifest["run_name"] == "smoke"
    assert manifest["status"] == "complete"
    assert {o["role"] for o in manifest["outputs"]} == {"concentration_field", "H_columns"}
    for o in manifest["outputs"]:
        assert len(o["sha256"]) == 64

    snap = Path(contract["output_dir"]) / "config.snapshot.yaml"
    assert snap.is_file()
    assert "dispersion" in snap.read_text()


def test_missing_declared_output_aborts_finalize(tmp_path: Path) -> None:
    rd = open_run_dir(stage="instrument", run_name="smoke", outputs_root=tmp_path)
    rd.record_output("obs.parquet", role="obs")
    with pytest.raises(RuntimeError, match="declared outputs are missing"):
        rd.finalize()


def test_read_upstream_by_role_and_path(tmp_path: Path) -> None:
    rd = open_run_dir(stage="dispersion", run_name="e2e", outputs_root=tmp_path)
    p = rd.path("concentration.nc")
    p.write_bytes(b"data")
    rd.record_output("concentration.nc", role="concentration_field")
    rd.finalize()

    up = read_upstream(rd.root)
    assert up.stage == "dispersion"
    assert up.run_name == "e2e"
    assert up.file("concentration_field").is_file()
    assert up.has("concentration_field")
    assert not up.has("obs")

    up2 = read_upstream(rd.root / "manifest.json")
    assert up2.file("concentration_field") == up.file("concentration_field")


def test_read_upstream_role_missing_raises(tmp_path: Path) -> None:
    rd = open_run_dir(stage="dispersion", run_name="e2e", outputs_root=tmp_path)
    p = rd.path("concentration.nc")
    p.write_bytes(b"data")
    rd.record_output("concentration.nc", role="concentration_field")
    rd.finalize()

    up = read_upstream(rd.root)
    with pytest.raises(KeyError, match="no role='bogus'"):
        up.file("bogus")


def test_contract_version_mismatch_rejected(tmp_path: Path) -> None:
    rd = open_run_dir(stage="dispersion", run_name="v", outputs_root=tmp_path)
    p = rd.path("concentration.nc")
    p.write_bytes(b"data")
    rd.record_output("concentration.nc", role="concentration_field")
    rd.finalize()

    manifest_path = rd.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract_version"] = "999.0"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="contract_version"):
        read_upstream(rd.root)


def test_inputs_carried_into_manifest(tmp_path: Path) -> None:
    up_root = tmp_path / "up"
    up = open_run_dir(stage="dispersion", run_name="ref", outputs_root=up_root)
    p = up.path("concentration.nc")
    p.write_bytes(b"data")
    up.record_output("concentration.nc", role="concentration_field")
    up.finalize()

    rd = open_run_dir(
        stage="instrument",
        run_name="ref",
        outputs_root=tmp_path / "down",
        inputs={"dispersion": str(up.root)},
    )
    p = rd.path("obs.parquet")
    p.write_bytes(b"parquetish")
    rd.record_output("obs.parquet", role="obs")
    rd.finalize()

    manifest = json.loads((rd.root / "manifest.json").read_text())
    assert manifest["inputs"] == {"dispersion": str(up.root)}
