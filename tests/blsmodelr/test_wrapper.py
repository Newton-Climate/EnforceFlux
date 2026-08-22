"""Subprocess-bridge tests for the bLSmodelR wrapper.

These tests do not require a real R installation: they inject a fake
``Rscript`` that reads the JSON contract and writes the CSV contract
directly, exercising every Python-side code path.

A second test runs the real ``Rscript r/run_bls.R --dry-run`` shim end to
end when Rscript is on PATH — this validates the R shim's JSON parser and
CSV writer, without touching bLSmodelR itself.
"""
from __future__ import annotations

import csv
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

from enforceflux.blsmodelr import (
    BlsInterval,
    BlsModelParams,
    BlsRequest,
    BlsSensor,
    BlsSource,
    BlsWrapper,
)


def _sample_request() -> BlsRequest:
    return BlsRequest(
        sensors=[BlsSensor("S1", 0.0, 0.0, 2.0),
                 BlsSensor("S2", 50.0, 0.0, 2.0)],
        sources=[BlsSource("cell_00", [(10.0, -5.0), (20.0, -5.0), (20.0, 5.0), (10.0, 5.0)]),
                 BlsSource("cell_01", [(30.0, -5.0), (40.0, -5.0), (40.0, 5.0), (30.0, 5.0)])],
        intervals=[BlsInterval(id="t0", u_star=0.35, L=-50.0, z0=0.05,
                               wind_dir_deg=270.0, wind_speed=3.0, z_ref=4.0)],
        model=BlsModelParams(n_particles=100, max_traj_s=60, seed=7),
    )


# ── plumbing tests with a fake Rscript ────────────────────────────────────


def _write_fake_rscript(tmp_path: Path) -> Path:
    """A shell 'Rscript' stand-in that mimics the shim's JSON->CSV contract."""
    script = tmp_path / "fake_rscript"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, csv\n"
        "args = sys.argv[1:]\n"
        "cfg = args[args.index('--config') + 1]\n"
        "out = args[args.index('--out') + 1]\n"
        "req = json.load(open(cfg))\n"
        "with open(out, 'w', newline='') as fh:\n"
        "    w = csv.writer(fh)\n"
        "    w.writerow(['sensor','source','interval','cxe','cxe_se','n_particles_used'])\n"
        "    for s in req['sensors']:\n"
        "        for src in req['sources']:\n"
        "            for iv in req['intervals']:\n"
        "                w.writerow([s['name'], src['name'], iv['id'],\n"
        "                            1.23e-3, 4.5e-5, req['model']['n_particles']])\n"
        "print('fake wrote', out)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_wrapper_roundtrips_json_and_csv(tmp_path):
    fake = _write_fake_rscript(tmp_path)
    wrapper = BlsWrapper({"rscript": str(fake), "workdir": tmp_path / "work"})
    result = wrapper.run(_sample_request())

    assert len(result) == 2 * 2 * 1  # sensors × sources × intervals
    assert set(result.sensor.tolist()) == {"S1", "S2"}
    assert set(result.source.tolist()) == {"cell_00", "cell_01"}
    assert np.all(result.cxe == 1.23e-3)
    assert np.all(result.n_particles_used == 100)

    # Request JSON must have been serialized to workdir with the contract keys.
    req_path = tmp_path / "work" / "request.json"
    assert req_path.exists()
    import json
    payload = json.loads(req_path.read_text())
    assert set(payload) == {"sensors", "sources", "intervals", "model"}
    assert payload["intervals"][0]["u_star"] == pytest.approx(0.35)


def test_missing_rscript_raises_actionable_error(tmp_path):
    wrapper = BlsWrapper({"rscript": str(tmp_path / "definitely_not_here")})
    with pytest.raises(FileNotFoundError, match="Rscript"):
        wrapper.run(_sample_request())


def test_subprocess_failure_surfaces_stderr(tmp_path):
    failing = tmp_path / "failing_rscript"
    failing.write_text("#!/usr/bin/env sh\necho 'boom' >&2\nexit 3\n")
    failing.chmod(failing.stat().st_mode | stat.S_IEXEC)
    wrapper = BlsWrapper({"rscript": str(failing)})
    with pytest.raises(RuntimeError, match="exit 3"):
        wrapper.run(_sample_request())


def test_csv_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad_rscript"
    bad.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, csv\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "with open(out, 'w', newline='') as fh:\n"
        "    csv.writer(fh).writerow(['sensor','source','interval'])\n"
    )
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
    wrapper = BlsWrapper({"rscript": str(bad), "workdir": tmp_path / "work"})
    with pytest.raises(RuntimeError, match="missing columns"):
        wrapper.run(_sample_request())


# ── live-shim smoke test (skipped if Rscript is not installed) ────────────


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")
def test_real_shim_dry_run(tmp_path):
    wrapper = BlsWrapper({"dry_run": True, "workdir": tmp_path / "work"})
    result = wrapper.run(_sample_request())
    assert len(result) == 2 * 2 * 1
    # Stub CxE is positive and finite.
    assert np.all(np.isfinite(result.cxe))
    assert np.all(result.cxe > 0)
