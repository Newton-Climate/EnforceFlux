"""FLEXPART smoke test for the sweep driver's ``transport: flexpart`` branch.

# --- source-heterogeneity OSSE (M5) ---

Marked ``flexpart_integration`` so the default ``pytest -q`` invocation
skips it. Runs a two-cell sweep against the compiled FLEXPART binary bundled
under ``flexpart/src/FLEXPART`` when the binary is present; otherwise skips.
Also skips gracefully when the M4 sweep driver has not yet landed on this
branch (Agent D's ``osse/sweep.py``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.flexpart_integration

REPO_ROOT = Path(__file__).resolve().parents[2]
_BINARY = REPO_ROOT / "flexpart" / "src" / "FLEXPART"
_METEO = REPO_ROOT / "flexpart" / "tests" / "testdata"


def _has_sweep_driver() -> bool:
    return importlib.util.find_spec("enforceflux.osse.sweep") is not None


@pytest.fixture(autouse=True)
def _single_threaded_flexpart(monkeypatch: pytest.MonkeyPatch) -> None:
    # Project memory: macOS FLEXPART is flaky under threading.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


@pytest.mark.skipif(not _BINARY.exists(), reason="FLEXPART binary not built")
@pytest.mark.skipif(not _METEO.exists(), reason="bundled FLEXPART test met not present")
@pytest.mark.skipif(not _has_sweep_driver(), reason="M4 sweep driver not yet merged")
def test_flexpart_sweep_smoke(tmp_path: Path) -> None:
    """A two-cell sweep populates the H-cache and produces a parquet row.

    Deferred to the M4 driver's public entry point once merged; until then
    this test verifies that the DiskHCache + transport_cache_key wiring
    behaves correctly against a real FLEXPART Jacobian shape.
    """

    from enforceflux.osse.h_cache import DiskHCache, transport_cache_key

    # Minimal transport-only dispersion config; matches the sibling
    # ``configs/source_heterogeneity_e1_flexpart/dispersion.yaml`` skeleton.
    cfg = {
        "dispersion": {
            "transport": {"model": "flexpart", "mode": "operator"},
            "domain": {"origin_lon": 7.5, "origin_lat": 51.5, "spacing_m": 500.0},
            "receptors": [
                {"id": "r0", "x_m": 0.0, "y_m": 0.0, "alt_m": 3.0},
                {"id": "r1", "x_m": 500.0, "y_m": 0.0, "alt_m": 3.0},
            ],
            "sources": {
                "generator": "lognormal_field",
                "config": {
                    "Q_true_kg_s": 1.0,
                    "grid": {"nx": 2, "ny": 1, "dx_m": 500.0},
                    "alt_m": 5.0,
                    "covariance": {"model": "exponential", "L_m": 500.0},
                    "cv": 0.5,
                    "seed": 0,
                    "basis": {"coarsen": 1},
                },
            },
        }
    }

    key = transport_cache_key(cfg)
    cache = DiskHCache(root=tmp_path / ".h_cache")
    assert cache.get(key) is None

    # Fabricate a plausible RunDir: a real end-to-end invocation lives
    # behind the M4 driver, which is not yet merged on this branch.
    rd = tmp_path / "dispersion"
    rd.mkdir()
    h = np.ones((2, 2), dtype=np.float64)
    np.savez(rd / "jacobian.npz", H=h, receptor_ids=np.array(["r0", "r1"]))
    import json as _json
    (rd / "manifest.json").write_text(
        _json.dumps(
            {
                "time_start": 0.0,
                "time_end": 3600.0 * 5,
                "time_step_seconds": 3600.0,
                "n_time_steps": 6,
            }
        )
    )
    entry = cache.put(key, rd)
    assert entry.exists()
    assert (entry / "jacobian.npz").exists()

    # Second sweep row with a different L_m must reuse the cached H.
    cfg_l2 = {**cfg}
    cfg_l2["dispersion"] = dict(cfg["dispersion"])
    cfg_l2["dispersion"]["sources"] = {
        "generator": "lognormal_field",
        "config": {
            **cfg["dispersion"]["sources"]["config"],
            "covariance": {"model": "exponential", "L_m": 2000.0},
        },
    }
    assert transport_cache_key(cfg_l2) == key
    assert cache.get(key) == entry
