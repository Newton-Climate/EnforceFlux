"""Tests for the on-disk H cache used by the FLEXPART sweep branch.

# --- source-heterogeneity OSSE (M5) ---
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from enforceflux.osse.h_cache import (
    CacheValidationError,
    DiskHCache,
    transport_cache_key,
)


def _base_cfg() -> dict:
    return {
        "dispersion": {
            "transport": {"model": "flexpart", "mode": "operator"},
            "met": {"records": [{"time": "2009-01-01T00:00", "wind_speed_m_s": 3.0}]},
            "domain": {"origin_lon": -121.75, "origin_lat": 39.15, "spacing_m": 100.0},
            "receptors": [
                {"id": "tower_n", "x_m": 0.0, "y_m": 650.0, "alt_m": 3.0},
                {"id": "tower_e", "x_m": 600.0, "y_m": 0.0, "alt_m": 3.0},
            ],
            "sources": {
                "generator": "lognormal_field",
                "config": {
                    "Q_true_kg_s": 2.78e-2,
                    "grid": {"nx": 8, "ny": 8, "dx_m": 100.0},
                    "alt_m": 2.0,
                    "covariance": {"model": "exponential", "L_m": 500.0},
                    "cv": 1.0,
                    "seed": 42,
                    "basis": {"coarsen": 2},
                },
            },
        }
    }


def _write_rundir(root: Path, n_receptors: int = 2, n_sources: int = 16) -> Path:
    rd = root / "dispersion"
    rd.mkdir()
    h = np.ones((n_receptors, n_sources), dtype=np.float64)
    np.savez(
        rd / "jacobian.npz",
        H=h,
        receptor_ids=np.array(["tower_n", "tower_e"]),
    )
    # A well-formed manifest with contiguous time coverage.
    manifest = {
        "time_start": 0.0,
        "time_end": 3600.0 * 5,
        "time_step_seconds": 3600.0,
        "n_time_steps": 6,
    }
    (rd / "manifest.json").write_text(json.dumps(manifest))
    return rd


def test_hit_and_miss(tmp_path: Path) -> None:
    cache = DiskHCache(root=tmp_path / "cache")
    cfg = _base_cfg()
    key = transport_cache_key(cfg)

    assert cache.get(key) is None, "empty cache must miss"

    rd = _write_rundir(tmp_path)
    entry = cache.put(key, rd)
    assert entry.exists()
    assert (entry / "jacobian.npz").exists()
    assert (entry / "manifest.json").exists()

    hit = cache.get(key)
    assert hit is not None
    assert hit == entry

    # A different key still misses.
    other_cfg = copy.deepcopy(cfg)
    other_cfg["dispersion"]["receptors"].append(
        {"id": "tower_s", "x_m": 0.0, "y_m": -650.0, "alt_m": 3.0}
    )
    assert cache.get(transport_cache_key(other_cfg)) is None


def test_key_ignores_source_fields(tmp_path: Path) -> None:
    base = _base_cfg()
    k0 = transport_cache_key(base)

    # Change L_m — must not invalidate.
    cfg_l = copy.deepcopy(base)
    cfg_l["dispersion"]["sources"]["config"]["covariance"]["L_m"] = 2000.0
    assert transport_cache_key(cfg_l) == k0

    # Change cv — must not invalidate.
    cfg_cv = copy.deepcopy(base)
    cfg_cv["dispersion"]["sources"]["config"]["cv"] = 0.25
    assert transport_cache_key(cfg_cv) == k0

    # Change seed — must not invalidate.
    cfg_seed = copy.deepcopy(base)
    cfg_seed["dispersion"]["sources"]["config"]["seed"] = 9999
    assert transport_cache_key(cfg_seed) == k0

    # Change basis coarsen — must not invalidate.
    cfg_basis = copy.deepcopy(base)
    cfg_basis["dispersion"]["sources"]["config"]["basis"] = {"coarsen": 4}
    assert transport_cache_key(cfg_basis) == k0

    # Change receptors — MUST invalidate.
    cfg_recv = copy.deepcopy(base)
    cfg_recv["dispersion"]["receptors"][0]["x_m"] = 123.0
    assert transport_cache_key(cfg_recv) != k0

    # Adding a receptor — MUST invalidate.
    cfg_add = copy.deepcopy(base)
    cfg_add["dispersion"]["receptors"].append(
        {"id": "tower_s", "x_m": 0.0, "y_m": -650.0, "alt_m": 3.0}
    )
    assert transport_cache_key(cfg_add) != k0

    # Changing the transport model — MUST invalidate.
    cfg_t = copy.deepcopy(base)
    cfg_t["dispersion"]["transport"]["model"] = "aermod"
    assert transport_cache_key(cfg_t) != k0


def test_put_refuses_time_gaps(tmp_path: Path) -> None:
    cache = DiskHCache(root=tmp_path / "cache")
    rd = _write_rundir(tmp_path)
    # Corrupt the manifest to advertise a gap: 6 steps claimed but end-time
    # implies 11.
    manifest_path = rd / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["time_end"] = 3600.0 * 10
    manifest_path.write_text(json.dumps(manifest))

    key = transport_cache_key(_base_cfg())
    with pytest.raises(CacheValidationError):
        cache.put(key, rd)
    # Nothing was written on failure.
    assert cache.get(key) is None
