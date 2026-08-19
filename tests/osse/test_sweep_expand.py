"""Unit tests for the M4 grid expansion + H-cache grouping."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from enforceflux.osse.sweep import expand_grid, group_by_h_cache


def test_grid_product_matches_manual_count():
    grid = {
        "dispersion.sources.config.covariance.L_m": [100, 250, 500],
        "dispersion.sources.config.cv": [0.5, 1.0],
        "dispersion.sources.config.seed": {"range": [0, 4]},
        "flux.inversion.prior_covariance.L_B_m": [800.0],
    }
    cells = expand_grid(grid)
    # 3 * 2 * 4 * 1 = 24
    assert len(cells) == 24
    for cell in cells:
        assert set(cell.keys()) == set(grid.keys())


def test_grid_nested_space_filling_expands():
    grid = {
        "instrument.network": {
            "kind": "nested_space_filling",
            "ns": [2, 4],
            "seeds": [0, 1, 2],
        },
    }
    cells = expand_grid(grid)
    assert len(cells) == 6
    seen = {(c["instrument.network"]["n"], c["instrument.network"]["seed"])
            for c in cells}
    assert seen == {(2, 0), (2, 1), (2, 2), (4, 0), (4, 1), (4, 2)}


def test_group_by_h_cache_collapses_reuse_keys():
    grid = {
        "dispersion.sources.config.covariance.L_m": [100, 500],
        "dispersion.sources.config.cv": [0.25, 1.0],
        "dispersion.sources.config.seed": {"range": [0, 3]},
    }
    cells = expand_grid(grid)
    # Reuse all three → all cells collapse to one group.
    g_all = group_by_h_cache(cells, list(grid.keys()))
    assert len(g_all) == 1
    # Reuse only seed → one group per (L, cv) pair.
    g_seed = group_by_h_cache(cells, ["dispersion.sources.config.seed"])
    assert len(g_seed) == 4


def test_cache_grouping_reuses_H(tmp_path):
    """Driver invokes _dispersion_full once per group, not once per row."""
    pytest.importorskip("pyarrow")
    from enforceflux.osse import sweep as sweep_mod

    REPO_ROOT = sweep_mod.REPO_ROOT
    base_dir = REPO_ROOT / "configs" / "source_heterogeneity_e0"

    # 2x2 grid over the E0 base, reused across everything → one group.
    grid = {
        "dispersion.sources.config.covariance.L_m": [400.0, 500.0],
        "dispersion.sources.config.cv": [0.1, 0.2],
    }
    reuse = ["dispersion.sources.config.covariance.L_m",
             "dispersion.sources.config.cv"]

    parquet = tmp_path / "sweep.parquet"
    cfg = sweep_mod.SweepConfig(
        name="cache_group_test",
        base_dispersion=base_dir / "dispersion.yaml",
        base_flux=base_dir / "flux.yaml",
        base_analysis=base_dir / "analysis.yaml",
        grid=grid,
        reuse_H_across=reuse,
        workers=1,
        parquet_out=parquet,
        outputs_root=tmp_path / "runs",
    )

    real_full = sweep_mod._dispersion_full
    real_shared = sweep_mod._dispersion_shared_h
    n_full = {"count": 0}
    n_shared = {"count": 0}

    def counting_full(*args, **kwargs):
        n_full["count"] += 1
        return real_full(*args, **kwargs)

    def counting_shared(*args, **kwargs):
        n_shared["count"] += 1
        return real_shared(*args, **kwargs)

    with patch.object(sweep_mod, "_dispersion_full", counting_full), \
         patch.object(sweep_mod, "_dispersion_shared_h", counting_shared):
        sweep_mod.run_sweep(cfg)

    assert n_full["count"] == 1, f"expected 1 full dispersion, got {n_full['count']}"
    assert n_shared["count"] == 3, f"expected 3 shared, got {n_shared['count']}"
