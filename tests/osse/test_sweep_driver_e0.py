"""Tiny end-to-end sweep on the E0 configs; asserts parquet schema + E_Q."""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
E0_DIR = REPO_ROOT / "configs" / "source_heterogeneity_e0"


REQUIRED_COLUMNS = [
    "L", "CV", "N", "layout_seed", "met_id", "realization", "transport",
    "L_B", "e_q", "dfs_total", "chi2_per_dof", "prior_influence",
    "ak_diag_mean", "inverse_crime_flag", "run_dir",
]


def test_sweep_driver_e0(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from enforceflux.osse.sweep import SweepConfig, run_sweep

    grid = {
        "dispersion.sources.config.covariance.L_m": [200, 500],
        "dispersion.sources.config.cv": [0.1, 0.5],
        "dispersion.sources.config.seed": {"range": [0, 3]},
    }
    reuse = ["dispersion.sources.config.covariance.L_m",
             "dispersion.sources.config.cv",
             "dispersion.sources.config.seed"]

    parquet = tmp_path / "sweep.parquet"
    cfg = SweepConfig(
        name="e0_tiny_sweep",
        base_dispersion=E0_DIR / "dispersion.yaml",
        base_flux=E0_DIR / "flux.yaml",
        base_analysis=E0_DIR / "analysis.yaml",
        grid=grid,
        reuse_H_across=reuse,
        workers=1,
        parquet_out=parquet,
        outputs_root=tmp_path / "runs",
    )
    out = run_sweep(cfg)
    assert out == parquet
    assert parquet.is_file()

    df = pd.read_parquet(parquet)
    # 2 * 2 * 3 = 12 rows
    assert len(df) == 12
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing column {col}"

    # For smooth fields (cv=0.1) E_Q should be small.
    smooth = df[df["CV"] == 0.1]
    assert len(smooth) == 6
    assert smooth["e_q"].max() < 0.05, (
        f"E_Q too large for cv=0.1: max = {smooth['e_q'].max()}"
    )
