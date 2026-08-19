"""Every sweep row carries the fields the optimal-estimation skill needs."""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
E0_DIR = REPO_ROOT / "configs" / "source_heterogeneity_e0"


def test_optimal_estimation_columns_non_null(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from enforceflux.osse.sweep import SweepConfig, run_sweep

    grid = {
        "dispersion.sources.config.covariance.L_m": [500],
        "dispersion.sources.config.cv": [0.1, 0.5],
    }
    reuse = ["dispersion.sources.config.covariance.L_m",
             "dispersion.sources.config.cv"]

    parquet = tmp_path / "sweep.parquet"
    cfg = SweepConfig(
        name="oe_report_test",
        base_dispersion=E0_DIR / "dispersion.yaml",
        base_flux=E0_DIR / "flux.yaml",
        base_analysis=E0_DIR / "analysis.yaml",
        grid=grid,
        reuse_H_across=reuse,
        workers=1,
        parquet_out=parquet,
        outputs_root=tmp_path / "runs",
    )
    run_sweep(cfg)

    df = pd.read_parquet(parquet)
    for col in ("dfs_total", "ak_diag_mean", "L_true", "L_B_m"):
        assert col in df.columns, f"missing column {col}"
        assert df[col].notna().all(), (
            f"column {col} has nulls: {df[col].tolist()}"
        )
    assert df["inverse_crime_flag"].notna().all()
    assert (df["dfs_total"] > 0).all()
