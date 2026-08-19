"""End-to-end E0 regression: sanity that posterior recovers Q_true.

Runs apps/dispersion_main.py, apps/flux_main.py, apps/analysis_main.py
in-process against configs/source_heterogeneity_e0/ and asserts E_Q < 0.02
for the smooth (cv=0.1) field.
"""
import json
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "source_heterogeneity_e0"
RUNS_ROOT = REPO_ROOT / "runs"


def _run_app(script: str, config: Path):
    saved_argv = sys.argv[:]
    sys.argv = [script, "--config", str(config)]
    try:
        runpy.run_path(str(REPO_ROOT / "apps" / script), run_name="__main__")
    finally:
        sys.argv = saved_argv


def _reset_run():
    from enforceflux.plugins.source_lognormal_field import clear_pending_writes

    clear_pending_writes()
    if str(REPO_ROOT / "apps") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "apps"))


@pytest.mark.slow
def test_e0_recovers_q_true():
    _reset_run()
    _run_app("dispersion_main.py", CONFIG_DIR / "dispersion.yaml")
    _run_app("flux_main.py", CONFIG_DIR / "flux.yaml")
    _run_app("analysis_main.py", CONFIG_DIR / "analysis.yaml")

    analysis_summary_path = (
        RUNS_ROOT / "source_heterogeneity_e0" / "analysis" / "summary.json"
    )
    summary = json.loads(analysis_summary_path.read_text())
    assert "source_heterogeneity" in summary
    e_q = float(summary["source_heterogeneity"]["E_Q"])
    assert e_q < 0.02, f"E_Q = {e_q} exceeds tolerance 0.02"
    assert summary["source_heterogeneity"]["inverse_crime_flag"] is False
