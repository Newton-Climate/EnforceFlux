"""Smoke tests for source-heterogeneity OSSE figures (M7)."""
from __future__ import annotations

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
pd = pytest.importorskip("pandas")

from enforceflux.analysis import heterogeneity_figures as hf


def _synthetic_sweep(seed: int = 0) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    Ls = [100.0, 500.0, 2000.0]
    CVs = [0.25, 1.0, 1.5]
    Ns = [2, 4, 8, 16, 32]
    transports = ["lpdm_lpdm", "les_lpdm", "les_gaussian"]
    seeds = [0, 1, 2]
    rows = []
    for L in Ls:
        for cv in CVs:
            for N in Ns:
                for t in transports:
                    for s in seeds:
                        # error decreases with N and L, floors for LES-* transports
                        base = 0.6 * (1.0 / np.sqrt(N)) * (1.0 + cv) * (500.0 / L)
                        floor = {"lpdm_lpdm": 0.0, "les_lpdm": 0.05,
                                 "les_gaussian": 0.12}[t]
                        e_q = base + floor + 0.03 * rng.standard_normal()
                        rows.append({
                            "L": L, "CV": cv, "N": N, "layout_seed": s,
                            "met_id": "M0", "realization": s, "transport": t,
                            "L_B": 800.0, "e_q": e_q,
                            "dfs_total": float(N) * 0.5,
                            "chi2_per_dof": 1.0 + 0.05 * rng.standard_normal(),
                            "prior_influence": 0.2,
                            "ak_diag_mean": 0.6,
                            "inverse_crime_flag": False,
                            "run_dir": f"runs/{L}_{cv}_{N}_{t}_{s}",
                        })
    return pd.DataFrame(rows)


class _Grid:
    def __init__(self, nx: int, ny: int, dx_m: float = 50.0):
        self.nx, self.ny, self.dx_m = nx, ny, dx_m


def _nonempty(*paths) -> None:
    for p in paths:
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 0, f"empty {p}"


def test_figure_source_quartet(tmp_path):
    rng = np.random.default_rng(1)
    grid = _Grid(32, 32, dx_m=25.0)
    Ls, CVs = [100.0, 2000.0], [0.25, 1.5]
    fields = {}
    for L in Ls:
        for cv in CVs:
            f = np.exp(cv * rng.standard_normal((grid.ny, grid.nx)))
            fields[(L, cv)] = f / f.mean()
    out = tmp_path / "fig1_quartet.png"
    png, md = hf.figure_source_quartet(fields, grid, out)
    _nonempty(png, md)
    body = md.read_text()
    assert "```csv" in body


def test_figure_phase_diagram(tmp_path):
    df = _synthetic_sweep()
    out = tmp_path / "fig2_phase.png"
    png, md = hf.figure_phase_diagram(df, out, epsilon=0.2)
    _nonempty(png, md)
    assert "P_success" in md.read_text() or "success" in md.read_text().lower()


def test_figure_n_min_curves(tmp_path):
    df = _synthetic_sweep()
    out = tmp_path / "fig3_nmin.png"
    png, md = hf.figure_n_min_curves(df, out, epsilon=0.3, p_star=0.5,
                                     n_bootstrap=20, seed=0)
    _nonempty(png, md)


def test_figure_scaling_collapse(tmp_path):
    df = _synthetic_sweep()
    out = tmp_path / "fig4_collapse.png"
    png, md = hf.figure_scaling_collapse(df, out, footprint_length_m=500.0,
                                         domain_area_m2=(5000.0) ** 2,
                                         epsilon=0.3)
    _nonempty(png, md)


def test_figure_error_floor(tmp_path):
    df = _synthetic_sweep()
    out = tmp_path / "fig5_floor.png"
    png, md = hf.figure_error_floor(df, out, L_select=500.0, CV_select=1.0)
    _nonempty(png, md)
    text = md.read_text()
    assert "les_lpdm" in text or "transport" in text


def test_inverse_crime_filter(tmp_path):
    df = _synthetic_sweep()
    df.loc[df.index[:10], "inverse_crime_flag"] = True
    out = tmp_path / "fig3_filtered.png"
    png, md = hf.figure_n_min_curves(df, out, epsilon=0.3, p_star=0.5,
                                     n_bootstrap=5, keep_inverse_crime=False)
    _nonempty(png, md)
