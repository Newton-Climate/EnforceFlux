"""Source-heterogeneity OSSE figures (memo Figs 1-5).

Each function reads the sweep parquet (or a dict of truth fields) and writes
a PNG plus a blind-accessible markdown description (orientation + CSV).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

# Columns produced by the sweep driver (M4).
_REQ_COLS = (
    "L", "CV", "N", "layout_seed", "met_id", "realization", "transport",
    "L_B", "e_q", "dfs_total", "chi2_per_dof", "prior_influence",
    "ak_diag_mean", "inverse_crime_flag", "run_dir",
)

_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lazy_mpl():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _lazy_pd():
    import pandas as pd
    return pd


def _filter_crime(df, keep_inverse_crime: bool):
    if keep_inverse_crime or "inverse_crime_flag" not in df.columns:
        return df
    return df[~df["inverse_crime_flag"].astype(bool)].copy()


def _write_description(path: Path, orientation: str, df_csv, caption: str = "") -> None:
    """Write a blind-accessible description: prose orientation + CSV block."""
    pd = _lazy_pd()
    buf = io.StringIO()
    if isinstance(df_csv, pd.DataFrame):
        df_csv.to_csv(buf, index=False, float_format="%.6g")
    else:
        buf.write(str(df_csv))
    text = (
        f"# Figure description (blind-accessible)\n\n"
        f"{orientation.strip()}\n\n"
    )
    if caption:
        text += f"{caption.strip()}\n\n"
    text += "```csv\n" + buf.getvalue().rstrip() + "\n```\n"
    path.write_text(text)


def _p_success(e_q: np.ndarray, epsilon: float) -> float:
    e_q = np.asarray(e_q, dtype=float)
    e_q = e_q[np.isfinite(e_q)]
    if e_q.size == 0:
        return np.nan
    return float(np.mean(np.abs(e_q) <= epsilon))


def _n_min_for_group(g, epsilon: float, p_star: float) -> float:
    """Smallest N whose P_success >= p_star; NaN if never reached."""
    by_N = g.groupby("N")["e_q"].apply(lambda s: _p_success(s.values, epsilon))
    by_N = by_N.sort_index()
    ok = by_N[by_N >= p_star]
    if len(ok) == 0:
        return np.nan
    return float(ok.index[0])


# ---------------------------------------------------------------------------
# Figure 1: source field quartet
# ---------------------------------------------------------------------------

def figure_source_quartet(
    fields: dict[tuple[float, float], np.ndarray],
    grid: Any,
    output: Path,
) -> tuple[Path, Path]:
    """2x2 quartet of truth fields at fixed Q_true, common colormap in F/F_bar."""
    plt = _lazy_mpl()
    pd = _lazy_pd()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    Ls = sorted({k[0] for k in fields})
    CVs = sorted({k[1] for k in fields})
    if len(Ls) < 2 or len(CVs) < 2:
        raise ValueError("figure_source_quartet needs >=2 L and >=2 CV values")
    L_lo, L_hi = Ls[0], Ls[-1]
    CV_lo, CV_hi = CVs[0], CVs[-1]

    corners = [
        (L_hi, CV_lo, "large L, low CV"),
        (L_hi, CV_hi, "large L, high CV"),
        (L_lo, CV_lo, "small L, low CV"),
        (L_lo, CV_hi, "small L, high CV"),
    ]

    normed = {}
    for key in [(L, CV) for L, CV, _ in corners]:
        if key not in fields:
            raise KeyError(f"fields is missing corner {key}")
        F = np.asarray(fields[key], dtype=float)
        fbar = F.mean()
        normed[key] = F / fbar if fbar > 0 else F.copy()

    vmax = max(float(v.max()) for v in normed.values())
    dx = float(getattr(grid, "dx_m", 1.0))
    ny, nx = next(iter(normed.values())).shape
    extent = (0.0, nx * dx, 0.0, ny * dx)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.2), constrained_layout=True)
    for ax, (L, CV, label) in zip(axes.flat, corners):
        im = ax.imshow(
            normed[(L, CV)], origin="lower", extent=extent,
            vmin=0.0, vmax=vmax, cmap="magma",
        )
        ax.set_title(f"L={L:g} m, CV={CV:g}\n({label})", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"$F/\bar{F}$ (dimensionless)")
    fig.suptitle("Truth flux fields at fixed total emission", fontsize=11)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    rows = []
    for (L, CV, label) in corners:
        v = normed[(L, CV)]
        rows.append({
            "L_m": L, "CV": CV, "corner": label,
            "F_over_Fbar_min": float(v.min()),
            "F_over_Fbar_median": float(np.median(v)),
            "F_over_Fbar_p95": float(np.percentile(v, 95)),
            "F_over_Fbar_max": float(v.max()),
            "fraction_above_2Fbar": float(np.mean(v > 2.0)),
        })
    tbl = pd.DataFrame(rows)

    desc = output.with_suffix(".md")
    orientation = (
        "A two-by-two grid of truth flux fields at the same domain-total emission. "
        "The top row shows large correlation length (fields organized into a few big "
        "patches); the bottom row shows small correlation length (fields organized into "
        "many small patches). The left column has low coefficient of variation (nearly "
        "uniform); the right column has high coefficient of variation (a handful of hot "
        "cells carry most of the emission). All four panels share a magma colormap in "
        "units of F over the field mean; brighter means locally higher emission."
    )
    _write_description(
        desc, orientation, tbl,
        caption="Per-corner summary statistics of F over its spatial mean.",
    )
    return output, desc


# ---------------------------------------------------------------------------
# Figure 2: phase diagram (L, N) per CV
# ---------------------------------------------------------------------------

def figure_phase_diagram(
    df,
    output: Path,
    epsilon: float = 0.2,
    keep_inverse_crime: bool = False,
) -> tuple[Path, Path]:
    """L (log x) vs N (y), one panel per CV; color = P_success. 0.9 contour."""
    plt = _lazy_mpl()
    pd = _lazy_pd()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = _filter_crime(df, keep_inverse_crime)
    grp = df.groupby(["CV", "L", "N"])["e_q"].apply(
        lambda s: _p_success(s.values, epsilon)
    ).reset_index(name="p_success")

    CVs = sorted(grp["CV"].unique())
    ncols = min(len(CVs), 4)
    nrows = int(np.ceil(len(CVs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows),
        constrained_layout=True, squeeze=False,
    )
    im = None
    for ax, cv in zip(axes.flat, CVs):
        sub = grp[grp["CV"] == cv]
        piv = sub.pivot(index="N", columns="L", values="p_success").sort_index().sort_index(axis=1)
        Ls = piv.columns.values.astype(float)
        Ns = piv.index.values.astype(float)
        Z = piv.values.astype(float)
        im = ax.pcolormesh(
            Ls, Ns, Z, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis",
        )
        if np.any(np.isfinite(Z)) and np.nanmax(Z) >= 0.9 >= np.nanmin(Z):
            try:
                cs = ax.contour(Ls, Ns, Z, levels=[0.9], colors="white", linewidths=1.5)
                ax.clabel(cs, fmt="P=0.9", fontsize=8)
            except Exception:
                pass
        ax.set_xscale("log")
        ax.set_xlabel("L [m]")
        ax.set_ylabel("N receptors")
        ax.set_title(f"CV = {cv:g}")
    for ax in axes.flat[len(CVs):]:
        ax.set_visible(False)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label(f"P_success (|E_Q| <= {epsilon:g})")
    fig.suptitle("Recovery phase diagram", fontsize=11)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    desc = output.with_suffix(".md")
    orientation = (
        "One panel per coefficient of variation. Each panel is a heatmap: correlation "
        "length L on a logarithmic horizontal axis, receptor count N on a linear "
        "vertical axis. The color shows the probability that the total-emission error "
        f"|E_Q| is within {epsilon:g}. A white contour marks the 0.90 success level. "
        "Cells sweep from unrecoverable (dark) at low N and small L to well-recovered "
        "(bright) at high N and large L."
    )
    _write_description(
        desc, orientation, grp.sort_values(["CV", "L", "N"]),
        caption="Per (CV, L, N) success probability used to render the heatmap.",
    )
    return output, desc


# ---------------------------------------------------------------------------
# Figure 3: N_min vs L, one curve per CV, with bootstrap bands
# ---------------------------------------------------------------------------

def figure_n_min_curves(
    df,
    output: Path,
    epsilon: float = 0.2,
    p_star: float = 0.9,
    keep_inverse_crime: bool = False,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> tuple[Path, Path]:
    """L (log x) vs N_min (log y); one curve per CV; bootstrap over layout_seed/realization."""
    plt = _lazy_mpl()
    pd = _lazy_pd()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = _filter_crime(df, keep_inverse_crime)
    rng = np.random.default_rng(seed)

    boot_cols = [c for c in ("layout_seed", "realization") if c in df.columns]
    rows = []
    for (cv, L), g in df.groupby(["CV", "L"]):
        n_min = _n_min_for_group(g, epsilon, p_star)
        boots = []
        if boot_cols and len(g) > 1:
            keys = g[boot_cols].drop_duplicates().values
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(keys), size=len(keys))
                pick = pd.DataFrame(keys[idx], columns=boot_cols)
                sub = g.merge(pick, on=boot_cols, how="inner")
                boots.append(_n_min_for_group(sub, epsilon, p_star))
        boots_arr = np.array([b for b in boots if np.isfinite(b)])
        lo = float(np.percentile(boots_arr, 16)) if boots_arr.size else np.nan
        hi = float(np.percentile(boots_arr, 84)) if boots_arr.size else np.nan
        rows.append({"CV": cv, "L": L, "n_min": n_min, "n_min_lo": lo, "n_min_hi": hi})
    tbl = pd.DataFrame(rows).sort_values(["CV", "L"])

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    CVs = sorted(tbl["CV"].unique())
    for i, cv in enumerate(CVs):
        sub = tbl[tbl["CV"] == cv].sort_values("L")
        ls = _LINESTYLES[i % len(_LINESTYLES)]
        mk = _MARKERS[i % len(_MARKERS)]
        ax.plot(sub["L"], sub["n_min"], linestyle=ls, marker=mk,
                label=f"CV = {cv:g}")
        if sub["n_min_lo"].notna().any():
            ax.fill_between(sub["L"], sub["n_min_lo"], sub["n_min_hi"], alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L [m]")
    ax.set_ylabel(f"N_min (|E_Q| <= {epsilon:g}, P >= {p_star:g})")
    ax.set_title("Receptors needed vs source correlation length")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    desc = output.with_suffix(".md")
    orientation = (
        "Log-log plot: correlation length L on the horizontal axis, minimum receptor "
        "count N_min on the vertical axis. One curve per coefficient of variation, "
        "distinguished by both line style and marker shape for colorblind access. "
        "Shaded bands span the 16th to 84th percentile from bootstrapping over layout "
        "seeds and realizations. Larger L or lower CV lowers N_min."
    )
    _write_description(
        desc, orientation, tbl,
        caption=f"N_min per (CV, L). epsilon={epsilon:g}, P*={p_star:g}.",
    )
    return output, desc


# ---------------------------------------------------------------------------
# Figure 4: scaling collapse d_N/L vs L/L_H
# ---------------------------------------------------------------------------

def figure_scaling_collapse(
    df,
    output: Path,
    footprint_length_m: float,
    domain_area_m2: float | None = None,
    epsilon: float = 0.2,
    keep_inverse_crime: bool = False,
) -> tuple[Path, Path]:
    """Replot success in dimensionless coords: d_N/L (y) vs L/L_H (x)."""
    plt = _lazy_mpl()
    pd = _lazy_pd()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = _filter_crime(df, keep_inverse_crime).copy()
    if domain_area_m2 is None:
        # Default: square domain 10x the footprint length. Explicit override recommended.
        domain_area_m2 = (10.0 * float(footprint_length_m)) ** 2

    grp = df.groupby(["CV", "L", "N"])["e_q"].apply(
        lambda s: _p_success(s.values, epsilon)
    ).reset_index(name="p_success")
    grp["d_N"] = np.sqrt(domain_area_m2 / grp["N"].astype(float))
    grp["dN_over_L"] = grp["d_N"] / grp["L"].astype(float)
    grp["L_over_LH"] = grp["L"].astype(float) / float(footprint_length_m)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    CVs = sorted(grp["CV"].unique())
    sc = None
    for i, cv in enumerate(CVs):
        sub = grp[grp["CV"] == cv]
        mk = _MARKERS[i % len(_MARKERS)]
        sc = ax.scatter(
            sub["L_over_LH"], sub["dN_over_L"], c=sub["p_success"],
            marker=mk, vmin=0.0, vmax=1.0, cmap="viridis",
            edgecolor="black", linewidth=0.4, s=45,
            label=f"CV = {cv:g}",
        )
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L / L_H$")
    ax.set_ylabel(r"$d_N / L$")
    ax.set_title("Dimensionless scaling collapse")
    ax.legend(frameon=False, loc="best")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"P_success (|E_Q| <= {epsilon:g})")
    fig.savefig(output, dpi=150)
    plt.close(fig)

    desc = output.with_suffix(".md")
    orientation = (
        "Scatter of every (CV, L, N) grid cell replotted in dimensionless coordinates: "
        "L divided by the footprint correlation length L_H on the horizontal log axis, "
        "and receptor spacing d_N divided by L on the vertical log axis. Each point is "
        "shaded by its success probability. Marker shape distinguishes CV so the plot "
        "reads without color. Dotted reference lines mark L/L_H = 1 and d_N/L = 1; "
        "successful recoveries collect at d_N/L below one and L/L_H above one."
    )
    _write_description(
        desc, orientation, grp.sort_values(["CV", "L", "N"]),
        caption=(
            f"Dimensionless coordinates. footprint_length_m={footprint_length_m:g}, "
            f"domain_area_m2={domain_area_m2:g}."
        ),
    )
    return output, desc


# ---------------------------------------------------------------------------
# Figure 5: error floor E_Q vs N for a selected (L, CV)
# ---------------------------------------------------------------------------

def figure_error_floor(
    df,
    output: Path,
    L_select: float | None = None,
    CV_select: float | None = None,
    keep_inverse_crime: bool = False,
) -> tuple[Path, Path]:
    """|E_Q| vs N, one curve per transport (LPDM->LPDM, LES->LPDM, LES->Gaussian)."""
    plt = _lazy_mpl()
    pd = _lazy_pd()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = _filter_crime(df, keep_inverse_crime).copy()
    if L_select is None:
        Ls = sorted(df["L"].unique())
        L_select = float(Ls[len(Ls) // 2])
    if CV_select is None:
        CVs = sorted(df["CV"].unique())
        CV_select = float(CVs[len(CVs) // 2])
    sub = df[(np.isclose(df["L"], L_select)) & (np.isclose(df["CV"], CV_select))]

    transports = sorted(sub["transport"].unique())
    tbl_rows = []
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for i, t in enumerate(transports):
        g = sub[sub["transport"] == t]
        agg = g.groupby("N")["e_q"].agg(
            median=lambda s: float(np.nanmedian(np.abs(s))),
            q16=lambda s: float(np.nanpercentile(np.abs(s), 16)),
            q84=lambda s: float(np.nanpercentile(np.abs(s), 84)),
        ).reset_index().sort_values("N")
        ls = _LINESTYLES[i % len(_LINESTYLES)]
        mk = _MARKERS[i % len(_MARKERS)]
        ax.plot(agg["N"], agg["median"], linestyle=ls, marker=mk, label=str(t))
        ax.fill_between(agg["N"], agg["q16"], agg["q84"], alpha=0.15)
        for _, r in agg.iterrows():
            tbl_rows.append({"transport": t, "N": int(r["N"]),
                             "median_absE_Q": r["median"],
                             "q16": r["q16"], "q84": r["q84"]})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N receptors")
    ax.set_ylabel(r"median $|E_Q|$")
    ax.set_title(f"Error floor at L={L_select:g} m, CV={CV_select:g}")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    tbl = _lazy_pd().DataFrame(tbl_rows)
    desc = output.with_suffix(".md")
    orientation = (
        "Log-log plot at a single (L, CV) regime: receptor count N on the horizontal "
        "axis, median absolute total-emission error on the vertical axis. Three curves "
        "compare experiment classes, distinguished by line style and marker shape: "
        "LPDM to LPDM (self-consistent), LES to LPDM (representativeness error), and "
        "LES to Gaussian (worst-case model form). Shaded bands span the 16th to 84th "
        "percentile across seeds. The LES curves plateau at large N; that plateau is "
        "the error floor."
    )
    _write_description(
        desc, orientation, tbl,
        caption=f"L_select={L_select:g}, CV_select={CV_select:g}.",
    )
    return output, desc
