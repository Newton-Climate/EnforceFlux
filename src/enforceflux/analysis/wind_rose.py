"""Wind-rose utilities for directional wind-frequency visualization."""
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for wind-rose plots. Install with: pip install matplotlib"
        )


def _wind_from_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction (degrees FROM) from u/v components.

    u: eastward wind component (m/s)
    v: northward wind component (m/s)
    """
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def _auto_speed_bins(speed: np.ndarray) -> np.ndarray:
    p95 = float(np.nanpercentile(speed, 95.0))
    top = max(1.0, p95)
    # 5 bins + open-ended top bin
    return np.array([0.0, 1.0, 3.0, 5.0, 8.0, top, np.inf], dtype=float)


def build_wind_rose(
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    direction_deg: np.ndarray | None = None,
    n_dir_bins: int = 16,
    speed_bins: Sequence[float] | None = None,
    calm_threshold: float = 0.2,
) -> dict:
    """Bin winds into direction/speed sectors for wind-rose plotting.

    Provide either:
    - u and v arrays, or
    - speed and direction_deg arrays (direction in meteorological degrees FROM).
    """
    if u is not None or v is not None:
        if u is None or v is None:
            raise ValueError("Provide both u and v, or neither")
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        v_arr = np.asarray(v, dtype=float).reshape(-1)
        if u_arr.shape != v_arr.shape:
            raise ValueError("u and v must have the same shape")
        speed_arr = np.hypot(u_arr, v_arr)
        dir_arr = _wind_from_direction_deg(u_arr, v_arr)
    else:
        if speed is None or direction_deg is None:
            raise ValueError("Provide either (u,v) or (speed,direction_deg)")
        speed_arr = np.asarray(speed, dtype=float).reshape(-1)
        dir_arr = np.asarray(direction_deg, dtype=float).reshape(-1) % 360.0
        if speed_arr.shape != dir_arr.shape:
            raise ValueError("speed and direction_deg must have the same shape")

    valid = np.isfinite(speed_arr) & np.isfinite(dir_arr)
    speed_arr = speed_arr[valid]
    dir_arr = dir_arr[valid]

    if speed_arr.size == 0:
        raise ValueError("No finite wind samples available")

    calm_mask = speed_arr < calm_threshold
    calm_fraction = float(np.mean(calm_mask))

    speed_use = speed_arr[~calm_mask]
    dir_use = dir_arr[~calm_mask]

    if speed_bins is None:
        s_bins = _auto_speed_bins(speed_use if speed_use.size > 0 else speed_arr)
    else:
        s_bins = np.asarray(speed_bins, dtype=float)
        if s_bins.ndim != 1 or s_bins.size < 2:
            raise ValueError("speed_bins must be a 1-D sequence with at least 2 values")
        if not np.all(np.diff(s_bins) > 0):
            raise ValueError("speed_bins must be strictly increasing")

    d_edges = np.linspace(0.0, 360.0, n_dir_bins + 1)
    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])

    table = np.zeros((n_dir_bins, len(s_bins) - 1), dtype=float)
    if speed_use.size > 0:
        d_idx = np.digitize(dir_use, d_edges, right=False) - 1
        d_idx[d_idx == n_dir_bins] = 0
        s_idx = np.digitize(speed_use, s_bins, right=False) - 1
        valid_bin = (d_idx >= 0) & (d_idx < n_dir_bins) & (s_idx >= 0) & (s_idx < len(s_bins) - 1)
        for i, j in zip(d_idx[valid_bin], s_idx[valid_bin]):
            table[i, j] += 1.0

    total = float(speed_arr.size)
    freq = 100.0 * table / total

    return {
        "frequency": freq,
        "direction_edges_deg": d_edges,
        "direction_centers_deg": d_centers,
        "speed_bins": s_bins,
        "calm_fraction": calm_fraction,
        "sample_count": int(total),
    }


def plot_wind_rose(
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    direction_deg: np.ndarray | None = None,
    n_dir_bins: int = 16,
    speed_bins: Sequence[float] | None = None,
    calm_threshold: float = 0.2,
    title: str = "Wind Rose",
    figsize: tuple[float, float] = (7.0, 7.0),
    cmap: str = "viridis",
):
    """Create a wind-rose plot and return (fig, ax, payload)."""
    _require_mpl()

    payload = build_wind_rose(
        u=u,
        v=v,
        speed=speed,
        direction_deg=direction_deg,
        n_dir_bins=n_dir_bins,
        speed_bins=speed_bins,
        calm_threshold=calm_threshold,
    )

    freq = payload["frequency"]
    d_centers = payload["direction_centers_deg"]
    d_edges = payload["direction_edges_deg"]
    s_bins = payload["speed_bins"]

    theta = np.radians(d_centers)
    width = np.radians(np.diff(d_edges))[0]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.95, freq.shape[1]))
    bottom = np.zeros(freq.shape[0], dtype=float)

    for k in range(freq.shape[1]):
        low = s_bins[k]
        high = s_bins[k + 1]
        if np.isinf(high):
            label = f">= {low:.1f} m/s"
        else:
            label = f"{low:.1f}-{high:.1f} m/s"
        ax.bar(theta, freq[:, k], width=width, bottom=bottom, color=colors[k], edgecolor="white", linewidth=0.5, label=label)
        bottom += freq[:, k]

    calm_pct = 100.0 * float(payload["calm_fraction"])
    ax.set_title(f"{title}\nCalm (<{calm_threshold} m/s): {calm_pct:.1f}%", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, ax, payload


def plot_wind_rose_from_netcdf(
    nc_path: str | Path,
    *,
    u_var: str = "u10",
    v_var: str = "v10",
    title: str | None = None,
    n_dir_bins: int = 16,
    speed_bins: Sequence[float] | None = None,
    calm_threshold: float = 0.2,
    figsize: tuple[float, float] = (7.0, 7.0),
    cmap: str = "viridis",
):
    """Load u/v winds from a NetCDF and plot a wind rose."""
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError("netCDF4 is required for NetCDF wind-rose plotting") from exc

    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    with Dataset(nc_path) as ds:
        if u_var not in ds.variables or v_var not in ds.variables:
            raise KeyError(f"Expected variables {u_var!r} and {v_var!r} in {nc_path}")
        u = np.asarray(ds.variables[u_var][:], dtype=float)
        v = np.asarray(ds.variables[v_var][:], dtype=float)

    resolved_title = title or f"Wind Rose ({nc_path.name})"
    return plot_wind_rose(
        u=u,
        v=v,
        n_dir_bins=n_dir_bins,
        speed_bins=speed_bins,
        calm_threshold=calm_threshold,
        title=resolved_title,
        figsize=figsize,
        cmap=cmap,
    )
