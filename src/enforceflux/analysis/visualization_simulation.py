"""Simulation concentration-map and animation utilities."""
from pathlib import Path
from typing import Sequence

import numpy as np

from enforceflux.analysis._viz_base import _make_fig, _require_mpl, _resolve, plt


def _auto_scale_values(
    values: np.ndarray,
    *,
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
    prefer_positive: bool = False,
) -> tuple[float, float]:
    """Compute robust vmin/vmax percentiles with safe fallbacks."""
    finite = values[np.isfinite(values)]
    if prefer_positive:
        finite = finite[finite > 0]

    if finite.size == 0:
        finite = values[np.isfinite(values)]

    if finite.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanpercentile(finite, lower_pct))
    vmax = float(np.nanpercentile(finite, upper_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = float(np.nanmax(finite))
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0, 1.0
    return vmin, vmax


def _build_norm(
    *,
    use_log_scale: bool,
    vmin: float,
    vmax: float,
):
    if not use_log_scale:
        return None, vmin, vmax

    _require_mpl()
    from matplotlib.colors import LogNorm

    if vmax <= 0:
        vmax = 1.0
    if vmin <= 0:
        vmin = max(vmax * 1e-6, 1e-6)
    if vmin >= vmax:
        vmin = max(vmax * 1e-3, 1e-6)
    return LogNorm(vmin=vmin, vmax=vmax), None, None


def _extract_surface_frames(
    concentration: np.ndarray,
    *,
    dim_names: Sequence[str] | None = None,
    level_index: int = 0,
    release_index: int = 0,
) -> np.ndarray:
    """Normalize concentration arrays to (time, y, x) for map plotting."""
    arr = np.asarray(concentration, dtype=float)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim < 3:
        raise ValueError(f"Unsupported concentration shape: {arr.shape}")

    def _find_name(candidates: tuple[str, ...], names: list[str]) -> int | None:
        for i, n in enumerate(names):
            low = n.lower()
            if any(c in low for c in candidates):
                return i
        return None

    names = (
        list(dim_names)
        if dim_names is not None and len(dim_names) == arr.ndim
        else [f"axis_{i}" for i in range(arr.ndim)]
    )

    time_ax = _find_name(("time",), names)
    lat_ax = _find_name(("lat", "latitude"), names)
    lon_ax = _find_name(("lon", "longitude"), names)

    if time_ax is None:
        time_ax = 0
    if lat_ax is None:
        lat_ax = arr.ndim - 2
    if lon_ax is None:
        lon_ax = arr.ndim - 1

    if len({time_ax, lat_ax, lon_ax}) != 3:
        time_ax, lat_ax, lon_ax = 0, arr.ndim - 2, arr.ndim - 1

    perm = [time_ax] + [ax for ax in range(arr.ndim) if ax not in {time_ax, lat_ax, lon_ax}] + [lat_ax, lon_ax]
    arr = np.transpose(arr, perm)
    ordered_names = [names[i] for i in perm]

    if arr.ndim == 3:
        return arr

    middle_count = arr.ndim - 3
    middle_names = ordered_names[1:1 + middle_count]
    indexer: list[int | slice] = [slice(None)] + [0] * middle_count + [slice(None), slice(None)]

    if middle_count == 1:
        indexer[1] = level_index
    else:
        rel_mid = _find_name(("release", "nageclass", "pointspec"), middle_names)
        lev_mid = _find_name(("height", "level", "z"), middle_names)
        if rel_mid is None:
            rel_mid = 0
        if lev_mid is None:
            lev_mid = middle_count - 1
        if lev_mid == rel_mid and middle_count > 1:
            lev_mid = middle_count - 1 if rel_mid != middle_count - 1 else 0
        indexer[1 + rel_mid] = release_index
        indexer[1 + lev_mid] = level_index

    try:
        out = arr[tuple(indexer)]
    except IndexError as exc:
        raise IndexError(
            f"Requested release_index={release_index} / level_index={level_index} "
            f"is out of bounds for concentration shape {arr.shape}."
        ) from exc

    if out.ndim != 3:
        raise ValueError(
            "Could not reduce concentration to (time, y, x). "
            f"Input shape {arr.shape} produced shape {out.shape}."
        )
    return out


def plot_simulation_heatmap(
    concentration: np.ndarray,
    *,
    dim_names: Sequence[str] | None = None,
    time_index: int = 0,
    level_index: int = 0,
    release_index: int = 0,
    lons: np.ndarray | None = None,
    lats: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    use_log_scale: bool = False,
    nonzero_scaling: bool = True,
    cmap: str = "magma",
    units: str = "",
    title: str = "CH4 concentration",
    ax=None,
    figsize: tuple = (7, 5),
):
    """Plot a 2-D heatmap of simulation concentration for one timestep."""
    frames = _extract_surface_frames(
        concentration,
        dim_names=dim_names,
        level_index=level_index,
        release_index=release_index,
    )
    if time_index < 0:
        time_index += frames.shape[0]
    if time_index < 0 or time_index >= frames.shape[0]:
        raise IndexError(f"time_index={time_index} out of range for {frames.shape[0]} frames")

    from scipy.ndimage import gaussian_filter
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)
    frame = gaussian_filter(frames[time_index].astype(float), sigma=1.0)

    if vmin is None or vmax is None:
        # Prefer nonzero values for sparse concentration maps.
        auto_vmin, auto_vmax = _auto_scale_values(
            frame,
            lower_pct=2.0,
            upper_pct=99.5,
            prefer_positive=nonzero_scaling,
        )
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    norm, vmin, vmax = _build_norm(use_log_scale=use_log_scale, vmin=float(vmin), vmax=float(vmax))
    frame_to_plot = np.where(frame > 0, frame, np.nan) if use_log_scale else frame
    cmap_obj = plt.get_cmap(cmap).copy() if use_log_scale else cmap
    if use_log_scale and hasattr(cmap_obj, "set_bad"):
        cmap_obj.set_bad("black")

    extent = None
    if lons is not None and lats is not None:
        extent = (
            float(np.min(lons)),
            float(np.max(lons)),
            float(np.min(lats)),
            float(np.max(lats)),
        )

    im = ax.imshow(
        frame_to_plot,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap=cmap_obj,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    cb = fig.colorbar(im, ax=ax)
    label = units if units else "Concentration"
    cb.set_label(f"{label} (log scale)" if use_log_scale else label)

    ax.set_title(f"{title} (t={time_index})")
    if extent is None:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.tight_layout()
    return fig, ax


def create_simulation_movie(
    concentration: np.ndarray,
    *,
    output_path: str | Path | None = None,
    dim_names: Sequence[str] | None = None,
    level_index: int = 0,
    release_index: int = 0,
    lons: np.ndarray | None = None,
    lats: np.ndarray | None = None,
    time_labels: Sequence[str] | None = None,
    fps: int = 5,
    use_log_scale: bool = False,
    nonzero_scaling: bool = True,
    cmap: str = "magma",
    units: str = "",
    title_prefix: str = "CH4 concentration",
    figsize: tuple = (7, 5),
):
    """Create an animation of concentration change over time."""
    _require_mpl()
    from matplotlib.animation import FuncAnimation

    frames = _extract_surface_frames(
        concentration,
        dim_names=dim_names,
        level_index=level_index,
        release_index=release_index,
    )
    n_frames = frames.shape[0]
    if n_frames == 0:
        raise ValueError("No frames available for animation")

    if time_labels is not None and len(time_labels) != n_frames:
        raise ValueError(
            f"time_labels length {len(time_labels)} must match number of frames {n_frames}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    auto_vmin, auto_vmax = _auto_scale_values(
        frames,
        lower_pct=2.0,
        upper_pct=99.5,
        prefer_positive=nonzero_scaling,
    )
    norm, im_vmin, im_vmax = _build_norm(
        use_log_scale=use_log_scale,
        vmin=auto_vmin,
        vmax=auto_vmax,
    )
    cmap_obj = plt.get_cmap(cmap).copy() if use_log_scale else cmap
    if use_log_scale and hasattr(cmap_obj, "set_bad"):
        cmap_obj.set_bad("black")

    def _frame_data(i: int) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        f = gaussian_filter(frames[i].astype(float), sigma=1.0)
        return np.where(f > 0, f, np.nan) if use_log_scale else f

    extent = None
    if lons is not None and lats is not None:
        extent = (
            float(np.min(lons)),
            float(np.max(lons)),
            float(np.min(lats)),
            float(np.max(lats)),
        )

    im = ax.imshow(
        _frame_data(0),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap=cmap_obj,
        norm=norm,
        vmin=im_vmin,
        vmax=im_vmax,
        extent=extent,
    )
    cb = fig.colorbar(im, ax=ax)
    label = units if units else "Concentration"
    cb.set_label(f"{label} (log scale)" if use_log_scale else label)

    if extent is None:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    def _frame_title(i: int) -> str:
        tag = time_labels[i] if time_labels is not None else f"t={i}"
        return f"{title_prefix} ({tag})"

    ax.set_title(_frame_title(0))

    def _update(i: int):
        im.set_data(_frame_data(i))
        ax.set_title(_frame_title(i))
        return (im,)

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=max(1, int(1000 / max(1, fps))),
        blit=False,
    )

    fig.tight_layout()

    if output_path is None:
        return anim

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    writer = "pillow" if suffix == ".gif" else "ffmpeg" if suffix == ".mp4" else None
    if writer is None:
        raise ValueError("output_path must end with .gif or .mp4")

    try:
        anim.save(str(out), writer=writer, fps=fps)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write animation to {out}. "
            "For GIF install pillow; for MP4 ensure ffmpeg is installed."
        ) from exc

    return out


def load_simulation_netcdf(
    nc_path: str | Path,
    *,
    variable_names: Sequence[str] = ("ch4_mixing_ratio", "ch4_concentration"),
) -> dict:
    """Load concentration and coordinates from an EnforceFlux/FLEXPART NetCDF."""
    try:
        from netCDF4 import Dataset, num2date
    except ImportError as exc:
        raise ImportError(
            "netCDF4 is required to read simulation outputs. Install with: pip install netCDF4"
        ) from exc

    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF output not found: {nc_path}")

    def _find_var(ds, candidates: Sequence[str]):
        for name in candidates:
            if name in ds.variables:
                return name
        return None

    with Dataset(nc_path) as ds:
        vname = _find_var(ds, variable_names)
        if vname is None:
            ch4_like = [n for n in ds.variables if "ch4" in n.lower()]
            if ch4_like:
                vname = ch4_like[0]
            else:
                raise KeyError(
                    "No CH4 concentration variable found in NetCDF. "
                    f"Tried {tuple(variable_names)}."
                )

        var = ds.variables[vname]
        concentration = np.asarray(var[:])
        dimensions = tuple(str(d) for d in var.dimensions)
        units = str(getattr(var, "units", ""))

        lon_name = _find_var(ds, ("longitude", "lon", "xlon"))
        lat_name = _find_var(ds, ("latitude", "lat", "ylat"))
        lons = np.asarray(ds.variables[lon_name][:]) if lon_name else None
        lats = np.asarray(ds.variables[lat_name][:]) if lat_name else None

        time_labels = None
        tname = _find_var(ds, ("time", "Times"))
        if tname and tname in ds.variables:
            tvar = ds.variables[tname]
            try:
                units_attr = getattr(tvar, "units")
                cal_attr = getattr(tvar, "calendar", "standard")
                dt_vals = num2date(tvar[:], units=units_attr, calendar=cal_attr)
                dt_iter = np.asarray(dt_vals, dtype=object).reshape(-1)
                time_labels = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in dt_iter]
            except Exception:
                try:
                    tvals = np.asarray(tvar[:]).reshape(-1)
                    time_labels = [str(v) for v in tvals]
                except Exception:
                    time_labels = None

    return {
        "concentration": concentration,
        "dimensions": dimensions,
        "lons": lons,
        "lats": lats,
        "time_labels": time_labels,
        "units": units,
        "variable_name": vname,
    }


def plot_simulation_heatmap_from_netcdf(
    nc_path: str | Path,
    *,
    time_index: int = 0,
    level_index: int = 0,
    release_index: int = 0,
    variable_names: Sequence[str] = ("ch4_mixing_ratio", "ch4_concentration"),
    vmin: float | None = None,
    vmax: float | None = None,
    use_log_scale: bool = False,
    nonzero_scaling: bool = True,
    cmap: str = "magma",
    title: str | None = None,
    ax=None,
    figsize: tuple = (7, 5),
):
    """Load simulation NetCDF and plot a concentration heatmap."""
    data = load_simulation_netcdf(nc_path, variable_names=variable_names)
    resolved_title = title or f"{data['variable_name']}"
    return plot_simulation_heatmap(
        data["concentration"],
        dim_names=data.get("dimensions"),
        time_index=time_index,
        level_index=level_index,
        release_index=release_index,
        lons=data["lons"],
        lats=data["lats"],
        vmin=vmin,
        vmax=vmax,
        use_log_scale=use_log_scale,
        nonzero_scaling=nonzero_scaling,
        cmap=cmap,
        units=data["units"],
        title=resolved_title,
        ax=ax,
        figsize=figsize,
    )


def create_simulation_movie_from_netcdf(
    nc_path: str | Path,
    *,
    output_path: str | Path,
    level_index: int = 0,
    release_index: int = 0,
    variable_names: Sequence[str] = ("ch4_mixing_ratio", "ch4_concentration"),
    fps: int = 5,
    use_log_scale: bool = False,
    nonzero_scaling: bool = True,
    cmap: str = "magma",
    title_prefix: str | None = None,
    figsize: tuple = (7, 5),
):
    """Load simulation NetCDF and create a concentration movie."""
    data = load_simulation_netcdf(nc_path, variable_names=variable_names)
    resolved_title = title_prefix or f"{data['variable_name']}"
    return create_simulation_movie(
        data["concentration"],
        output_path=output_path,
        dim_names=data.get("dimensions"),
        level_index=level_index,
        release_index=release_index,
        lons=data["lons"],
        lats=data["lats"],
        time_labels=data["time_labels"],
        fps=fps,
        use_log_scale=use_log_scale,
        nonzero_scaling=nonzero_scaling,
        cmap=cmap,
        units=data["units"],
        title_prefix=resolved_title,
        figsize=figsize,
    )
