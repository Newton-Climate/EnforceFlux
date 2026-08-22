"""Source discretization and Jacobian assembly for bLSmodelR.

Two responsibilities:

1. Turn a source region (rectangular grid or arbitrary polygons in metres) into
   the list of :class:`BlsSource` cells that :class:`BlsWrapper` expects, plus
   the parallel array of cell areas (m^2).
2. Reshape the long-form :class:`BlsRunResult` returned by the wrapper into
   the ``(n_sensors, n_sources)`` Jacobian block ``g`` used by the framework's
   ``ForwardModelResult``.

Kept intentionally dependency-free (numpy + stdlib) so it can be imported
without pulling in the plugin/transport layers.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from enforceflux.blsmodelr.wrapper import BlsRunResult, BlsSource


# ── source discretization ────────────────────────────────────────────────


def build_source_grid(
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    nx: int,
    ny: int,
    name_prefix: str = "cell",
) -> tuple[list[BlsSource], np.ndarray]:
    """Discretize a rectangular region into ``nx * ny`` cell polygons.

    The region ``[x_bounds] x [y_bounds]`` is split into ``nx`` columns and
    ``ny`` rows of equal-size rectangles. The centre of cell ``(i, j)`` sits
    at ``(x_min + (i+0.5)*dx, y_min + (j+0.5)*dy)`` — i.e. the rectangles
    are centred on grid points, not corners.

    Ordering (row-major, y fastest):
        k = i * ny + j    with  i in [0, nx),  j in [0, ny)

    So ``sources[0]`` corresponds to ``(i=0, j=0)`` — the ``(x_min, y_min)``
    corner cell — and ``cell_area_m2[k]`` is the area of ``sources[k]``.
    Every cell has the same area for a regular grid, but the array is
    returned per-cell for a uniform interface with :func:`build_source_polygons`.

    Each polygon is a 4-corner rectangle wound CCW, and names are
    zero-padded to 4 digits (``f'{name_prefix}_{k:04d}'``).
    """
    if nx < 1 or ny < 1:
        raise ValueError(f"nx and ny must be >= 1, got nx={nx}, ny={ny}")
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    if not (x_max > x_min and y_max > y_min):
        raise ValueError(
            f"x_bounds and y_bounds must be increasing, got {x_bounds}, {y_bounds}"
        )
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    area = dx * dy

    sources: list[BlsSource] = []
    areas = np.full(nx * ny, area, dtype=float)
    for i in range(nx):
        x0 = x_min + i * dx
        x1 = x0 + dx
        for j in range(ny):
            y0 = y_min + j * dy
            y1 = y0 + dy
            k = i * ny + j
            # CCW starting at (x0, y0)
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            sources.append(BlsSource(name=f"{name_prefix}_{k:04d}", polygon_xy=poly))
    return sources, areas


def _shoelace_area(vertices: Iterable[tuple[float, float]]) -> float:
    verts = list(vertices)
    if len(verts) < 3:
        raise ValueError(f"polygon needs >= 3 vertices, got {len(verts)}")
    xs = np.asarray([v[0] for v in verts], dtype=float)
    ys = np.asarray([v[1] for v in verts], dtype=float)
    return 0.5 * float(np.abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def build_source_polygons(
    *,
    polygons: list[tuple[str, list[tuple[float, float]]]],
    areas_m2: list[float] | None = None,
) -> tuple[list[BlsSource], np.ndarray]:
    """Wrap a user-supplied ``(name, vertices)`` polygon list.

    Vertices are in metres in the same local frame as the sensors. If
    ``areas_m2`` is omitted, each polygon's area is computed with the
    shoelace formula (winding-agnostic, via ``abs``).
    """
    if areas_m2 is not None and len(areas_m2) != len(polygons):
        raise ValueError(
            f"areas_m2 length {len(areas_m2)} does not match polygons "
            f"length {len(polygons)}"
        )
    sources = [BlsSource(name=name, polygon_xy=list(verts)) for name, verts in polygons]
    if areas_m2 is None:
        areas = np.asarray([_shoelace_area(verts) for _, verts in polygons], dtype=float)
    else:
        areas = np.asarray(areas_m2, dtype=float)
    return sources, areas


# ── Jacobian assembly ────────────────────────────────────────────────────


def jacobian_from_bls_result(
    *,
    result: BlsRunResult,
    sensor_order: list[str],
    source_order: list[str],
    interval_reduce: str = "mean",
) -> np.ndarray:
    """Reshape the long-form CxE table into a ``(n_sensors, n_sources)`` matrix.

    Rows follow ``sensor_order``; columns follow ``source_order``. For each
    ``(sensor, source)`` pair, the CxE values across the intervals present in
    ``result`` are collapsed via ``interval_reduce``:

    * ``"mean"`` — arithmetic mean of the intervals that appear.
    * ``"sum"``  — sum over the intervals that appear (use this when the
      intervals are contiguous time bins and you want the total
      residence-time-weighted response).

    If a ``(sensor, source)`` pair has **no** intervals at all in the result,
    a :class:`ValueError` is raised listing the missing pairs. Partial
    coverage (some intervals present, some not) is not an error: the reduction
    is over the present ones only.

    Units are unchanged from :attr:`BlsRunResult.cxe`.
    """
    if interval_reduce not in ("mean", "sum"):
        raise ValueError(
            f"interval_reduce must be 'mean' or 'sum', got {interval_reduce!r}"
        )

    # Bucket cxe values by (sensor, source), keyed for existence check.
    buckets: dict[tuple[str, str], list[float]] = {}
    for s, src, val in zip(result.sensor, result.source, result.cxe):
        key = (str(s), str(src))
        buckets.setdefault(key, []).append(float(val))

    missing: list[tuple[str, str]] = []
    g = np.zeros((len(sensor_order), len(source_order)), dtype=float)
    for i, sens in enumerate(sensor_order):
        for j, src in enumerate(source_order):
            vals = buckets.get((sens, src))
            if not vals:
                missing.append((sens, src))
                continue
            arr = np.asarray(vals, dtype=float)
            g[i, j] = arr.sum() if interval_reduce == "sum" else arr.mean()

    if missing:
        raise ValueError(
            f"jacobian_from_bls_result: no intervals found for "
            f"{len(missing)} (sensor, source) pair(s): {missing}"
        )
    return g
