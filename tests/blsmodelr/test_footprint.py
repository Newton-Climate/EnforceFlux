"""Unit tests for enforceflux.blsmodelr.footprint."""
from __future__ import annotations

import numpy as np
import pytest

from enforceflux.blsmodelr import BlsRunResult, BlsSource
from enforceflux.blsmodelr.footprint import (
    build_source_grid,
    build_source_polygons,
    jacobian_from_bls_result,
)


# ── build_source_grid ────────────────────────────────────────────────────


def test_grid_count_and_areas():
    sources, areas = build_source_grid(
        x_bounds=(0.0, 20.0), y_bounds=(0.0, 30.0), nx=2, ny=3
    )
    assert len(sources) == 6
    assert areas.shape == (6,)
    # each cell is 10 x 10 = 100 m^2
    assert np.allclose(areas, 100.0)


def test_grid_row_major_ordering_first_cell_is_xmin_ymin():
    sources, _ = build_source_grid(
        x_bounds=(0.0, 20.0), y_bounds=(0.0, 30.0), nx=2, ny=3
    )
    # k=0 -> (i=0, j=0) -> (x_min, y_min) corner cell
    first = sources[0]
    assert first.name == "cell_0000"
    assert first.polygon_xy[0] == (0.0, 0.0)
    assert first.polygon_xy[2] == (10.0, 10.0)
    # y is the fast axis: k=1 -> (i=0, j=1) -> (x_min, y_min+dy)
    assert sources[1].polygon_xy[0] == (0.0, 10.0)
    # k=ny -> (i=1, j=0) -> (x_min+dx, y_min)
    assert sources[3].polygon_xy[0] == (10.0, 0.0)


def test_grid_polygon_is_ccw_rectangle():
    sources, _ = build_source_grid(
        x_bounds=(0.0, 2.0), y_bounds=(0.0, 2.0), nx=1, ny=1
    )
    poly = sources[0].polygon_xy
    assert len(poly) == 4
    xs = np.array([p[0] for p in poly])
    ys = np.array([p[1] for p in poly])
    # Signed shoelace area is positive for CCW winding
    signed = 0.5 * (np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    assert signed > 0


def test_grid_rejects_bad_bounds():
    with pytest.raises(ValueError):
        build_source_grid(x_bounds=(1.0, 0.0), y_bounds=(0.0, 1.0), nx=1, ny=1)
    with pytest.raises(ValueError):
        build_source_grid(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), nx=0, ny=1)


# ── build_source_polygons ────────────────────────────────────────────────


def test_shoelace_unit_square():
    _, areas = build_source_polygons(
        polygons=[("unit", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]
    )
    assert np.isclose(areas[0], 1.0)


def test_shoelace_l_shape():
    # L-shape total area = 3 (2x2 minus 1x1 corner)
    l_poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    sources, areas = build_source_polygons(polygons=[("L", l_poly)])
    assert sources[0].name == "L"
    assert np.isclose(areas[0], 3.0)


def test_shoelace_cw_winding_still_positive():
    # Same square but wound clockwise; abs() should give area 1
    _, areas = build_source_polygons(
        polygons=[("cw", [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])]
    )
    assert np.isclose(areas[0], 1.0)


def test_areas_override_used_when_supplied():
    _, areas = build_source_polygons(
        polygons=[("a", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])],
        areas_m2=[42.0],
    )
    assert areas[0] == 42.0


def test_areas_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_source_polygons(
            polygons=[("a", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])],
            areas_m2=[1.0, 2.0],
        )


# ── jacobian_from_bls_result ─────────────────────────────────────────────


def _make_result(rows):
    """rows = list of (sensor, source, interval, cxe)."""
    sensor = np.array([r[0] for r in rows], dtype=object)
    source = np.array([r[1] for r in rows], dtype=object)
    interval = np.array([r[2] for r in rows], dtype=object)
    cxe = np.asarray([r[3] for r in rows], dtype=float)
    return BlsRunResult(
        sensor=sensor,
        source=source,
        interval=interval,
        cxe=cxe,
        cxe_se=np.zeros_like(cxe),
        n_particles_used=np.zeros(len(rows), dtype=int),
    )


def test_jacobian_happy_path_shape_and_order():
    rows = [
        ("s1", "src_a", "i0", 1.0),
        ("s1", "src_b", "i0", 2.0),
        ("s2", "src_a", "i0", 3.0),
        ("s2", "src_b", "i0", 4.0),
    ]
    g = jacobian_from_bls_result(
        result=_make_result(rows),
        sensor_order=["s1", "s2"],
        source_order=["src_a", "src_b"],
    )
    assert g.shape == (2, 2)
    assert np.allclose(g, [[1.0, 2.0], [3.0, 4.0]])


def test_jacobian_reorder_columns():
    rows = [
        ("s1", "src_a", "i0", 1.0),
        ("s1", "src_b", "i0", 2.0),
    ]
    g = jacobian_from_bls_result(
        result=_make_result(rows),
        sensor_order=["s1"],
        source_order=["src_b", "src_a"],
    )
    assert np.allclose(g, [[2.0, 1.0]])


def test_jacobian_mean_vs_sum_over_intervals():
    rows = [
        ("s1", "src_a", "i0", 2.0),
        ("s1", "src_a", "i1", 4.0),
    ]
    res = _make_result(rows)
    g_mean = jacobian_from_bls_result(
        result=res, sensor_order=["s1"], source_order=["src_a"], interval_reduce="mean"
    )
    g_sum = jacobian_from_bls_result(
        result=res, sensor_order=["s1"], source_order=["src_a"], interval_reduce="sum"
    )
    assert np.isclose(g_mean[0, 0], 3.0)
    assert np.isclose(g_sum[0, 0], 6.0)
    assert not np.isclose(g_mean[0, 0], g_sum[0, 0])


def test_jacobian_missing_pair_raises_with_list():
    rows = [("s1", "src_a", "i0", 1.0)]
    with pytest.raises(ValueError) as exc:
        jacobian_from_bls_result(
            result=_make_result(rows),
            sensor_order=["s1", "s2"],
            source_order=["src_a", "src_b"],
        )
    msg = str(exc.value)
    # missing: (s1, src_b), (s2, src_a), (s2, src_b)
    assert "src_b" in msg
    assert "s2" in msg


def test_jacobian_bad_reduce_raises():
    rows = [("s1", "src_a", "i0", 1.0)]
    with pytest.raises(ValueError):
        jacobian_from_bls_result(
            result=_make_result(rows),
            sensor_order=["s1"],
            source_order=["src_a"],
            interval_reduce="median",
        )


def test_jacobian_partial_interval_coverage_is_ok():
    # (s1, src_a) has 2 intervals, (s1, src_b) has 1 — no error, mean over present.
    rows = [
        ("s1", "src_a", "i0", 2.0),
        ("s1", "src_a", "i1", 4.0),
        ("s1", "src_b", "i0", 10.0),
    ]
    g = jacobian_from_bls_result(
        result=_make_result(rows),
        sensor_order=["s1"],
        source_order=["src_a", "src_b"],
    )
    assert np.isclose(g[0, 0], 3.0)
    assert np.isclose(g[0, 1], 10.0)


# ── smoke: BlsSource shape is preserved ──────────────────────────────────


def test_grid_returns_bls_source_instances():
    sources, _ = build_source_grid(
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), nx=1, ny=1
    )
    assert isinstance(sources[0], BlsSource)


# ── interval_reduce="none": one observation row per interval ──────────────


def _two_interval_rows():
    return [
        ("s1", "src_a", "i0", 1.0), ("s1", "src_b", "i0", 2.0),
        ("s2", "src_a", "i0", 3.0), ("s2", "src_b", "i0", 4.0),
        ("s1", "src_a", "i1", 10.0), ("s1", "src_b", "i1", 20.0),
        ("s2", "src_a", "i1", 30.0), ("s2", "src_b", "i1", 40.0),
    ]


def test_jacobian_none_keeps_the_interval_axis():
    g = jacobian_from_bls_result(
        result=_make_result(_two_interval_rows()),
        sensor_order=["s1", "s2"],
        source_order=["src_a", "src_b"],
        interval_reduce="none",
        interval_order=["i0", "i1"],
    )
    assert g.shape == (2, 2, 2)
    assert np.allclose(g[0], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(g[1], [[10.0, 20.0], [30.0, 40.0]])


def test_jacobian_none_follows_the_requested_interval_order():
    g = jacobian_from_bls_result(
        result=_make_result(_two_interval_rows()),
        sensor_order=["s1", "s2"],
        source_order=["src_a", "src_b"],
        interval_reduce="none",
        interval_order=["i1", "i0"],
    )
    assert np.allclose(g[0], [[10.0, 20.0], [30.0, 40.0]])


def test_jacobian_none_averages_to_the_mean_path():
    kw = dict(result=_make_result(_two_interval_rows()),
              sensor_order=["s1", "s2"], source_order=["src_a", "src_b"])
    stacked = jacobian_from_bls_result(
        **kw, interval_reduce="none", interval_order=["i0", "i1"])
    assert np.allclose(stacked.mean(axis=0),
                       jacobian_from_bls_result(**kw, interval_reduce="mean"))


def test_jacobian_none_requires_an_interval_order():
    with pytest.raises(ValueError, match="interval_order must name"):
        jacobian_from_bls_result(
            result=_make_result(_two_interval_rows()),
            sensor_order=["s1", "s2"], source_order=["src_a", "src_b"],
            interval_reduce="none",
        )


def test_jacobian_none_refuses_a_missing_interval():
    """A gap cannot be averaged over here; a zero row would be a fake datum."""
    rows = [r for r in _two_interval_rows() if not (r[0] == "s2" and r[2] == "i1")]
    with pytest.raises(ValueError, match="every"):
        jacobian_from_bls_result(
            result=_make_result(rows),
            sensor_order=["s1", "s2"], source_order=["src_a", "src_b"],
            interval_reduce="none", interval_order=["i0", "i1"],
        )
