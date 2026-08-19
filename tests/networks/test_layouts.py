"""Tests for enforceflux.networks.layouts (M3)."""

from dataclasses import dataclass

import numpy as np
import pytest

from enforceflux.networks.layouts import nested_sequence, random, space_filling


@dataclass(frozen=True)
class _Grid:
    nx: int
    ny: int
    dx_m: float
    origin_x_m: float
    origin_y_m: float


DOMAIN = _Grid(nx=64, ny=64, dx_m=50.0, origin_x_m=0.0, origin_y_m=0.0)


def _extent(d: _Grid) -> tuple[float, float]:
    return d.nx * d.dx_m, d.ny * d.dx_m


def test_space_filling_covers_domain():
    N = 16
    rng = np.random.default_rng(0)
    recs = space_filling(N, DOMAIN, rng)
    assert len(recs) == N
    pts = np.array([[r.x_m, r.y_m] for r in recs])
    # All inside the domain.
    Lx, Ly = _extent(DOMAIN)
    assert (pts[:, 0] >= 0).all() and (pts[:, 0] <= Lx).all()
    assert (pts[:, 1] >= 0).all() and (pts[:, 1] <= Ly).all()

    # Max nearest-neighbour distance bound: for N points on a square domain
    # covered by a near-square grid, the worst nearest-neighbour spacing is
    # bounded by ~ Lx / sqrt(N) * sqrt(2) (diagonal of one cell). Allow a small
    # slack factor to keep the assertion robust to grid clipping when N is not
    # a perfect square.
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    theoretical = Lx / np.sqrt(N) * np.sqrt(2) * 1.5
    assert nn.max() <= theoretical


def test_nested_sequence_is_nested():
    def layout(n, dom, rng):
        return space_filling(n, dom, rng)

    seq = nested_sequence([2, 4, 8], layout, seed=7, domain=DOMAIN)
    ids2 = [r.id for r in seq[2]]
    ids4 = [r.id for r in seq[4]]
    ids8 = [r.id for r in seq[8]]
    # Prefix nesting: each smaller layout is the prefix of the larger.
    assert ids2 == ids8[:2]
    assert ids4 == ids8[:4]
    # Positional nesting as well.
    xy8 = [(r.x_m, r.y_m) for r in seq[8]]
    xy4 = [(r.x_m, r.y_m) for r in seq[4]]
    xy2 = [(r.x_m, r.y_m) for r in seq[2]]
    assert xy2 == xy8[:2]
    assert xy4 == xy8[:4]


def test_random_reproducible_with_seed():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = random(20, DOMAIN, rng1)
    b = random(20, DOMAIN, rng2)
    for ra, rb in zip(a, b):
        assert ra.x_m == pytest.approx(rb.x_m)
        assert ra.y_m == pytest.approx(rb.y_m)
    # Different seed → different draws.
    c = random(20, DOMAIN, np.random.default_rng(124))
    assert any(ra.x_m != rc.x_m or ra.y_m != rc.y_m for ra, rc in zip(a, c))
