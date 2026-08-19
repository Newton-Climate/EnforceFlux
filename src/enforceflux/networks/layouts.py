"""Receptor-network layout generators for OSSE experiments (M3).

Layouts return ``list[RunReceptor]`` from the transport run-config contract
without modifying it. ``domain`` is duck-typed on a ``FieldGrid``-like object
(``origin_x_m``, ``origin_y_m``, ``nx``, ``ny``, ``dx_m``) so this milestone
does not hard-depend on M1's ``source_fields.lognormal_gp.FieldGrid`` landing
first; the real ``FieldGrid`` satisfies the protocol.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from enforceflux.transport.run_config import RunReceptor


@runtime_checkable
class _DomainLike(Protocol):
    origin_x_m: float
    origin_y_m: float
    nx: int
    ny: int
    dx_m: float


def _bounds(domain: _DomainLike) -> tuple[float, float, float, float]:
    x0 = float(domain.origin_x_m)
    y0 = float(domain.origin_y_m)
    x1 = x0 + float(domain.nx) * float(domain.dx_m)
    y1 = y0 + float(domain.ny) * float(domain.dx_m)
    return x0, x1, y0, y1


def _receptor(idx: int, x: float, y: float) -> RunReceptor:
    return RunReceptor(id=f"r_{idx:05d}", x_m=float(x), y_m=float(y))


def space_filling(
    N: int, domain: _DomainLike, rng: np.random.Generator
) -> list[RunReceptor]:
    """Deterministic space-filling layout.

    Uses a near-square grid clipped to N points, laid out on the interior of
    the domain so all receptors are inside the bounds. ``rng`` is accepted for
    API symmetry with :func:`random` and :func:`nested_sequence` but only used
    for tie-breaking when N is not a perfect square.
    """
    if N <= 0:
        return []
    x0, x1, y0, y1 = _bounds(domain)
    # Choose a grid nx_g * ny_g >= N with as square an aspect as possible.
    nx_g = int(np.ceil(np.sqrt(N)))
    ny_g = int(np.ceil(N / nx_g))
    # Centers of a nx_g by ny_g partition of the domain interior.
    xs = x0 + (np.arange(nx_g) + 0.5) * (x1 - x0) / nx_g
    ys = y0 + (np.arange(ny_g) + 0.5) * (y1 - y0) / ny_g
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    if pts.shape[0] > N:
        # Drop excess points from the trailing row, permuted by rng for
        # determinism given a fixed seed.
        perm = rng.permutation(pts.shape[0])
        keep = np.sort(perm[:N])
        pts = pts[keep]
    return [_receptor(i, x, y) for i, (x, y) in enumerate(pts)]


def random(
    N: int, domain: _DomainLike, rng: np.random.Generator
) -> list[RunReceptor]:
    """Uniform random layout inside the domain."""
    if N <= 0:
        return []
    x0, x1, y0, y1 = _bounds(domain)
    xs = rng.uniform(x0, x1, size=N)
    ys = rng.uniform(y0, y1, size=N)
    return [_receptor(i, x, y) for i, (x, y) in enumerate(zip(xs, ys))]


def nested_sequence(
    ns: Sequence[int],
    layout: Callable[[int, _DomainLike, np.random.Generator], list[RunReceptor]],
    seed: int,
    domain: _DomainLike | None = None,
) -> dict[int, list[RunReceptor]]:
    """Return ``{n: receptors}`` where each larger layout is a superset.

    The largest ``n`` is drawn from the ``layout`` function; smaller layouts
    are prefixes of that draw so nesting holds by construction. ``domain`` is
    optional only so callers with partial-application patterns can pre-bind
    it into ``layout`` — if ``domain`` is ``None``, ``layout`` must already
    have the domain bound.
    """
    ns_sorted = sorted(set(int(n) for n in ns))
    if not ns_sorted:
        return {}
    rng = np.random.default_rng(seed)
    n_max = ns_sorted[-1]
    if domain is None:
        full = layout(n_max, rng)  # type: ignore[call-arg]
    else:
        full = layout(n_max, domain, rng)
    out: dict[int, list[RunReceptor]] = {}
    for n in ns_sorted:
        out[n] = list(full[:n])
    return out
