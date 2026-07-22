"""``AermodModel`` — the Python-driven entry point to the AERMOD-style model.

One object turns a :class:`~enforceflux.aermod.config.AermodConfig` plus a set
of sources into either

* a **Jacobian** ``G`` (receptor × source) of concentration per unit emission,
  which is what the inversion machinery consumes, or
* a **concentration field** on a receptor grid, for forward simulation and
  plume visualization,

and, because the kernel is JAX, into **gradients of either** with respect to
any input — see :meth:`AermodModel.response_fn` and
:meth:`AermodModel.sensitivity_to_met`.

The heavy lifting is a single scalar kernel
(:func:`~enforceflux.aermod.dispersion.chi_over_q`) that is ``vmap``-ed over the
source, receptor, and meteorology axes and ``jit``-compiled once per shape, so a
grid of 10⁴ receptors over 24 hours costs one fused kernel launch rather than a
Python loop.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from enforceflux.aermod.config import (
    AermodConfig,
    Receptor,
    ReceptorGrid,
    StackParameters,
)
from enforceflux.aermod.dispersion import (
    ReceptorArray,
    SourceArray,
    StackArray,
    chi_over_q,
)
from enforceflux.aermod.meteorology import MetState, stack_met_states


@dataclass(frozen=True)
class GridField:
    """Concentration field on a regular receptor grid.

    ``values`` has shape ``(n_met, n_y, n_x)`` — one horizontal slice per
    meteorological hour, at the grid's receptor height.
    """

    x: np.ndarray
    y: np.ndarray
    z: float
    values: np.ndarray
    units: str
    meta: dict[str, Any]


def receptors_from_grid(grid: ReceptorGrid) -> tuple[list[Receptor], np.ndarray, np.ndarray]:
    """Expand a :class:`ReceptorGrid` into receptors plus its x/y axes."""
    nx = int(np.floor((grid.x_max - grid.x_min) / grid.spacing_m)) + 1
    ny = int(np.floor((grid.y_max - grid.y_min) / grid.spacing_m)) + 1
    xs = grid.x_min + grid.spacing_m * np.arange(nx)
    ys = grid.y_min + grid.spacing_m * np.arange(ny)
    receptors = [
        Receptor(id=f"grid_{iy:04d}_{ix:04d}", x=float(x), y=float(y), z=grid.height_m)
        for iy, y in enumerate(ys)
        for ix, x in enumerate(xs)
    ]
    return receptors, xs, ys


def receptors_from_instruments(
    instruments: Iterable[Any], path_samples: int = 1
) -> list[Receptor]:
    """Build receptors from :class:`~enforceflux.instrument.Instrument` deployments.

    Point instruments become one receptor. With ``path_samples > 1``, an
    instrument that carries open-path geometry (``path_length_m`` and
    ``path_bearing_deg``, used by the ``line_integral`` operators) is sampled at
    evenly spaced points along its path; the samples share a ``group`` so
    :class:`AermodModel` averages them back into a single path-averaged
    observation.
    """
    receptors: list[Receptor] = []
    for inst in instruments:
        path_length = float(getattr(inst, "path_length_m", 0.0) or 0.0)
        is_path = path_samples > 1 and path_length > 0.0
        if not is_path:
            receptors.append(
                Receptor(
                    id=str(inst.id),
                    x=float(inst.x),
                    y=float(inst.y),
                    z=float(getattr(inst, "z", 0.0)),
                    group=str(inst.id),
                )
            )
            continue

        bearing = np.deg2rad(float(getattr(inst, "path_bearing_deg", 0.0)))
        # Bearing is clockwise from north; the path is centred on (x, y).
        ux, uy = np.sin(bearing), np.cos(bearing)
        offsets = np.linspace(-0.5, 0.5, path_samples) * path_length
        weight = 1.0 / path_samples
        for k, offset in enumerate(offsets):
            receptors.append(
                Receptor(
                    id=f"{inst.id}_p{k:02d}",
                    x=float(inst.x) + float(offset) * ux,
                    y=float(inst.y) + float(offset) * uy,
                    z=float(getattr(inst, "z", 0.0)),
                    weight=weight,
                    group=str(inst.id),
                )
            )
    return receptors


class AermodModel:
    """AERMOD-style dispersion driven entirely from Python objects.

    Parameters
    ----------
    config:
        The run specification. Meteorology, release geometry, unit convention,
        and the numerical options all live here.
    """

    def __init__(self, config: AermodConfig) -> None:
        self.config = config
        self._met_state: MetState = stack_met_states(config.met)
        kernel = functools.partial(_kernel_with_options, options=config.options)
        # source axis, then receptor axis, then meteorology axis.
        per_source = jax.vmap(kernel, in_axes=(None, 0, 0, None))
        per_receptor = jax.vmap(per_source, in_axes=(0, None, None, None))
        self._response = jax.jit(jax.vmap(per_receptor, in_axes=(None, None, None, 0)))

    # ── Core evaluation ──────────────────────────────────────────────────────

    @property
    def met_state(self) -> MetState:
        """Derived boundary-layer parameters, one entry per configured hour."""
        return self._met_state

    def response_fn(self):
        """The raw jitted, differentiable response function.

        Signature ``(ReceptorArray, SourceArray, StackArray, MetState) ->
        (n_met, n_receptor, n_source)`` in s m⁻³ (χ/Q, *before* the configured
        unit scaling). Every argument is a JAX pytree of arrays, so this can be
        passed straight to ``jax.grad``/``jax.jacobian`` with respect to source
        positions, stack geometry, or meteorology::

            fn = model.response_fn()
            d_conc_d_met = jax.grad(lambda m: fn(rec, src, stk, m).sum())(model.met_state)
        """
        return self._response

    def unit_response(
        self, sources: Sequence[Any], receptors: Sequence[Receptor] | None = None
    ) -> np.ndarray:
        """χ/Q for every (hour, receptor, source), in ``config.concentration_units``.

        Shape ``(n_met, n_receptor, n_source)``.
        """
        receptors = self._resolve_receptors(receptors)
        rec = _receptor_arrays(receptors)
        src, stk = self._source_arrays(sources)
        raw = np.asarray(self._response(rec, src, stk, self._met_state))
        return raw * self.config.unit_scale

    # ── Inversion-facing Jacobian ────────────────────────────────────────────

    def jacobian(
        self, sources: Sequence[Any], receptors: Sequence[Receptor] | None = None
    ) -> np.ndarray:
        """Observation × source Jacobian ``∂y/∂Q``.

        The plume is always solved separately for every meteorological hour —
        nothing here averages the meteorology. ``config.reduce`` only decides
        what happens to the resulting hour axis:

        ``"stack"``
            Every (hour, receptor) pair becomes its own observation row →
            ``(n_met · n_obs, n_source)``. This is the time-resolved form: it
            keeps each hour as an independent measurement, which is where the
            information content of a varying wind actually lives. Row order is
            hour-major; :meth:`observation_labels` names the rows.
        ``"mean"`` (default)
            ``(n_obs, n_source)``, the Jacobian of a *period-mean*
            concentration. Correct only when one observation genuinely averages
            the whole window; it discards the time variation otherwise.
        ``"max"``
            ``(n_obs, n_source)``, the worst-case hour per receptor — a
            screening statistic, not an observation operator.
        ``"none"``
            ``(n_met, n_obs, n_source)``, unreduced, for callers doing their own
            time handling.

        Receptors sharing a ``group`` are combined with their ``weight`` into a
        single row first (open-path averaging).
        """
        receptors = self._resolve_receptors(receptors)
        response = self.unit_response(sources, receptors)  # (n_met, n_rec, n_src)
        grouped = _apply_groups(response, receptors)  # (n_met, n_obs, n_src)
        # Emission units: the caller's flux times this factor is kg s-1.
        grouped = grouped * self.config.emission_scale_to_kg_s

        if self.config.reduce == "none":
            return grouped
        if self.config.reduce == "stack":
            n_met, n_obs, n_source = grouped.shape
            return grouped.reshape(n_met * n_obs, n_source)
        if self.config.reduce == "max":
            return grouped.max(axis=0)
        return grouped.mean(axis=0)

    def observation_labels(
        self, receptors: Sequence[Receptor] | None = None
    ) -> list[tuple[str | None, str]]:
        """``(timestamp, receptor_id)`` for each row of a ``reduce="stack"`` Jacobian.

        Stacking makes rows anonymous — this keeps the bookkeeping honest, in
        the same hour-major order the Jacobian uses.
        """
        receptors = self._resolve_receptors(receptors)
        groups = _group_ids(receptors)
        return [(met.timestamp, group) for met in self.config.met for group in groups]

    # ── Forward simulation ───────────────────────────────────────────────────

    def concentrations(
        self,
        sources: Sequence[Any],
        receptors: Sequence[Receptor] | None = None,
        emissions: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Total concentration at each receptor, summed over sources.

        Shape ``(n_met, n_receptor)``. ``emissions`` defaults to each source's
        ``flux_true`` (scaled by ``config.emission_scale_to_kg_s``).
        """
        receptors = self._resolve_receptors(receptors)
        response = self.unit_response(sources, receptors)
        q = self._emissions(sources, emissions)
        return response @ q

    def grid_field(
        self,
        sources: Sequence[Any],
        grid: ReceptorGrid | None = None,
        emissions: Sequence[float] | None = None,
    ) -> GridField:
        """Concentration field over a regular grid, one slice per hour."""
        grid = grid or self.config.grid
        if grid is None:
            raise ValueError(
                "grid_field needs a ReceptorGrid, either passed in or set as "
                "AermodConfig.grid."
            )
        receptors, xs, ys = receptors_from_grid(grid)
        values = self.concentrations(sources, receptors, emissions)
        field = np.asarray(values).reshape(len(self.config.met), len(ys), len(xs))
        return GridField(
            x=xs,
            y=ys,
            z=grid.height_m,
            values=field,
            units=self.config.concentration_units,
            meta={
                "model": "aermod",
                "n_sources": len(list(sources)),
                "n_met": len(self.config.met),
                "spacing_m": grid.spacing_m,
            },
        )

    # ── Differentiability helpers ────────────────────────────────────────────

    def sensitivity_to_met(
        self,
        sources: Sequence[Any],
        receptors: Sequence[Receptor] | None = None,
        emissions: Sequence[float] | None = None,
    ) -> MetState:
        """∂(total concentration)/∂(meteorological parameter), per hour.

        Returns a :class:`MetState` whose fields hold the derivative of the
        summed receptor concentration with respect to that parameter — the kind
        of thing that is painful with a Fortran model and free here.
        """
        receptors = self._resolve_receptors(receptors)
        rec = _receptor_arrays(receptors)
        src, stk = self._source_arrays(sources)
        q = jnp.asarray(self._emissions(sources, emissions))
        scale = self.config.unit_scale

        def total(met_state: MetState) -> jnp.ndarray:
            response = self._response(rec, src, stk, met_state) * scale
            return jnp.sum(response @ q)

        return jax.grad(total)(self._met_state)

    # ── Internals ────────────────────────────────────────────────────────────

    def _resolve_receptors(
        self, receptors: Sequence[Receptor] | None
    ) -> list[Receptor]:
        resolved = list(receptors) if receptors is not None else list(self.config.receptors)
        if not resolved:
            raise ValueError(
                "No receptors: pass them explicitly, or set AermodConfig.receptors."
            )
        return resolved

    def _source_arrays(self, sources: Sequence[Any]) -> tuple[SourceArray, StackArray]:
        sources = list(sources)
        if not sources:
            raise ValueError("AermodModel needs at least one source")
        stacks = [self._stack_for(s) for s in sources]
        source = SourceArray(
            x=jnp.asarray([float(s.x) for s in sources]),
            y=jnp.asarray([float(s.y) for s in sources]),
            base_elevation=jnp.asarray([float(getattr(s, "z", 0.0)) for s in sources]),
        )
        stack = StackArray(
            height=jnp.asarray([st.height_m for st in stacks]),
            diameter=jnp.asarray([st.diameter_m for st in stacks]),
            exit_velocity=jnp.asarray([st.exit_velocity_m_s for st in stacks]),
            exit_temperature=jnp.asarray([st.exit_temperature_k for st in stacks]),
        )
        return source, stack

    def _stack_for(self, source: Any) -> StackParameters:
        return self.config.stack_for(str(getattr(source, "id", "")))

    def _emissions(
        self, sources: Sequence[Any], emissions: Sequence[float] | None
    ) -> np.ndarray:
        sources = list(sources)
        if emissions is None:
            try:
                values = [float(s.flux_true) for s in sources]
            except AttributeError:
                raise ValueError(
                    "Sources carry no 'flux_true'; pass emissions=[...] explicitly."
                ) from None
        else:
            values = [float(v) for v in emissions]
            if len(values) != len(sources):
                raise ValueError(
                    f"emissions has {len(values)} entries for {len(sources)} sources"
                )
        return np.asarray(values) * self.config.emission_scale_to_kg_s


def _kernel_with_options(receptor, source, stack, met_state, *, options):
    return chi_over_q(receptor, source, stack, met_state, options)


def _receptor_arrays(receptors: Sequence[Receptor]) -> ReceptorArray:
    return ReceptorArray(
        x=jnp.asarray([r.x for r in receptors]),
        y=jnp.asarray([r.y for r in receptors]),
        z=jnp.asarray([r.z for r in receptors]),
    )


def _group_ids(receptors: Sequence[Receptor]) -> list[str]:
    """Observation ids in row order: each receptor's ``group``, else its ``id``."""
    groups: list[str] = []
    for r in receptors:
        key = r.group if r.group is not None else r.id
        if key not in groups:
            groups.append(key)
    return groups


def _apply_groups(response: np.ndarray, receptors: Sequence[Receptor]) -> np.ndarray:
    """Weighted-average receptors sharing a ``group`` into single observations."""
    groups = _group_ids(receptors)
    index = {key: i for i, key in enumerate(groups)}
    if len(groups) == len(receptors):
        return response

    weights = np.zeros((len(groups), len(receptors)))
    for j, r in enumerate(receptors):
        key = r.group if r.group is not None else r.id
        weights[index[key], j] = r.weight
    # Normalize so each observation is a weighted mean, not a sum.
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / np.where(row_sums == 0.0, 1.0, row_sums)
    return np.einsum("ij,mjs->mis", weights, response)
