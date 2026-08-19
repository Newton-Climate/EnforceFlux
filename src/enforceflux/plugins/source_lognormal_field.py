"""Log-Gaussian source-field generator plugin (M2).

Expands a ``sources.generator: lognormal_field`` block into one ``RunSource``
per fine truth-grid cell, and stages ``truth_field.nc`` and
``basis_mapping.npz`` for the dispersion driver to drain into its RunDir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from enforceflux.core.base import ISourceModel
from enforceflux.source_fields.basis import (
    BasisMapping,
    save_mapping,
    uniform_coarse_basis,
)
from enforceflux.source_fields.lognormal_gp import (
    FieldGrid,
    LognormalFieldSpec,
    sample_lognormal_field,
)
from enforceflux.transport.run_config import RunSource


# Module-level buffer of pending on-disk writes; the dispersion driver drains
# it into the current RunDir immediately after `RunConfig` is built.
_pending_writes: list[tuple[str, Callable[[Path], None]]] = []


def drain_pending_writes(run_dir_root: Path) -> list[tuple[str, str]]:
    """Write staged artefacts under ``run_dir_root`` and return (relpath, role) list."""
    written: list[tuple[str, str]] = []
    while _pending_writes:
        relpath, writer = _pending_writes.pop(0)
        target = Path(run_dir_root) / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        writer(target)
        role = Path(relpath).stem
        written.append((relpath, role))
    return written


def clear_pending_writes() -> None:
    _pending_writes.clear()


def _spec_from_config(cfg: dict[str, Any]) -> LognormalFieldSpec:
    g = cfg["grid"]
    grid = FieldGrid(
        nx=int(g["nx"]),
        ny=int(g["ny"]),
        dx_m=float(g["dx_m"]),
        origin_x_m=float(g.get("origin_x_m", 0.0)),
        origin_y_m=float(g.get("origin_y_m", 0.0)),
    )
    cov = cfg.get("covariance") or {}
    return LognormalFieldSpec(
        grid=grid,
        Q_true_kg_s=float(cfg["Q_true_kg_s"]),
        L_m=float(cov.get("L_m", 500.0)),
        cv=float(cfg["cv"]),
        covariance=str(cov.get("model", "exponential")),
        matern_nu=float(cov.get("matern_nu", 1.5)),
        f_emit=cfg.get("f_emit"),
        seed=int(cfg.get("seed", 0)),
    )


def _write_truth_field(F: np.ndarray, grid: FieldGrid, L_m: float, cv: float,
                       Q_true_kg_s: float, covariance: str, seed: int
                       ) -> Callable[[Path], None]:
    def _writer(target: Path) -> None:
        from netCDF4 import Dataset

        with Dataset(target, "w", format="NETCDF4") as ds:
            ds.createDimension("y", grid.ny)
            ds.createDimension("x", grid.nx)
            xs, ys = grid.cell_centers()
            v_x = ds.createVariable("x", "f8", ("x",))
            v_y = ds.createVariable("y", "f8", ("y",))
            v_x[:] = xs
            v_y[:] = ys
            v_F = ds.createVariable("F_true", "f8", ("y", "x"))
            v_F[:, :] = F
            v_F.units = "kg s-1 per cell"
            v_area = ds.createVariable("cell_area_m2", "f8", ("y", "x"))
            v_area[:, :] = grid.cell_areas_m2()
            ds.setncattr("Q_true_kg_s", float(Q_true_kg_s))
            ds.setncattr("L_true_m", float(L_m))
            ds.setncattr("cv", float(cv))
            ds.setncattr("covariance_model", covariance)
            ds.setncattr("seed", int(seed))
            ds.setncattr("nx", int(grid.nx))
            ds.setncattr("ny", int(grid.ny))
            ds.setncattr("dx_m", float(grid.dx_m))
            ds.setncattr("origin_x_m", float(grid.origin_x_m))
            ds.setncattr("origin_y_m", float(grid.origin_y_m))

    return _writer


def _write_basis(mapping: BasisMapping) -> Callable[[Path], None]:
    def _writer(target: Path) -> None:
        save_mapping(target, mapping)
    return _writer


class LognormalFieldSource(ISourceModel):
    """Source plugin that samples a lognormal random field and returns one
    :class:`RunSource` per fine cell."""

    def build_sources(self, config: dict[str, Any], domain: Any = None):  # type: ignore[override]
        spec = _spec_from_config(config)
        rng = np.random.default_rng(spec.seed)
        F = sample_lognormal_field(spec, rng)
        coarsen = int((config.get("basis") or {}).get("coarsen", 1))
        mapping = uniform_coarse_basis(spec.grid, coarsen=coarsen)

        _pending_writes.append((
            "truth_field.nc",
            _write_truth_field(
                F, spec.grid, spec.L_m, spec.cv, spec.Q_true_kg_s,
                spec.covariance, spec.seed,
            ),
        ))
        _pending_writes.append(("basis_mapping.npz", _write_basis(mapping)))

        alt_m = float(config.get("alt_m", 2.0))
        prior_mean = float((config.get("prior") or {}).get("mean_kg_s_per_cell", 0.0))
        areas = spec.grid.cell_areas_m2()
        xs, ys = spec.grid.cell_centers()

        sources: list[RunSource] = []
        for j in range(spec.grid.ny):
            for i in range(spec.grid.nx):
                sources.append(
                    RunSource(
                        id=f"cell_{j * spec.grid.nx + i:05d}",
                        x_m=float(xs[i]),
                        y_m=float(ys[j]),
                        emission_rate_kg_s=float(F[j, i] * areas[j, i]),
                        altitude_m=alt_m,
                        prior_mean_kg_s=prior_mean,
                        prior_std_kg_s=None,
                    )
                )
        return sources
