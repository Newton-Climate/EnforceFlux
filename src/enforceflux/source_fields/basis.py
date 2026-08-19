"""Truth-grid to inversion-basis projector `W` and mass-conserving aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .lognormal_gp import FieldGrid


@dataclass(frozen=True)
class BasisMapping:
    # W is the coarse-from-fine aggregator: entries are 1 on the fine cells that
    # belong to each coarse cell, 0 elsewhere, so W @ (F_fine * area_fine) is the
    # total emission (kg/s) per coarse cell — memo eq. 33.
    W: np.ndarray                    # (n_coarse, n_fine)
    fine_cell_areas_m2: np.ndarray   # (n_fine,)
    coarse_cell_areas_m2: np.ndarray # (n_coarse,)
    coarse_centers_m: np.ndarray     # (n_coarse, 2)


def uniform_coarse_basis(fine_grid: FieldGrid, coarsen: int) -> BasisMapping:
    if coarsen < 1:
        raise ValueError("coarsen must be >= 1.")
    if fine_grid.nx % coarsen or fine_grid.ny % coarsen:
        raise ValueError(
            f"Fine grid {fine_grid.nx}x{fine_grid.ny} must divide evenly by coarsen={coarsen}."
        )
    nxc, nyc = fine_grid.nx // coarsen, fine_grid.ny // coarsen
    n_fine = fine_grid.nx * fine_grid.ny
    n_coarse = nxc * nyc

    W = np.zeros((n_coarse, n_fine), dtype=float)
    for iy in range(fine_grid.ny):
        for ix in range(fine_grid.nx):
            fine_idx = iy * fine_grid.nx + ix
            coarse_idx = (iy // coarsen) * nxc + (ix // coarsen)
            W[coarse_idx, fine_idx] = 1.0

    fine_areas = np.full(n_fine, fine_grid.dx_m * fine_grid.dx_m, dtype=float)
    coarse_dx = fine_grid.dx_m * coarsen
    coarse_areas = np.full(n_coarse, coarse_dx * coarse_dx, dtype=float)

    xc = fine_grid.origin_x_m + (np.arange(nxc) + 0.5) * coarse_dx
    yc = fine_grid.origin_y_m + (np.arange(nyc) + 0.5) * coarse_dx
    XX, YY = np.meshgrid(xc, yc, indexing="xy")
    centers = np.column_stack([XX.ravel(), YY.ravel()])

    return BasisMapping(
        W=W,
        fine_cell_areas_m2=fine_areas,
        coarse_cell_areas_m2=coarse_areas,
        coarse_centers_m=centers,
    )


def polygon_basis(fine_grid: FieldGrid, polygons: list[Any]) -> BasisMapping:
    raise NotImplementedError(
        "polygon_basis is deferred; only uniform_coarse_basis is available in M1."
    )


def project_flux_to_coarse(F_fine: np.ndarray, mapping: BasisMapping) -> np.ndarray:
    """Emission-conserving aggregation: x_coarse = W @ (F_fine * area_fine)."""
    flat = F_fine.ravel()
    if flat.shape[0] != mapping.fine_cell_areas_m2.shape[0]:
        raise ValueError(
            f"F_fine has {flat.shape[0]} cells but mapping expects "
            f"{mapping.fine_cell_areas_m2.shape[0]}."
        )
    emissions = flat * mapping.fine_cell_areas_m2
    x_coarse = mapping.W @ emissions
    total_fine = float(emissions.sum())
    total_coarse = float(x_coarse.sum())
    if total_fine != 0.0:
        assert abs(total_coarse - total_fine) / abs(total_fine) < 1e-12, (
            f"Mass not conserved: fine={total_fine}, coarse={total_coarse}"
        )
    return x_coarse


def save_mapping(path: Path, mapping: BasisMapping) -> None:
    np.savez(
        Path(path),
        W=mapping.W,
        fine_cell_areas_m2=mapping.fine_cell_areas_m2,
        coarse_cell_areas_m2=mapping.coarse_cell_areas_m2,
        coarse_centers_m=mapping.coarse_centers_m,
    )


def load_mapping(path: Path) -> BasisMapping:
    data = np.load(Path(path))
    return BasisMapping(
        W=data["W"],
        fine_cell_areas_m2=data["fine_cell_areas_m2"],
        coarse_cell_areas_m2=data["coarse_cell_areas_m2"],
        coarse_centers_m=data["coarse_centers_m"],
    )
