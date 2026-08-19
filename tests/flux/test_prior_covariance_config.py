import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps"))

from enforceflux.source_fields.basis import save_mapping, uniform_coarse_basis
from enforceflux.source_fields.lognormal_gp import FieldGrid


def _stage(tmp_path):
    grid = FieldGrid(nx=4, ny=4, dx_m=100.0)
    mapping = uniform_coarse_basis(grid, coarsen=2)
    disp = tmp_path / "d"
    disp.mkdir()
    save_mapping(disp / "basis_mapping.npz", mapping)
    G = np.eye(4, 16)
    np.savez(disp / "jacobian.npz",
             G=G,
             row_labels=np.asarray([f"r{i}" for i in range(4)]),
             column_labels=np.asarray([f"c{i}" for i in range(16)]),
             units=np.asarray("x"))
    from netCDF4 import Dataset
    with Dataset(disp / "truth_field.nc", "w", format="NETCDF4") as ds:
        ds.createDimension("y", 4); ds.createDimension("x", 4)
        v = ds.createVariable("F_true", "f8", ("y", "x"))
        v[:, :] = 1.0
        va = ds.createVariable("cell_area_m2", "f8", ("y", "x"))
        va[:, :] = grid.cell_areas_m2()
        ds.setncattr("L_true_m", 100.0)
    (disp / "manifest.json").write_text(json.dumps({
        "contract_version": "1.0", "stage": "dispersion",
        "run_name": "t", "status": "complete",
        "output_dir": str(disp),
        "outputs": [
            {"path": "jacobian.npz", "role": "jacobian"},
            {"path": "basis_mapping.npz", "role": "basis_mapping"},
            {"path": "truth_field.nc", "role": "truth_field"},
        ],
    }))
    return disp


def test_gaussian_process_prior_used(tmp_path):
    from enforceflux.runs import read_upstream
    from flux_inputs import build_from_prebuilt_operator

    disp = _stage(tmp_path)
    up = read_upstream(disp)
    cfg = {"inversion": {"prior_covariance": {"model": "gaussian_process",
                                              "sigma_kg_s": 2.0, "L_B_m": 300.0}},
           "observations": {"default_sigma": 1.0}}
    (*_, Sa, diag) = build_from_prebuilt_operator(cfg, up)[-3:]
    (G, y, Se, names, _, meta, n_flux, xp, Sa, diag) = build_from_prebuilt_operator(cfg, up)
    assert Sa.ndim == 2
    assert Sa.shape == (4, 4)
    # Diagonal equals sigma^2.
    assert np.allclose(np.diag(Sa), 4.0)
    # Off-diagonals decay with distance.
    assert Sa[0, 1] > 0
    assert Sa[0, 3] < Sa[0, 1]


def test_diagonal_prior_when_not_gp(tmp_path):
    from enforceflux.runs import read_upstream
    from flux_inputs import build_from_prebuilt_operator

    disp = _stage(tmp_path)
    up = read_upstream(disp)
    cfg = {"inversion": {"prior_variance": 0.25,
                         "prior_covariance": {"model": "diagonal"}},
           "observations": {"default_sigma": 1.0}}
    (G, y, Se, names, _, meta, n_flux, xp, Sa, diag) = build_from_prebuilt_operator(cfg, up)
    assert Sa.ndim == 1
    assert np.allclose(Sa, 0.25)
