import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Make apps/ importable for flux_inputs helpers.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps"))

from enforceflux.source_fields.basis import save_mapping, uniform_coarse_basis
from enforceflux.source_fields.lognormal_gp import FieldGrid, LognormalFieldSpec, sample_lognormal_field


def _stage_dispersion(tmp_path, L_true_m=500.0):
    grid = FieldGrid(nx=4, ny=4, dx_m=200.0, origin_x_m=-400.0, origin_y_m=-400.0)
    spec = LognormalFieldSpec(
        grid=grid, Q_true_kg_s=1.0e-2, L_m=L_true_m, cv=0.1, seed=1
    )
    rng = np.random.default_rng(1)
    F = sample_lognormal_field(spec, rng)
    mapping = uniform_coarse_basis(grid, coarsen=2)
    # Simple synthetic H_fine: 6 observations x 16 fine cells with positive weights.
    rng2 = np.random.default_rng(2)
    G_fine = rng2.uniform(0.1, 1.0, size=(6, 16))

    # Persist as if produced by a dispersion RunDir.
    disp_root = tmp_path / "disp"
    disp_root.mkdir()
    np.savez(disp_root / "jacobian.npz",
             G=G_fine,
             row_labels=np.asarray(["r0", "r1", "r2", "r3", "r4", "r5"]),
             column_labels=np.asarray([f"cell_{i:05d}" for i in range(16)]),
             units=np.asarray("ng m-3 / (kg s-1)"))
    save_mapping(disp_root / "basis_mapping.npz", mapping)

    from netCDF4 import Dataset
    with Dataset(disp_root / "truth_field.nc", "w", format="NETCDF4") as ds:
        ds.createDimension("y", 4)
        ds.createDimension("x", 4)
        v = ds.createVariable("F_true", "f8", ("y", "x"))
        v[:, :] = F
        va = ds.createVariable("cell_area_m2", "f8", ("y", "x"))
        va[:, :] = grid.cell_areas_m2()
        ds.setncattr("L_true_m", float(L_true_m))

    manifest = {
        "contract_version": "1.0",
        "stage": "dispersion",
        "run_name": "test",
        "status": "complete",
        "output_dir": str(disp_root),
        "outputs": [
            {"path": "jacobian.npz", "role": "jacobian", "sha256": None},
            {"path": "basis_mapping.npz", "role": "basis_mapping", "sha256": None},
            {"path": "truth_field.nc", "role": "truth_field", "sha256": None},
        ],
    }
    (disp_root / "manifest.json").write_text(json.dumps(manifest))
    return disp_root, G_fine, mapping, F


def test_H_coarse_shape_matches_W(tmp_path):
    from enforceflux.runs import read_upstream
    from flux_inputs import build_from_prebuilt_operator

    disp_root, G_fine, mapping, _ = _stage_dispersion(tmp_path)
    up = read_upstream(disp_root)
    cfg = {"inversion": {"prior_covariance": {"model": "gaussian_process",
                                              "sigma_kg_s": 1.0e-3, "L_B_m": 800.0}},
           "observations": {"default_sigma": 0.1, "add_noise": False}}
    (G_coarse, y_obs, Se, source_names, _, _, n_flux,
     x_prior, Sa, diag) = build_from_prebuilt_operator(cfg, up)
    assert G_coarse.shape == (G_fine.shape[0], mapping.W.shape[0])
    assert Sa.shape == (mapping.W.shape[0], mapping.W.shape[0])
    assert n_flux == 1
    assert diag["L_B_m"] == 800.0
    assert diag["inverse_crime_flag"] is False


def test_inverse_crime_flag_when_L_B_equals_L_true(tmp_path):
    from enforceflux.runs import read_upstream
    from flux_inputs import build_from_prebuilt_operator

    disp_root, *_ = _stage_dispersion(tmp_path, L_true_m=500.0)
    up = read_upstream(disp_root)
    cfg = {"inversion": {"prior_covariance": {"model": "gaussian_process",
                                              "sigma_kg_s": 1.0e-3, "L_B_m": 500.0}},
           "observations": {"default_sigma": 0.1, "add_noise": False}}
    (*_, diag) = build_from_prebuilt_operator(cfg, up)
    assert diag["L_true_m"] == 500.0
    assert diag["L_B_m"] == 500.0
    assert diag["inverse_crime_flag"] is True


def test_instrument_total_only_uniform_template_is_truth_independent(tmp_path):
    from netCDF4 import Dataset
    from enforceflux.runs import read_upstream
    from flux_inputs import build_from_prebuilt_operator_with_instrument

    disp_root, G_fine, mapping, _ = _stage_dispersion(tmp_path)
    obs = tmp_path / "obs.nc"
    with Dataset(obs, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("instrument", G_fine.shape[0])
        ds.createVariable("y_obs", "f8", ("time", "instrument"))[:] = 1.0
        ds.createVariable("valid_mask", "i1", ("time", "instrument"))[:] = 1
        ds.createVariable("noise_variance", "f8", ("time", "instrument"))[:] = 0.1

    cfg = {
        "input": {"time_reduce": "mean"},
        "inversion": {
            "total_only": True,
            "total_template": "uniform",
            "prior_covariance": {"model": "diagonal", "sigma_kg_s": 0.01},
        },
    }
    result = build_from_prebuilt_operator_with_instrument(
        cfg, read_upstream(disp_root), obs
    )
    G_total, _, _, names, _, _, _, _, _, diagnostics = result
    area_weights = mapping.fine_cell_areas_m2 / mapping.fine_cell_areas_m2.sum()
    assert G_total.shape == (G_fine.shape[0], 1)
    assert np.allclose(G_total[:, 0], G_fine @ area_weights)
    assert names == ["Q_total"]
    assert diagnostics["inversion_template"] == "uniform"
