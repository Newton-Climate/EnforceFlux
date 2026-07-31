"""Time-resolved flux inversion, exercised per transport model.

The flux inversion (apps/flux_main.py) turns each source's per-timestep
concentration field into a block-Toeplitz convolution operator G and solves for
an emission-rate *time series* per source. These tests drive that path with the
real transport backends:

* **AERMOD** is pure-Python/JAX, so its test always runs.
* **FLEXPART** (Lagrangian — the repo's stand-in for STILT) and **MicroHH** (LES)
  need compiled binaries, so they sit behind integration markers and ``skipif``
  guards; ``make test`` deselects them by default.

Each model produces a per-source, time-resolved concentration NetCDF (FLEXPART
does so natively via its ``pointspec`` release axis; AERMOD and MicroHH are run
once per source and stacked). We then synthesise observations from a known
time-varying truth, invert, and check the well-observed windows recover it.
Using ``synthetic_from_truth`` keeps the forward and inverse operators
identical, so the assertion targets the inversion pipeline and each model's
field plumbing (dimension names, units, coordinate layout) — not the backend's
transport physics.
"""
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "apps", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Skip conditions for the binary-backed backends ──────────────────────────
FLEXPART_BIN = REPO_ROOT / "flexpart" / "src" / "FLEXPART"
FLEXPART_MET = REPO_ROOT / "flexpart" / "tests" / "testdata"
MICROHH_BIN = REPO_ROOT / "microhh" / "build" / "microhh"
ERA5_DIR = REPO_ROOT / "runs" / "sacramento_valley_2020" / "meteo_april_week"

requires_flexpart = pytest.mark.skipif(
    not (FLEXPART_BIN.exists() and FLEXPART_MET.is_dir()),
    reason="FLEXPART binary or bundled test met data not present",
)
requires_microhh = pytest.mark.skipif(
    not (
        MICROHH_BIN.exists()
        and ERA5_DIR.is_dir()
        and importlib.util.find_spec("eccodes") is not None
    ),
    reason="MicroHH binary or ERA5 GRIB test data not present",
)

SOURCE_LON, SOURCE_LAT = -121.75, 39.15


# ── Truth generator and the shared inversion driver ─────────────────────────
def _time_varying_truth(n_sources: int, n_time: int) -> np.ndarray:
    """A distinct emission-rate time series per source (kg/s).

    Source 0 is steady; source 1 carries a single-window pulse; any further
    sources get a late ramp. Guarantees genuine time variation to recover.
    """
    truth = np.zeros((n_sources, n_time), dtype=float)
    truth[0, :] = 0.5
    if n_sources > 1:
        truth[1, n_time // 2] = 1.0
    for j in range(2, n_sources):
        truth[j, -2:] = 0.25
    return truth


def _write_per_source_nc(path: Path, fields, source_ids) -> None:
    """Stack per-source ``(time, y, x)`` fields into the multi-release NetCDF
    that ``flux_helpers.prepare_sim_transport`` reads.

    The canonical metric axes (``x``/``y``) are written as the ``longitude`` /
    ``latitude`` coordinate variables; receptors are then addressed in the same
    metric frame, keeping the round-trip self-consistent without a projection.
    """
    from netCDF4 import Dataset

    ref = fields[0]
    n_time, ny, nx = ref.values.shape
    stacked = np.stack([f.values for f in fields], axis=0)  # (releases, time, y, x)

    with Dataset(path, "w") as ds:
        ds.createDimension("releases", len(fields))
        ds.createDimension("time", n_time)
        ds.createDimension("latitude", ny)
        ds.createDimension("longitude", nx)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = np.asarray(ref.y)
        ds.createVariable("longitude", "f8", ("longitude",))[:] = np.asarray(ref.x)
        var = ds.createVariable(
            "ch4_mixing_ratio", "f8", ("releases", "time", "latitude", "longitude")
        )
        var[:] = stacked
        ds.source_ids = ",".join(source_ids)
        ds.n_point_sources = len(fields)


def _invert_nc(nc_path, receptors_lonlat, tmp_path, *, sigma=1.0):
    """Run flux_main's receptors-mode time-resolved inversion on a NetCDF.

    Reads the source/time counts from the file, injects a time-varying truth via
    ``synthetic_from_truth`` (so y = G @ truth exactly), inverts, and returns the
    recovery diagnostics.
    """
    from netCDF4 import Dataset

    from flux_helpers import build_prior, infer_time_size, prepare_sim_transport
    from flux_inputs import build_from_receptors_mode
    from enforceflux.inversion import oe_from_linear

    with Dataset(nc_path) as ds:
        _, _, _, cvar, n_sources, _ = prepare_sim_transport(ds, "ch4_mixing_ratio")
        n_time = infer_time_size(cvar)

    truth = _time_varying_truth(n_sources, n_time)
    cfg = {
        "input": {
            "mode": "simulation_receptors",
            "simulation_netcdf": str(nc_path),
            "variable_name": "ch4_mixing_ratio",
            "level_index": 0,
        },
        "receptors": [
            {"id": f"r{i}", "lon": float(lon), "lat": float(lat)}
            for i, (lon, lat) in enumerate(receptors_lonlat)
        ],
        "observations": {
            "mode": "synthetic_from_truth",
            "truth_flux_kg_s": truth.tolist(),
            "default_sigma": sigma,
            "add_noise": False,
        },
        "inversion": {
            "method": "linear",
            "prior_flux_kg_s": 0.0,
            "prior_variance": 1.0e2,  # loose: let the data drive the observable windows
        },
    }

    G, y_obs, Se, names, vname, sim_nc, obs_meta, n_flux = build_from_receptors_mode(cfg)
    x_prior, Sa = build_prior(cfg, n_sources, n_flux)
    result = oe_from_linear(G=G, y=y_obs, x_prior=x_prior, Sa=Sa, Se=Se, source_names=names)

    x_opt = np.asarray(result.x_posterior, dtype=float).reshape(n_sources, n_flux)
    post_sigma = np.sqrt(np.maximum(np.diag(result.posterior_cov), 0.0)).reshape(
        n_sources, n_flux
    )
    return {
        "G": G,
        "x_opt": x_opt,
        "truth": truth,
        "post_sigma": post_sigma,
        "prior_sigma": float(np.sqrt(cfg["inversion"]["prior_variance"])),
        "n_flux": n_flux,
        "n_sources": n_sources,
    }


def _assert_time_resolved_recovery(diag) -> None:
    """The inversion must be time-resolved, causal, and recover observed windows."""
    n_sources, n_flux = diag["n_sources"], diag["n_flux"]

    # 1. State really is a per-source time series, not a single scalar.
    assert n_flux > 1, "inversion collapsed to a single flux window"
    assert diag["x_opt"].shape == (n_sources, n_flux)

    # 2. G is block lower-triangular in time: emission in window k cannot affect
    #    an earlier observation time t < k. Each receptor's rows span n_flux.
    n_receptors = diag["G"].shape[0] // n_flux
    for j in range(n_sources):
        for i in range(n_receptors):
            block = diag["G"][i * n_flux : (i + 1) * n_flux, j * n_flux : (j + 1) * n_flux]
            assert np.allclose(np.triu(block, k=1), 0.0), "G block is not causal"

    # 3. Windows the data strongly constrains (>99% uncertainty reduction) must
    #    recover the injected truth; weakly observed / edge windows may fall back
    #    to the prior. The 1e-2 tolerance leaves margin for a stochastic backend
    #    (FLEXPART's Monte-Carlo particle noise); AERMOD recovers to ~1e-6.
    well_observed = diag["post_sigma"] < 0.1 * diag["prior_sigma"]
    assert well_observed.sum() >= n_sources, "no window was constrained by the data"
    err = np.abs(diag["x_opt"] - diag["truth"])[well_observed]
    assert np.all(err < 1e-2), f"observed-window recovery too coarse: max err {err.max():.2e}"


# ── AERMOD (always runs) ────────────────────────────────────────────────────
def _aermod_field(tmp_path, source_x, source_y, tag):
    """One AERMOD simulation field (time, y, x) for a single unit-rate source."""
    import yaml

    from enforceflux.transport import TransportRunConfig, run_transport

    met_records = [
        {
            "time": f"2020-03-31T0{h}:00",
            "wind_speed_m_s": speed,
            "wind_direction_deg": 270.0,  # from the west → plume toward +x (east)
            "mixing_height_m": mix,
            "friction_velocity_m_s": 0.3,
            "sensible_heat_flux_w_m2": 100.0,
        }
        for h, (speed, mix) in enumerate(
            [(3.0, 800.0), (2.5, 700.0), (2.0, 600.0), (3.5, 900.0), (2.2, 500.0)]
        )
    ]
    blob = {
        "transport": {
            "model": "aermod",
            "mode": "simulation",
            "start": "2020-03-31T00:00:00",
            "end": "2020-03-31T05:00:00",
        },
        "met": {"records": met_records},
        "domain": {
            "origin_lon": SOURCE_LON,
            "origin_lat": SOURCE_LAT,
            "x_min": -2600.0,
            "x_max": 2600.0,
            "y_min": -2200.0,
            "y_max": 2200.0,
            "spacing_m": 200.0,
            "receptor_height_m": 2.0,
        },
        "sources": [
            {
                "id": tag,
                "x_m": float(source_x),
                "y_m": float(source_y),
                "alt_m": 5.0,
                "emission_rate_kg_s": 1.0,  # unit rate → field is the unit response
            }
        ],
        "receptors": [{"id": "probe", "x_m": 800.0, "y_m": 0.0, "alt_m": 2.0}],
        "output": {"path": str(tmp_path / f"aermod_{tag}.nc")},
    }
    cfg_path = tmp_path / f"run_{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(blob))
    return run_transport(TransportRunConfig.from_file(cfg_path)).field


def test_aermod_time_resolved_flux_inversion(tmp_path):
    # Two sources at distinct crosswind positions; three receptors downwind.
    sources = [("srcA", 0.0, 0.0), ("srcB", 0.0, 1000.0)]
    fields = [_aermod_field(tmp_path, x, y, tag) for tag, x, y in sources]
    nc = tmp_path / "aermod_per_source.nc"
    _write_per_source_nc(nc, fields, [s[0] for s in sources])

    receptors_xy = [(800.0, 0.0), (800.0, 1000.0), (800.0, 500.0)]
    diag = _invert_nc(nc, receptors_xy, tmp_path)
    _assert_time_resolved_recovery(diag)


# ── FLEXPART (STILT stand-in): native multi-release output ──────────────────
@requires_flexpart
@pytest.mark.flexpart_integration
def test_flexpart_time_resolved_flux_inversion(tmp_path):
    """FLEXPART is the repo's Lagrangian particle model (STILT stand-in). A
    forward run with per-release output writes a native ``(pointspec, time, lat,
    lon)`` NetCDF — exactly the multi-source, time-resolved field flux_main
    consumes, so no stacking is needed."""
    from enforceflux.flexpart.simulation import FlexpartSimulation, PointSource, SimulationConfig

    start = datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2009, 1, 1, 3, 0, 0, tzinfo=timezone.utc)  # bundled met window
    sources = [
        PointSource(
            id="srcA", lon=7.5, lat=51.5, alt_m=5.0,
            emission_rate_kg_s=1.0, start=start, end=end, n_particles=5000,
        ),
        PointSource(
            id="srcB", lon=7.9, lat=51.5, alt_m=5.0,
            emission_rate_kg_s=1.0, start=start, end=end, n_particles=5000,
        ),
    ]
    cfg = SimulationConfig(
        executable=FLEXPART_BIN.resolve(),
        options_dir=(REPO_ROOT / "flexpart" / "tests" / "default_options").resolve(),
        available_file=(REPO_ROOT / "flexpart" / "tests" / "default_winds" / "AVAILABLE").resolve(),
        meteo_dir=FLEXPART_MET.resolve(),
        run_dir=(tmp_path / "fp_run").resolve(),
        start=start,
        end=end,
        output_step_s=3600,
        domain_lon_min=-25.0, domain_lat_min=10.0,
        domain_lon_max=60.0, domain_lat_max=75.0,
        domain_dx=1.0, domain_dy=1.0,
        heights_m=[100.0, 500.0, 1000.0, 50000.0],
        sources=sources,
        output_path=(tmp_path / "fp_out.nc").resolve(),
        species_name="CH4",
        species_number=24,
        nxshift=0,
        ldirect=1,
        output_per_source=True,  # per-release pointspec axis
    )
    nc = FlexpartSimulation(cfg).run()

    # Receptors ringing the two sources so the plume is sampled whatever the
    # (bundled) wind direction is.
    receptors_lonlat = [
        (7.5, 51.7), (7.7, 51.5), (7.5, 51.3),
        (7.9, 51.7), (8.1, 51.5), (7.9, 51.3),
    ]
    diag = _invert_nc(nc, receptors_lonlat, tmp_path)
    _assert_time_resolved_recovery(diag)


# ── MicroHH (LES): real backend, per-source runs ────────────────────────────
def _microhh_field(tmp_path, tag, source_x, source_y):
    """One MicroHH LES concentration field (time, y, x) for a single source."""
    import yaml

    from enforceflux.transport import TransportRunConfig, run_transport

    blob = {
        "transport": {"model": "microhh", "mode": "simulation",
                      "start": "2020-04-01T00:00:00", "end": "2020-04-01T05:00:00"},
        "met": {"era5": {"meteo_dir": str(ERA5_DIR), "surface_roughness_m": 0.15}},
        "domain": {
            "origin_lon": SOURCE_LON, "origin_lat": SOURCE_LAT,
            "x_min": -2600.0, "x_max": 2600.0, "y_min": -2200.0, "y_max": 2200.0,
            "spacing_m": 200.0, "receptor_height_m": 2.0,
        },
        "sources": [{"id": tag, "x_m": float(source_x), "y_m": float(source_y),
                     "alt_m": 5.0, "emission_rate_kg_s": 1.0}],
        "receptors": [{"id": "probe", "x_m": 800.0, "y_m": 0.0, "alt_m": 2.0}],
        "microhh": {
            "executable": str(MICROHH_BIN),
            "case_dir": str(tmp_path / f"microhh_case_{tag}"),
            "num_workers": 1,
            # Small grid + short run so this real LES completes in minutes; kept
            # opt-in behind the microhh_integration marker. run_transport reads
            # the LES timing from this block (not a separate simulation block).
            "grid": {"itot": 32, "jtot": 32, "ktot": 32, "xsize": 1600.0, "ysize": 1600.0},
            "spinup_seconds": 0,
            "runtime_seconds": 2400,  # a frame every sampletime → 5 output times
            "sampletime": 600,
        },
        "output": {"path": str(tmp_path / f"microhh_{tag}.nc")},
    }
    cfg_path = tmp_path / f"run_microhh_{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(blob))
    return run_transport(TransportRunConfig.from_file(cfg_path)).field


@requires_microhh
@pytest.mark.microhh_integration
def test_microhh_time_resolved_flux_inversion(tmp_path):
    sources = [("srcA", 0.0, 0.0), ("srcB", 0.0, 1000.0)]
    fields = [_microhh_field(tmp_path, tag, x, y) for tag, x, y in sources]
    nc = tmp_path / "microhh_per_source.nc"
    _write_per_source_nc(nc, fields, [s[0] for s in sources])

    receptors_xy = [(800.0, 0.0), (800.0, 1000.0), (800.0, 500.0)]
    diag = _invert_nc(nc, receptors_xy, tmp_path)
    _assert_time_resolved_recovery(diag)
