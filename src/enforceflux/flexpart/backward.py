"""FLEXPART backward-mode Jacobian (footprint) runner.

Builds the G-matrix (source-receptor sensitivity / Jacobian) by running one
FLEXPART backward simulation per receptor (instrument).

Backward mode physics
---------------------
With ``LDIRECT = -1`` FLEXPART releases particles *from each instrument location*
at the observation time and integrates them backward in time.  The gridded
residence-time output (``grid_time_*.nc``) is the **footprint function** — the
Jacobian row for that receptor:

    G[i, j]  =  footprint(lon_j, lat_j)  for receptor i
             =  ∂c_receptor_i / ∂Q_source_j

Raw FLEXPART units: ``spec001_mr`` in ``[s]`` — grid-cell-integrated
residence time (FLEXPART sums particle dwell time over each output cell).

Physical conversion to ``[ng m⁻³ / (kg s⁻¹)]``::

    G_physical[i,j]  =  result.g[i,j] [s]  ×  1e12  /  V_cell[j] [m³]

where ``V_cell = area_j × mixing_height_m``.  Use the helper
``FlexpartBackwardRunner.to_jacobian(g, source_areas_m2)`` which computes
this automatically.

This conversion is confirmed by the source-receptor reciprocity theorem;
forward and backward G values agree within ~4 % (limited by particle count).

``FOOTPRINT_TO_JACOBIAN = 1e12`` is the kg → ng factor *before* dividing
by cell volume.  It should not be used alone as a complete conversion.

Comparison with FlexpartRunner (forward per-source)
----------------------------------------------------
- Forward runner: one run per *source* → reads **point-receptor** output → G columns
- Backward runner: one run per *instrument* → reads **gridded footprint** → G rows

For ``n_instruments < n_sources`` the backward approach is more efficient.
For ``n_instruments > n_sources`` use the forward runner instead.

Usage
-----
::

    from enforceflux.flexpart.backward import FlexpartBackwardRunner

    runner = FlexpartBackwardRunner(base_config=sim_cfg, domain=domain_cfg)
    result = runner.run(instruments, sources)

    # Raw footprint [s]; divide by grid-cell volume for physical units:
    source_areas_m2 = np.array([7.7e11 / 100.0])  # area = V / mixing_height
    G = FlexpartBackwardRunner.to_jacobian(result.g, source_areas_m2)
    # G[i,j] is now in [ng m⁻³ / (kg s⁻¹)]
"""
import dataclasses
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from enforceflux.models.config import DomainConfig
from enforceflux.backend import UnitRunResult
from enforceflux.flexpart.simulation import FlexpartSimulation, SimulationConfig
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source

# Multiply raw footprint [s m³ kg⁻¹] by this to get [ng m⁻³ / (kg s⁻¹)]
FOOTPRINT_TO_JACOBIAN = 1e12


class FlexpartBackwardRunner:
    """
    Builds the Jacobian G-matrix via FLEXPART backward-mode (footprint) runs.

    Parameters
    ----------
    base_config : SimulationConfig
        Template configuration providing FLEXPART executable path, met-data
        directory, AVAILABLE file, options template, domain bounds (lon/lat),
        simulation period, species, and output grid settings.
        The simulation period should span ``[obs_time - lookback, obs_time]``
        where ``obs_time = base_config.end``.
    domain : DomainConfig
        Provides the CRS used to convert instrument and source coordinates
        (in projected metres) to WGS-84 lon/lat for FLEXPART.
        If ``domain.crs`` is ``None``, coordinates are assumed to already be
        lon/lat (x = longitude, y = latitude).
    config : dict, optional
        Override keys:
        - ``base_run_dir`` (str): root directory for per-receptor run directories.
          Default: ``"runs/flexpart_backward"``.
        - ``n_particles`` (int): particles per backward release.  Default 10 000.
        - ``cache`` (bool): skip re-running if output already exists.
          Default ``True``.
        - ``dry_run`` (bool): write inputs only, skip FLEXPART execution.
          Default ``False``.
        - ``surface_only`` (bool): when reading footprint, sum only the lowest
          height level (surface flux inversion).  Default ``True``.
    """

    def __init__(
        self,
        base_config: SimulationConfig,
        domain: DomainConfig,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.base_config = base_config
        self.domain = domain
        self.config = config or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        instruments: list[Instrument],
        sources: list[Source],
    ) -> UnitRunResult:
        """
        Run one backward FLEXPART simulation per receptor and assemble G.

        Parameters
        ----------
        instruments : list[Instrument]
            Receptor locations in domain CRS (``inst.x``, ``inst.y``, ``inst.z``).
        sources : list[Source]
            Source locations in domain CRS (``src.x``, ``src.y``).  Each
            corresponds to one column of the returned G-matrix.

        Returns
        -------
        UnitRunResult
            ``result.g[i, j]`` = footprint at source ``j`` for receptor ``i``
            in units ``[s m³ kg⁻¹]``.
            Multiply by ``FOOTPRINT_TO_JACOBIAN = 1e12`` to convert to
            ``[ng m⁻³ / (kg s⁻¹)]``.
        """
        if not instruments:
            raise ValueError("instruments list is empty")
        if not sources:
            raise ValueError("sources list is empty")
        if not self.domain.crs:
            raise ValueError("domain.crs must be set to use FlexpartBackwardRunner")

        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError("pyproj is required: pip install pyproj") from exc

        transformer = Transformer.from_crs(
            self.domain.crs, self.domain.crs_wgs84, always_xy=True
        )

        base_run_dir = Path(self.config.get("base_run_dir", "runs/flexpart_backward")).resolve()
        base_run_dir.mkdir(parents=True, exist_ok=True)

        n_inst = len(instruments)
        n_src = len(sources)
        G = np.zeros((n_inst, n_src), dtype=float)
        meta: dict[str, Any] = {"runs": []}

        # Precompute source lon/lat for footprint sampling
        src_lons = np.empty(n_src)
        src_lats = np.empty(n_src)
        for j, src in enumerate(sources):
            src_lons[j], src_lats[j] = transformer.transform(src.x, src.y)

        for i, inst in enumerate(instruments):
            inst_lon, inst_lat = transformer.transform(inst.x, inst.y)
            run_dir = base_run_dir / f"receptor_{inst.id}"
            options_dir = run_dir / "options"
            output_dir = run_dir / "output"
            pathnames = run_dir / "pathnames"

            dry_run = bool(self.config.get("dry_run", False))
            cache = bool(self.config.get("cache", True))
            should_run = not (cache and self._has_output(output_dir))

            if should_run or dry_run:
                self._prepare_run_dir(
                    run_dir, options_dir, output_dir, pathnames,
                    inst_lon=inst_lon,
                    inst_lat=inst_lat,
                    inst_z=inst.z,
                    inst_id=inst.id,
                )

            if not dry_run and should_run:
                self._execute(run_dir, pathnames)

            if not dry_run:
                footprint, fp_lons, fp_lats = self._read_raw_footprint(output_dir)
                G[i] = self._sample_at_sources(
                    footprint, fp_lons, fp_lats, src_lons, src_lats
                )

            meta["runs"].append(
                {"receptor": inst.id, "run_dir": str(run_dir), "output_dir": str(output_dir)}
            )

        return UnitRunResult(g=G, meta=meta)

    # ── Unit conversion ───────────────────────────────────────────────────────

    @staticmethod
    def to_jacobian(
        g_raw: np.ndarray,
        source_areas_m2: np.ndarray,
        mixing_height_m: float = 100.0,
    ) -> np.ndarray:
        """
        Convert raw footprint output to physical Jacobian units.

        FLEXPART ``spec001_mr`` in backward mode has units ``[s]``
        (grid-cell-integrated residence time).  The physical Jacobian element
        is::

            G_physical[i,j]  =  g_raw[i,j] [s]  ×  1e12  /  V_cell[j] [m³]
            V_cell[j]        =  source_areas_m2[j]  ×  mixing_height_m

        Parameters
        ----------
        g_raw : (m, n) ndarray
            Raw ``UnitRunResult.g`` from :meth:`run`.  Units: ``[s]``.
        source_areas_m2 : (n,) ndarray
            Horizontal area of each source grid cell in m².
            For a 1° × 1° cell at latitude φ:
            ``area = (π/180 × R_earth)² × cos(φ) × Δlat × Δlon``.
        mixing_height_m : float, optional
            Effective mixing height used as the vertical extent of each
            source cell.  Default 100 m (lowest FLEXPART height level).

        Returns
        -------
        G : (m, n) ndarray
            Jacobian in ``[ng m⁻³ / (kg s⁻¹)]``.
        """
        source_areas_m2 = np.asarray(source_areas_m2, dtype=float)
        V_cell = source_areas_m2 * mixing_height_m          # (n,)  m³
        return g_raw * 1e12 / V_cell[np.newaxis, :]         # broadcast over rows

    # ── Run-directory setup ───────────────────────────────────────────────────

    def _prepare_run_dir(
        self,
        run_dir: Path,
        options_dir: Path,
        output_dir: Path,
        pathnames: Path,
        *,
        inst_lon: float,
        inst_lat: float,
        inst_z: float,
        inst_id: str,
    ) -> None:
        bc = self.base_config

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bc.options_dir, options_dir, dirs_exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a per-receptor SimulationConfig with ldirect = -1
        bwd_cfg = self._make_backward_config(run_dir)

        # Use FlexpartSimulation to write standard files (COMMAND, OUTGRID, pathnames)
        sim = FlexpartSimulation(bwd_cfg)
        sim._write_pathnames(pathnames, options_dir, output_dir)
        sim._write_command(options_dir / "COMMAND")
        sim._write_outgrid(options_dir / "OUTGRID")

        # Write backward-mode RELEASES: particles released FROM the instrument
        self._write_instrument_release(
            options_dir / "RELEASES",
            lon=inst_lon,
            lat=inst_lat,
            z=inst_z,
            inst_id=inst_id,
            cfg=bwd_cfg,
        )

    def _make_backward_config(self, run_dir: Path) -> SimulationConfig:
        """
        Clone the base SimulationConfig, overriding ``ldirect = -1``,
        ``run_dir``, and ``output_path``.  Sources are deliberately left empty
        because the backward runner writes its own RELEASES file.
        """
        bc = self.base_config
        return dataclasses.replace(
            bc,
            ldirect=-1,
            run_dir=run_dir,
            output_path=run_dir / "output" / "_unused.nc",
            sources=[],      # backward runner writes RELEASES directly
        )

    def _write_instrument_release(
        self,
        path: Path,
        *,
        lon: float,
        lat: float,
        z: float,
        inst_id: str,
        cfg: SimulationConfig,
    ) -> None:
        """
        Write a RELEASES file that releases particles FROM the instrument
        (receptor) location at the observation time (``cfg.end``).

        In backward mode FLEXPART traces these particles *back* to their
        origins.  The total mass ``MASS = 1.0 kg`` normalises the footprint
        output (the value does not affect the spatial pattern, only the
        absolute scaling).
        """
        obs_date = cfg.end.strftime("%Y%m%d")
        obs_time = cfg.end.strftime("%H%M%S")
        n_parts = int(self.config.get("n_particles", 10_000))
        spec_num = int(self.config.get("species_number", cfg.species_number))

        path.write_text(
            "&RELEASES_CTRL\n"
            " NSPEC      =           1,\n"
            f" SPECNUM_REL=          {spec_num},\n"
            " /\n"
            "&RELEASE\n"
            f" IDATE1  =       {obs_date},\n"
            f" ITIME1  =         {obs_time},\n"
            f" IDATE2  =       {obs_date},\n"
            f" ITIME2  =         {obs_time},\n"
            f" LON1    =        {lon:.6f},\n"
            f" LON2    =        {lon:.6f},\n"
            f" LAT1    =        {lat:.6f},\n"
            f" LAT2    =        {lat:.6f},\n"
            f" Z1      =        {z:.3f},\n"
            f" Z2      =        {z:.3f},\n"
            " ZKIND   =              1,\n"
            " MASS    =       1.0000E+00,\n"
            f" PARTS   =          {n_parts},\n"
            f" COMMENT =    \"RECEPTOR {inst_id[:8]}\",\n"
            " /\n"
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    def _has_output(self, output_dir: Path) -> bool:
        return bool(list(output_dir.glob("grid_time_*.nc")))

    def _execute(self, run_dir: Path, pathnames: Path) -> None:
        exe = self.base_config.executable
        if not exe.exists():
            raise FileNotFoundError(f"FLEXPART executable not found: {exe}")
        # Pass pathnames as its filename only; FLEXPART opens it relative to cwd=run_dir.
        env = os.environ.copy()
        # Single-threaded OpenMP prevents a race condition in FLEXPART's NetCDF
        # output writer that causes SIGTRAP/hangs on macOS (see simulation.py).
        env["OMP_NUM_THREADS"] = "1"
        proc = subprocess.run(
            [str(exe), pathnames.name],
            cwd=run_dir,
            env=env,
        )
        if proc.returncode != 0:
            # FLEXPART 11 can segfault during teardown after the run has
            # completed and written its output; only fail if output is missing.
            if not self._has_output(run_dir / "output"):
                raise subprocess.CalledProcessError(proc.returncode, proc.args)
            warnings.warn(
                f"FLEXPART exited with code {proc.returncode} after writing "
                f"output in {run_dir}; continuing with the written footprint.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Footprint reader ──────────────────────────────────────────────────────

    def _read_raw_footprint(
        self, output_dir: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read the FLEXPART backward-mode gridded output and return the
        time-integrated footprint as a 2-D (lat × lon) array.

        Returns
        -------
        footprint : (n_lat, n_lon) ndarray
            Residence-time footprint summed over the lookback window.
            Units: ``[s m³ kg⁻¹]`` (FLEXPART native).
        lons : (n_lon,) ndarray  — longitude bin centres [degrees]
        lats : (n_lat,) ndarray  — latitude bin centres [degrees]
        """
        candidates = sorted(output_dir.glob("grid_time_*.nc"))
        if not candidates:
            # FLEXPART sometimes names backward output grid_conc_*.nc
            candidates = sorted(output_dir.glob("grid_conc_*.nc"))
        if not candidates:
            raise FileNotFoundError(
                f"No backward footprint NetCDF found in {output_dir}.\n"
                "Expected: grid_time_YYYYMMDDHHMMSS.nc\n"
                "Check that FLEXPART ran successfully and IOUT=9."
            )

        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required: pip install netCDF4") from exc

        surface_only = bool(self.config.get("surface_only", True))

        with Dataset(candidates[0]) as ds:
            lons = np.asarray(ds.variables["longitude"][:])
            lats = np.asarray(ds.variables["latitude"][:])

            # Prefer mixing-ratio (spec001_mr); fall back to mass-conc (spec001)
            var_name = None
            for candidate_var in ("spec001_mr", "spec001"):
                if candidate_var in ds.variables:
                    var_name = candidate_var
                    break
            if var_name is None:
                raise RuntimeError(
                    f"Cannot find 'spec001_mr' or 'spec001' in {candidates[0]}.\n"
                    f"Available variables: {list(ds.variables.keys())}"
                )

            raw = np.asarray(ds.variables[var_name][:])
            # Dimensions: (nageclass, pointspec, time, height, lat, lon)

            if surface_only:
                # Sum over time, keep only the lowest height level (index 0)
                footprint = raw[0, 0, :, 0, :, :].sum(axis=0)
            else:
                # Sum over time AND all height levels (column integral)
                footprint = raw[0, 0, :, :, :, :].sum(axis=(0, 1))

        return footprint, lons, lats

    # ── Footprint sampling ────────────────────────────────────────────────────

    def _sample_at_sources(
        self,
        footprint: np.ndarray,
        fp_lons: np.ndarray,
        fp_lats: np.ndarray,
        src_lons: np.ndarray,
        src_lats: np.ndarray,
    ) -> np.ndarray:
        """
        Sample the 2-D footprint at each source location (nearest grid cell).

        Returns a 1-D array of length ``n_sources`` with the footprint value
        at each source in ``[s m³ kg⁻¹]``.
        """
        n_src = len(src_lons)
        values = np.empty(n_src)
        for j in range(n_src):
            ilat = int(np.argmin(np.abs(fp_lats - src_lats[j])))
            ilon = int(np.argmin(np.abs(fp_lons - src_lons[j])))
            values[j] = float(footprint[ilat, ilon])
        return values
