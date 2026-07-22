import os
import subprocess
import shutil
from datetime import datetime as _DT
from pathlib import Path
from typing import Any

import numpy as np

from enforceflux.backend import UnitEmissionRunner
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source


class FlexpartRunner(UnitEmissionRunner):
    model_name = "FLEXPART"
    output_glob = "*.nc"
    default_run_dir = "runs/flexpart"

    def _prepare_run(
        self,
        run_dir: Path,
        output_dir: Path,
        source: Source,
        instruments: list[Instrument],
        transformer: Any,
    ) -> None:
        options_dir = run_dir / "options"
        pathnames_path = run_dir / "pathnames"

        options_template = self._resolve_path(self.config.get("options_dir", "flexpart/options"))
        if not options_template.exists():
            raise FileNotFoundError(f"Options directory not found: {options_template}")
        pathnames_template = self.config.get("pathnames_template")
        if pathnames_template:
            template_path = self._resolve_path(pathnames_template)
            if not template_path.exists():
                raise FileNotFoundError(f"Pathnames template not found: {template_path}")

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(options_template, options_dir, dirs_exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        available_file = self._resolve_path(self.config.get("available_file", "flexpart/AVAILABLE"))
        meteo_dir = self._resolve_path(self.config.get("meteo_dir", "./inputs"))
        if not self.config.get("dry_run", False):
            if not available_file.exists():
                raise FileNotFoundError(f"AVAILABLE file not found: {available_file}")
            if not meteo_dir.exists():
                raise FileNotFoundError(f"Meteorological input directory not found: {meteo_dir}")

        self._write_pathnames(
            pathnames_path,
            options_dir=options_dir,
            output_dir=output_dir,
            meteo_dir=meteo_dir,
            available_file=available_file,
        )

        source_lon, source_lat = transformer.transform(source.x, source.y)
        self._write_releases(
            options_dir / "RELEASES",
            source_lon=source_lon,
            source_lat=source_lat,
            source_alt=source.z,
            source_id=source.id,
        )
        self._write_receptors(
            options_dir / "RECEPTORS",
            instruments=instruments,
            transformer=transformer,
        )

    def _execute(self, run_dir: Path) -> None:
        pathnames_path = run_dir / "pathnames"
        executable = self._resolve_path(self.config.get("executable", "flexpart/src/FLEXPART"))
        if not executable.exists():
            raise FileNotFoundError(f"FLEXPART executable not found: {executable}")

        env_overrides = {k: str(v) for k, v in (self.config.get("env", {}) or {}).items()}
        env = os.environ.copy()
        env.update(env_overrides)
        command = [str(executable), str(pathnames_path)]
        subprocess.run(command, cwd=run_dir, check=True, env=env)

    def _read_receptor_values(self, output_dir: Path, instruments: list[Instrument]) -> np.ndarray:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("netCDF4 is required to read FLEXPART outputs") from exc

        output_file = self._find_output_file(output_dir)
        with Dataset(output_file) as dataset:
            receptor_var = self._select_receptor_variable(dataset)
            data = np.asarray(dataset.variables[receptor_var][:])
            receptor_axis = self._find_receptor_axis(dataset.variables[receptor_var].dimensions)
            if receptor_axis is None:
                raise RuntimeError(
                    f"Unable to locate receptor dimension for variable {receptor_var}"
                )
            axes = tuple(i for i in range(data.ndim) if i != receptor_axis)
            if axes:
                data = data.mean(axis=axes)

            data = np.asarray(data).reshape(-1)
            names = self._read_receptor_names(dataset)
            if names:
                lookup = {name.strip(): idx for idx, name in enumerate(names)}
                mapped = []
                for inst in instruments:
                    if inst.id not in lookup:
                        raise RuntimeError(
                            "Receptor names do not match instruments. "
                            f"Missing: {inst.id}. Available: {sorted(lookup.keys())}"
                        )
                    mapped.append(data[lookup[inst.id]])
                return np.asarray(mapped)

            if data.size != len(instruments):
                raise RuntimeError(
                    f"Receptor output size {data.size} does not match instruments {len(instruments)}"
                )

            return data

    def _find_output_file(self, output_dir: Path) -> Path:
        preferred = self.config.get("receptor_output_file")
        if preferred:
            path = Path(preferred)
            if not path.is_absolute():
                path = output_dir / path
            if not path.exists():
                raise FileNotFoundError(f"Receptor output file not found: {path}")
            return path

        candidates = sorted(output_dir.glob("*.nc"))
        if not candidates:
            raise FileNotFoundError(f"No NetCDF outputs found in {output_dir}")

        for candidate in candidates:
            try:
                from netCDF4 import Dataset
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("netCDF4 is required to read FLEXPART outputs") from exc
            with Dataset(candidate) as dataset:
                if any("receptor" in dim.lower() for dim in dataset.dimensions):
                    return candidate
        return candidates[-1]

    def _select_receptor_variable(self, dataset: Any) -> str:
        preferred = self.config.get("receptor_variable")
        if preferred and preferred in dataset.variables:
            return preferred

        candidates = []
        for name, var in dataset.variables.items():
            if any("receptor" in dim.lower() for dim in var.dimensions):
                candidates.append(name)
        if not candidates:
            available = ", ".join(sorted(dataset.variables.keys()))
            raise RuntimeError(
                "No receptor variable found in output. Available variables: "
                f"{available}"
            )
        return candidates[0]

    def _find_receptor_axis(self, dimensions: tuple[str, ...]) -> int | None:
        for idx, dim in enumerate(dimensions):
            if "receptor" in dim.lower():
                return idx
        return None

    def _read_receptor_names(self, dataset: Any) -> list[str] | None:
        for name, var in dataset.variables.items():
            if "receptor" not in name.lower():
                continue
            if var.dtype.kind in {"S", "U"}:
                arr = np.asarray(var[:])
                return self._decode_char_array(arr)
        return None

    def _decode_char_array(self, arr: np.ndarray) -> list[str]:
        if arr.ndim == 1:
            return [self._decode_entry(item) for item in arr]
        if arr.ndim == 2:
            return [self._decode_entry(item) for item in arr]
        flat = arr.reshape(arr.shape[0], -1)
        return [self._decode_entry(item) for item in flat]

    def _decode_entry(self, item: Any) -> str:
        if isinstance(item, bytes):
            return item.decode("utf-8").strip()
        if isinstance(item, np.ndarray):
            if item.dtype.kind == "S":
                return b"".join(item.tolist()).decode("utf-8").strip()
            if item.dtype.kind == "U":
                return "".join(item.tolist()).strip()
        return str(item).strip()

    def _write_pathnames(
        self,
        path: Path,
        options_dir: Path,
        output_dir: Path,
        meteo_dir: Path,
        available_file: Path,
    ) -> None:
        # FLEXPART truncates each pathnames entry to a fixed 120-char buffer
        # (com_mod.f90), which breaks runs under deep temp dirs. FLEXPART runs
        # with cwd=run_dir and options/ and output/ are direct children, so
        # write them relative to keep entries short. See simulation.py.
        run_dir = path.parent
        lines = [
            f"{os.path.relpath(options_dir, run_dir)}/",
            f"{os.path.relpath(output_dir, run_dir)}/",
            str(meteo_dir),
            str(available_file),
        ]
        path.write_text("\n".join(lines) + "\n")

    def _write_releases(
        self,
        path: Path,
        source_lon: float,
        source_lat: float,
        source_alt: float,
        source_id: str,
    ) -> None:
        start_date = int(self.config.get("release_start_date", 20120101))
        start_time = int(self.config.get("release_start_time", 90000))
        end_date = int(self.config.get("release_end_date", start_date))
        end_time = int(self.config.get("release_end_time", start_time))
        z1 = float(self.config.get("release_z1", source_alt))
        z2 = float(self.config.get("release_z2", z1))
        zkind = int(self.config.get("release_zkind", 1))
        parts = int(self.config.get("parts", 10000))
        species_num = int(self.config.get("species_number", 24))

        # unit_emission_rate is now in kg/s.  MASS (total kg released) = rate × duration.
        # This gives G[i,j] units of [concentration / (kg/s)] — physically correct for a
        # continuous emission source.
        unit_kg_s = float(self.config.get("unit_emission_rate", 1.0))
        t0 = _DT.strptime(f"{start_date}{start_time:06d}", "%Y%m%d%H%M%S")
        t1 = _DT.strptime(f"{end_date}{end_time:06d}", "%Y%m%d%H%M%S")
        duration_s = max(abs((t1 - t0).total_seconds()), 1.0)
        mass = unit_kg_s * duration_s

        lines = [
            "***************************************************************************************************************",
            "*                                                                                                             *",
            "*   Input file for the Lagrangian particle dispersion model FLEXPART                                          *",
            "***************************************************************************************************************",
            "&RELEASES_CTRL",
            f" NSPEC      =           1,",
            f" SPECNUM_REL=          {species_num},",
            " /",
            "&RELEASE",
            f" IDATE1  =       {start_date},",
            f" ITIME1  =         {start_time:06d},",
            f" IDATE2  =       {end_date},",
            f" ITIME2  =         {end_time:06d},",
            f" LON1    =        {source_lon:.6f},",
            f" LON2    =        {source_lon:.6f},",
            f" LAT1    =        {source_lat:.6f},",
            f" LAT2    =        {source_lat:.6f},",
            f" Z1      =        {z1:.3f},",
            f" Z2      =        {z2:.3f},",
            f" ZKIND   =              {zkind},",
            f" MASS    =       {mass:.4E},",
            f" PARTS   =          {parts},",
            f" COMMENT =    \"SOURCE {source_id}\",",
            " /",
        ]
        path.write_text("\n".join(lines) + "\n")

    def _write_receptors(
        self,
        path: Path,
        instruments: list[Instrument],
        transformer: Any,
    ) -> None:
        lines: list[str] = []
        for inst in instruments:
            lon, lat = transformer.transform(inst.x, inst.y)
            name = f"{inst.id}"[:16]
            lines.append("&RECEPTORS")
            lines.append(f" RECEPTOR=\"{name:<16}\",")
            lines.append(f" LAT=  {lat:.7f},")
            lines.append(f" LON= {lon:.7f},")
            lines.append(f" ALT=  {inst.z:.3f},")
            lines.append(" /")
        path.write_text("\n".join(lines) + "\n")
