"""Download ERA5 reanalysis from the Copernicus CDS for use as FLEXPART meteorology.

Requirements (install once):
    pip install cdsapi eccodes

CDS credentials (one-time setup):
    Create a free account at https://cds.climate.copernicus.eu/
    Copy your UID and API key from your profile page into ~/.cdsapirc:

        url: https://cds.climate.copernicus.eu/api
        key: <uid>:<api-key>

Usage::

    from enforceflux.meteo import ERA5Downloader

    dl = ERA5Downloader(output_dir="inputs/meteo")
    available_file = dl.download(
        start="2020-06-15T00:00",
        end="2020-06-15T18:00",
        bbox=(-124, 36, -118, 41),   # (lon_min, lat_min, lon_max, lat_max)
    )
    # available_file → Path("inputs/meteo/AVAILABLE")
"""
import calendar
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

log = logging.getLogger(__name__)

# ── ERA5 variable lists ───────────────────────────────────────────────────────

# 3-D analysis fields (pressure levels).
# FLEXPART needs winds, temperature, humidity, and cloud microphysics.
_PL_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
]

# Standard pressure levels (hPa) — enough vertical resolution for FLEXPART.
_DEFAULT_PRESSURE_LEVELS = [
    "1000", "925", "850", "700", "600",
    "500",  "400", "300", "250", "200",
    "150",  "100",  "50",  "30",  "20", "10",
]

# 3-D model-level fields (ECMWF IFS hybrid levels) needed by FLEXPART.
# Parameter IDs follow ECMWF MARS conventions for ERA5 complete.
_ML_PARAM_IDS = "130/131/132/133/246/247/248"

# 2-D instantaneous single-level fields (analysed every hour in ERA5).
_SFC_INSTANT = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "boundary_layer_height",
    "sea_surface_temperature",
    "total_cloud_cover",
]

# 2-D accumulated flux fields — ERA5 accumulates from 00 UTC and 12 UTC.
# FLEXPART uses the hour-to-hour difference to get the instantaneous flux.
_SFC_ACCUMULATED = [
    "large_scale_precipitation",
    "convective_precipitation",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "eastward_turbulent_surface_stress",
    "northward_turbulent_surface_stress",
]

# Time-invariant fields downloaded once (no time/date in request).
# "geopotential" is the correct CDS variable name for surface geopotential (z / orography).
_STATIC_VARIABLES = [
    "land_sea_mask",
    "geopotential",
]


# ── Helper types ──────────────────────────────────────────────────────────────

@dataclass
class DownloadResult:
    """Paths to downloaded data and the generated AVAILABLE file."""
    available_file: Path
    meteo_dir: Path
    n_timesteps: int
    files: list[Path] = field(default_factory=list)


# ── Main class ────────────────────────────────────────────────────────────────

class ERA5Downloader:
    """Download ERA5 from CDS and produce FLEXPART-ready GRIB files + AVAILABLE.

    Parameters
    ----------
    output_dir:
        Directory to write GRIB files and the AVAILABLE index.
        Created if it does not exist.
    timestep_hours:
        Temporal resolution to request.  ERA5 is 1-hourly but FLEXPART
        typically uses 3-hourly to keep run times reasonable.
    pressure_levels:
        List of pressure levels (hPa as strings) to download.
        Defaults to :data:`_DEFAULT_PRESSURE_LEVELS`.
    vertical_mode:
        ``"pressure_levels"`` (default) uses CDS pressure-level products.
        ``"model_levels"`` uses ERA5 complete model-level fields (required
        by FLEXPART's ECMWF reader).
    model_level_grid_deg:
        Grid spacing in degrees for model-level requests. If ``None``, CDS
        default grid is used.
    model_level_allow_global_fallback:
        If ``True``, failed area-clipped model-level requests retry globally.
        If ``False``, fail fast to avoid silently downloading huge global files.
    cleanup_raw_daily_grib:
        If ``True``, delete intermediate daily raw GRIBs after merging/splitting
        into FLEXPART timestep files.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        timestep_hours: int = 3,
        pressure_levels: list[str] | None = None,
        vertical_mode: Literal["pressure_levels", "model_levels"] = "pressure_levels",
        model_level_grid_deg: float | None = 0.25,
        model_level_allow_global_fallback: bool = False,
        cleanup_raw_daily_grib: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestep_hours = timestep_hours
        self.pressure_levels = pressure_levels or _DEFAULT_PRESSURE_LEVELS
        self.vertical_mode = vertical_mode
        self.model_level_grid_deg = model_level_grid_deg
        self.model_level_allow_global_fallback = model_level_allow_global_fallback
        self.cleanup_raw_daily_grib = cleanup_raw_daily_grib

    # ── Public API ────────────────────────────────────────────────────────────

    def download(
        self,
        start: str | datetime,
        end: str | datetime,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> DownloadResult:
        """Download ERA5 for the given period and return a :class:`DownloadResult`.

        Parameters
        ----------
        start, end:
            Simulation time window (inclusive).  Strings must be ISO-8601
            (e.g. ``"2020-06-15T00:00"``).  Data is padded by one extra
            timestep on each side so FLEXPART can interpolate at the boundaries.
        bbox:
            ``(lon_min, lat_min, lon_max, lat_max)`` in degrees WGS-84.
            If *None*, downloads the global field (large; not recommended).
        """
        _require_cdsapi()
        _require_eccodes()

        t0, t1 = _parse_dt(start), _parse_dt(end)
        # Pad by one timestep on each side so FLEXPART can interpolate.
        t0_dl = t0 - timedelta(hours=self.timestep_hours)
        t1_dl = t1 + timedelta(hours=self.timestep_hours)

        area = _bbox_to_area(bbox) if bbox else None

        log.info("Downloading ERA5 %s → %s", t0_dl.isoformat(), t1_dl.isoformat())

        all_files: list[Path] = []

        # Static fields (land-sea mask, orography) — downloaded once.
        static_path = self.output_dir / "EA_static.grib"
        if static_path.exists() and not _is_grib_file_readable(static_path):
            log.warning("Static GRIB appears corrupted, re-downloading: %s", static_path.name)
            static_path.unlink(missing_ok=True)

        if not static_path.exists():
            log.info("Fetching static fields → %s", static_path.name)
            self._fetch_static(static_path, area=area)
        else:
            log.info("Static file already present, skipping.")

        # Download analysis + flux fields day-by-day (CDS queue is shorter
        # for per-day requests than multi-week requests).
        for day_start, day_end in _day_windows(t0_dl, t1_dl):
            date_str = day_start.strftime("%Y-%m-%d")
            times = _hour_list(day_start, day_end, self.timestep_hours)
            if not times:
                continue

            # Skip the entire day if all per-timestep EA files already exist.
            date_int = int(date_str.replace("-", ""))
            expected = [
                self.output_dir / f"EA{date_int:08d}{int(t[:2]):02d}"
                for t in times
            ]
            if all(p.exists() for p in expected):
                log.info("All EA files for %s already present — skipping.", date_str)
                all_files.extend(expected)
                continue

            # 3-D upper-air fields (pressure-level or model-level).
            if self.vertical_mode == "model_levels":
                pl_raw = self.output_dir / f"ERA5_ml_{date_str}.grib"
                if pl_raw.exists() and not _is_grib_file_readable(pl_raw):
                    log.warning("Corrupted GRIB detected, re-downloading: %s", pl_raw.name)
                    pl_raw.unlink(missing_ok=True)
                if not pl_raw.exists():
                    log.info("Fetching model-level fields %s times=%s", date_str, times)
                    self._fetch_model_levels(pl_raw, date_str, times, area=area)
            else:
                pl_raw = self.output_dir / f"ERA5_pl_{date_str}.grib"
                if pl_raw.exists() and not _is_grib_file_readable(pl_raw):
                    log.warning("Corrupted GRIB detected, re-downloading: %s", pl_raw.name)
                    pl_raw.unlink(missing_ok=True)
                if not pl_raw.exists():
                    log.info("Fetching pressure-level fields %s times=%s", date_str, times)
                    self._fetch_pressure_levels(pl_raw, date_str, times, area=area)

            # 2-D single-level fields (instantaneous + accumulated together).
            sl_raw = self.output_dir / f"ERA5_sl_{date_str}.grib"
            if sl_raw.exists() and not _is_grib_file_readable(sl_raw):
                log.warning("Corrupted GRIB detected, re-downloading: %s", sl_raw.name)
                sl_raw.unlink(missing_ok=True)
            if not sl_raw.exists():
                log.info("Fetching single-level fields %s times=%s", date_str, times)
                self._fetch_single_levels(sl_raw, date_str, times, area=area)

            # Split per-timestep and merge PL + SL into one file per step.
            day_files = self._merge_and_split(pl_raw, sl_raw, static_path, date_str, times)
            all_files.extend(day_files)

            if self.cleanup_raw_daily_grib:
                pl_raw.unlink(missing_ok=True)
                sl_raw.unlink(missing_ok=True)

        all_files.sort(key=lambda p: p.name)

        # Generate AVAILABLE file.
        available = self._write_available(all_files)

        return DownloadResult(
            available_file=available,
            meteo_dir=self.output_dir,
            n_timesteps=len(all_files),
            files=all_files,
        )

    # ── CDS fetch helpers ─────────────────────────────────────────────────────

    def _fetch_pressure_levels(
        self, dest: Path, date: str, times: list[str], *, area: list[float] | None
    ) -> None:
        import cdsapi
        c = cdsapi.Client(quiet=True)
        req: dict = {
            "product_type": "reanalysis",
            "variable": _PL_VARIABLES,
            "pressure_level": self.pressure_levels,
            "date": date,
            "time": times,
            "format": "grib",
        }
        if area:
            req["area"] = area
        c.retrieve("reanalysis-era5-pressure-levels", req, str(dest))

    def _fetch_model_levels(
        self, dest: Path, date: str, times: list[str], *, area: list[float] | None
    ) -> None:
        """Fetch ERA5 model-level analysis fields via ERA5 complete.

        FLEXPART's ECMWF input reader expects model-level data with hybrid
        coefficients (GRIB key ``pv``), which are absent in pressure-level
        products.
        """
        import cdsapi

        c = cdsapi.Client(quiet=True)
        req: dict = {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "an",
            "levtype": "ml",
            "levelist": "1/to/137",
            "param": _ML_PARAM_IDS,
            "date": date,
            "time": times,
            "format": "grib",
        }
        if self.model_level_grid_deg is not None:
            req["grid"] = [float(self.model_level_grid_deg), float(self.model_level_grid_deg)]
        if area:
            req["area"] = area

        try:
            c.retrieve("reanalysis-era5-complete", req, str(dest))
        except Exception:
            if not area:
                raise
            if not self.model_level_allow_global_fallback:
                raise RuntimeError(
                    "Model-level ERA5 request with area clipping failed and global fallback is disabled. "
                    "This is intentional to avoid very large global downloads. "
                    "Check CDS request constraints or explicitly enable global fallback."
                )
            # Some CDS environments reject area subsetting for ERA5 complete;
            # optionally retry globally.
            log.warning(
                "Model-level request with area clipping failed for %s; retrying global field.",
                date,
            )
            req.pop("area", None)
            c.retrieve("reanalysis-era5-complete", req, str(dest))

    def _fetch_single_levels(
        self, dest: Path, date: str, times: list[str], *, area: list[float] | None
    ) -> None:
        import cdsapi
        c = cdsapi.Client(quiet=True)
        req: dict = {
            "product_type": "reanalysis",
            "variable": _SFC_INSTANT + _SFC_ACCUMULATED,
            "date": date,
            "time": times,
            "format": "grib",
        }
        if area:
            req["area"] = area
        c.retrieve("reanalysis-era5-single-levels", req, str(dest))

    def _fetch_static(self, dest: Path, *, area: list[float] | None) -> None:
        import cdsapi
        c = cdsapi.Client(quiet=True)
        req: dict = {
            "product_type": "reanalysis",
            "variable": _STATIC_VARIABLES,
            "year": "2000",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "grib",
        }
        if area:
            req["area"] = area
        c.retrieve("reanalysis-era5-single-levels", req, str(dest))

    # ── GRIB splitting + merging ──────────────────────────────────────────────

    def _merge_and_split(
        self,
        pl_path: Path,
        sl_path: Path,
        static_path: Path,
        date_str: str,
        times: list[str],
    ) -> list[Path]:
        """For each timestep, write one GRIB file containing PL + SL + static."""
        import eccodes

        # Collect messages from PL file, keyed by (date, time).
        pl_msgs: dict[tuple[int, int], list[bytes]] = {}
        _collect_grib_messages(pl_path, pl_msgs)

        sl_msgs: dict[tuple[int, int], list[bytes]] = {}
        _collect_grib_messages(sl_path, sl_msgs)

        # Static messages are appended to every timestep file.
        static_raw: list[bytes] = []
        if static_path.exists():
            with open(static_path, "rb") as fh:
                while True:
                    msg = eccodes.codes_grib_new_from_file(fh)
                    if msg is None:
                        break
                    buf = eccodes.codes_get_message(msg)
                    static_raw.append(buf)
                    eccodes.codes_release(msg)

        out_files: list[Path] = []
        date_int = int(date_str.replace("-", ""))

        for time_str in times:
            if ":" in time_str:
                hh = int(time_str[:2])
                mm = int(time_str[3:5])
            else:
                hh = int(time_str[:2])
                mm = 0
            time_int = hh * 100 + mm   # HHMM as integer (GRIB dataTime)

            fname = f"EA{date_int:08d}{hh:02d}"
            out_path = self.output_dir / fname

            if out_path.exists():
                out_files.append(out_path)
                continue

            msgs: list[bytes] = []
            msgs.extend(pl_msgs.get((date_int, time_int), []))
            msgs.extend(sl_msgs.get((date_int, time_int), []))
            msgs.extend(static_raw)

            if not msgs:
                log.warning("No GRIB messages found for %s %04d — skipping.", date_str, time_int)
                continue

            with open(out_path, "wb") as fh:
                for raw in msgs:
                    fh.write(raw)

            log.debug("Wrote %s (%d messages)", fname, len(msgs))
            out_files.append(out_path)

        return out_files

    # ── AVAILABLE file ────────────────────────────────────────────────────────

    def _write_available(self, files: list[Path]) -> Path:
        """Write the FLEXPART AVAILABLE index file.

        Format (3 header lines then data):
            YYYYMMDD HHMMSS      filename
        where filename is bare (no directory), and FLEXPART prepends the
        meteo_dir from the pathnames file at runtime.
        """
        available_path = self.output_dir / "AVAILABLE"
        lines = [
            "XXXXXX EMPTY LINES XXXXXXXXX",
            "XXXXXX EMPTY LINES XXXXXXXX",
            "YYYYMMDD HHMMSS      name of the file (up to 255 characters)",
        ]
        for fp in files:
            dt = _filename_to_dt(fp.name)
            if dt is None:
                log.warning("Could not parse datetime from filename %s — skipping.", fp.name)
                continue
            date_str = dt.strftime("%Y%m%d")
            time_str = dt.strftime("%H%M%S")
            lines.append(f"{date_str} {time_str}      {fp.name}")

        available_path.write_text("\n".join(lines) + "\n")
        log.info("Wrote AVAILABLE with %d entries → %s", len(files), available_path)
        return available_path


# ── Module-level helpers ──────────────────────────────────────────────────────

def _require_cdsapi() -> None:
    try:
        import cdsapi  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is required to download ERA5 data.\n"
            "Install it with:  pip install cdsapi\n"
            "Then add your CDS credentials to ~/.cdsapirc — see\n"
            "  https://cds.climate.copernicus.eu/how-to-api"
        ) from exc


def _require_eccodes() -> None:
    try:
        import eccodes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The eccodes Python package is required to split GRIB files.\n"
            "Install it with:  pip install eccodes\n"
            "The system eccodes library (brew install eccodes / apt eccodes) "
            "must also be present."
        ) from exc


def _parse_dt(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return s.replace(tzinfo=timezone.utc) if s.tzinfo is None else s
    dt = datetime.fromisoformat(s.rstrip("Z"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _bbox_to_area(bbox: tuple[float, float, float, float]) -> list[float]:
    """Convert (lon_min, lat_min, lon_max, lat_max) → CDS [N, W, S, E]."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return [lat_max, lon_min, lat_min, lon_max]


def _day_windows(
    start: datetime, end: datetime
) -> Iterator[tuple[datetime, datetime]]:
    """Yield (day_start, day_end) for each calendar day in [start, end]."""
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end:
        day_end = current.replace(hour=23, minute=59, second=59)
        yield current, min(day_end, end)
        current += timedelta(days=1)


def _hour_list(day_start: datetime, day_end: datetime, step: int) -> list[str]:
    """Return HH:00 strings for every `step` hours within the day window."""
    hours = []
    h = day_start.hour - (day_start.hour % step)  # align to step boundary
    while h <= min(day_end.hour, 23):
        hours.append(f"{h:02d}:00")
        h += step
    return hours


def _collect_grib_messages(
    grib_path: Path, store: dict[tuple[int, int], list[bytes]]
) -> None:
    """Read all messages from *grib_path* into *store* keyed by (validityDate, validityTime).

    Using validityTime (not dataTime) ensures accumulated fields (precipitation,
    heat fluxes, surface stresses) are placed at the timestep they *end* at,
    which is what FLEXPART expects.  For instantaneous fields dataTime==validityTime
    so the behaviour is unchanged.
    """
    import eccodes

    with open(grib_path, "rb") as fh:
        while True:
            msg = eccodes.codes_grib_new_from_file(fh)
            if msg is None:
                break
            vdate = eccodes.codes_get(msg, "validityDate")   # YYYYMMDD int
            vtime = eccodes.codes_get(msg, "validityTime")   # HHMM int (e.g. 0, 300, 600)
            raw = eccodes.codes_get_message(msg)
            eccodes.codes_release(msg)
            store.setdefault((vdate, vtime), []).append(raw)


def _is_grib_file_readable(path: Path) -> bool:
    """Best-effort GRIB integrity check using ecCodes message iteration."""
    try:
        import eccodes
    except ImportError:
        # If eccodes is unavailable we cannot validate; keep existing behavior.
        return True

    try:
        with open(path, "rb") as fh:
            while True:
                msg = eccodes.codes_grib_new_from_file(fh)
                if msg is None:
                    break
                eccodes.codes_release(msg)
        return True
    except Exception:
        return False


def _filename_to_dt(name: str) -> datetime | None:
    """Parse ``EA{YYYYMMDD}{HH}`` → datetime, return None on failure."""
    if not name.startswith("EA") or len(name) < 12:
        return None
    try:
        date_part = name[2:10]   # YYYYMMDD
        hour_part = name[10:12]  # HH
        return datetime(
            int(date_part[:4]),
            int(date_part[4:6]),
            int(date_part[6:8]),
            int(hour_part),
            tzinfo=timezone.utc,
        )
    except (ValueError, IndexError):
        return None


def is_flexpart_meteo_compatible(
    meteo_dir: str | Path,
    available_file: str | Path | None = None,
) -> bool:
    """Return True if meteorology appears compatible with FLEXPART ECMWF mode.

    The check inspects the first file listed in AVAILABLE and verifies that it
    carries hybrid-level coefficients (GRIB key ``pv``).
    """
    try:
        import eccodes
    except ImportError:
        return False

    meteo_path = Path(meteo_dir)
    available_path = Path(available_file) if available_file else (meteo_path / "AVAILABLE")
    if not available_path.exists():
        return False

    first_name = _first_available_filename(available_path)
    if first_name is None:
        return False

    met_file = meteo_path / first_name
    if not met_file.exists():
        return False

    with open(met_file, "rb") as fh:
        gid = eccodes.codes_grib_new_from_file(fh)
        if gid is None:
            return False
        try:
            try:
                eccodes.codes_get_size(gid, "pv")
                return True
            except Exception:
                return False
        finally:
            eccodes.codes_release(gid)


def available_covers_window(
    available_file: str | Path,
    start: str | datetime,
    end: str | datetime,
    *,
    timestep_hours: int = 3,
) -> bool:
    """Return True if AVAILABLE contains all timestamps in [start, end]."""
    apath = Path(available_file)
    if not apath.exists():
        return False

    t0 = _parse_dt(start)
    t1 = _parse_dt(end)
    if t1 < t0:
        return False

    available_steps = _available_timestamps(apath)
    if not available_steps:
        return False

    step = timedelta(hours=timestep_hours)
    current = t0
    while current <= t1:
        if current not in available_steps:
            return False
        current += step
    return True


def _available_timestamps(available_file: Path) -> set[datetime]:
    out: set[datetime] = set()
    for line in available_file.read_text().splitlines()[3:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        ds, ts = parts[0], parts[1]
        if len(ds) != 8 or len(ts) < 4:
            continue
        try:
            dt = datetime(
                int(ds[0:4]), int(ds[4:6]), int(ds[6:8]),
                int(ts[0:2]), int(ts[2:4]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        out.add(dt)
    return out


def _first_available_filename(available_file: Path) -> str | None:
    lines = available_file.read_text().splitlines()
    for line in lines[3:]:
        parts = line.split()
        if not parts:
            continue
        return parts[-1]
    return None
