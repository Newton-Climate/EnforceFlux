"""FLEXPART source types: point and diffuse emission sources."""
import math
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PointSource:
    """Single-location methane release (e.g. landfill, well pad)."""

    id: str
    lon: float
    lat: float
    alt_m: float
    emission_rate_kg_s: float   # kg s⁻¹ — total emission rate
    start: datetime
    end: datetime
    n_particles: int = 10_000


@dataclass
class DiffuseSource:
    """Area emission discretised into a regular lat/lon grid of FLEXPART releases.

    Typical use: rice paddy fields, wetlands, agricultural zones.
    Each cell produces one RELEASES block; total mass is
    ``flux × cell_area × duration``.
    """

    id: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    alt_m: float
    emission_flux_kg_m2_s: float   # kg m⁻² s⁻¹
    start: datetime
    end: datetime
    cell_size_deg: float = 0.1
    n_particles_per_cell: int = 1_000

    def cells(self) -> list[tuple[float, float, float, float, float]]:
        """Return (lon1, lat1, lon2, lat2, mass_kg) for each discretised cell."""
        R = 6_371_000.0  # Earth radius in metres
        duration = (self.end - self.start).total_seconds()
        result: list[tuple[float, float, float, float, float]] = []
        lon = self.lon_min
        while lon < self.lon_max - 1e-9:
            lon2 = min(lon + self.cell_size_deg, self.lon_max)
            lat = self.lat_min
            while lat < self.lat_max - 1e-9:
                lat2 = min(lat + self.cell_size_deg, self.lat_max)
                lat_c = math.radians((lat + lat2) / 2.0)
                dx = math.radians(lon2 - lon) * R * math.cos(lat_c)
                dy = math.radians(lat2 - lat) * R
                area_m2 = abs(dx * dy)
                mass_kg = self.emission_flux_kg_m2_s * area_m2 * duration
                result.append((lon, lat, lon2, lat2, mass_kg))
                lat += self.cell_size_deg
            lon += self.cell_size_deg
        return result
