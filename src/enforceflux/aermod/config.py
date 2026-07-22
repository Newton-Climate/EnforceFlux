"""Python-first configuration objects for the AERMOD-style dispersion model.

Everything the model needs is expressed as plain dataclasses, so a run can be
driven entirely from Python::

    from enforceflux.aermod import AermodConfig, AermodModel, StackParameters, SurfaceMet

    cfg = AermodConfig(
        met=[SurfaceMet(wind_speed_m_s=3.0, wind_direction_deg=225.0, stability_class="D")],
        default_stack=StackParameters(height_m=10.0),
    )
    g = AermodModel(cfg).jacobian(sources, receptors)

The same structures round-trip through dicts (:meth:`AermodConfig.from_dict`),
which is what the registry plugins and the JSON/YAML configs use — the dict form
is a serialization of the Python API, never a separate input language.

Coordinates are Cartesian metres: ``x`` east, ``y`` north, ``z`` above ground.
AERMOD's dispersion algebra is dimensional, so a projected CRS (UTM et al.) is
required — geographic degrees will silently produce nonsense.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

STABILITY_CLASSES = ("A", "B", "C", "D", "E", "F")


def _require_keys(blob: dict, keys: Sequence[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


def _opt_float(blob: dict, key: str) -> float | None:
    value = blob.get(key)
    return None if value is None else float(value)


@dataclass(frozen=True)
class StackParameters:
    """Release geometry and exhaust state for one source.

    Defaults describe a passive ground-level release (no plume rise): zero stack
    height, zero exit velocity, and ambient exit temperature. Give
    ``diameter_m``/``exit_velocity_m_s``/``exit_temperature_k`` to activate
    Briggs momentum and buoyancy rise.
    """

    height_m: float = 0.0
    diameter_m: float = 0.0
    exit_velocity_m_s: float = 0.0
    exit_temperature_k: float = 0.0  # 0 → ambient (neutrally buoyant release)

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "StackParameters":
        return cls(
            height_m=float(blob.get("height_m", 0.0)),
            diameter_m=float(blob.get("diameter_m", 0.0)),
            exit_velocity_m_s=float(blob.get("exit_velocity_m_s", 0.0)),
            exit_temperature_k=float(blob.get("exit_temperature_k", 0.0)),
        )


@dataclass(frozen=True)
class SurfaceMet:
    """One hour (or one steady condition) of boundary-layer meteorology.

    Only ``wind_speed_m_s`` and ``wind_direction_deg`` are required. Everything
    else is either defaulted or derived by
    :func:`~enforceflux.aermod.meteorology.derive_met_state`: given a Pasquill
    stability class the Monin-Obukhov length follows from Golder (1972), and
    ``u*``, ``w*``, and the mixing height follow from surface-layer similarity.
    Supplying a measured value always overrides the derivation.
    """

    wind_speed_m_s: float
    wind_direction_deg: float  # meteorological convention: direction wind blows *from*
    temperature_k: float = 293.15
    stability_class: str | None = None  # "A".."F"
    monin_obukhov_length_m: float | None = None
    mixing_height_m: float | None = None
    surface_roughness_m: float = 0.1
    friction_velocity_m_s: float | None = None
    convective_velocity_m_s: float | None = None
    sensible_heat_flux_w_m2: float | None = None
    reference_height_m: float = 10.0
    # Potential-temperature gradient above the mixed layer, used for stable
    # plume rise and for capping vertical growth.
    potential_temperature_gradient_k_m: float = 0.01
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.stability_class is not None:
            cls_ = str(self.stability_class).strip().upper()
            if cls_ not in STABILITY_CLASSES:
                raise ValueError(
                    f"stability_class must be one of {STABILITY_CLASSES}, got {self.stability_class!r}"
                )
            object.__setattr__(self, "stability_class", cls_)
        if self.stability_class is None and self.monin_obukhov_length_m is None:
            raise ValueError(
                "SurfaceMet needs either stability_class ('A'..'F') or "
                "monin_obukhov_length_m to define boundary-layer stability."
            )
        if self.wind_speed_m_s < 0.0:
            raise ValueError("wind_speed_m_s must be non-negative")
        if self.surface_roughness_m <= 0.0:
            raise ValueError("surface_roughness_m must be positive")
        if self.monin_obukhov_length_m is not None and self.monin_obukhov_length_m == 0.0:
            raise ValueError("monin_obukhov_length_m must be non-zero (positive=stable)")

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "SurfaceMet":
        _require_keys(blob, ["wind_speed_m_s", "wind_direction_deg"], "aermod.met")
        return cls(
            wind_speed_m_s=float(blob["wind_speed_m_s"]),
            wind_direction_deg=float(blob["wind_direction_deg"]),
            temperature_k=float(blob.get("temperature_k", 293.15)),
            stability_class=blob.get("stability_class"),
            monin_obukhov_length_m=_opt_float(blob, "monin_obukhov_length_m"),
            mixing_height_m=_opt_float(blob, "mixing_height_m"),
            surface_roughness_m=float(blob.get("surface_roughness_m", 0.1)),
            friction_velocity_m_s=_opt_float(blob, "friction_velocity_m_s"),
            convective_velocity_m_s=_opt_float(blob, "convective_velocity_m_s"),
            sensible_heat_flux_w_m2=_opt_float(blob, "sensible_heat_flux_w_m2"),
            reference_height_m=float(blob.get("reference_height_m", 10.0)),
            potential_temperature_gradient_k_m=float(
                blob.get("potential_temperature_gradient_k_m", 0.01)
            ),
            timestamp=blob.get("timestamp"),
        )


@dataclass(frozen=True)
class Receptor:
    """A point at which concentration is evaluated."""

    id: str
    x: float
    y: float
    z: float = 0.0
    # Weight used when several receptors are averaged into one observation
    # (path-averaged open-path instruments sample a line as N points).
    weight: float = 1.0
    group: str | None = None  # observation this receptor belongs to

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "Receptor":
        _require_keys(blob, ["id", "x", "y"], "aermod.receptor")
        return cls(
            id=str(blob["id"]),
            x=float(blob["x"]),
            y=float(blob["y"]),
            z=float(blob.get("z", 0.0)),
            weight=float(blob.get("weight", 1.0)),
            group=blob.get("group"),
        )


@dataclass(frozen=True)
class ReceptorGrid:
    """Regular Cartesian receptor grid for forward simulations."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    spacing_m: float
    height_m: float = 0.0

    def __post_init__(self) -> None:
        if self.spacing_m <= 0.0:
            raise ValueError("ReceptorGrid.spacing_m must be positive")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("ReceptorGrid extents must satisfy max > min")

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "ReceptorGrid":
        _require_keys(
            blob, ["x_min", "x_max", "y_min", "y_max", "spacing_m"], "aermod.grid"
        )
        return cls(
            x_min=float(blob["x_min"]),
            x_max=float(blob["x_max"]),
            y_min=float(blob["y_min"]),
            y_max=float(blob["y_max"]),
            spacing_m=float(blob["spacing_m"]),
            height_m=float(blob.get("height_m", 0.0)),
        )


@dataclass(frozen=True)
class DispersionOptions:
    """Numerical knobs for the dispersion kernel.

    ``reflections`` is the number of image sources used to enforce no-flux
    boundaries at the ground and at the mixing height; 3 is ample for the
    well-mixed limit. ``min_wind_speed_m_s`` implements AERMOD's low-wind floor
    (the steady-state plume equation is singular as ``u → 0``).
    """

    reflections: int = 3
    min_wind_speed_m_s: float = 0.5
    min_sigma_v_m_s: float = 0.2
    min_sigma_w_m_s: float = 0.02
    gradual_plume_rise: bool = True
    convective_bigaussian: bool = True
    # Skewness of the CBL vertical-velocity PDF and the ratio σ_wj / |w̄j|
    # (Weil et al. 1997 closure, as adopted by AERMOD).
    cbl_skewness: float = 0.105
    cbl_sigma_ratio: float = 2.0

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "DispersionOptions":
        defaults = cls()
        kwargs = {
            name: type(getattr(defaults, name))(blob[name])
            for name in cls.__dataclass_fields__
            if name in blob
        }
        return cls(**kwargs)


@dataclass(frozen=True)
class AermodConfig:
    """Complete specification of an AERMOD-style run.

    Parameters
    ----------
    met:
        One or more :class:`SurfaceMet` conditions. Multiple entries are treated
        as independent hours; results are reduced across them by ``reduce``.
    default_stack:
        Release geometry applied to every source without an explicit entry in
        ``stacks``.
    stacks:
        Per-source overrides keyed by ``Source.id``.
    emission_scale_to_kg_s:
        Multiplier converting the caller's flux units into kg s⁻¹ (e.g.
        ``1/3600`` when fluxes are kg hr⁻¹).
    concentration_units:
        ``"s_per_m3"`` (default) leaves the response as χ/Q in s m⁻³;
        ``"ug_m3_per_g_s"`` scales to µg m⁻³ per g s⁻¹, AERMOD's native output;
        ``"ppb_ch4_per_kg_s"`` converts to a CH₄ mixing-ratio enhancement.
    reduce:
        What :meth:`~enforceflux.aermod.model.AermodModel.jacobian` does with
        the hour axis — every hour is always solved separately regardless.
        ``"stack"`` makes each (hour, receptor) pair its own observation row and
        is what a time-resolved OSSE wants; ``"mean"`` (default) gives the
        Jacobian of a period-mean observation; ``"max"`` is a screening
        statistic; ``"none"`` leaves the axis in place.
    """

    met: tuple[SurfaceMet, ...]
    default_stack: StackParameters = field(default_factory=StackParameters)
    stacks: dict[str, StackParameters] = field(default_factory=dict)
    receptors: tuple[Receptor, ...] = ()
    grid: ReceptorGrid | None = None
    options: DispersionOptions = field(default_factory=DispersionOptions)
    emission_scale_to_kg_s: float = 1.0
    concentration_units: str = "s_per_m3"
    reduce: str = "mean"
    receptor_path_samples: int = 1

    _UNIT_SCALE = {
        # χ/Q is computed in s m⁻³ (kg m⁻³ per kg s⁻¹).
        "s_per_m3": 1.0,
        "ug_m3_per_g_s": 1.0e6,
        # CH₄: 1 kg m⁻³ ≈ 1/(0.016 kg mol⁻¹) mol m⁻³; at 293.15 K, 1 atm air is
        # 41.6 mol m⁻³, so 1 kg m⁻³ CH₄ ≈ 1.503e9 ppb.
        "ppb_ch4_per_kg_s": 1.5028e9,
        # The canonical cross-model unit (see enforceflux.transport.canonical).
        "ng_m3_per_kg_s": 1.0e12,
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "met", tuple(self.met))
        object.__setattr__(self, "receptors", tuple(self.receptors))
        if not self.met:
            raise ValueError("AermodConfig requires at least one SurfaceMet entry")
        if self.concentration_units not in self._UNIT_SCALE:
            raise ValueError(
                f"Unknown concentration_units={self.concentration_units!r}. "
                f"Expected one of {sorted(self._UNIT_SCALE)}"
            )
        if self.reduce not in ("stack", "mean", "max", "none"):
            raise ValueError(
                f"Unknown reduce={self.reduce!r}; expected 'stack', 'mean', 'max', or 'none'"
            )
        if self.receptor_path_samples < 1:
            raise ValueError("receptor_path_samples must be >= 1")

    @property
    def unit_scale(self) -> float:
        """Multiplier from internal χ/Q [s m⁻³] to ``concentration_units``."""
        return self._UNIT_SCALE[self.concentration_units]

    def stack_for(self, source_id: str) -> StackParameters:
        return self.stacks.get(source_id, self.default_stack)

    def with_met(self, met: Iterable[SurfaceMet]) -> "AermodConfig":
        """Return a copy driven by different meteorology (everything else fixed)."""
        return replace(self, met=tuple(met))

    # ── Serialization ────────────────────────────────────────────────────────

    @classmethod
    def from_dict(
        cls, blob: dict[str, Any], *, met: Sequence[SurfaceMet] | None = None
    ) -> "AermodConfig":
        """Build a config from a dict, optionally with meteorology supplied directly.

        ``met`` takes precedence over ``blob["met"]`` and is how already-built
        :class:`SurfaceMet` objects enter — e.g. the ERA5 adapter's output (see
        :func:`enforceflux.meteo.adapters.to_aermod`).
        """
        if met is not None:
            met = tuple(met)
        else:
            met_blob = blob.get("met")
            if met_blob is None:
                raise ValueError(
                    "AERMOD config requires a 'met' entry: a mapping (single condition), "
                    "a list of hourly conditions, or an 'era5' block naming ERA5 "
                    "meteorology to read."
                )
            if isinstance(met_blob, dict):
                met_blob = [met_blob]
            met = tuple(SurfaceMet.from_dict(m) for m in met_blob)

        stacks = {
            str(k): StackParameters.from_dict(v)
            for k, v in (blob.get("stacks") or {}).items()
        }
        default_stack = StackParameters.from_dict(blob.get("default_stack") or {})
        receptors = tuple(
            Receptor.from_dict(r) for r in (blob.get("receptors") or [])
        )
        grid_blob = blob.get("grid")
        options_blob = blob.get("options") or {}

        return cls(
            met=met,
            default_stack=default_stack,
            stacks=stacks,
            receptors=receptors,
            grid=ReceptorGrid.from_dict(grid_blob) if grid_blob else None,
            options=DispersionOptions.from_dict(options_blob),
            emission_scale_to_kg_s=float(blob.get("emission_scale_to_kg_s", 1.0)),
            concentration_units=str(blob.get("concentration_units", "s_per_m3")),
            reduce=str(blob.get("reduce", "mean")).lower(),
            receptor_path_samples=int(blob.get("receptor_path_samples", 1)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AermodConfig":
        """Load from a JSON or YAML file (YAML needs ``pyyaml`` installed)."""
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # imported lazily: YAML is optional

            blob = yaml.safe_load(text)
        else:
            blob = json.loads(text)
        if not isinstance(blob, dict):
            raise ValueError(f"AERMOD config at {path} must parse to a mapping")
        # Allow either a bare config or one nested under an "aermod" key.
        return cls.from_dict(blob.get("aermod", blob))
