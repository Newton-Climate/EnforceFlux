"""Translate the shared run config into each model's native configuration.

The shared YAML is authoritative; this is where it becomes what each backend
actually expects. AERMOD takes a config dict directly. FLEXPART and MicroHH are
driven by their own YAML schemas, so those are **generated** into the run
directory — written to disk rather than passed in memory, so a run that fails
inside a Fortran binary leaves behind exactly the input file it was given.

Meteorology always goes through :mod:`enforceflux.meteo`: one
:class:`~enforceflux.meteo.record.MetSeries` is read once and adapted per
model, so the models cannot end up on different weather.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from enforceflux.meteo.record import MetRecord, MetSeries
from enforceflux.models.source import Source
from enforceflux.transport.run_config import TransportRunConfig


# ── Meteorology ──────────────────────────────────────────────────────────────


def build_met_series(run: TransportRunConfig) -> MetSeries:
    """The run's meteorology, from an ``era5:`` block or inline records."""
    from enforceflux.meteo import met_series_from_era5

    met = run.met
    if not met:
        raise ValueError(
            "The run config has no 'met' section. Provide met.era5 (a GRIB "
            "directory to read) or met.records (inline canonical records)."
        )

    if "era5" in met:
        era5 = dict(met["era5"])
        geographic = sorted({"longitude", "latitude"} & set(era5))
        if geographic:
            raise ValueError(
                f"met.era5 declares {geographic}. The met column is taken at "
                "domain.origin_lon/origin_lat — the run's single geographic "
                "anchor — so remove these."
            )
        lon, lat = run.origin_lonlat
        meteo_dir = era5.pop("meteo_dir", None)
        if meteo_dir is None:
            raise ValueError("met.era5 requires 'meteo_dir'")
        era5.setdefault("start", run.start)
        era5.setdefault("end", run.end)
        return met_series_from_era5(run.resolve(meteo_dir), lon, lat, **era5)

    if "records" in met:
        records = tuple(_met_record(r, run) for r in met["records"])
        lon, lat = run.origin_lonlat
        return MetSeries(
            records=records,
            longitude=lon,
            latitude=lat,
            provenance={"source": "inline"},
        )

    raise ValueError(
        f"Unrecognised met section keys {sorted(met)}; expected 'era5' or 'records'."
    )


def _met_record(blob: dict[str, Any], run: TransportRunConfig) -> MetRecord:
    missing = [k for k in ("wind_speed_m_s", "wind_direction_deg") if k not in blob]
    if missing:
        raise ValueError(f"met.records[] entry is missing {missing}")
    time = blob.get("time")
    return MetRecord(
        time=(
            datetime.fromisoformat(str(time).rstrip("Z")).replace(tzinfo=timezone.utc)
            if time
            else (run.start or datetime(2020, 1, 1, tzinfo=timezone.utc))
        ),
        wind_speed_m_s=float(blob["wind_speed_m_s"]),
        wind_direction_deg=float(blob["wind_direction_deg"]),
        temperature_k=float(blob.get("temperature_k", 293.15)),
        mixing_height_m=float(blob.get("mixing_height_m", 800.0)),
        friction_velocity_m_s=float(blob.get("friction_velocity_m_s", 0.3)),
        sensible_heat_flux_w_m2=float(blob.get("sensible_heat_flux_w_m2", 0.0)),
        surface_roughness_m=float(blob.get("surface_roughness_m", 0.1)),
        surface_pressure_pa=float(blob.get("surface_pressure_pa", 100000.0)),
    )


# ── Shared geometry ──────────────────────────────────────────────────────────


def projected_sources(run: TransportRunConfig) -> list[Source]:
    """Run sources as :class:`Source` objects in local metres (for AERMOD).

    Positions are already Cartesian metres about the origin, so this is a
    straight adaptation — no projection is involved.
    """
    sources: list[Source] = []
    for item in run.sources:
        x, y = item.x_m, item.y_m
        sources.append(
            Source(
                id=item.id,
                kind="point",
                x=float(x),
                y=float(y),
                z=item.altitude_m,
                flux_true=item.emission_rate_kg_s,
                flux_prior_mean=(
                    item.prior_mean_kg_s
                    if item.prior_mean_kg_s is not None
                    else item.emission_rate_kg_s
                ),
                flux_prior_std=(
                    item.prior_std_kg_s
                    if item.prior_std_kg_s is not None
                    else max(item.emission_rate_kg_s, 1.0e-12)
                ),
            )
        )
    return sources


def projected_receptors(run: TransportRunConfig) -> list[dict[str, Any]]:
    """Run receptors as AERMOD receptor dicts in local metres."""
    receptors = []
    for item in run.receptors:
        x, y = item.x_m, item.y_m
        receptors.append(
            {
                "id": item.id,
                "x": float(x),
                "y": float(y),
                "z": item.altitude_m or run.domain.receptor_height_m,
            }
        )
    return receptors


def projected_grid(run: TransportRunConfig) -> dict[str, float]:
    """The domain as an AERMOD receptor grid in local metres."""
    return {
        "x_min": float(run.domain.x_min),
        "x_max": float(run.domain.x_max),
        "y_min": float(run.domain.y_min),
        "y_max": float(run.domain.y_max),
        "spacing_m": run.domain.spacing_m,
        "height_m": run.domain.receptor_height_m,
    }


# ── AERMOD ───────────────────────────────────────────────────────────────────


def aermod_config(run: TransportRunConfig, series: MetSeries) -> dict[str, Any]:
    """The plugin config for :mod:`enforceflux.plugins.transport_aermod`.

    Emissions arrive in kg s⁻¹ and the canonical unit is ng m⁻³, so the
    concentration convention is pinned here rather than left to the user — a
    run's output must be comparable across models.
    """
    from enforceflux.meteo import to_aermod

    options = run.options
    config: dict[str, Any] = {
        "concentration_units": "ng_m3_per_kg_s",
        "emission_scale_to_kg_s": 1.0,
        "reduce": options.get("reduce", "stack"),
        "receptor_path_samples": int(options.get("receptor_path_samples", 1)),
        "default_stack": options.get("default_stack", {}),
        "stacks": options.get("stacks", {}),
    }
    if "options" in options:
        config["options"] = options["options"]

    config["receptors"] = projected_receptors(run)
    config["grid"] = projected_grid(run)
    # Already-built SurfaceMet objects, the in-process equivalent of the
    # plugin's 'era5' block — the canonical series has been read once upstream.
    config["met_objects"] = to_aermod(series)
    return config


# ── FLEXPART ─────────────────────────────────────────────────────────────────


def write_flexpart_config(
    run: TransportRunConfig, series: MetSeries, run_dir: Path
) -> Path:
    """Generate a native FLEXPART simulation YAML and return its path."""
    import yaml

    from enforceflux.meteo import to_flexpart

    options = run.options
    required = ["executable", "options_dir"]
    missing = [k for k in required if k not in options]
    if missing:
        raise ValueError(
            f"Running FLEXPART needs a 'flexpart:' block with {missing} — the "
            "compiled binary and its options template have no counterpart in the "
            "other models, so they cannot be shared keys."
        )

    if not series.provenance.get("meteo_dir"):
        raise ValueError(
            "FLEXPART reads ERA5 GRIB directly, so it cannot be driven by inline "
            "'met.records' — those only reach the models that take scalar "
            "boundary-layer parameters (AERMOD, MicroHH). Point 'met.era5' at a "
            "GRIB directory to run FLEXPART from this config."
        )
    met_source = to_flexpart(series, start=run.start, end=run.end)
    start = run.start or series.start
    end = run.end or series.end
    dlon, dlat = run.domain.spacing_deg

    blob = {
        "flexpart": {
            "executable": str(run.resolve(options["executable"])),
            "options_dir": str(run.resolve(options["options_dir"])),
            "available_file": str(met_source.available_file),
            "meteo_dir": str(met_source.meteo_dir),
            "run_dir": str(run_dir / "flexpart_run"),
        },
        "simulation": {
            "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "output_step_seconds": int(options.get("output_step_seconds", 3600)),
            "sync_seconds": int(options.get("sync_seconds", 900)),
            "nxshift": int(options.get("nxshift", -9999)),
        },
        # FLEXPART is lon/lat-gridded, so the Cartesian domain is projected
        # out here. These bounds are derived from the origin + extent, never
        # declared, so they cannot disagree with the metres above.
        "domain": {
            "lon_min": run.domain.lon_min,
            "lat_min": run.domain.lat_min,
            "lon_max": run.domain.lon_max,
            "lat_max": run.domain.lat_max,
            "dx": float(options.get("dx", dlon)),
            "dy": float(options.get("dy", dlat)),
            "heights_m": list(run.domain.heights_m),
        },
        "species": {
            "name": str(options.get("species_name", "CH4")),
            "number": int(options.get("species_number", 24)),
        },
        "sources": [
            {
                "type": "point",
                "id": source.id,
                **dict(zip(("lon", "lat"), run.to_lonlat(source.x_m, source.y_m))),
                "alt_m": source.altitude_m,
                "emission_rate_kg_s": source.emission_rate_kg_s,
                "n_particles": int(options.get("n_particles", 100_000)),
            }
            for source in run.sources
        ],
        "output": {
            "path": str(run_dir / "flexpart_native.nc"),
            "compress": run.output.compress,
            "per_source": bool(options.get("per_source", False)),
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "flexpart_generated.yaml"
    path.write_text(yaml.safe_dump(blob, sort_keys=False))
    return path


# ── MicroHH ──────────────────────────────────────────────────────────────────


def write_microhh_config(
    run: TransportRunConfig, series: MetSeries, run_dir: Path
) -> Path:
    """Generate a native MicroHH case YAML and return its path.

    The LES box is aligned with the mean wind (its ``x_bearing_deg`` comes from
    the same met series every other model uses), and the forcing block is the
    canonical met adapted by :func:`enforceflux.meteo.to_microhh_forcing`.
    """
    import yaml

    from enforceflux.meteo import microhh_box_bearing, to_microhh_forcing

    options = run.options
    if "executable" not in options:
        raise ValueError(
            "Running MicroHH needs a 'microhh:' block with 'executable' — the "
            "compiled LES binary has no counterpart in the other models."
        )

    reduce = str(options.get("met_reduce", "daytime_mean"))
    forcing = to_microhh_forcing(
        series,
        reduce=reduce,
        min_directional_consistency=float(options.get("min_directional_consistency", 0.6)),
    )
    bearing = microhh_box_bearing(series, reduce=reduce)

    grid = dict(options.get("grid") or {})
    itot = int(grid.get("itot", 192))
    jtot = int(grid.get("jtot", 96))
    ktot = int(grid.get("ktot", 64))
    xsize = float(grid.get("xsize", 3840.0))
    ysize = float(grid.get("ysize", 1920.0))
    zsize = float(grid.get("zsize", 2048.0))

    start = run.start or series.start

    blob = {
        "microhh": {
            "executable": str(run.resolve(options["executable"])),
            "case_dir": str(run_dir / "microhh_case"),
            # MPI ranks. Decomposed into (npx,npy) against the grid when the
            # case loads, so an invalid count is rejected there, not mid-run.
            "num_workers": int(options.get("num_workers", 1)),
        },
        "simulation": {
            "name": str(options.get("case_name", "transport_run")),
            "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "spinup_seconds": int(options.get("spinup_seconds", 1800)),
            "runtime_seconds": int(options.get("runtime_seconds", 3600)),
            "sampletime": int(options.get("sampletime", 60)),
        },
        "grid": {
            "itot": itot, "jtot": jtot, "ktot": ktot,
            "xsize": xsize, "ysize": ysize, "zsize": zsize,
            # Optional near-surface stretching; absent keys leave it uniform.
            **{k: float(grid[k]) for k in ("dz_surface_m", "dz_max_m") if k in grid},
        },
        # MicroHH re-projects lon/lat into its own wind-aligned box, so the
        # case file is written in lon/lat about the same shared origin.
        "domain": {
            "origin_lon": run.domain.origin_lon,
            "origin_lat": run.domain.origin_lat,
            "x_bearing_deg": float(options.get("x_bearing_deg", bearing)),
            "source_x0": float(options.get("source_x0", 0.15 * xsize)),
            "source_y0": float(options.get("source_y0", 0.5 * ysize)),
        },
        "forcing": {
            "u_geo": forcing.u_geo,
            "v_geo": forcing.v_geo,
            "z0m": forcing.z0m,
            "z0h": forcing.z0h,
            "thl_surface_K": forcing.thl_surface_K,
            "thl_lapse_K_per_m": forcing.thl_lapse_K_per_m,
            "boundary_layer_height_m": forcing.boundary_layer_height_m,
            "inversion_strength_K": forcing.inversion_strength_K,
            "inversion_depth_m": forcing.inversion_depth_m,
            "surface_heat_flux_K_m_s": forcing.surface_heat_flux_K_m_s,
        },
        "species": {
            "name": str(options.get("scalar_name", "ch4")),
            "emission_scale": float(options.get("emission_scale", 1.0)),
        },
        "sources": [
            {
                "id": source.id,
                **dict(zip(("lon", "lat"), run.to_lonlat(source.x_m, source.y_m))),
                "alt_m": source.altitude_m,
                "emission_rate_kg_s": source.emission_rate_kg_s,
                "sigma_x_m": float(options.get("source_sigma_m", 25.0)),
                "sigma_y_m": float(options.get("source_sigma_m", 25.0)),
                "sigma_z_m": float(options.get("source_sigma_m", 25.0)),
            }
            for source in run.sources
        ],
        "instruments": [
            {
                "id": receptor.id,
                **dict(zip(("lon", "lat"), run.to_lonlat(receptor.x_m, receptor.y_m))),
                "alt_m": receptor.altitude_m or run.domain.receptor_height_m,
            }
            for receptor in run.receptors
        ],
        "output": {"path": str(run_dir / "microhh_native.nc")},
    }

    # Area sources realised as a 2-D surface flux instead of volumetric blobs.
    # Keyed by source id so the emission rate stays declared once, in the shared
    # `sources:` block, and the model block only says HOW to realise it.
    surface = dict(options.get("surface_flux_sources") or {})
    if surface:
        by_id = {s.id: s for s in run.sources}
        unknown = sorted(set(surface) - set(by_id))
        if unknown:
            raise ValueError(
                f"microhh.surface_flux_sources names {unknown}, which are not in "
                f"`sources:` (have {sorted(by_id)})."
            )
        blob["surface_flux_patches"] = [
            {
                "id": sid,
                **dict(zip(("lon", "lat"),
                           run.to_lonlat(by_id[sid].x_m, by_id[sid].y_m))),
                "side_m": float(spec["side_m"]),
                "emission_rate_kg_s": by_id[sid].emission_rate_kg_s,
            }
            for sid, spec in surface.items()
        ]
        # Those sources are now the bottom boundary condition; leaving them in
        # the [source] block as well would emit everything twice.
        blob["sources"] = [s for s in blob["sources"] if s["id"] not in surface]

    # Explicit cross-section planes. Without these the slice follows sources[0],
    # so two runs with different source geometry slice different planes.
    cross = {k[6:]: float(options[k]) for k in ("cross_xy_m", "cross_xz_m") if k in options}
    if cross:
        blob["cross"] = cross

    # Raw .ini passthrough ({section: {key: value}}), for MicroHH modules the
    # canonical config has no vocabulary for — e.g. [dump] 3-D field output.
    if options.get("extra_ini"):
        blob["extra_ini"] = dict(options["extra_ini"])

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "microhh_generated.yaml"
    path.write_text(yaml.safe_dump(blob, sort_keys=False))
    return path


def default_window(series: MetSeries) -> tuple[datetime, datetime]:
    """A one-day window from a series, for configs that omit start/end."""
    return series.start, series.start + timedelta(days=1)
