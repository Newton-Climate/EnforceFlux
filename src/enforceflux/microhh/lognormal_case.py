"""Write a MicroHH case whose surface flux is a lognormal ``field`` (M6b).

Generalizes ``examples/microhh_paddy_spatial_basis.py``: instead of tiling nine
per-patch tracers, this writes a single ``ch4`` scalar and encodes the whole
spatial pattern into the ``<scalar>_bot_in.<time>`` surface-flux binary. The
integrated emission over the field equals ``Q_true_kg_s`` exactly (kinematic
flux = mass flux / rho, matching MicroHH's ``sbcbot=flux`` convention).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from enforceflux.source_fields.lognormal_gp import FieldGrid


R_D = 287.04
P_BOT = 100_000.0


def _reference_density_kg_m3(template_ini: str) -> float:
    """Recover the near-surface density MicroHH will use from the ini's temperature.

    Prefer ``thl_surface_K`` if the template carries it as a comment; otherwise
    fall back to a standard 289.6 K, matching the paddy-basis example.
    """
    m = re.search(r"thl_surface_K\s*=\s*([0-9.eE+-]+)", template_ini)
    T = float(m.group(1)) if m else 289.5965302519878
    return P_BOT / (R_D * T)


def _rewrite_ini_single_scalar(text: str, scalar: str) -> str:
    """Collapse any tiled tracer list back to a single scalar name."""
    def repl(key: str, s: str) -> str:
        return re.sub(rf"^(\s*{key}\s*=\s*).*$", rf"\1{scalar}", s, flags=re.MULTILINE)
    for key in ("slist", "sbot_2d_list", "scalar_outflow", "limitlist", "fluxlimit_list"):
        text = repl(key, text)
    # crosslist may carry '<scalar>,<scalar>_path'; keep the _path companion.
    text = re.sub(
        r"^(\s*crosslist\s*=\s*).*$",
        rf"\1{scalar},{scalar}_path",
        text,
        flags=re.MULTILINE,
    )
    return text


def _bot_time_stamp(cfg_ini: str) -> str:
    """MicroHH reads the sbot_2d file at t = savetime intervals; the first file
    is stamped ``0000000``. Only the initial (t=0) file is required."""
    return "0000000"


def write_lognormal_case(
    template: Path,
    out_dir: Path,
    field: np.ndarray,
    grid: FieldGrid,
    Q_true_kg_s: float,
    *,
    scalar: str = "ch4",
) -> Path:
    """Copy ``template`` into ``out_dir`` and write a single-scalar case whose
    surface emission field equals ``field`` (kg s-1 per cell) renormalised to
    ``Q_true_kg_s``. Returns the path to the written ``.ini``.

    ``field`` is ``(ny, nx)`` on the caller's ``grid``; it is rasterised
    directly onto the MicroHH cross-section, so the two must share ``dx_m``
    and cell count.
    """
    template = Path(template)
    out_dir = Path(out_dir)
    if not template.is_dir():
        raise FileNotFoundError(f"MicroHH template dir does not exist: {template}")

    inis = sorted(template.glob("*.ini"))
    if not inis:
        raise FileNotFoundError(f"No .ini in template {template}")
    if len(inis) > 1:
        raise ValueError(f"Ambiguous template — multiple .ini in {template}: {inis}")
    name = inis[0].stem

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in template.iterdir():
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)

    ini_text = (out_dir / f"{name}.ini").read_text()
    ini_text = _rewrite_ini_single_scalar(ini_text, scalar)
    (out_dir / f"{name}.ini").write_text(ini_text)

    if field.shape != (grid.ny, grid.nx):
        raise ValueError(
            f"field shape {field.shape} does not match grid {(grid.ny, grid.nx)}"
        )

    areas = grid.cell_areas_m2()
    total = float((field * areas).sum())
    if total <= 0.0:
        raise ValueError("field integrates to zero; cannot renormalise to Q_true.")
    # field is a mass flux (kg m-2 s-1) whose area-integral must equal Q_true;
    # rescale by the ratio and the surface bcondition file expects kinematic
    # flux = mass_flux / rho.
    mass_flux_kg_m2_s = field * (Q_true_kg_s / total)

    rho = _reference_density_kg_m3(ini_text)
    kinematic = mass_flux_kg_m2_s / rho          # what sbot_2d expects

    ts = _bot_time_stamp(ini_text)
    bot_path = out_dir / f"{scalar}_bot_in.{ts}"
    kinematic.astype(np.float64).tofile(bot_path)

    return out_dir / f"{name}.ini"


__all__ = ["write_lognormal_case"]
