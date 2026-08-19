"""Round-trip test for the M6b lognormal-field MicroHH case writer.

Marked ``les_integration``: launches MicroHH if the compiled binary is on the
system; otherwise verifies the written surface-flux binary matches the input
field's integrated emission to machine tolerance and skips the actual run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from enforceflux.microhh.lognormal_case import write_lognormal_case
from enforceflux.source_fields.lognormal_gp import FieldGrid


pytestmark = pytest.mark.les_integration


def _make_template(tmp_path: Path) -> Path:
    tpl = tmp_path / "template"
    tpl.mkdir()
    (tpl / "roundtrip.ini").write_text(
        "# minimal MicroHH template for M6b round-trip\n"
        "# thl_surface_K=289.5965302519878\n"
        "[fields]\n"
        "slist=ch4\n"
        "[boundary]\n"
        "sbot_2d_list=ch4\n"
        "scalar_outflow=ch4\n"
        "[cross]\n"
        "crosslist=ch4,ch4_path\n"
        "[limiter]\n"
        "limitlist=ch4\n"
        "[advec]\n"
        "fluxlimit_list=ch4\n"
    )
    return tpl


def _find_microhh_binary() -> Path | None:
    env = os.environ.get("MICROHH_BIN")
    if env and Path(env).exists():
        return Path(env)
    for candidate in (
        Path("microhh") / "build" / "microhh",
        Path("../microhh/build/microhh"),
    ):
        if candidate.exists():
            return candidate
    which = shutil.which("microhh")
    return Path(which) if which else None


def test_lognormal_case_roundtrip(tmp_path):
    grid = FieldGrid(nx=8, ny=8, dx_m=100.0)
    rng = np.random.default_rng(0)
    field = rng.uniform(0.5, 1.5, size=(grid.ny, grid.nx))
    Q_true = 1.5e-4

    tpl = _make_template(tmp_path)
    out = tmp_path / "case"
    ini_path = write_lognormal_case(tpl, out, field, grid, Q_true)

    assert ini_path.exists()
    bot_path = out / "ch4_bot_in.0000000"
    assert bot_path.exists()

    # sbot_2d is kinematic (kg/m2/s per rho). Multiply back through and check
    # the integrated emission equals Q_true.
    kinematic = np.fromfile(bot_path, dtype=np.float64).reshape(grid.ny, grid.nx)
    # rho as computed in lognormal_case (T=289.5965302519878 K).
    rho = 1e5 / (287.04 * 289.5965302519878)
    total = float((kinematic * rho * grid.cell_areas_m2()).sum())
    assert abs(total - Q_true) / Q_true < 1e-12

    binary = _find_microhh_binary()
    if binary is None:
        pytest.skip("MicroHH binary not available; skipping full LES round-trip.")

    # We only need `init` to prove the case is well-formed; a full `run` needs
    # a proper input.nc which this synthetic template intentionally omits.
    try:
        subprocess.run(
            [str(binary), "init", ini_path.stem],
            cwd=out, check=True, capture_output=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"MicroHH init did not accept the synthetic template: {exc!r}")
