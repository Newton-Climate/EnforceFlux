"""Run nine independently scaled 100 m rice-paddy patches as LES tracers.

The passive tracers share one turbulent flow, producing the source-basis
responses needed for spatial OP/EC DFS calculations.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import netCDF4 as nc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "runs" / "sacramento_paddy_ec_30m" / "paddy" / "microhh_case"
OUT = ROOT / "runs" / "sacramento_paddy_spatial_basis" / "microhh_case"
NAME = "transport_run"
N = 3
TOTAL_KG_S = 1.35e-4


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ini = (TEMPLATE / f"{NAME}.ini").read_text()
    tracers = [f"ch4_{j}{i}" for j in range(N) for i in range(N)]
    joined = ",".join(tracers)
    ini = ini.replace("slist=ch4", f"slist={joined}")
    ini = ini.replace("sbot_2d_list=ch4", f"sbot_2d_list={joined}")
    ini = ini.replace("scalar_outflow=ch4", f"scalar_outflow={joined}")
    ini = ini.replace("crosslist=ch4,ch4_path", f"crosslist={joined}")
    ini = ini.replace("limitlist=ch4", f"limitlist={joined}")
    (OUT / f"{NAME}.ini").write_text(ini)

    shutil.copy2(TEMPLATE / f"{NAME}_input.nc", OUT / f"{NAME}_input.nc")
    with nc.Dataset(OUT / f"{NAME}_input.nc", "a") as ds:
        init = ds.groups["init"]
        z = len(ds.dimensions["z"])
        for tracer in tracers:
            init.createVariable(tracer, "f8", ("z",))[:] = np.zeros(z)
            init.createVariable(f"{tracer}_inflow", "f8", ("z",))[:] = np.zeros(z)

    # 20 m LES grid, exactly five cells along each edge of a 100 m patch.
    rho = 100000.0 / (287.04 * 289.5965302519878)
    flux = (TOTAL_KG_S / 9) / (100.0 * 100.0) / rho
    for j in range(N):
        for i in range(N):
            field = np.zeros((96, 192), dtype=np.float64)
            # Patches tile x,y = [810,1110] m around the field centre (960 m).
            field[40 + 5*j:45 + 5*j, 40 + 5*i:45 + 5*i] = flux
            field.tofile(OUT / f"ch4_{j}{i}_bot_in.0000000")

    exe = ROOT / "microhh" / "build" / "microhh"
    launch = ["mpirun", "-n", "4", str(exe)]
    subprocess.run([*launch, "init", NAME], cwd=OUT, check=True)
    subprocess.run([*launch, "run", NAME], cwd=OUT, check=True)


if __name__ == "__main__":
    main()
