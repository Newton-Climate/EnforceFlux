"""Two-phase driver for the fine-grid optical-C_n^2 case: spin-up -> restart -> sample.

The base MicroHH writer emits only the scalar cross-sections at an integer
cadence, and the runner always starts from t=0. Optical C_n^2 needs something
different:

  * the TEMPERATURE field (`th`) written as an xy cross-section at beam height,
  * at a 0.5 s cadence (sub-second — the YAML loader would round it to 0),
  * over a short window that starts from a fully spun-up state, not from rest.

This driver gets there in two phases against the SAME case directory, patching
the generated .ini text rather than modifying the library:

  Phase 1 (spin-up):  starttime=0, endtime=spinup, savetime=spinup, coarse
                      cadence. `init` + `run` -> MicroHH writes restart files
                      (th.<spinup>, u/v/w/ch4.<spinup>, ...).
  Phase 2 (sample):   starttime=spinup, endtime=spinup+runtime, sampletime=0.5,
                      dtmax=0.5, and crosslist patched to include `th`. `run`
                      ONLY (no init, no clean) so MicroHH warm-starts from the
                      Phase-1 restart. -> th.xy.<lvl>.<time> every 0.5 s.

Then:
    python examples/optical_scintillation_psd.py \
        --th-field ../runs/microhh_optical_cn2_fine/th.xy.<lvl>.<time> \
        --itot 480 --jtot 240 --spacing-m 4

Usage:
    python examples/optical_cn2_fine_run.py                     # dry-run: patch + print
    python examples/optical_cn2_fine_run.py --execute           # actually run both phases
    python examples/optical_cn2_fine_run.py --execute --skip-spinup   # reuse a saved restart
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

from enforceflux.microhh import load_microhh_config
from enforceflux.microhh.case import write_case
from enforceflux.microhh.runner import MicroHHRunner

REPO = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO / "examples" / "microhh_optical_cn2_fine.yaml"


def _set_key(ini: str, section: str, key: str, value: str) -> str:
    """Set `key=value` inside [section]: replace the line if present, else insert.

    Line-based so it never disturbs other keys or sections. Raises if the
    section is absent (the base writer always emits [time]/[cross]/... so a
    miss means a typo, not a valid case).
    """
    out, in_sec, done, header_at = [], False, False, None
    header = f"[{section}]"
    for line in ini.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_sec and not done:            # leaving target section, key never seen
                out.append(f"{key}={value}")
                done = True
            in_sec = (stripped == header)
            if in_sec:
                header_at = len(out)
        elif in_sec and not done and re.match(rf"\s*{re.escape(key)}\s*=", line):
            out.append(f"{key}={value}")
            done = True
            continue
        out.append(line)
    if in_sec and not done:                    # section was the file's last one
        out.append(f"{key}={value}")
        done = True
    if not done:
        if header_at is None:
            raise KeyError(f"section [{section}] not found in .ini")
        out.insert(header_at + 1, f"{key}={value}")
    return "\n".join(out) + "\n"


def _beam_level(cfg) -> int:
    z = cfg.grid.levels()
    return int(np.argmin(np.abs(z - (cfg.cross_xy_m or cfg.sources[0].alt_m))))


def patch_ini(ini: str, *, starttime: int, endtime: int, savetime: int,
              sampletime: str, dtmax: str, scalar: str, add_th_cross: bool) -> str:
    ini = _set_key(ini, "time", "starttime", str(starttime))
    ini = _set_key(ini, "time", "endtime", str(endtime))
    ini = _set_key(ini, "time", "savetime", str(savetime))
    ini = _set_key(ini, "time", "dtmax", dtmax)
    for sec in ("stats", "cross", "column"):
        ini = _set_key(ini, sec, "sampletime", sampletime)
    if add_th_cross:
        # Add temperature to the xy cross list. The base writer already sets
        # `xy` to the beam height in METRES (cfg.cross_xy_m); MicroHH resolves
        # that to the nearest level, so we leave `xy` untouched.
        ini = _set_key(ini, "cross", "crosslist", f"th,{scalar},{scalar}_path")
    return ini


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_YAML)
    ap.add_argument("--execute", action="store_true",
                    help="actually invoke MicroHH (default: dry-run, write patched .ini only)")
    ap.add_argument("--skip-spinup", action="store_true",
                    help="reuse an existing restart at t=spinup; run only the sampling phase")
    args = ap.parse_args()

    cfg = load_microhh_config(args.config)
    spinup, runtime = cfg.spinup_s, cfg.runtime_s
    scalar = cfg.scalar_name
    kbeam = _beam_level(cfg)
    exe = cfg.executable

    ini_path = cfg.case_dir / f"{cfg.case_name}.ini"

    def run_phase(args_list: list[str]) -> None:
        cmd = [str(exe), *args_list]
        print(f"    $ {' '.join(cmd)}   (cwd={cfg.case_dir})")
        if args.execute:
            subprocess.run(cmd, cwd=str(cfg.case_dir), check=True)

    print(f"case dir       : {cfg.case_dir}")
    print(f"grid           : {cfg.grid.itot}x{cfg.grid.jtot}x{cfg.grid.ktot}  dx={cfg.grid.dx} m")
    print(f"beam level idx : {kbeam}  (z={cfg.grid.levels()[kbeam]:.1f} m)")
    print(f"spin-up / window: {spinup} s / {runtime} s   sampling dt=0.5 s\n")

    runner = MicroHHRunner(cfg)

    # ── Phase 1: spin-up ────────────────────────────────────────────────────
    if not args.skip_spinup:
        write_case(cfg)                      # base .ini + input.nc
        ini = ini_path.read_text()
        ini = patch_ini(ini, starttime=0, endtime=spinup, savetime=spinup,
                        sampletime="60", dtmax="6.", scalar=scalar,
                        add_th_cross=False)
        ini_path.write_text(ini)
        print("[phase 1] spin-up  (save restart at t=%d)" % spinup)
        if args.execute:
            runner.clean_outputs()           # fresh start; keeps .ini + input.nc
        run_phase(["init", cfg.case_name])
        run_phase(["run", cfg.case_name])
    else:
        print("[phase 1] skipped — reusing restart at t=%d" % spinup)

    # ── Phase 2: sampling window (warm-start, temperature cross at 0.5 s) ────
    if not ini_path.exists():
        write_case(cfg)
    ini = ini_path.read_text()
    ini = patch_ini(ini, starttime=spinup, endtime=spinup + runtime, savetime=spinup + runtime,
                    sampletime="0.5", dtmax="0.5", scalar=scalar,
                    add_th_cross=True)
    ini_path.write_text(ini)
    print("\n[phase 2] sampling window  (NO init/clean — warm-start from restart)")
    print(f"          crosslist patched to include `th` at level {kbeam}")
    run_phase(["run", cfg.case_name])        # run only: reads restart at starttime

    print("\nDone." if args.execute else "\nDry-run complete: patched .ini written, nothing executed.")
    print("Next:")
    print("  python examples/optical_scintillation_psd.py \\")
    print(f"      --th-field {cfg.case_dir}/th.xy.{kbeam:03d}.{spinup+runtime:07d} \\")
    print(f"      --itot {cfg.grid.itot} --jtot {cfg.grid.jtot} --spacing-m {cfg.grid.dx:g}")


if __name__ == "__main__":
    main()
