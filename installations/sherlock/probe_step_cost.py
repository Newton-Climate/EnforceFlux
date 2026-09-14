"""Summarise MicroHH per-step cost for the Sherlock scaling probes.

MicroHH's `<case>.out` prints CPUDT = wall seconds for each block of
`outputiter` (50) iterations. Median over full blocks, skipping the first
(startup) row and the trailing partial block.
"""
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
rows = []
for out in sorted(root.glob("sherlock_probe_*/dispersion/concentration_microhh/microhh_case/*.out")):
    run = out.parts[-5]
    m = re.match(r"sherlock_probe_dx(\d+)_n(\d+)", run)
    lines = [l.split() for l in out.read_text().splitlines()[1:] if l.strip()]
    blocks, prev = [], 0
    for f in lines:
        it, cpudt = int(f[0]), float(f[2])
        if it - prev == 50:
            blocks.append(cpudt / 50)
        prev = it
    if not blocks:
        continue
    rows.append((int(m[1]), int(m[2]), statistics.median(blocks) * 1e3,
                 float(lines[-1][3]), len(blocks)))

print("dx_m,ranks,ms_per_step,dt_s,n_blocks,speedup_vs_4,efficiency")
base = {dx: ms * n for dx, n, ms, *_ in rows if n == 4}
for dx, n, ms, dt, nb in sorted(rows):
    b = base.get(dx)
    sp = f"{b / 4 / ms:.2f}" if b else ""  # b/4 = the 4-rank ms per step
    eff = f"{b / (ms * n):.2f}" if b else ""
    print(f"{dx},{n},{ms:.1f},{dt:.3g},{nb},{sp},{eff}")
