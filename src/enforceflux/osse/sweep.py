"""OSSE Monte-Carlo sweep driver (M4).

Loads a ``sweep.yaml`` referencing base dispersion/flux/analysis YAMLs, expands
the outer product of ``sweep.grid.*``, runs the three stages per cell (reusing
H per ``sweep.cache.reuse_H_across`` bucket), and writes one parquet row per
realization.

Row schema (columns):

    L, CV, N, layout_seed, met_id, realization, transport,
    L_B, e_q, dfs_total, chi2_per_dof, prior_influence, ak_diag_mean,
    inverse_crime_flag, run_dir
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import runpy
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_APPS_DIR = REPO_ROOT / "apps"


# ---------------------------------------------------------------------------
# H-cache protocol
# ---------------------------------------------------------------------------


class HCache(Protocol):
    def get(self, key: str) -> Path | None: ...
    def put(self, key: str, path: Path) -> None: ...


class InMemoryHCache:
    """Process-local cache; Agent F's M5 disk-backed cache is a drop-in."""

    def __init__(self) -> None:
        self._store: dict[str, Path] = {}

    def get(self, key: str) -> Path | None:
        return self._store.get(key)

    def put(self, key: str, path: Path) -> None:
        self._store[key] = Path(path)


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _expand_grid_entry(key: str, spec: Any) -> list[Any]:
    """Expand a single grid entry into a list of concrete values."""
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        if "range" in spec:
            a, b = spec["range"]
            return list(range(int(a), int(b)))
        if spec.get("kind") == "nested_space_filling":
            ns = [int(n) for n in spec["ns"]]
            seeds = [int(s) for s in spec["seeds"]]
            return [{"n": n, "seed": s, "kind": "nested_space_filling"}
                    for n in ns for s in seeds]
        raise ValueError(f"grid[{key!r}]: unsupported dict spec {spec!r}")
    # Scalar → single-element list.
    return [spec]


def expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of grid entries → list of ``{dotted_path: value}``."""
    keys = list(grid.keys())
    per_key = [_expand_grid_entry(k, grid[k]) for k in keys]
    cells: list[dict[str, Any]] = []
    for combo in itertools.product(*per_key):
        cells.append(dict(zip(keys, combo)))
    return cells


def _h_cache_key(cell: dict[str, Any], reuse_across: Iterable[str]) -> str:
    """SHA of the cell values EXCLUDING the reuse-across keys.

    Two cells that differ only in the reuse-across keys share an H, so their
    key collapses to the same string.
    """
    reuse = set(reuse_across)
    stable = {k: cell[k] for k in sorted(cell) if k not in reuse}
    blob = json.dumps(stable, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def group_by_h_cache(
    cells: list[dict[str, Any]], reuse_across: Iterable[str]
) -> dict[str, list[int]]:
    """Return ``{hkey: [row_indices]}`` preserving cell order within a group."""
    groups: dict[str, list[int]] = {}
    for i, cell in enumerate(cells):
        groups.setdefault(_h_cache_key(cell, reuse_across), []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _apply_overrides(
    base_dispersion: dict, base_flux: dict, base_analysis: dict,
    cell: dict[str, Any],
) -> tuple[dict, dict, dict]:
    """Apply dotted overrides. Keys prefixed with dispersion./flux./analysis./
    instrument. route into the matching config."""
    d = copy.deepcopy(base_dispersion)
    f = copy.deepcopy(base_flux)
    a = copy.deepcopy(base_analysis)
    for k, v in cell.items():
        if k == "instrument.network":
            _materialize_network(d, v)
            continue
        prefix = k.split(".", 1)[0]
        # Dotted paths mirror the YAML wrapper: the first segment IS the
        # wrapper's outer key, so apply the full path (not the tail) to the
        # matching config.
        if prefix == "dispersion":
            _set_dotted(d, k, v)
        elif prefix == "flux":
            _set_dotted(f, k, v)
        elif prefix == "analysis":
            _set_dotted(a, k, v)
        elif prefix == "instrument":
            # Unrecognized instrument.* keys: no-op for M4.
            continue
        else:
            raise ValueError(f"override key must start with dispersion./flux./"
                             f"analysis./instrument.; got {k!r}")
    return d, f, a


def _materialize_network(dispersion_cfg: dict, spec: Any) -> None:
    """Replace ``dispersion.receptors`` with a nested_sequence layout draw."""
    if not isinstance(spec, dict) or spec.get("kind") != "nested_space_filling":
        return
    from enforceflux.networks.layouts import nested_sequence, space_filling
    from enforceflux.source_fields.lognormal_gp import FieldGrid

    src_cfg = ((dispersion_cfg.get("dispersion") or {}).get("sources") or {}).get("config") or {}
    g = src_cfg.get("grid") or {}
    domain = FieldGrid(
        nx=int(g.get("nx", 64)), ny=int(g.get("ny", 64)),
        dx_m=float(g.get("dx_m", 50.0)),
        origin_x_m=float(g.get("origin_x_m", 0.0)),
        origin_y_m=float(g.get("origin_y_m", 0.0)),
    )
    n = int(spec["n"])
    seed = int(spec["seed"])
    layouts = nested_sequence([n], space_filling, seed, domain=domain)
    receptors = layouts[n]
    alt_m = float(
        (dispersion_cfg.get("dispersion") or {}).get("domain", {}).get("receptor_height_m", 3.0)
    )
    dispersion_cfg.setdefault("dispersion", {})["receptors"] = [
        {"id": r.id, "x_m": r.x_m, "y_m": r.y_m, "alt_m": alt_m} for r in receptors
    ]


# ---------------------------------------------------------------------------
# Row execution
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    name: str
    base_dispersion: Path
    base_flux: Path
    base_analysis: Path
    grid: dict[str, Any]
    reuse_H_across: list[str] = field(default_factory=list)
    workers: int = 1
    parquet_out: Path = Path("runs/sweep/sweep.parquet")
    outputs_root: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: Path) -> "SweepConfig":
        blob = yaml.safe_load(Path(path).read_text()) or {}
        run_meta = blob.get("run") or {}
        sweep = blob.get("sweep") or {}
        base = sweep.get("base") or {}
        yaml_dir = Path(path).resolve().parent
        def _rp(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else (yaml_dir / q).resolve()
        parquet = _rp(str((sweep.get("output") or {}).get("parquet", "runs/sweep/sweep.parquet")))
        return cls(
            name=str(run_meta.get("name", "sweep")),
            base_dispersion=_rp(str(base["dispersion"])),
            base_flux=_rp(str(base["flux"])),
            base_analysis=_rp(str(base["analysis"])),
            grid=dict(sweep.get("grid") or {}),
            reuse_H_across=list((sweep.get("cache") or {}).get("reuse_H_across") or []),
            workers=int((sweep.get("parallel") or {}).get("workers", 1)),
            parquet_out=parquet,
            outputs_root=parquet.parent / "runs",
        )


def _row_id(idx: int, cell: dict[str, Any]) -> str:
    blob = json.dumps(cell, sort_keys=True, default=str).encode()
    tag = hashlib.sha256(blob).hexdigest()[:8]
    return f"r{idx:05d}_{tag}"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def _write_stage_yaml(
    scratch: Path, stage: str, base_blob: dict,
    row_name: str, outputs_root: Path,
    inputs: dict[str, Path] | None = None,
) -> Path:
    blob = copy.deepcopy(base_blob)
    blob["run"] = {"name": row_name, "outputs_root": str(outputs_root)}
    blob["inputs"] = {k: str(v) for k, v in (inputs or {}).items()}
    out = scratch / f"{stage}.yaml"
    out.write_text(yaml.safe_dump(blob, sort_keys=False))
    return out


def _run_app(script: str, config: Path) -> None:
    """Invoke apps/<script>.py in-process with ``--config <config>``."""
    saved_argv = sys.argv[:]
    saved_path = sys.path[:]
    if str(_APPS_DIR) not in sys.path:
        sys.path.insert(0, str(_APPS_DIR))
    sys.argv = [script, "--config", str(config)]
    try:
        # Import fresh so module-level state (pending writes) resets each call.
        from enforceflux.plugins.source_lognormal_field import clear_pending_writes
        clear_pending_writes()
        runpy.run_path(str(_APPS_DIR / script), run_name="__main__")
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path


# ---------------------------------------------------------------------------
# H reuse: shared-H dispersion writes a new per-row RunDir sharing jacobian.npz
# ---------------------------------------------------------------------------


def _dispersion_full(
    scratch: Path, base: dict, row_name: str, outputs_root: Path,
) -> Path:
    disp_yaml = _write_stage_yaml(scratch, "dispersion", base, row_name, outputs_root)
    _run_app("dispersion_main.py", disp_yaml)
    return outputs_root / row_name / "dispersion"


def _dispersion_shared_h(
    scratch: Path, base: dict, row_name: str, outputs_root: Path,
    cached_disp_dir: Path,
) -> Path:
    """Copy H from cache, generate row-specific truth_field/basis_mapping."""
    from enforceflux.plugins.source_lognormal_field import (
        LognormalFieldSource, clear_pending_writes, drain_pending_writes,
    )
    from enforceflux.runs import open_run_dir

    row_disp = outputs_root / row_name / "dispersion"
    if row_disp.exists():
        shutil.rmtree(row_disp)
    row_disp.mkdir(parents=True, exist_ok=True)

    run_dir = open_run_dir(
        stage="dispersion", run_name=row_name,
        outputs_root=str(outputs_root),
        inputs={},
    )

    src_cfg = (base.get("dispersion") or {}).get("sources", {}).get("config") or {}
    clear_pending_writes()
    LognormalFieldSource().build_sources(src_cfg, None)
    for relpath, role in drain_pending_writes(run_dir.root):
        run_dir.record_output(relpath, role=role)

    # Copy the cached Jacobian in.
    jac_src = cached_disp_dir / "jacobian.npz"
    if not jac_src.is_file():
        raise RuntimeError(
            f"cached dispersion RunDir {cached_disp_dir} has no jacobian.npz"
        )
    shutil.copy2(jac_src, run_dir.path("jacobian.npz"))
    run_dir.record_output("jacobian.npz", role="jacobian")
    run_dir.snapshot_config({"reused_from": str(cached_disp_dir)})
    run_dir.finalize()
    return row_disp


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_sweep(
    sweep_cfg: SweepConfig,
    *, hcache: HCache | None = None,
    scratch_root: Path | None = None,
) -> Path:
    """Execute the sweep and write ``sweep_cfg.parquet_out``. Returns its path."""
    if hcache is None:
        hcache = InMemoryHCache()

    base_disp = _load_yaml(sweep_cfg.base_dispersion)
    base_flux = _load_yaml(sweep_cfg.base_flux)
    base_ana = _load_yaml(sweep_cfg.base_analysis)

    cells = expand_grid(sweep_cfg.grid)
    groups = group_by_h_cache(cells, sweep_cfg.reuse_H_across)

    transport = str(
        ((base_disp.get("dispersion") or {}).get("transport") or {}).get("model", "aermod")
    ).strip().lower()

    # --- M5 flexpart hook (Agent F) ---
    if transport == "flexpart":
        raise NotImplementedError(
            "flexpart branch is Agent F's M5 territory"
        )
    # --- end M5 hook ---

    scratch_root = Path(scratch_root or (sweep_cfg.outputs_root / "_scratch"))
    scratch_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    workers = max(1, int(sweep_cfg.workers))

    # For each cache group: first row is "full", the rest share H.
    for hkey, row_indices in groups.items():
        first_idx = row_indices[0]
        first_row = _run_row(
            first_idx, cells[first_idx], base_disp, base_flux, base_ana,
            sweep_cfg, scratch_root, hcache, hkey, shared_h=False, transport=transport,
        )
        rows.append(first_row)
        rest = row_indices[1:]
        if not rest:
            continue
        if workers == 1:
            for idx in rest:
                rows.append(_run_row(
                    idx, cells[idx], base_disp, base_flux, base_ana,
                    sweep_cfg, scratch_root, hcache, hkey, shared_h=True,
                    transport=transport,
                ))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        _run_row_pickleable,
                        idx, cells[idx],
                        str(sweep_cfg.base_dispersion),
                        str(sweep_cfg.base_flux),
                        str(sweep_cfg.base_analysis),
                        str(sweep_cfg.outputs_root),
                        str(scratch_root),
                        list(sweep_cfg.reuse_H_across),
                        hkey,
                        str(_cached_dir_for(hcache, hkey)),
                        transport,
                    )
                    for idx in rest
                ]
                for fut in futures:
                    rows.append(fut.result())

    _write_parquet(sweep_cfg.parquet_out, rows)
    return sweep_cfg.parquet_out


def _cached_dir_for(hcache: HCache, hkey: str) -> Path:
    p = hcache.get(hkey)
    if p is None:
        raise RuntimeError(f"H-cache miss for group {hkey}")
    return p


def _run_row_pickleable(
    idx, cell, base_disp_path, base_flux_path, base_ana_path,
    outputs_root, scratch_root, reuse_across, hkey, cached_disp_dir, transport,
):
    base_disp = _load_yaml(Path(base_disp_path))
    base_flux = _load_yaml(Path(base_flux_path))
    base_ana = _load_yaml(Path(base_ana_path))
    sweep_cfg = SweepConfig(
        name="", base_dispersion=Path(base_disp_path),
        base_flux=Path(base_flux_path), base_analysis=Path(base_ana_path),
        grid={}, reuse_H_across=list(reuse_across), workers=1,
        parquet_out=Path("/tmp/_ignored.parquet"),
        outputs_root=Path(outputs_root),
    )
    hcache = InMemoryHCache()
    hcache.put(hkey, Path(cached_disp_dir))
    return _run_row(
        idx, cell, base_disp, base_flux, base_ana, sweep_cfg,
        Path(scratch_root), hcache, hkey, shared_h=True, transport=transport,
    )


def _run_row(
    idx: int, cell: dict[str, Any],
    base_disp: dict, base_flux: dict, base_ana: dict,
    sweep_cfg: SweepConfig, scratch_root: Path, hcache: HCache,
    hkey: str, *, shared_h: bool, transport: str,
) -> dict[str, Any]:
    row_name = _row_id(idx, cell)
    row_scratch = scratch_root / row_name
    row_scratch.mkdir(parents=True, exist_ok=True)

    d_cfg, f_cfg, a_cfg = _apply_overrides(base_disp, base_flux, base_ana, cell)
    outputs_root = sweep_cfg.outputs_root

    if shared_h:
        cached = hcache.get(hkey)
        if cached is None:
            # Fallback to full if cache miss.
            disp_dir = _dispersion_full(row_scratch, d_cfg, row_name, outputs_root)
            hcache.put(hkey, disp_dir)
        else:
            disp_dir = _dispersion_shared_h(
                row_scratch, d_cfg, row_name, outputs_root, cached,
            )
    else:
        disp_dir = _dispersion_full(row_scratch, d_cfg, row_name, outputs_root)
        hcache.put(hkey, disp_dir)

    flux_yaml = _write_stage_yaml(
        row_scratch, "flux", f_cfg, row_name, outputs_root,
        inputs={"dispersion": disp_dir},
    )
    _run_app("flux_main.py", flux_yaml)
    flux_dir = outputs_root / row_name / "flux"

    ana_yaml = _write_stage_yaml(
        row_scratch, "analysis", a_cfg, row_name, outputs_root,
        inputs={"dispersion": disp_dir, "flux": flux_dir},
    )
    _run_app("analysis_main.py", ana_yaml)
    ana_dir = outputs_root / row_name / "analysis"

    return _collect_row(idx, cell, disp_dir, flux_dir, ana_dir, transport)


# ---------------------------------------------------------------------------
# Row → parquet columns
# ---------------------------------------------------------------------------


def _collect_row(
    idx: int, cell: dict[str, Any],
    disp_dir: Path, flux_dir: Path, ana_dir: Path, transport: str,
) -> dict[str, Any]:
    flux_summary = json.loads((flux_dir / "summary.json").read_text())
    ana_summary_path = ana_dir / "summary.json"
    ana_summary = (
        json.loads(ana_summary_path.read_text()) if ana_summary_path.is_file() else {}
    )
    sh = ana_summary.get("source_heterogeneity") or {}

    # Diagnostic bits: dfs_total / ak_diag_mean from flux matrices, always.
    dfs_total: float | None = None
    ak_diag_mean: float | None = None
    chi2_per_dof: float | None = ana_summary.get("chi2_per_dof")
    prior_influence: float | None = ana_summary.get("prior_influence")
    matrices_path = flux_dir / "matrices.npz"
    if matrices_path.is_file():
        with np.load(matrices_path) as mats:
            A = np.asarray(mats["averaging_kernel"])
            if A.ndim == 2:
                diag = np.diag(A)
            else:
                diag = A
            dfs_total = float(np.sum(diag))
            ak_diag_mean = float(np.mean(diag))

    # Grid values → memoized columns.
    L = _cell_get(cell, "dispersion.sources.config.covariance.L_m")
    cv = _cell_get(cell, "dispersion.sources.config.cv")
    seed = _cell_get(cell, "dispersion.sources.config.seed")
    L_B = _cell_get(cell, "flux.inversion.prior_covariance.L_B_m") or flux_summary.get("L_B_m")

    net = cell.get("instrument.network")
    if isinstance(net, dict):
        N = int(net.get("n", 0))
        layout_seed = int(net.get("seed", 0))
    else:
        N = None
        layout_seed = None

    return {
        "L": _to_float(L),
        "CV": _to_float(cv),
        "N": N,
        "layout_seed": layout_seed,
        "met_id": _cell_get(cell, "dispersion.transport.met_id") or "default",
        "realization": _to_int(seed) if seed is not None else int(idx),
        "transport": transport,
        "L_B": _to_float(L_B),
        "L_true": _to_float(sh.get("L_true_m") or flux_summary.get("L_true_m")),
        "L_B_m": _to_float(L_B),
        "e_q": _to_float(sh.get("E_Q")),
        "dfs_total": dfs_total,
        "chi2_per_dof": _to_float(chi2_per_dof),
        "prior_influence": _to_float(prior_influence),
        "ak_diag_mean": ak_diag_mean,
        "inverse_crime_flag": bool(
            sh.get("inverse_crime_flag", flux_summary.get("inverse_crime_flag", False))
        ),
        "run_dir": str(disp_dir.parent),
    }


def _cell_get(cell: dict[str, Any], key: str) -> Any:
    return cell.get(key)


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Parquet writer (lazy pandas/pyarrow imports)
# ---------------------------------------------------------------------------


REQUIRED_COLUMNS = [
    "L", "CV", "N", "layout_seed", "met_id", "realization", "transport",
    "L_B", "e_q", "dfs_total", "chi2_per_dof", "prior_influence",
    "ak_diag_mean", "inverse_crime_flag", "run_dir",
]


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df.to_parquet(path, index=False)
