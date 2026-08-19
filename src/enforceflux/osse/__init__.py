"""enforceflux.osse — runs the source → instrument → transport → inversion pipeline."""
from enforceflux.osse.run import OSSEOutput, run_osse
# --- source-heterogeneity OSSE (M4) ---
from enforceflux.osse.sweep import (
    HCache,
    InMemoryHCache,
    SweepConfig,
    expand_grid,
    group_by_h_cache,
    run_sweep,
)
# --- end M4 ---

__all__ = [
    "OSSEOutput",
    "run_osse",
    # --- source-heterogeneity OSSE (M4) ---
    "HCache",
    "InMemoryHCache",
    "SweepConfig",
    "expand_grid",
    "group_by_h_cache",
    "run_sweep",
    # --- end M4 ---
]
