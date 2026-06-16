"""Shared matplotlib helpers for analysis visualization modules."""
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization.\n"
            "Install with:  pip install matplotlib"
        )


def _make_fig(figsize=None):
    """Create a fresh (fig, ax) pair."""
    _require_mpl()
    return plt.subplots(figsize=figsize)


def _resolve(ax):
    """Return (fig, ax): create new if ax is None, otherwise reuse."""
    _require_mpl()
    if ax is None:
        return _make_fig()
    return ax.get_figure(), ax
