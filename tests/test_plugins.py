import importlib.metadata as metadata

import pytest

from enforceflux.core.base import (
    IInversionEngine,
    IInstrumentModel,
    ISourceModel,
    ITransportModel,
)
from enforceflux.utils.plugin_registry import list_plugins


def _has_entry_points(group: str) -> bool:
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return len(eps.select(group=group)) > 0
    return group in eps and len(eps[group]) > 0


@pytest.mark.skipif(
    not _has_entry_points("enforceflux.source"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_source_plugins():
    plugins = list_plugins("enforceflux.source", base_cls=ISourceModel)
    assert "static" in plugins


@pytest.mark.skipif(
    not _has_entry_points("enforceflux.instrument"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_instrument_plugins():
    plugins = list_plugins("enforceflux.instrument", base_cls=IInstrumentModel)
    assert "static" in plugins


@pytest.mark.skipif(
    not _has_entry_points("enforceflux.transport"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_transport_plugins():
    plugins = list_plugins("enforceflux.transport", base_cls=ITransportModel)
    assert "gaussian" in plugins
    assert "flexpart" in plugins


@pytest.mark.skipif(
    not _has_entry_points("enforceflux.inversion"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_inversion_plugins():
    plugins = list_plugins("enforceflux.inversion", base_cls=IInversionEngine)
    assert "bayesian" in plugins
