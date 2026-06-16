import importlib.metadata as metadata

import pytest

from enforceflux.core.base import (
    IFluxEstimator,
    IInversionEngine,
    IInstrumentModel,
    ISourceModel,
    ITransportOperator,
    ITransportSimulation,
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
    not _has_entry_points("enforceflux.transport_operator"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_transport_operator_plugins():
    plugins = list_plugins("enforceflux.transport_operator", base_cls=ITransportOperator)
    assert "gaussian" in plugins
    assert "flexpart" in plugins


@pytest.mark.skipif(
    not _has_entry_points("enforceflux.transport_simulation"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_transport_simulation_plugins():
    plugins = list_plugins(
        "enforceflux.transport_simulation", base_cls=ITransportSimulation
    )
    assert "flexpart" in plugins


@pytest.mark.skipif(
    not _has_entry_points("enforceflux.inversion"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_inversion_plugins():
    plugins = list_plugins("enforceflux.inversion", base_cls=IInversionEngine)
    assert "bayesian" in plugins


@pytest.mark.skipif(
    not _has_entry_points("enforceflux.flux"),
    reason="Entry points unavailable; install package to test plugins",
)
def test_flux_plugins():
    plugins = list_plugins("enforceflux.flux", base_cls=IFluxEstimator)
    assert "inversion" in plugins
    assert "eddy_covariance" in plugins
    assert "flux_gradient" in plugins
