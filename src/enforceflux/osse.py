from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enforceflux.config import ProjectConfig
from enforceflux.core.base import (
    ForwardModelResult,
    IInversionEngine,
    IInstrumentModel,
    ISourceModel,
    ITransportModel,
)
from enforceflux.metrics import MetricResults, compute_metrics
from enforceflux.retrieval.inversion import InversionResult
from enforceflux.utils.plugin_registry import get_plugin, normalize_plugin_name


@dataclass(frozen=True)
class OSSEOutput:
    g: np.ndarray
    y: np.ndarray
    x_true: np.ndarray
    x_prior: np.ndarray
    inversion: InversionResult
    metrics: MetricResults


def run_osse(config: ProjectConfig) -> OSSEOutput:
    source_component = config.component("source")
    instrument_component = config.component("instrument")
    transport_component = config.component("transport")
    inversion_component = config.component("inversion")

    source_plugin_name = normalize_plugin_name(
        "enforceflux.source", source_component.plugin
    )
    instrument_plugin_name = normalize_plugin_name(
        "enforceflux.instrument", instrument_component.plugin
    )
    transport_plugin_name = normalize_plugin_name(
        "enforceflux.transport", transport_component.plugin
    )
    inversion_plugin_name = normalize_plugin_name(
        "enforceflux.inversion", inversion_component.plugin
    )

    source_cls = get_plugin("enforceflux.source", source_plugin_name, ISourceModel)
    instrument_cls = get_plugin(
        "enforceflux.instrument", instrument_plugin_name, IInstrumentModel
    )
    transport_cls = get_plugin(
        "enforceflux.transport", transport_plugin_name, ITransportModel
    )
    inversion_cls = get_plugin(
        "enforceflux.inversion", inversion_plugin_name, IInversionEngine
    )

    sources = source_cls().build_sources(source_component.config, config.domain)
    instruments = instrument_cls().build_instruments(
        instrument_component.config, config.domain
    )
    transport = transport_cls()

    forward: ForwardModelResult = transport.build_forward_operator(
        sources, instruments, config.domain, transport_component.config
    )
    g = forward.g

    x_true = np.array([src.flux_true for src in sources])
    x_prior = np.array([src.flux_prior_mean for src in sources])

    noise_std = np.array([inst.effective_noise_std for inst in instruments])
    r = np.diag(noise_std**2)

    rng = np.random.default_rng(config.random_seed)
    noise = rng.normal(0.0, noise_std)
    y = g @ x_true + noise

    s_a = np.diag([src.flux_prior_std**2 for src in sources])

    inversion_engine = inversion_cls()
    inversion = inversion_engine.invert(g=g, y=y, x_prior=x_prior, s_a=s_a, r=r)

    r_cond = float(inversion_component.config.get("r_cond", 1e-10))
    metrics = compute_metrics(
        g=g,
        fisher=inversion.fisher_information,
        posterior_cov=inversion.posterior_cov,
        averaging_kernel=inversion.averaging_kernel,
        r_cond=r_cond,
    )

    return OSSEOutput(
        g=g,
        y=y,
        x_true=x_true,
        x_prior=x_prior,
        inversion=inversion,
        metrics=metrics,
    )
