from dataclasses import dataclass

import numpy as np

from enforceflux.models.config import ProjectConfig
from enforceflux.core.base import (
    ForwardModelResult,
    IInversionEngine,
    IInstrumentModel,
    ISourceModel,
    ITransportOperator,
)
from enforceflux.analysis.metrics import MetricResults, compute_metrics
from enforceflux.instrument import InstrumentOperator
from enforceflux.inversion.result import InversionResult
from enforceflux.utils.plugin_registry import get_plugin, normalize_plugin_name


@dataclass(frozen=True)
class OSSEOutput:
    g: np.ndarray         # (m, n) instrument-modified forward operator (H_g)
    y: np.ndarray         # (m,) simulated observations (NaN where invalid)
    x_true: np.ndarray
    x_prior: np.ndarray
    inversion: InversionResult
    metrics: MetricResults


def run_osse(config: ProjectConfig) -> OSSEOutput:
    source_component = config.component("source")
    instrument_component = config.component("instrument")
    transport_component = config.component("transport_operator")
    inversion_component = config.component("inversion")

    source_plugin_name = normalize_plugin_name(
        "enforceflux.source", source_component.plugin
    )
    instrument_plugin_name = normalize_plugin_name(
        "enforceflux.instrument", instrument_component.plugin
    )
    transport_plugin_name = normalize_plugin_name(
        "enforceflux.transport_operator", transport_component.plugin
    )
    inversion_plugin_name = normalize_plugin_name(
        "enforceflux.inversion", inversion_component.plugin
    )

    source_cls = get_plugin("enforceflux.source", source_plugin_name, ISourceModel)
    instrument_cls = get_plugin(
        "enforceflux.instrument", instrument_plugin_name, IInstrumentModel
    )
    transport_cls = get_plugin(
        "enforceflux.transport_operator", transport_plugin_name, ITransportOperator
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

    x_true = np.array([src.flux_true for src in sources])
    x_prior = np.array([src.flux_prior_mean for src in sources])

    rng = np.random.default_rng(config.random_seed)
    op = InstrumentOperator(instruments, rng=rng)
    obs = op.simulate_observations(forward.g, x_true)

    valid = obs.valid_mask
    if not valid.any():
        raise RuntimeError(
            "No valid observations after applying instrument operator. "
            "Check instrument configuration (dropout probability, detection limits)."
        )

    g_inv = obs.H_g[valid]
    y_inv = obs.y_obs[valid]
    r_inv = obs.R[np.ix_(valid, valid)]

    s_a = np.diag([src.flux_prior_std**2 for src in sources])

    inversion_engine = inversion_cls()
    inversion = inversion_engine.invert(g=g_inv, y=y_inv, x_prior=x_prior, s_a=s_a, r=r_inv)

    r_cond = float(inversion_component.config.get("r_cond", 1e-10))
    metrics = compute_metrics(
        g=g_inv,
        fisher=inversion.fisher_information,
        posterior_cov=inversion.posterior_cov,
        averaging_kernel=inversion.averaging_kernel,
        r_cond=r_cond,
    )

    return OSSEOutput(
        g=obs.H_g,
        y=obs.y_obs,
        x_true=x_true,
        x_prior=x_prior,
        inversion=inversion,
        metrics=metrics,
    )
