"""Model-agnostic transport runs: one config in, one result shape out.

Any of the three transport models runs from the same YAML — the model is a
single line in it — and returns the same :class:`TransportRunResult`, with
simulation output normalised to a canonical ``concentration(time, y, x)``
NetCDF in ng m⁻³::

    from enforceflux.transport import TransportRunConfig, run_transport

    run = TransportRunConfig.from_file("apps/transport_main.yaml")
    result = run_transport(run)
    print(result.summary())

See :mod:`enforceflux.transport.run_config` for the schema,
:mod:`enforceflux.transport.canonical` for the output contract, and
``apps/transport_main.py`` for the CLI.
"""

from enforceflux.transport.canonical import (
    CANONICAL_UNITS,
    CanonicalField,
    read_canonical,
    write_canonical,
)
from enforceflux.transport.run_config import (
    DomainProjection,
    RunDomain,
    RunOutput,
    RunReceptor,
    RunSource,
    TransportRunConfig,
)
from enforceflux.transport.runner import TransportRunResult, run_transport

__all__ = [
    "CANONICAL_UNITS",
    "CanonicalField",
    "DomainProjection",
    "RunDomain",
    "RunOutput",
    "RunReceptor",
    "RunSource",
    "TransportRunConfig",
    "TransportRunResult",
    "read_canonical",
    "run_transport",
    "write_canonical",
]
