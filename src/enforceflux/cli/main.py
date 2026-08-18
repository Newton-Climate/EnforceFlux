"""Unified `enforceflux` CLI.

Every subcommand takes a YAML config plus per-stage overrides. Under the hood
each subcommand delegates to the existing ``apps/*_main.py`` module's ``main``
function; the CLI is the single entry point and the ``apps/`` scripts are
retained as import targets.

Usage:
    enforceflux <command> [args...]

Commands:
    met         Prepare met/forcing inputs (ERA5 → canonical MetSeries).
    dispersion  Forward transport → 4-D concentration + H columns.
    instrument  Apply instrument operator to concentration field → virtual obs.
    obs         Ingest real observation files → obs artifact (same schema).
    flux        Run inversion / flux estimator on obs + H → posterior.
    analysis    Post-hoc diagnostics (Rodgers hygiene: DFS, χ²/dof, AK, ...).
    osse        Legacy end-to-end OSSE from a single JSON config.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APPS = _REPO_ROOT / "apps"
if str(_APPS) not in sys.path:
    sys.path.insert(0, str(_APPS))


def _delegate(module_name: str, argv: list[str]) -> int:
    """Import ``apps/<module_name>.py`` and invoke its ``main``.

    Handles both ``main(argv)`` and ``main()`` signatures by patching
    ``sys.argv`` for the latter.
    """
    module = importlib.import_module(module_name)
    main = module.main
    try:
        rc = main(argv)
    except TypeError:
        saved = sys.argv
        sys.argv = [module_name, *argv]
        try:
            rc = main()
        finally:
            sys.argv = saved
    return int(rc or 0)


def _cmd_met(argv: list[str]) -> int:
    return _delegate("met_main", argv)


def _cmd_dispersion(argv: list[str]) -> int:
    return _delegate("dispersion_main", argv)


def _cmd_instrument(argv: list[str]) -> int:
    return _delegate("instrument_main", argv)


def _cmd_obs(argv: list[str]) -> int:
    return _delegate("obs_main", argv)


def _cmd_flux(argv: list[str]) -> int:
    return _delegate("flux_main", argv)


def _cmd_analysis(argv: list[str]) -> int:
    return _delegate("analysis_main", argv)


def _cmd_osse(argv: list[str]) -> int:
    return _delegate("osse_main", argv)


_COMMANDS = {
    "met": _cmd_met,
    "dispersion": _cmd_dispersion,
    "instrument": _cmd_instrument,
    "obs": _cmd_obs,
    "flux": _cmd_flux,
    "analysis": _cmd_analysis,
    "osse": _cmd_osse,
}


def _print_usage() -> None:
    print(__doc__, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_usage()
        return 0 if args else 2
    cmd, rest = args[0], args[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"unknown command: {cmd!r}", file=sys.stderr)
        _print_usage()
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
