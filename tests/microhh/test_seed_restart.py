"""Warm-restart seeding: donor flow state in, donor output out.

Every realization in the rice-paddy sweep warm-starts from one donor case so
they share a bitwise-identical turbulent flow. That only isolates the emission
field if the donor's *output* stays behind — its cross-sections carry the
donor's CH4, and `canonical.from_microhh` reads cross-sections as the
observation window.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from enforceflux.microhh.runner import MicroHHRunner


STAMP = "0003600"


class _Runner:
    """Bind `_seed_restart` to a config without constructing a full case."""

    def __init__(self, config):
        self.config = config

    _seed_restart = MicroHHRunner._seed_restart


def _config(tmp_path: Path) -> SimpleNamespace:
    donor = tmp_path / "donor"
    case = tmp_path / "case"
    donor.mkdir()
    case.mkdir()
    return SimpleNamespace(
        restart_from_dir=donor,
        restart_time_s=int(STAMP),
        case_dir=case,
        scalar_name="ch4",
    )


def _populate_donor(donor: Path) -> None:
    # Restart state: bare `<name>.<stamp>`.
    for name in ("u", "v", "w", "th", "time", "h2o"):
        (donor / f"{name}.{STAMP}").write_bytes(b"DONOR-STATE")
    # The donor's own emitted scalar, which must not be inherited.
    (donor / f"ch4.{STAMP}").write_bytes(b"DONOR-CH4")
    (donor / f"ch4_gradbot.{STAMP}").write_bytes(b"DONOR-GRADBOT")
    # Donor OUTPUT sharing the same stamp — the contamination vector.
    (donor / f"ch4.xy.000.00003.{STAMP}").write_bytes(b"DONOR-XY")
    (donor / f"ch4.xz.000.00012.{STAMP}").write_bytes(b"DONOR-XZ")
    (donor / f"ch4_path.xy.000.{STAMP}").write_bytes(b"DONOR-PATH")


def _populate_initialized_case(case: Path) -> None:
    """What `microhh init` leaves behind: zeroed scalar stamped 0000000."""
    (case / "ch4.0000000").write_bytes(b"ZERO")
    (case / "ch4_gradbot.0000000").write_bytes(b"ZERO-GRADBOT")


def test_donor_flow_state_is_adopted(tmp_path):
    cfg = _config(tmp_path)
    _populate_donor(cfg.restart_from_dir)
    _populate_initialized_case(cfg.case_dir)

    _Runner(cfg)._seed_restart()

    for name in ("u", "v", "w", "th", "time"):
        copied = cfg.case_dir / f"{name}.{STAMP}"
        assert copied.is_file(), f"{name} restart state was not adopted"
        assert copied.read_bytes() == b"DONOR-STATE"


def test_donor_cross_sections_are_not_copied(tmp_path):
    """The regression: `*.<stamp>` also matches the donor's cross-sections.

    Copied through, the donor's CH4 field became frame one of this run's
    observation window — identical across every realization, and wrong.
    """
    cfg = _config(tmp_path)
    _populate_donor(cfg.restart_from_dir)
    _populate_initialized_case(cfg.case_dir)

    _Runner(cfg)._seed_restart()

    leaked = sorted(
        p.name for p in cfg.case_dir.iterdir()
        if ".xy." in p.name or ".xz." in p.name
    )
    assert leaked == [], f"donor output leaked into the restarted case: {leaked}"


def test_emitted_scalar_is_zeroed_at_the_restart_stamp(tmp_path):
    """The donor's CH4 is replaced by the freshly initialized zero field."""
    cfg = _config(tmp_path)
    _populate_donor(cfg.restart_from_dir)
    _populate_initialized_case(cfg.case_dir)

    _Runner(cfg)._seed_restart()

    assert (cfg.case_dir / f"ch4.{STAMP}").read_bytes() == b"ZERO"
    assert (cfg.case_dir / f"ch4_gradbot.{STAMP}").read_bytes() == b"ZERO-GRADBOT"


def test_missing_donor_state_is_refused(tmp_path):
    cfg = _config(tmp_path)
    # Only the scalar, none of the flow state.
    (cfg.restart_from_dir / f"ch4.{STAMP}").write_bytes(b"DONOR-CH4")
    _populate_initialized_case(cfg.case_dir)

    with pytest.raises(FileNotFoundError, match="missing"):
        _Runner(cfg)._seed_restart()
