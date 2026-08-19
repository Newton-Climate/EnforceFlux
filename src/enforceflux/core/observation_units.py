from dataclasses import dataclass, fields
from math import isclose


@dataclass(frozen=True)
class ObservationSpec:
    variable: str
    unit: str
    averaging_window_s: float
    height_m: float
    path_length_m: float


_NUMERIC_FIELDS = ("averaging_window_s", "height_m", "path_length_m")
_STRING_FIELDS = ("variable", "unit")
_REL_TOL = 1e-9


def assert_matches(actual: ObservationSpec, expected: ObservationSpec) -> None:
    mismatches: list[str] = []
    for name in _STRING_FIELDS:
        a, e = getattr(actual, name), getattr(expected, name)
        if a != e:
            mismatches.append(f"{name}: actual={a!r} expected={e!r}")
    for name in _NUMERIC_FIELDS:
        a, e = getattr(actual, name), getattr(expected, name)
        if not isclose(a, e, rel_tol=_REL_TOL, abs_tol=_REL_TOL):
            mismatches.append(f"{name}: actual={a!r} expected={e!r}")
    if mismatches:
        raise ValueError(
            "ObservationSpec mismatch on fields: " + "; ".join(mismatches)
        )


__all__ = ["ObservationSpec", "assert_matches"]
