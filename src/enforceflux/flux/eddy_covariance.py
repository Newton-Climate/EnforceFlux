"""Source-free eddy covariance retrieval helpers."""
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from enforceflux.core.base import FluxResult, IFluxEstimator


@dataclass(frozen=True)
class EddyCovarianceWindow:
    """One EC averaging window in already aligned instrument space.

    This object is intentionally source-free: it represents what an EC tower
    computed from tower-side data products, not an OSSE transport operator.
    """

    flux: float | None = None
    covariance_wc: float | None = None
    w_prime: np.ndarray | None = field(default=None, compare=False, repr=False)
    c_prime: np.ndarray | None = field(default=None, compare=False, repr=False)
    qc_passed: bool = True
    n_samples: int | None = None
    timestamp_s: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class EddyCovarianceFluxEstimator(IFluxEstimator):
    """Estimate EC fluxes from preprocessed tower-side windows.

    This is not a transport operator. It consumes flux-window summaries or
    covariance products that have already been aligned into EC observation
    space and returns one flux estimate per window.
    """

    def estimate(self, observations: Any, config: dict[str, Any]) -> FluxResult:
        windows = self._coerce_windows(observations)
        unit_scale = float(config.get("unit_scale", 1.0))
        min_samples = int(config.get("min_samples", 1))
        reject_failed_qc = bool(config.get("reject_failed_qc", True))

        flux = np.full(len(windows), np.nan, dtype=float)
        valid_mask = np.zeros(len(windows), dtype=bool)
        methods: list[str] = []

        for idx, window in enumerate(windows):
            methods.append("invalid")
            if reject_failed_qc and not window.qc_passed:
                continue

            value, method = self._window_flux(window)
            methods[-1] = method
            if value is None:
                continue

            n_samples = self._sample_count(window)
            if n_samples < min_samples:
                methods[-1] = "insufficient_samples"
                continue

            flux[idx] = value * unit_scale
            valid_mask[idx] = True

        return FluxResult(
            flux=flux,
            meta={
                "valid_mask": valid_mask,
                "methods": methods,
                "timestamps_s": np.array([w.timestamp_s for w in windows], dtype=object),
                "n_samples": np.array([self._sample_count(w) for w in windows], dtype=int),
            },
        )

    def _window_flux(self, window: EddyCovarianceWindow) -> tuple[float | None, str]:
        if window.flux is not None:
            return float(window.flux), "flux"
        if window.covariance_wc is not None:
            return float(window.covariance_wc), "covariance_wc"
        if window.w_prime is not None and window.c_prime is not None:
            w_prime = np.asarray(window.w_prime, dtype=float)
            c_prime = np.asarray(window.c_prime, dtype=float)
            if w_prime.shape != c_prime.shape:
                raise ValueError("w_prime and c_prime must have the same shape.")
            if w_prime.size == 0:
                return None, "empty_primes"
            return float(np.mean(w_prime * c_prime)), "covariance_from_primes"
        return None, "missing_flux_input"

    def _sample_count(self, window: EddyCovarianceWindow) -> int:
        if window.n_samples is not None:
            return int(window.n_samples)
        if window.w_prime is not None:
            return int(np.asarray(window.w_prime).size)
        if window.c_prime is not None:
            return int(np.asarray(window.c_prime).size)
        return 1

    def _coerce_windows(self, observations: Any) -> list[EddyCovarianceWindow]:
        if isinstance(observations, EddyCovarianceWindow):
            return [observations]
        if isinstance(observations, Mapping):
            return [self._window_from_mapping(observations)]
        if isinstance(observations, Iterable) and not isinstance(observations, (str, bytes)):
            windows: list[EddyCovarianceWindow] = []
            for item in observations:
                if isinstance(item, EddyCovarianceWindow):
                    windows.append(item)
                elif isinstance(item, Mapping):
                    windows.append(self._window_from_mapping(item))
                else:
                    raise TypeError(
                        "EC observations must be EddyCovarianceWindow objects or mappings."
                    )
            if windows:
                return windows
        raise TypeError(
            "EC observations must be an EddyCovarianceWindow, a mapping, or an iterable of them."
        )

    def _window_from_mapping(self, data: Mapping[str, Any]) -> EddyCovarianceWindow:
        return EddyCovarianceWindow(
            flux=data.get("flux"),
            covariance_wc=data.get("covariance_wc"),
            w_prime=None if data.get("w_prime") is None else np.asarray(data["w_prime"], dtype=float),
            c_prime=None if data.get("c_prime") is None else np.asarray(data["c_prime"], dtype=float),
            qc_passed=bool(data.get("qc_passed", True)),
            n_samples=data.get("n_samples"),
            timestamp_s=data.get("timestamp_s"),
            meta=dict(data.get("meta", {})),
        )
