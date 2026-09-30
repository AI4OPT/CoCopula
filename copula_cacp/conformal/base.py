"""Shared machinery for the conformalizers.

All methods operate on a matrix of 99 predictive quantiles and adjust the
(li, ui) pair belonging to each coverage level by a conformal correction
s_hat, giving Eq. (6) / Eq. (17) of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def quantile_index_bounds(conf_level: float, n_q: int = 99) -> tuple[int, int]:
    """Column indices of the lower/upper quantile for a coverage level.

    With 99 stored quantiles (0.01 ... 0.99), a target coverage c maps to the
    alpha/2 and 1-alpha/2 columns.
    """
    c = float(np.round(conf_level, 2))
    lb = (1 - c) / 2
    ub = 1 - lb
    li = int(np.ceil(lb * n_q)) - 1
    ui = int(np.floor(ub * n_q))
    return li, ui


@dataclass
class FitData:
    actual: np.ndarray                      # (T,)
    forecast: np.ndarray                    # (Q, T)
    cov_past: Optional[np.ndarray] = None   # (T, D) context vectors, or None


class BaseConformalizer:
    def __init__(self,
                 alpha_values: Optional[np.ndarray] = None,
                 value_bounds: Optional[Tuple[float, float]] = None,
                 enforce_monotonic: bool = True):
        self.alpha_values = (alpha_values if alpha_values is not None
                             else np.round(np.linspace(0.02, 0.98, 49), 2))
        self.fit_data: Optional[FitData] = None
        self.value_bounds = value_bounds
        self.enforce_monotonic = enforce_monotonic

    def fit(self, actual: np.ndarray, forecast: np.ndarray,
            past_cov: Optional[np.ndarray] = None):
        assert forecast.ndim == 2, "forecast must be (Q, T)"
        assert actual.ndim == 1 and actual.shape[-1] == forecast.shape[-1], \
            "actual must match T"
        if past_cov is not None:
            assert past_cov.shape[0] == actual.shape[0], \
                "past_cov must match time length T"
        self.fit_data = FitData(actual=actual, forecast=forecast, cov_past=past_cov)
        self._fit_impl()
        return self

    def _fit_impl(self) -> None:
        return

    def _finalize_quantiles(self, q: np.ndarray) -> np.ndarray:
        if self.enforce_monotonic:
            q = np.maximum.accumulate(q)
        if self.value_bounds is not None:
            lo, hi = self.value_bounds
            q = np.clip(q, lo, hi)
        return q

    def update(self, forecast_q: np.ndarray,
               point_cov: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def batch_forecast(self, forecasts: np.ndarray,
                       future_cov: Optional[np.ndarray] = None) -> np.ndarray:
        """Calibrate a block of test steps.  forecasts (Q, T) -> (T, Q)."""
        Tt = forecasts.shape[1]
        out = np.empty((Tt, forecasts.shape[0]))
        for t in range(Tt):
            pcov = None if future_cov is None else future_cov[t].reshape(1, -1)
            qvec = self.update(forecasts[:, t].copy(), pcov)
            out[t] = self._finalize_quantiles(qvec)
        return out

    # -- helper shared by every weighted variant ------------------------
    @staticmethod
    def _weighted_quantile(values: np.ndarray, q: float, w: np.ndarray) -> float:
        """Weighted empirical quantile, Eq. (16)."""
        order = np.argsort(values)
        v, ww = values[order], w[order]
        cdf = np.cumsum(ww) - 0.5 * ww
        cdf /= cdf[-1]
        return float(np.interp(q, cdf, v))
