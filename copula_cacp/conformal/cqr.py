"""Conformalized Quantile Regression (Appendix A.2)."""
from __future__ import annotations

import numpy as np

from .base import BaseConformalizer, quantile_index_bounds


class CQRConformalizer(BaseConformalizer):
    """Unweighted split-conformal calibration of an interval, Eq. (11)-(13)."""

    def _fit_impl(self):
        A = self.fit_data.actual
        F = self.fit_data.forecast
        conf_scores = {}
        for alpha in self.alpha_values:
            c = np.round(1 - alpha, 2)
            li, ui = quantile_index_bounds(c)
            score = np.maximum(F[li] - A, A - F[ui])   # Eq. (11)
            conf_scores[alpha] = np.quantile(score, c)  # Eq. (12)
        self.conformity_scores = conf_scores

    def update(self, forecast_q, point_cov=None):
        for alpha in self.alpha_values:
            c = np.round(1 - alpha, 2)
            li, ui = quantile_index_bounds(c)
            delta = self.conformity_scores[alpha]
            forecast_q[li] -= delta
            forecast_q[ui] += delta
        return forecast_q
