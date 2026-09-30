"""CACP with kernel similarity weights -- the variant reported in the paper.

Weights come from a radial basis function on the context vectors, Eq. (7):

    psi(c_t, c_tau) = exp( -gamma * || c_t - c_tau ||^2 )

and the conformal correction is the weighted quantile of the calibration
conformity scores, Eq. (15)-(16).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from .base import BaseConformalizer, quantile_index_bounds


class KernelConformalizer(BaseConformalizer):
    def __init__(self, kernel: str = "rbf", gamma: float = 1.0,
                 degree: int = 3, coef0: float = 1.0, **kw):
        super().__init__(**kw)
        self.kernel, self.gamma, self.degree, self.coef0 = kernel, gamma, degree, coef0
        self.scaler = MinMaxScaler()

    def _fit_impl(self):
        if self.fit_data.cov_past is None:
            raise ValueError("KernelConformalizer requires past covariates.")
        self.Xref = self.scaler.fit_transform(self.fit_data.cov_past)

    def _sim(self, x_new: np.ndarray) -> np.ndarray:
        """Normalised similarity of every calibration point to the test point."""
        X = self.Xref
        if self.kernel == "rbf":
            d = cdist(X, x_new, "sqeuclidean").flatten()
            s = np.exp(-self.gamma * d)
        elif self.kernel == "laplacian":
            d = cdist(X, x_new, "cityblock").flatten()
            s = np.exp(-self.gamma * d)
        elif self.kernel == "linear":
            s = (X @ x_new.T).flatten()
        elif self.kernel == "poly":
            s = ((X @ x_new.T + self.coef0) ** self.degree).flatten()
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        ssum = s.sum()
        return s / ssum if ssum != 0 else np.ones_like(s) / len(s)

    def update(self, forecast_q, point_cov=None):
        if point_cov is None:
            raise ValueError("KernelConformalizer.update requires point_cov.")
        x = self.scaler.transform(point_cov)
        w = self._sim(x)
        A, F = self.fit_data.actual, self.fit_data.forecast
        for alpha in self.alpha_values:
            c = np.round(1 - alpha, 2)
            li, ui = quantile_index_bounds(c)
            score = np.maximum(F[li] - A, A - F[ui])
            delta = self._weighted_quantile(score, c, w)
            forecast_q[li] -= delta
            forecast_q[ui] += delta
        return forecast_q
