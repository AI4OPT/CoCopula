"""Evaluation metrics (Appendix B.3).

All predictions are (T, 99) matrices of quantiles normalised by the system
capacity; y is the matching (T,) vector of normalised actuals.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

DEFAULT_LEVELS: Sequence[float] = (0.9, 0.8, 0.7, 0.6)


def level_indices(level: float, n_q: int = 99) -> Tuple[int, int]:
    """Columns of the lower/upper quantile for a target coverage level."""
    c = round(level, 2)
    lb = (1 - c) / 2
    ub = 1 - lb
    li = int(np.ceil(lb * n_q)) - 1
    ui = int(np.floor(ub * n_q))
    return li, ui


def picp_aiw(y: np.ndarray, pred: np.ndarray,
             levels: Iterable[float] = DEFAULT_LEVELS
             ) -> Tuple[Dict[float, float], Dict[float, float]]:
    """Prediction Interval Coverage Probability, Eq. (18), and width, Eq. (19)."""
    picp, aiw = {}, {}
    for c in levels:
        li, ui = level_indices(c)
        L, U = pred[:, li], pred[:, ui]
        picp[c] = float(((y >= L) & (y <= U)).mean())
        aiw[c] = float(np.mean(U - L))
    return picp, aiw


def winkler_score(y: np.ndarray, pred: np.ndarray,
                  levels: Iterable[float] = DEFAULT_LEVELS) -> Dict[float, float]:
    """Winkler score, Eq. (21)-(22)."""
    out = {}
    for c in levels:
        li, ui = level_indices(c)
        L, U = pred[:, li], pred[:, ui]
        alpha = 1 - c
        width = U - L
        penalty = 2 / alpha * ((L - y) * (y < L) + (y - U) * (y > U))
        out[c] = float(np.mean(width + penalty))
    return out


def mean_winkler(y: np.ndarray, pred: np.ndarray,
                 levels: Iterable[float] = DEFAULT_LEVELS) -> float:
    """Scalar objective used for hyperparameter selection."""
    d = winkler_score(y, pred, levels)
    return float(np.mean(list(d.values()))) if d else float("inf")


def hourly_picp(y: np.ndarray, pred: np.ndarray, hours: np.ndarray,
                level: float = 0.8) -> Dict[int, float]:
    """Coverage broken down by hour of day (Figure 2b)."""
    li, ui = level_indices(level)
    out = {}
    for h in np.unique(hours):
        m = hours == h
        if not m.any():
            continue
        L, U = pred[m, li], pred[m, ui]
        out[int(h)] = float(((y[m] >= L) & (y[m] <= U)).mean())
    return out


def coverage_width_curve(y: np.ndarray, pred: np.ndarray,
                         levels: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(coverage, width) pairs across many target levels -- Figure 2a."""
    cov, wid = [], []
    for c in levels:
        li, ui = level_indices(c)
        L, U = pred[:, li], pred[:, ui]
        finite = np.isfinite(y) & np.isfinite(L) & np.isfinite(U)
        if not finite.any():
            cov.append(np.nan)
            wid.append(np.nan)
            continue
        cov.append(float(((y[finite] >= L[finite]) & (y[finite] <= U[finite])).mean()))
        wid.append(float(np.mean(U[finite] - L[finite])))
    return np.asarray(cov), np.asarray(wid)
