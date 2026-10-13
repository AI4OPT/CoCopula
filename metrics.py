"""Evaluation metrics: PICP, AIW and Winkler score (WS).

P is a (T, 99) matrix of capacity-normalized quantiles, y the (T,) actuals.
The reported numbers are computed within each 7-day test window and then
averaged over the windows, every window weighted equally.
"""
import numpy as np
import pandas as pd

from conformal import interval_columns

COVERAGES = (0.9, 0.8, 0.7, 0.6)


def picp(y, P, c):
    li, ui = interval_columns(c)
    return float(np.mean((y >= P[:, li]) & (y <= P[:, ui])))


def aiw(y, P, c):
    li, ui = interval_columns(c)
    return float(np.mean(P[:, ui] - P[:, li]))


def winkler(y, P, c):
    li, ui = interval_columns(c)
    L, U, alpha = P[:, li], P[:, ui], 1 - c
    penalty = 2 / alpha * ((L - y) * (y < L) + (y - U) * (y > U))
    return float(np.mean(U - L + penalty))


def mean_winkler(y, P):
    """Validation objective for choosing the CACP kernel and gamma."""
    return float(np.mean([winkler(y, P, c) for c in COVERAGES]))


def window_metrics(y, P, times, start, window_days=7):
    """Table of PICP / AIW / WS per coverage, averaged over consecutive test windows."""
    rows = []
    t0, step = pd.Timestamp(start), pd.Timedelta(days=window_days)
    while t0 <= times[-1]:
        m = np.asarray((times >= t0) & (times < t0 + step))
        if m.any():
            for c in COVERAGES:
                rows.append({"coverage": c, "PICP": picp(y[m], P[m], c),
                             "AIW": aiw(y[m], P[m], c), "WS": winkler(y[m], P[m], c)})
        t0 += step
    return pd.DataFrame(rows).groupby("coverage", sort=False).mean()
