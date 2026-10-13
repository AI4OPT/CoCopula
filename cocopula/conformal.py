"""Stage 2: conformal calibration of a fleet-level quantile forecast.

Both methods widen (or narrow) each central interval [q_{a/2}, q_{1-a/2}] by a
correction s computed from the conformity scores of past hours

    s_t = max(q_{a/2,t} - y_t,  y_t - q_{1-a/2,t}).

    CQR   s = (1 - a) empirical quantile of the scores
    CACP  s = (1 - a) weighted quantile of the scores, with weights
          psi(c_t, c_tau) = exp(-gamma * ||c_t - c_tau||)   (RBF: squared L2,
          Laplacian: L1) between the context of the test hour and of each
          calibration hour.

Every pair of quantiles (2%, 4%, ..., 98% central intervals) is calibrated;
the result is made monotone and clipped to [0, 1] (capacity-normalized).
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

ALPHAS = np.round(np.linspace(0.02, 0.98, 49), 2)


def interval_columns(coverage, n_levels=99):
    """Columns of the lower and upper quantile of a central interval."""
    c = round(coverage, 2)
    lower = (1 - c) / 2
    return int(np.ceil(lower * n_levels)) - 1, int(np.floor((1 - lower) * n_levels))


def scores(y, Q, li, ui):
    return np.maximum(Q[li] - y, y - Q[ui])


def finalize(P):
    """Monotone quantiles in [0, 1].  P: (T, 99)."""
    return np.clip(np.maximum.accumulate(P, axis=1), 0.0, 1.0)


def no_calibration(Q_test):
    """Raw forecast: sorted quantiles clipped to [0, 1].  Returns (T, 99)."""
    return np.clip(np.sort(Q_test, axis=0), 0.0, 1.0).T


def cqr(y_cal, Q_cal, Q_test):
    """Conformalized quantile regression.  Q: (99, T)  ->  (T_test, 99)."""
    P = Q_test.T.copy()
    for a in ALPHAS:
        c = round(1 - a, 2)
        li, ui = interval_columns(c)
        s = np.quantile(scores(y_cal, Q_cal, li, ui), c)
        P[:, li] -= s
        P[:, ui] += s
    return finalize(P)


def kernel_weights(X_cal, X_test, kernel, gamma):
    """Normalized similarity of every calibration hour to every test hour.  (n_cal, T_test)."""
    scaler = MinMaxScaler().fit(X_cal)
    A, B = scaler.transform(X_cal), scaler.transform(X_test)
    d = cdist(A, B, "sqeuclidean" if kernel == "rbf" else "cityblock")
    W = np.exp(-gamma * d)
    total = W.sum(axis=0)
    W = np.where(total > 0, W / np.where(total > 0, total, 1), 1.0 / len(A))
    return W


def weighted_quantile(values_sorted, weights_sorted, q):
    """Weighted quantile with a mid-point CDF, for one test hour."""
    cdf = np.cumsum(weights_sorted) - 0.5 * weights_sorted
    return np.interp(q, cdf / cdf[-1], values_sorted)


def cacp(y_cal, Q_cal, X_cal, Q_test, X_test, kernel="rbf", gamma=1.0):
    """Context-aware conformal prediction.  Q: (99, T), X: (T, k)  ->  (T_test, 99)."""
    W = kernel_weights(X_cal, X_test, kernel, gamma)
    P = Q_test.T.copy()
    for a in ALPHAS:
        c = round(1 - a, 2)
        li, ui = interval_columns(c)
        s = scores(y_cal, Q_cal, li, ui)
        order = np.argsort(s)
        Ws = W[order]
        for t in range(P.shape[0]):
            delta = weighted_quantile(s[order], Ws[:, t], c)
            P[t, li] -= delta
            P[t, ui] += delta
    return finalize(P)
