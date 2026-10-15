"""Stage 2: conformal calibration of a fleet-level quantile forecast (Section III-C of the paper).

Both methods widen (or narrow) each central interval [q^{a/2}_t, q^{1-a/2}_t] of a test hour t
by a correction s_hat computed from the conformity scores of the calibration hours tau

    s_tau = max(q^{a/2}_tau - y_tau,  y_tau - q^{1-a/2}_tau),

    C~_t = [q^{a/2}_t - s_hat,  q^{1-a/2}_t + s_hat].

    CQR   finite-sample conformal quantile of the n calibration scores:
              s_hat = s_(k),   k = ceil((n + 1)(1 - a)),   s_hat = +inf if k > n.
    CACP  finite-sample weighted quantile.  Each calibration hour gets the weight
              w_tau = psi(c_t, c_tau) = exp(-gamma * ||c_t - c_tau||)
          (RBF: squared L2, Laplacian: L1) between the contexts of the test and
          calibration hours.  The test hour keeps its own weight w_t = psi(c_t, c_t) = 1
          as a point mass at +inf, so
              p_tau = w_tau / (sum_tau' w_tau' + w_t),
              s_hat = inf{ s in {s_tau} : sum_{s_tau <= s} p_tau >= 1 - a },
          and s_hat = +inf if no calibration score satisfies the condition.

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
    """Conformalized quantile regression with the finite-sample quantile.  Q: (99, T)  ->  (T_test, 99).

    s = the ceil((n + 1)(1 - a))-th smallest calibration score (+inf if that exceeds n).
    """
    P = Q_test.T.copy()
    for a in ALPHAS:
        c = round(1 - a, 2)
        li, ui = interval_columns(c)
        sc = np.sort(scores(y_cal, Q_cal, li, ui)); n = sc.size
        k = int(np.ceil((n + 1) * c))
        s = sc[k - 1] if k <= n else np.inf
        P[:, li] -= s
        P[:, ui] += s
    return finalize(P)


def kernel_weights(X_cal, X_test, kernel, gamma):
    """Unnormalized similarity of every calibration hour to every test hour.  (n_cal, T_test)."""
    scaler = MinMaxScaler().fit(X_cal)
    A, B = scaler.transform(X_cal), scaler.transform(X_test)
    d = cdist(A, B, "sqeuclidean" if kernel == "rbf" else "cityblock")
    return np.exp(-gamma * d)


def cacp(y_cal, Q_cal, X_cal, Q_test, X_test, kernel="rbf", gamma=1.0):
    """Context-aware conformal prediction with the finite-sample weighted quantile.

    The test hour carries its own weight psi(c_t, c_t) = 1 as a point mass at +inf
    (Tibshirani et al., 2019): s_t = the smallest calibration score whose cumulative
    normalized weight reaches 1 - a, or +inf if it never does.
    Q: (99, T), X: (T, k)  ->  (T_test, 99).
    """
    W = kernel_weights(X_cal, X_test, kernel, gamma)
    norm = W.sum(axis=0) + 1.0                                   # + weight of the test point itself
    P = Q_test.T.copy()
    for a in ALPHAS:
        c = round(1 - a, 2)
        li, ui = interval_columns(c)
        s = scores(y_cal, Q_cal, li, ui)
        order = np.argsort(s)
        cdf = np.cumsum(W[order], axis=0) / norm                 # (n_cal, T_test)
        reached = cdf >= c
        idx = np.argmax(reached, axis=0)
        delta = np.where(reached.any(axis=0), s[order][idx], np.inf)
        P[:, li] -= delta
        P[:, ui] += delta
    return finalize(P)
