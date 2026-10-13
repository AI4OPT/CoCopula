"""Stage 1: aggregation of site-level quantile forecasts with a Gaussian copula.

    1. PIT of every observation through its own forecast:  u_it = F_it(x_it)
    2. Sigma = correlation of Phi^-1(u) across sites
    3. Monte Carlo: draw Z ~ N(0, Sigma), map U = Phi(Z) through every site's
       inverse CDF F_it^-1(U_i), sum over sites, and read the 99 quantiles
       of the summed samples.

The independent baseline (Ind) is the same procedure with Sigma = I.
CDFs and inverse CDFs are piecewise linear in the 99 quantiles, with linear
extrapolation beyond the 1% and 99% levels.
"""
import numpy as np
from scipy.stats import norm

LEVELS = np.arange(0.01, 1, 0.01)        # the 99 quantile levels


def pit(x, quantiles):
    """u_t = F_t(x_t) for one site.  x: (T,), quantiles: (99, T)  ->  (T,)."""
    order = np.argsort(quantiles, axis=0, kind="stable")
    xs = np.take_along_axis(quantiles, order, axis=0)          # sorted quantile values
    ys = LEVELS[order]
    hi = np.clip((xs < x).sum(axis=0), 1, len(LEVELS) - 1)     # bracketing index
    lo = hi - 1
    cols = np.arange(x.size)
    x_lo, x_hi = xs[lo, cols], xs[hi, cols]
    y_lo, y_hi = ys[lo, cols], ys[hi, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (y_hi - y_lo) / (x_hi - x_lo) * (x - x_lo) + y_lo


def all_pits(actuals, quantiles):
    """PIT values of every site and hour, clipped to [1e-4, 0.999].  (N, T)."""
    return np.stack([np.clip(pit(actuals[i], np.asarray(quantiles[i])), 0.0001, 0.999)
                     for i in range(actuals.shape[0])])


def fit_sigma(u):
    """Gaussian-copula correlation from PIT values u (N, T_fit).

    Hours where any site gives a non-finite normal score are left out.
    """
    z = norm.ppf(u)
    keep = np.all(np.isfinite(z), axis=0)
    return np.corrcoef(z[:, keep])


def sample_uniforms(sigma, n_samples=1000, seed=42):
    """Copula sample U = Phi(Z), Z ~ N(0, Sigma).  (n_samples, N)."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(len(sigma)), sigma, n_samples)
    return norm.cdf(z)


def aggregate(quantiles, uniforms, start, stop, block=256):
    """Fleet-level quantiles for hours start..stop-1.  Returns (99, stop - start).

    quantiles: (N, 99, T) site forecasts, uniforms: (S, N) copula sample.
    """
    n_levels = len(LEVELS)
    # inverse-CDF interpolation constants; they depend only on the uniforms
    hi = np.clip(np.searchsorted(LEVELS, uniforms.T), 1, n_levels - 1)   # (N, S)
    lo = hi - 1
    dx = LEVELS[hi] - LEVELS[lo]
    du = uniforms.T - LEVELS[lo]

    out = np.empty((n_levels, stop - start))
    for a in range(start, stop, block):
        b = min(a + block, stop)
        Q = np.asarray(quantiles[:, :, a:b])                      # (N, 99, hours)
        total = 0.0
        for i in range(Q.shape[0]):                                # sum over sites
            y_lo, y_hi = Q[i][lo[i]], Q[i][hi[i]]                  # (S, hours)
            total = total + ((y_hi - y_lo) / dx[i][:, None] * du[i][:, None] + y_lo)
        out[:, a - start:b - start] = np.quantile(total, LEVELS, axis=0)
    return out
