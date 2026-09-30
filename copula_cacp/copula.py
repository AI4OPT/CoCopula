"""Gaussian-copula aggregation of site-level probabilistic forecasts.

Implements Step 1 of the paper (Section 2.1):

  1. Probability integral transform of the observations through their own
     marginal predictive CDFs, followed by a Gaussian transformation
         z_hat[i, t] = Phi^-1( F_hat[i, t]( x[i, t] ) )                 Eq. (4)
  2. Dependence matrix Sigma_hat = corr(Z_hat) across sites.
  3. Monte-Carlo evaluation of the aggregate predictive distribution: draw
     Z ~ MVN(0, Sigma_hat), map each component back through the marginal
     inverse CDF, and sum over sites.

The numerics are kept identical to the implementation that produced the
published results (linear interpolation with extrapolation for both the PIT
and its inverse, 1000 Monte-Carlo draws, fixed seed).  The one change is that
the multivariate-normal draw is cached: the original code re-drew it with the
same seed at every time step, which is the same sample but repeats a Cholesky
factorisation of Sigma for each of the 8760 hours.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import norm

try:  # optional, only for progress bars
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kw):
        return it


class GaussianCopulaModel:
    """Aggregate N marginal quantile forecasts into one predictive distribution.

    Parameters
    ----------
    n_components:
        Number of sites.  Overwritten by :meth:`fit`.
    sigma:
        Dependence matrix.  Defaults to the identity (independence).
    n_samples:
        Monte-Carlo sample size used to approximate the aggregate distribution.
    seed:
        Seed for the multivariate-normal draw.
    """

    def __init__(self, n_components: int = 1, sigma: np.ndarray | None = None,
                 n_samples: int = 1000, seed: int = 42):
        self.n_components = n_components
        self.sigma = sigma if sigma is not None else np.identity(n_components)
        self.quantile_levels = np.arange(0.01, 1, 0.01)   # the 99 NREL levels
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self._uniform_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # dependence structure
    # ------------------------------------------------------------------
    def fit(self, actuals: np.ndarray, marginals: np.ndarray,
            time_mask: np.ndarray | None = None, progress: bool = True):
        """Estimate Sigma from historical actuals and their marginal CDFs.

        Parameters
        ----------
        actuals:   (N, T) realised generation per site.
        marginals: (N, 99, T) quantile forecasts per site.  May be a memmap:
                   only one site is materialised at a time, so the full
                   multi-gigabyte tensor never has to fit in memory.
        time_mask: optional boolean mask over the time axis, applied lazily so
                   that callers do not have to slice the whole tensor first.
        """
        self.n_components = actuals.shape[0]
        a_all = actuals[:, time_mask] if time_mask is not None else actuals
        T = a_all.shape[1]
        normal_margins = np.zeros((self.n_components, T), dtype=float)

        for i in tqdm(range(self.n_components), disable=not progress, desc="PIT"):
            m_i = marginals[i]                                   # (99, T_full)
            m_i = np.asarray(m_i[:, time_mask] if time_mask is not None else m_i)
            a_i = np.asarray(a_all[i])

            interpolators = [
                self.interpolation(m_i[:, t], self.quantile_levels)
                for t in range(T)
            ]
            uniform_values = np.array([
                interpolators[t](a_i[t]) for t in range(T)
            ])
            normal_margins[i, :] = norm.ppf(np.clip(uniform_values, 0.0001, 0.999))

        self.sigma = np.corrcoef(self.clean_vectors(normal_margins))
        self._uniform_cache = None   # Sigma changed -> resample
        return self

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def predict(self, Q: np.ndarray) -> np.ndarray:
        """Aggregate quantiles for one time step.

        Parameters
        ----------
        Q: (N, 99) marginal quantiles of every site at a single time step.

        Returns
        -------
        (99,) quantiles of the aggregate predictive distribution.
        """
        simulated_samples = self.make_samples(Q)
        return np.quantile(simulated_samples, self.quantile_levels)

    def multi_step_pred(self, marginals: np.ndarray,
                        time_mask: np.ndarray | None = None,
                        chunk: int = 512, progress: bool = True) -> np.ndarray:
        """Aggregate a whole horizon.

        Parameters
        ----------
        marginals: (N, 99, T).  May be a memmap; time steps are materialised in
                   blocks of `chunk` so peak memory stays bounded.
        time_mask: optional boolean mask over the time axis.

        Returns
        -------
        (99, T_selected) aggregate quantiles.
        """
        idx = (np.flatnonzero(time_mask) if time_mask is not None
               else np.arange(marginals.shape[2]))
        out = np.empty((99, idx.size), dtype=float)

        with tqdm(total=idx.size, disable=not progress, desc="aggregate") as bar:
            for start in range(0, idx.size, chunk):
                sel = idx[start:start + chunk]
                # One bulk read per block instead of one per time step.
                if (sel[-1] - sel[0] + 1) == sel.size:      # contiguous: cheap slice
                    block = np.asarray(marginals[:, :, sel[0]:sel[-1] + 1])
                    local = sel - sel[0]
                else:
                    block = np.asarray(marginals[:, :, sel])
                    local = np.arange(sel.size)
                for k, t in enumerate(local):
                    out[:, start + k] = self.predict(block[:, :, t])
                bar.update(len(sel))
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def interpolation(x, y, itype: str = "linear"):
        """Interpolator used for both the PIT and its inverse."""
        if itype == "CubicSpline":
            return CubicSpline(x, y, extrapolate=True)
        if itype == "linear":
            return interp1d(x, y, kind="linear", fill_value="extrapolate")
        raise ValueError("Interpolation type is not valid! Use 'linear' or 'CubicSpline'.")

    def mvn_generator(self, n_samples: int | None = None, seed: int | None = None) -> np.ndarray:
        """Draw from MVN(0, Sigma).  Cached: the draw is identical every call."""
        n_samples = self.n_samples if n_samples is None else n_samples
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(np.zeros(self.n_components), self.sigma, n_samples)

    @staticmethod
    def transform_to_uniform(normal_samples: np.ndarray) -> np.ndarray:
        return norm.cdf(normal_samples)

    def _uniforms(self) -> np.ndarray:
        """Copula sample in uniform space, (n_samples, N).  Computed once."""
        if self._uniform_cache is None or self._uniform_cache.shape[1] != self.n_components:
            self._uniform_cache = self.transform_to_uniform(self.mvn_generator())
        return self._uniform_cache

    def make_samples(self, Q: np.ndarray) -> np.ndarray:
        """Fleet-level Monte-Carlo sample for one time step, shape (n_samples,)."""
        inverse_uniforms = self._uniforms()

        interpolations = [
            self.interpolation(self.quantile_levels, Q[i, :])
            for i in range(self.n_components)
        ]
        simulated_generation = np.array([
            interpolations[i](inverse_uniforms[:, i])
            for i in range(self.n_components)
        ])
        return np.sum(simulated_generation, axis=0)

    @staticmethod
    def clean_vectors(vectors: np.ndarray) -> list[np.ndarray]:
        """Drop time steps where any site produced a non-finite transform."""
        stacked = np.stack(vectors)
        mask = np.all(np.isfinite(stacked), axis=0)
        return [vec[mask] for vec in vectors]
