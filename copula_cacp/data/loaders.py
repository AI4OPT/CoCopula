"""Loading NREL PERFORM arrays and the context (covariate) tensors.

Three data *levels* are used by the pipeline:

  "sites"   (N, ...)  per-site actuals and quantile forecasts -- input to the
                      copula aggregation (step 1).
  "system"  (1, ...)  NREL's own fleet-level forecast -- the reference baseline.
  "copula"  (1, ...)  the aggregated forecast produced by step 1, paired with
                      the true system-level actuals.

Context vectors always come from the *system* covariate files, since both the
NREL and copula fleet-level forecasts describe the same physical aggregate.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .. import config


def minmax_scale_3d(x: np.ndarray) -> np.ndarray:
    """MinMax-scale an (n, k, t) tensor per feature across (n, t)."""
    n, k, t = x.shape
    reshaped = x.transpose(1, 0, 2).reshape(k, -1).T   # (n*t, k)
    scaled = MinMaxScaler().fit_transform(reshaped)
    return scaled.T.reshape(k, n, t).transpose(1, 0, 2)


def _solar_dir(iso: str, level: str) -> str:
    return os.path.join(config.NREL_ROOT, iso.upper(), "solar", "day-ahead", level)


def load_covariates(iso: str, covariates: Optional[List[str]],
                    cov_prefix: Optional[str] = None) -> Optional[np.ndarray]:
    """Concatenate and scale the requested covariate blocks -> (N, sum_k, T)."""
    if not covariates:
        return None
    prefix = cov_prefix or f"{iso.lower()}_system"
    blocks = []
    for c in covariates:
        path = os.path.join(config.COV_ROOT, f"{prefix}_{c}.npy")
        if not os.path.exists(path):
            available = sorted(
                os.path.basename(p)
                for p in glob.glob(os.path.join(config.COV_ROOT, f"{prefix}_*.npy"))
            )
            raise FileNotFoundError(
                f"Covariate file not found: {path}\n"
                f"Available for prefix '{prefix}':\n - " + "\n - ".join(available)
            )
        blocks.append(np.load(path))
    return minmax_scale_3d(np.concatenate(blocks, axis=1))


def load_dataset(
    iso: str,
    level: str,
    covariates: Optional[List[str]] = None,
    mmap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], pd.DatetimeIndex]:
    """Return (actuals, marginals, capacity, covariates, time_index).

    Shapes: actuals (N, T), marginals (N, 99, T), capacity (N, 1),
    covariates (N, k, T) or None, time_index length T.

    `mmap=True` returns the site-level marginals as a read-only memmap.  That
    tensor is ~5 GB for MISO, so anything touching all sites at once should
    memory-map it and slice, rather than reading it whole.
    """
    iso_u = iso.upper()
    level = level.lower()
    start, end = config.time_bounds(iso_u)
    mode = "r" if mmap else None

    if level in ("sites", "system"):
        d = _solar_dir(iso_u, level)
        actuals = np.load(os.path.join(d, "actuals.npy"), mmap_mode=mode)
        marginals = np.load(os.path.join(d, "marginals.npy"), mmap_mode=mode)
        capacity = np.load(os.path.join(d, "capacity.npy"))
        cov_prefix = f"{iso.lower()}_{level}"

    elif level == "copula":
        # Aggregated quantiles from step 1, evaluated against the true
        # system-level actuals and normalised by the system capacity.
        sysd = _solar_dir(iso_u, "system")
        actuals = np.load(os.path.join(sysd, "actuals.npy"))
        capacity = np.load(os.path.join(sysd, "capacity.npy"))
        path = config.copula_path(iso_u)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Aggregated forecast not found: {path}\n"
                f"Run step 1 first:  python scripts/step1_aggregate.py --iso {iso_u}"
            )
        marg = np.load(path)
        T = actuals.shape[1]
        marginals = marg.reshape(1, 99, T)
        cov_prefix = f"{iso.lower()}_system"

    else:
        raise ValueError(f"Invalid level: {level!r}. Use sites, system or copula.")

    cov = load_covariates(iso_u, covariates, cov_prefix)
    time_index = pd.date_range(start=start, end=end, freq="h")

    if marginals.shape[-1] != len(time_index):
        raise ValueError(
            f"{iso_u}/{level}: {marginals.shape[-1]} time steps but the "
            f"{config.year(iso_u)} hourly index has {len(time_index)}."
        )
    return actuals, marginals, capacity, cov, time_index
