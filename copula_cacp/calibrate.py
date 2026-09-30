"""Step 2: rolling-window conformal calibration of a fleet-level forecast.

The test period is walked one day at a time.  For each test day:

  calibration = everything from `train_start` up to the validation week
  validation  = the last `val_days` before the test day
  test        = the day itself

Hyperparameters and the context subset are chosen on validation, the winner is
refit on calibration+validation, and the calibrated quantiles for the test day
are appended to the output.  This reproduces "calibration is performed using
all information available prior to the test day" (Appendix B.1).

Night hours (zero actual generation) are dropped throughout, as in the source
pipeline; they carry no information and would inflate coverage.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .conformal.factory import SEARCH_SPACES, make_conformalizer, requires_covariates
from .data.loaders import load_dataset
from .tuning import all_nonempty_subsets, product_grid, search_winkler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _prepare_series(iso: str, level: str, site_idx: int = 0):
    """Capacity-normalised actuals/quantiles with night hours removed."""
    actuals, marginals, capacity, _, time_index = load_dataset(iso, level, None)

    y_full = actuals[site_idx].astype(float)
    Q_full = marginals[site_idx].astype(float)
    cap = float(capacity[site_idx])
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(f"{iso}/{level}: invalid capacity {cap}")

    y_norm = y_full / cap
    Q_norm = Q_full / cap

    mask = y_norm > 0                      # drop night / zero-generation hours
    return y_norm[mask], Q_norm[:, mask], time_index[mask], mask, cap


def _context_matrix(iso: str, level: str, covariates: List[str],
                    mask: np.ndarray, site_idx: int = 0) -> Optional[np.ndarray]:
    """(T_masked, k) context matrix aligned with the masked time axis."""
    _, _, _, cov, _ = load_dataset(iso, level, covariates)
    if cov is None:
        return None
    return np.transpose(cov[site_idx], (1, 0))[mask, :]


def _save(preds: np.ndarray, times: pd.DatetimeIndex,
          out_npy: str, out_times: str) -> None:
    os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
    np.save(out_npy, preds)
    np.save(out_times, times.values)


# ---------------------------------------------------------------------------
# uncalibrated pass-through (the "NREL" and "Copula" rows of Table 1)
# ---------------------------------------------------------------------------
def export_passthrough(iso: str, level: str, out_npy: str, out_times: str,
                       *, first_train_end: Optional[str] = None,
                       test_window_days: int = config.TEST_WINDOW_DAYS,
                       verbose: int = 1) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Emit the raw quantiles over the same test windows, no correction applied.

    Quantiles are sorted and clipped to [0, 1] so that the comparison against
    the calibrated methods is like-for-like.
    """
    y, Q, ti, _, _ = _prepare_series(iso, level)
    train_end = pd.Timestamp(first_train_end or config.first_train_end(iso))
    offset = pd.DateOffset(days=int(test_window_days))

    preds_all: List[np.ndarray] = []
    times_all: List[pd.DatetimeIndex] = []
    while train_end <= ti[-1]:
        test_end = train_end + offset
        m = (ti >= train_end) & (ti < test_end)
        if m.sum():
            pred = np.clip(np.apply_along_axis(np.sort, 0, Q[:, m]), 0.0, 1.0)
            preds_all.append(pred.T)
            times_all.append(ti[m])
        train_end = test_end

    if not preds_all:
        raise RuntimeError(f"[{iso}/{level}] pass-through produced no windows.")

    preds = np.vstack(preds_all)
    times = pd.DatetimeIndex(np.concatenate([t.values for t in times_all]))
    _save(preds, times, out_npy, out_times)
    if verbose:
        print(f"[OK] {out_npy}  shape={preds.shape}", flush=True)
    return preds, times


# ---------------------------------------------------------------------------
# conformal calibration
# ---------------------------------------------------------------------------
def _tune_window(
    method: str,
    grid: Dict[str, Any],
    cov_superset: Optional[List[str]],
    cov_cache: Dict[Tuple[str, ...], Optional[np.ndarray]],
    y_cal: np.ndarray, Q_cal: np.ndarray,
    y_val: np.ndarray, Q_val: np.ndarray,
    calib_mask: np.ndarray, val_mask: np.ndarray,
    *,
    cov_selection: bool,
    value_bounds: Optional[Tuple[float, float]],
    verbose: int,
    log_path: Optional[str],
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    """Pick (hyperparameters, covariate subset) by validation Winkler."""
    needs_cov = requires_covariates(method)
    if needs_cov:
        if not cov_superset:
            raise ValueError(f"Method '{method}' requires context covariates.")
        cov_candidates: List[Optional[List[str]]] = (
            all_nonempty_subsets(cov_superset) if cov_selection else [list(cov_superset)]
        )
    else:
        cov_candidates = [None]

    candidates = product_grid(grid)
    best = (float("inf"), {}, None)

    for covs in cov_candidates:
        X = cov_cache[() if covs is None else tuple(covs)]
        cov_cal = None if X is None else X[calib_mask, :]
        cov_val = None if X is None else X[val_mask, :]

        params, score = search_winkler(
            method, candidates,
            y_cal, Q_cal, cov_cal,
            y_val, Q_val, cov_val,
            value_bounds=value_bounds,
            verbose=verbose,
            log_path=log_path,
        )
        if score < best[0]:
            best = (score, params, covs)

    _, best_params, best_covs = best
    return best_params, best_covs


def export_conformalized(
    iso: str,
    level: str,
    method: str,
    out_npy: str,
    out_times: str,
    *,
    covariates: Optional[List[str]] = None,
    site_idx: int = 0,
    train_start: Optional[str] = None,
    first_train_end: Optional[str] = None,
    test_window_days: int = config.TEST_WINDOW_DAYS,
    val_days: int = config.VAL_DAYS,
    refit_includes_val: bool = True,
    value_bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
    cov_selection: bool = True,
    max_windows: Optional[int] = None,
    log_dir: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Roll the calibration forward one test window at a time."""
    method = method.lower()
    y, Q, ti, mask, cap = _prepare_series(iso, level, site_idx)

    train_start_ts = pd.Timestamp(train_start or config.train_start(iso))
    train_end = pd.Timestamp(first_train_end or config.first_train_end(iso))
    test_window = pd.DateOffset(days=int(test_window_days))
    val_window = pd.DateOffset(days=int(val_days))

    # Pre-load every covariate subset once; the per-window loop only slices.
    cov_cache: Dict[Tuple[str, ...], Optional[np.ndarray]] = {(): None}
    if requires_covariates(method):
        if not covariates:
            raise ValueError(f"Method '{method}' requires --covariates.")
        subsets = all_nonempty_subsets(covariates) if cov_selection else [list(covariates)]
        for covs in subsets:
            cov_cache[tuple(covs)] = _context_matrix(iso, level, covs, mask, site_idx)

    grid = SEARCH_SPACES.get(method, {})
    preds_all: List[np.ndarray] = []
    times_all: List[pd.DatetimeIndex] = []
    n_windows = 0

    if verbose:
        print(f"[INFO] {iso}/{level} method={method} "
              f"train_start={train_start_ts.date()} test_from={train_end.date()} "
              f"val_days={val_days} cov_selection={cov_selection}", flush=True)

    while True:
        val_start = train_end - val_window
        test_start = train_end
        test_end = train_end + test_window
        if test_start > ti[-1]:
            break

        calib_mask = (ti >= train_start_ts) & (ti < val_start)
        val_mask = (ti >= val_start) & (ti < train_end)
        test_mask = (ti >= test_start) & (ti < test_end)

        if calib_mask.sum() < 50 or test_mask.sum() == 0:
            train_end = test_end
            continue

        y_cal, Q_cal = y[calib_mask], Q[:, calib_mask]
        y_val, Q_val = y[val_mask], Q[:, val_mask]

        log_path = None
        if log_dir:
            tag = f"{iso}_{level}_{method}_{test_start.date()}"
            log_path = os.path.join(log_dir, f"{tag}.csv")

        best_params, best_covs = _tune_window(
            method, grid, covariates, cov_cache,
            y_cal, Q_cal, y_val, Q_val, calib_mask, val_mask,
            cov_selection=cov_selection,
            value_bounds=value_bounds,
            verbose=verbose,
            log_path=log_path,
        )

        fit_mask = (calib_mask | val_mask) if refit_includes_val else calib_mask
        X = cov_cache[() if best_covs is None else tuple(best_covs)]
        X_fit = None if X is None else X[fit_mask, :]
        X_test = None if X is None else X[test_mask, :]

        conf = make_conformalizer(
            method, **{**best_params, "value_bounds": value_bounds,
                       "enforce_monotonic": True}
        )
        conf.fit(y[fit_mask], Q[:, fit_mask], past_cov=X_fit)
        pred = conf.batch_forecast(Q[:, test_mask], future_cov=X_test)

        preds_all.append(pred)
        times_all.append(ti[test_mask])
        n_windows += 1

        if verbose >= 2:
            print(f"[WINDOW] {test_start.date()}  covs={best_covs or '-'}  "
                  f"params={best_params}", flush=True)

        train_end = test_end
        if max_windows is not None and n_windows >= max_windows:
            break

    if not preds_all:
        raise RuntimeError("No predictions produced; check the date windows.")

    preds = np.vstack(preds_all)
    times = pd.DatetimeIndex(np.concatenate([t.values for t in times_all]))
    _save(preds, times, out_npy, out_times)
    if verbose:
        print(f"[OK] {out_npy}  shape={preds.shape}  "
              f"{times[0]} -> {times[-1]}  ({n_windows} windows)", flush=True)
    return preds, times
