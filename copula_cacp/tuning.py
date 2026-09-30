"""Per-window hyperparameter and context-subset selection.

For every rolling test day the calibration set is split into a core part and
the most recent week (validation).  Every (hyperparameter, covariate subset)
candidate is fit on the core part and scored on the validation week by mean
Winkler score; the winner is refit on core+validation before predicting.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .conformal.factory import make_conformalizer
from .evaluate.metrics import mean_winkler


def product_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a parameter grid; [{}] when the grid is empty."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    vals = [list(param_grid[k]) for k in keys]
    return [dict(zip(keys, comb)) for comb in itertools.product(*vals)]


def all_nonempty_subsets(items: List[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for r in range(1, len(items) + 1):
        out.extend(list(itertools.combinations(items, r)))
    return [list(x) for x in out]


def search_winkler(
    method: str,
    candidates: List[Dict[str, Any]],
    y_train: np.ndarray, Q_train: np.ndarray, cov_train: Optional[np.ndarray],
    y_val: np.ndarray, Q_val: np.ndarray, cov_val: Optional[np.ndarray],
    *,
    value_bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
    verbose: int = 0,
    log_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], float]:
    """Grid search over `candidates`, scored by validation Winkler."""
    best_params: Dict[str, Any] = {}
    best_score = float("inf")
    total = len(candidates)

    fh = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        fh = open(log_path, "w", encoding="utf-8")
        fh.write("trial,params,val_winkler\n")

    for i, params in enumerate(candidates, start=1):
        try:
            conf = make_conformalizer(method, **{**params, "value_bounds": value_bounds})
            conf.fit(y_train, Q_train, past_cov=cov_train)
            pred = conf.batch_forecast(Q_val, future_cov=cov_val)
            score = mean_winkler(y_val, pred)
        except Exception as e:                       # a bad candidate must not kill the run
            if verbose:
                print(f"[TUNE] ({i}/{total}) {params} -> FAIL {type(e).__name__}: {e}",
                      flush=True)
            score = float("inf")

        if verbose > 1:
            print(f"[TUNE] ({i}/{total}) {params} -> val_winkler={score:.6f}", flush=True)
        if fh:
            fh.write(f"{i},{repr(params)},{score:.6f}\n")
        if score < best_score:
            best_params, best_score = params, score

    if fh:
        fh.close()
    return best_params, best_score
