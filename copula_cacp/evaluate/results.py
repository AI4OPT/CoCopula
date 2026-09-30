"""Loading step-2 outputs and aligning them onto a common test timeline."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config
from ..data.loaders import load_dataset


def load_actuals(iso: str) -> Tuple[pd.Series, float]:
    """Capacity-normalised system actuals indexed by timestamp."""
    actuals, _, capacity, _, time_index = load_dataset(iso, "system", None)
    cap = float(capacity[0])
    return pd.Series(actuals[0].astype(float) / cap, index=time_index), cap


def load_run(iso: str, source: str, method: str
             ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Return (preds (T, 99), times) for one (source, method) cell."""
    p = config.pred_path(iso, source, method)
    t = config.times_path(iso, source, method)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing predictions: {p}\n"
            f"Run:  python scripts/step2_calibrate.py --iso {iso} "
            f"--source {source} --method {method}"
        )
    preds = np.load(p)
    times = pd.DatetimeIndex(np.load(t, allow_pickle=True))
    return preds, times


def load_aligned(
    iso: str,
    runs: Optional[List[Tuple[str, str, str]]] = None,
    require_all: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DatetimeIndex]:
    """Load several runs and restrict them to their common timestamps.

    Returns (y, {label: preds}, times) where every array shares one time axis,
    so metrics are computed on exactly the same test points for every method.
    """
    runs = runs or config.PAPER_METHODS
    y_series, _ = load_actuals(iso)

    loaded: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]] = {}
    for source, method, label in runs:
        try:
            loaded[label] = load_run(iso, source, method)
        except FileNotFoundError:
            if require_all:
                raise
            print(f"[WARN] {iso}: skipping missing run {label}", flush=True)

    if not loaded:
        raise RuntimeError(f"{iso}: no runs available.")

    common = None
    for _, times in loaded.values():
        common = times if common is None else common.intersection(times)
    common = common.sort_values()

    out: Dict[str, np.ndarray] = {}
    for label, (preds, times) in loaded.items():
        pos = pd.Series(np.arange(len(times)), index=times)
        out[label] = preds[pos.loc[common].to_numpy()]

    y = y_series.loc[common].to_numpy()
    return y, out, common
