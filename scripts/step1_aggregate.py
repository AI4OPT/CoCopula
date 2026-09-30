#!/usr/bin/env python3
"""Step 1 -- Gaussian-copula aggregation of site-level forecasts (Section 2.1).

Fits the dependence matrix on the history available at the time and produces
the fleet-level predictive quantiles.  Following Appendix B.1, the first fit
uses the first six months and the correlation structure is refreshed monthly:

    fit on months 1..6   -> predict months 1..7
    fit on months 1..7   -> predict month 8
    ...
    fit on months 1..11  -> predict month 12

Output: <AGG_DIR>/<iso>_solar_copula.npy   of shape (99, T)
        <AGG_DIR>/<iso>_sigma.npz          the monthly dependence matrices
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_cacp import config
from copula_cacp.copula import GaussianCopulaModel
from copula_cacp.data.loaders import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iso", required=True, choices=config.ISOS + [i.lower() for i in config.ISOS])
    p.add_argument("--start_month", type=int, default=config.COPULA_TRAIN_MONTHS,
                   help="Last month of the initial correlation fit (default: 6).")
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Monte-Carlo sample size for the aggregate distribution.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Override the output .npy path.")
    p.add_argument("--max_sites", type=int, default=None,
                   help="Debug: aggregate only the first N sites.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Debug: aggregate only the first N time steps.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iso = args.iso.upper()
    progress = not args.quiet

    # Memory-mapped: the MISO site tensor is ~5 GB and is consumed in slices.
    actuals, marginals, _, _, time_index = load_dataset(iso, "sites", None, mmap=True)
    if args.max_sites:
        actuals = actuals[: args.max_sites]
        marginals = marginals[: args.max_sites]
    if args.max_steps:
        actuals = actuals[:, : args.max_steps]
        marginals = marginals[:, :, : args.max_steps]
        time_index = time_index[: args.max_steps]

    n_sites, T = actuals.shape
    print(f"[INFO] {iso}: {n_sites} sites, {T} hourly steps, "
          f"year {config.year(iso)}", flush=True)

    copula = GaussianCopulaModel(n_samples=args.n_samples, seed=args.seed)
    predictions = np.full((99, T), np.nan, dtype=float)
    sigmas = {}

    # ---- initial fit: months 1..start_month, predict months 1..start_month+1
    t0 = time.time()
    train_mask = time_index.month <= args.start_month
    forecast_mask = time_index.month <= (args.start_month + 1)

    print(f"[FIT ] months 1-{args.start_month}  "
          f"({train_mask.sum()} steps)", flush=True)
    copula.fit(actuals, marginals, time_mask=train_mask, progress=progress)
    sigmas[f"month_{args.start_month}"] = copula.sigma

    print(f"[PRED] months 1-{args.start_month + 1}  "
          f"({forecast_mask.sum()} steps)", flush=True)
    predictions[:, forecast_mask] = copula.multi_step_pred(
        marginals, time_mask=forecast_mask, progress=progress)

    # ---- monthly refits
    for month in range(args.start_month + 1, 12):
        forecast_mask = time_index.month == (month + 1)
        if not forecast_mask.any():
            continue
        train_mask = time_index.month <= month

        print(f"[FIT ] months 1-{month} ({train_mask.sum()} steps)", flush=True)
        copula.fit(actuals, marginals, time_mask=train_mask, progress=progress)
        sigmas[f"month_{month}"] = copula.sigma

        print(f"[PRED] month {month + 1} ({forecast_mask.sum()} steps)", flush=True)
        predictions[:, forecast_mask] = copula.multi_step_pred(
            marginals, time_mask=forecast_mask, progress=progress)

    out = args.out or config.copula_path(iso)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, predictions)
    np.savez(config.sigma_path(iso), **sigmas)

    covered = np.isfinite(predictions[0]).sum()
    print(f"[OK] {out}  shape={predictions.shape}  "
          f"{covered}/{T} steps populated  "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
