"""Calibrate a fleet-level forecast (stage 2) and report PICP / AIW / WS.

    python scripts/run_calibration.py --data_dir DATA --iso MISO --forecast out/miso_copula.npy --method cacp --out_dir out/

--forecast   nrel  (NREL's own fleet forecast)  or  a (99, T) file from scripts/run_aggregation.py
--method     none  (no calibration), cqr, or cacp

Protocol: hours with zero generation are dropped and everything is divided by
the fleet capacity.  July-December is split into consecutive 7-day test
windows.  For each window, the 7 days before it are the validation week and
all earlier hours are the calibration set.  CACP picks its kernel and gamma
on the validation week (lowest mean Winkler score), then both CQR and CACP
are refit on calibration + validation and applied to the test window.

Outputs in --out_dir, prefixed <forecast name>_<method> (e.g. miso_copula_cacp, miso_nrel_cqr):
    _quantiles.npy  (T_test, 99) calibrated quantiles (fraction of capacity)
    _times.npy      time stamps of the test hours
    _metrics.csv    PICP, AIW and WS per target coverage
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

from cocopula.conformal import cacp, cqr, no_calibration
from cocopula.data import YEAR, hours, load_context, load_system
from cocopula.metrics import mean_winkler, window_metrics

KERNELS = ["rbf", "laplacian"]
GAMMAS = [0.5, 1.0, 2.0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--iso", required=True, choices=["MISO", "SPP", "ERCOT"])
    p.add_argument("--forecast", required=True, help="'nrel' or a .npy file from scripts/run_aggregation.py")
    p.add_argument("--method", required=True, choices=["none", "cqr", "cacp"])
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    # ---- data: capacity-normalized, night hours removed
    actuals, nrel, capacity = load_system(args.data_dir, args.iso)
    if args.forecast == "nrel":
        Q, name = nrel, f"{args.iso.lower()}_nrel"
    else:
        Q, name = np.load(args.forecast), os.path.splitext(os.path.basename(args.forecast))[0]
    day = actuals > 0
    y, Q, times = actuals[day] / capacity, Q[:, day] / capacity, hours(args.iso)[day]
    X = load_context(args.iso)[day] if args.method == "cacp" else None

    # ---- rolling 7-day test windows from 1 July
    year = YEAR[args.iso]
    first, last = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-07-01")
    week = pd.Timedelta(days=7)
    preds, test_times = [], []
    while last <= times[-1]:
        cal = np.asarray((times >= first) & (times < last - week))
        val = np.asarray((times >= last - week) & (times < last))
        test = np.asarray((times >= last) & (times < last + week))
        if test.any() and cal.sum() >= 50:
            fit = cal | val
            if args.method == "none":
                P = no_calibration(Q[:, test])
            elif args.method == "cqr":
                P = cqr(y[fit], Q[:, fit], Q[:, test])
            else:
                # choose kernel and gamma on the validation week
                best, best_ws = None, np.inf
                for kernel in KERNELS:
                    for gamma in GAMMAS:
                        P_val = cacp(y[cal], Q[:, cal], X[cal], Q[:, val], X[val], kernel, gamma)
                        ws = mean_winkler(y[val], P_val)
                        if ws < best_ws:
                            best, best_ws = (kernel, gamma), ws
                P = cacp(y[fit], Q[:, fit], X[fit], Q[:, test], X[test], *best)
                print(f"  week of {last.date()}: kernel = {best[0]}, gamma = {best[1]}")
            preds.append(P)
            test_times.append(times[test])
        last += week

    P = np.vstack(preds)
    t = pd.DatetimeIndex(np.concatenate([tt.values for tt in test_times]))
    y_test = pd.Series(y, index=times).loc[t].to_numpy()

    # ---- save and report
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"{name}_{args.method}")
    np.save(prefix + "_quantiles.npy", P)
    np.save(prefix + "_times.npy", t.values)
    table = window_metrics(y_test, P, t, f"{year}-07-01")
    table.to_csv(prefix + "_metrics.csv")
    print(f"\n{args.iso} | forecast = {name} | method = {args.method} | {len(t)} test hours")
    print(table.round(4).to_string())
    print(f"\nsaved {prefix}_{{quantiles.npy, times.npy, metrics.csv}}")


if __name__ == "__main__":
    main()
