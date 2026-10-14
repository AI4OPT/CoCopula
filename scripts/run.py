"""Run one method on one ISO: aggregation, calibration and evaluation.

    python scripts/run.py --data_dir DATA_DIR --iso MISO --aggregation copula --calibration cacp --out_dir out/

--aggregation  nrel         NREL's fleet-level forecast (no aggregation)
               independent  site forecasts aggregated with Sigma = I
               copula       site forecasts aggregated with a Gaussian copula
--calibration  cqr or cacp; leave it out for no calibration

The aggregated fleet forecast is saved as out/<iso>_<aggregation>.npy and reused
by later runs.  Results are saved as out/<iso>_<aggregation>[_<calibration>]_*:
    _quantiles.npy  (T_test, 99) calibrated quantiles (fraction of capacity)
    _times.npy      time stamps of the test hours
    _metrics.csv    PICP, AIW and Winkler score per target coverage
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

from cocopula.data import YEAR, load_system
from cocopula.metrics import window_metrics
from cocopula.pipeline import aggregate_fleet, calibrate


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--iso", required=True, choices=["MISO", "SPP", "ERCOT"])
    p.add_argument("--aggregation", required=True, choices=["nrel", "independent", "copula"])
    p.add_argument("--calibration", default=None, choices=["cqr", "cacp"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples", type=int, default=1000, help="Monte Carlo samples (aggregation)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    iso = args.iso

    # ---- stage 1: fleet-level forecast
    actuals, nrel, capacity = load_system(args.data_dir, iso)
    if args.aggregation == "nrel":
        fleet = nrel
    else:
        path = os.path.join(args.out_dir, f"{iso.lower()}_{args.aggregation}.npy")
        if os.path.exists(path):
            print(f"using the saved fleet forecast {path}")
            fleet = np.load(path)
        else:
            fleet = aggregate_fleet(args.data_dir, iso, args.aggregation, args.n_samples, args.seed)
            np.save(path, fleet)

    # ---- stage 2: calibration (optional) and evaluation
    P, y, t = calibrate(iso, fleet, actuals, capacity, args.calibration)
    name = f"{iso.lower()}_{args.aggregation}" + (f"_{args.calibration}" if args.calibration else "")
    prefix = os.path.join(args.out_dir, name)
    np.save(prefix + "_quantiles.npy", P)
    np.save(prefix + "_times.npy", t.values)
    table = window_metrics(y, P, t, f"{YEAR[iso]}-07-01")
    table.to_csv(prefix + "_metrics.csv")

    print(f"\n{iso} | aggregation = {args.aggregation} | calibration = {args.calibration or 'none'} "
          f"| {len(t)} test hours | {(time.time() - t0) / 60:.1f} min")
    print(table.round(4).to_string())
    print(f"\nsaved {prefix}_{{quantiles.npy, times.npy, metrics.csv}}")


if __name__ == "__main__":
    main()
