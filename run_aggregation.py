"""Aggregate site-level quantile forecasts into a fleet-level forecast (stage 1).

    python run_aggregation.py --data_dir DATA --iso MISO --dependence copula --out out/miso_copula.npy

--dependence copula       Gaussian copula.  Sigma is fitted on January-June and
                          used for January-July; it is then refitted each month
                          on all earlier months (months 1..m -> month m+1).
--dependence independent  Sigma = I (sites treated as independent).

The output is a (99, 8760) array of fleet-level quantiles (same units as the data).
"""
import argparse
import os
import time

import numpy as np

from copula import aggregate, all_pits, fit_sigma, sample_uniforms
from data import hours, load_sites


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--iso", required=True, choices=["MISO", "SPP", "ERCOT"])
    p.add_argument("--dependence", default="copula", choices=["copula", "independent"])
    p.add_argument("--out", required=True, help="output .npy file")
    p.add_argument("--n_samples", type=int, default=1000, help="Monte Carlo samples S")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t_start = time.time()
    actuals, quantiles = load_sites(args.data_dir, args.iso)
    month = hours(args.iso).month.to_numpy()
    n_sites, T = actuals.shape
    print(f"{args.iso}: {n_sites} sites, {T} hours, dependence = {args.dependence}")

    fleet = np.full((99, T), np.nan)

    if args.dependence == "independent":
        U = sample_uniforms(np.identity(n_sites), args.n_samples, args.seed)
        fleet[:] = aggregate(quantiles, U, 0, T)

    else:
        print("computing PIT values ...")
        u = all_pits(actuals, quantiles)
        # (fit on months 1..m, forecast months [first, last])
        schedule = [(6, 1, 7)] + [(m, m + 1, m + 1) for m in range(7, 12)]
        for fit_until, first, last in schedule:
            sigma = fit_sigma(u[:, month <= fit_until])
            U = sample_uniforms(sigma, args.n_samples, args.seed)
            idx = np.flatnonzero((month >= first) & (month <= last))
            fleet[:, idx[0]:idx[-1] + 1] = aggregate(quantiles, U, idx[0], idx[-1] + 1)
            print(f"  Sigma fitted on months 1-{fit_until} -> forecast months {first}-{last}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, fleet)
    print(f"saved {args.out}  {fleet.shape}  ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
