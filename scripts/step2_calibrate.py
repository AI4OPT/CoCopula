#!/usr/bin/env python3
"""Step 2 -- conformal calibration of one fleet-level forecast (Section 2.2).

One invocation handles a single (iso, source, method) cell of Table 1:

    --source nrel    NREL's own system-level forecast
    --source copula  the step-1 aggregation of the site-level forecasts

    --method none    uncalibrated pass-through   ("NREL" / "Copula")
    --method cqr     standard split conformal    ("+CQR")
    --method kernel  CACP, RBF/Laplacian weights ("+CACP", the paper default)
    --method knn     CACP, k-NN weights
    --method kmeans  CACP, cluster weights

Output: <PRED_DIR>/<ISO>/<source>_<method>.npy of shape (T_test, 99), plus a
matching _times.npy.  All values are normalised by system capacity.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_cacp import config
from copula_cacp.calibrate import export_conformalized, export_passthrough
from copula_cacp.conformal.factory import requires_covariates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iso", required=True,
                   choices=config.ISOS + [i.lower() for i in config.ISOS])
    p.add_argument("--source", required=True, choices=config.SOURCES)
    p.add_argument("--method", required=True, choices=config.METHODS)
    p.add_argument("--covariates", nargs="*", default=config.DEFAULT_COVARIATES,
                   help="Context blocks forming the context vector c "
                        "(default: solarity time historical).")
    p.add_argument("--no_cov_selection", action="store_true",
                   help="Use the full covariate set instead of tuning the subset.")
    p.add_argument("--test_window_days", type=int, default=config.TEST_WINDOW_DAYS)
    p.add_argument("--val_days", type=int, default=config.VAL_DAYS)
    p.add_argument("--first_train_end", default=None,
                   help="Start of the test period (default: six months in).")
    p.add_argument("--max_windows", type=int, default=None,
                   help="Debug: stop after N test days.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--log_dir", default=None, help="Write per-window tuning CSVs here.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iso = args.iso.upper()
    # "nrel" source reads NREL's system forecast; "copula" reads step-1 output.
    level = "system" if args.source == "nrel" else "copula"

    out_dir = args.out_dir or os.path.join(config.PRED_DIR, iso)
    os.makedirs(out_dir, exist_ok=True)
    tag = config.run_tag(args.source, args.method)
    out_npy = os.path.join(out_dir, f"{tag}.npy")
    out_times = os.path.join(out_dir, f"{tag}_times.npy")

    if os.path.exists(out_npy) and not args.overwrite:
        print(f"[SKIP] {out_npy} exists. Pass --overwrite to rerun.", flush=True)
        return

    if args.method == "none":
        export_passthrough(
            iso, level, out_npy, out_times,
            first_train_end=args.first_train_end,
            test_window_days=args.test_window_days,
            verbose=args.verbose,
        )
        return

    covariates = args.covariates if requires_covariates(args.method) else None
    export_conformalized(
        iso, level, args.method, out_npy, out_times,
        covariates=covariates,
        first_train_end=args.first_train_end,
        test_window_days=args.test_window_days,
        val_days=args.val_days,
        cov_selection=not args.no_cov_selection,
        max_windows=args.max_windows,
        log_dir=args.log_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
