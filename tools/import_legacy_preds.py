#!/usr/bin/env python3
"""Import the original run's prediction arrays into this repo's layout.

The published figures were made from prediction arrays scattered across the
old trees under inconsistent names.  This tool copies them into
`<PRED_DIR>/<ISO>/<source>_<method>.npy` so that `step3_table.py` and
`step4_figures.py` run against them unchanged -- no recomputation.

Filenames are not trusted.  A conformal correction shifts the (li, ui) column
pair of each level but never moves the median column 49, so the median is an
exact fingerprint of which forecast an array was calibrated from.  Every
import is checked against both candidate bases and rejected if the array does
not track the base its registry entry claims.

    python tools/import_legacy_preds.py --verify        # check only
    python tools/import_legacy_preds.py                 # check and write
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_cacp import config
from copula_cacp.data.loaders import load_dataset

LEGACY_PYCP = "/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/CACP/PyCP"
LEGACY_ARRAY = "/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/CACP/PyCP_run/preds_array"

# (iso, source, method) -> legacy file stem, or None to rebuild from the base.
#
# MISO/SPP come from the six-month-calibration runs (test period July-December)
# that the paper's figures used.  ERCOT has no copula-calibrated arrays on
# disk, so only its NREL rows can be imported; those come from the preds_array
# export, whose test period starts in March.
REGISTRY = {
    ("MISO", "nrel", "none"):     None,
    ("MISO", "nrel", "cqr"):      f"{LEGACY_PYCP}/system_cqr",
    ("MISO", "nrel", "kernel"):   f"{LEGACY_PYCP}/system7",
    ("MISO", "copula", "none"):   None,
    ("MISO", "copula", "cqr"):    f"{LEGACY_PYCP}/copula_cqr",
    # miso_copula7 is a byte-equal duplicate of this file (max|delta| = 9e-15).
    ("MISO", "copula", "kernel"): f"{LEGACY_PYCP}/miso_copula",

    ("SPP", "nrel", "none"):      None,
    ("SPP", "nrel", "cqr"):       f"{LEGACY_PYCP}/spp_system_cqr",
    ("SPP", "nrel", "kernel"):    f"{LEGACY_PYCP}/spp_system_cacp",
    ("SPP", "copula", "none"):    None,
    ("SPP", "copula", "cqr"):     f"{LEGACY_PYCP}/spp_copula_cqr",
    ("SPP", "copula", "kernel"):  f"{LEGACY_PYCP}/spp_copula_cacp",

    ("ERCOT", "nrel", "none"):    None,
    ("ERCOT", "nrel", "cqr"):     f"{LEGACY_ARRAY}/ercot_system/cqr",
    ("ERCOT", "nrel", "kernel"):  f"{LEGACY_ARRAY}/ercot_system/kernel",
    ("ERCOT", "copula", "none"):  None,
}

# Cells the original run never wrote to disk, so they cannot be recreated from
# stored predictions -- they need step 2 of the pipeline to be run.
UNAVAILABLE = {
    ("ERCOT", "copula", "cqr"): "no ercot copula+CQR array exists",
    ("ERCOT", "copula", "kernel"): "no ercot copula+CACP array exists",
}

MEDIAN_COL = 49       # never moved by a conformal correction
RATIO_MIN = 5.0       # claimed base must fit at least this much better


def _bases(iso: str):
    """Capacity-normalised (time_index, actuals, NREL quantiles, copula quantiles)."""
    actuals, marginals, capacity, _, ti = load_dataset(iso, "system", None)
    cap = float(capacity[0])
    nrel = marginals[0] / cap

    cop_path = config.copula_path(iso)
    legacy_cop = (f"/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/"
                  f"nrel_forecasts/{iso.lower()}_solar_copula.npy")
    path = cop_path if os.path.exists(cop_path) else legacy_cop
    cop = np.load(path) / cap if os.path.exists(path) else None
    return ti, actuals[0] / cap, nrel, cop, path


def _positions(ti: pd.DatetimeIndex, times: pd.DatetimeIndex) -> np.ndarray:
    return pd.Series(np.arange(len(ti)), index=ti).loc[times].to_numpy()


def _check_base(pred, pos, nrel, cop, claimed):
    """Return (ok, detail) by comparing the untouched median to both bases."""
    d_nrel = float(np.abs(pred[:, MEDIAN_COL] - nrel[MEDIAN_COL, pos]).mean())
    d_cop = (float(np.abs(pred[:, MEDIAN_COL] - cop[MEDIAN_COL, pos]).mean())
             if cop is not None else np.inf)
    winner = "nrel" if d_nrel < d_cop else "copula"
    best, other = (d_nrel, d_cop) if winner == "nrel" else (d_cop, d_nrel)
    ratio = other / best if best > 0 else np.inf
    ok = (winner == claimed) and ratio >= RATIO_MIN
    return ok, (f"|med-NREL|={d_nrel:.2e} |med-Copula|={d_cop:.2e} "
                f"-> {winner}-based (x{ratio:.0f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="Check without writing.")
    ap.add_argument("--isos", nargs="+", default=config.ISOS)
    ap.add_argument("--out_dir", default=config.PRED_DIR)
    args = ap.parse_args()

    failures, written, missing = 0, 0, []

    for iso in [i.upper() for i in args.isos]:
        ti, y, nrel, cop, cop_path = _bases(iso)
        test_from = pd.Timestamp(config.first_train_end(iso))
        print(f"\n=== {iso} ===  copula source: {os.path.basename(cop_path)}")

        for (u_iso, u_src, u_meth), why in UNAVAILABLE.items():
            if u_iso == iso:
                missing.append(f"{iso} {u_src}+{u_meth}: {why}")

        for (r_iso, source, method), stem in REGISTRY.items():
            if r_iso != iso:
                continue
            label = f"{source}+{method}"
            base = nrel if source == "nrel" else cop
            if base is None:
                missing.append(f"{iso} {label} (no base array)")
                continue

            if stem is None:
                # Uncalibrated pass-through, rebuilt from the base quantiles on
                # the same night-masked test period the calibrated runs used.
                m = (y > 0) & np.asarray(ti >= test_from)
                pred = np.clip(np.sort(base[:, m], axis=0), 0.0, 1.0).T
                times = ti[m]
                detail = f"rebuilt from base, n={len(times)}"
                ok = True
            else:
                p, t = f"{stem}.npy", f"{stem}_times.npy"
                if not (os.path.exists(p) and os.path.exists(t)):
                    missing.append(f"{iso} {label} ({os.path.basename(stem)} absent)")
                    continue
                pred = np.load(p)
                times = pd.DatetimeIndex(np.load(t, allow_pickle=True))
                ok, detail = _check_base(pred, _positions(ti, times), nrel, cop, source)
                detail += f", n={len(times)}"

            status = "OK  " if ok else "FAIL"
            print(f"  {status} {label:16s} {detail}")
            if not ok:
                failures += 1
                continue

            if not args.verify:
                out = os.path.join(args.out_dir, iso)
                os.makedirs(out, exist_ok=True)
                np.save(os.path.join(out, f"{source}_{method}.npy"), pred)
                np.save(os.path.join(out, f"{source}_{method}_times.npy"), times.values)
                written += 1

    if missing:
        print("\nNot available on disk:")
        for m in missing:
            print(f"  - {m}")

    print(f"\n{'verified' if args.verify else 'imported'}: "
          f"{written} written, {failures} failed, {len(missing)} missing")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
