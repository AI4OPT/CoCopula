#!/usr/bin/env python3
"""Check this repo against the code that produced the published results.

Two comparisons, both on a small slice so they run on a login node:

  copula   -- Sigma and the aggregated quantiles must match the original
              Aggregated_Forecasting/Copula.py bit-for-bit (up to float noise).
  conformal-- CQR and CACP-kernel corrections must match the original
              PyCP conformalizers.

Point --legacy_copula / --legacy_pycp at the original trees.  Either check is
skipped (not failed) if its tree is absent.

    python tests/test_equivalence.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_cacp.conformal.factory import make_conformalizer
from copula_cacp.copula import GaussianCopulaModel
from copula_cacp.data.loaders import load_dataset

DEFAULT_LEGACY_COPULA = "/storage/home/hcoda1/1/amoradi30/Aggregated_Forecasting"
DEFAULT_LEGACY_PYCP = "/storage/home/hcoda1/1/amoradi30/r-phentenryck3-1/CACP/PyCP"

ISO, N_SITES, N_FIT, N_PRED = "MISO", 6, 240, 24


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(label: str, a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    finite = np.isfinite(a) & np.isfinite(b)
    same_nan = (np.isfinite(a) == np.isfinite(b)).all()
    diff = np.abs(a[finite] - b[finite])
    worst = float(diff.max()) if diff.size else 0.0
    ok = same_nan and worst <= tol
    print(f"  {'MATCH' if ok else 'DIFFER'}  {label}: max|delta|={worst:.3e} "
          f"(n={finite.sum()}, nan-pattern {'same' if same_nan else 'DIFFERENT'})")
    return ok


def check_copula(legacy_dir: str) -> bool | None:
    path = os.path.join(legacy_dir, "Copula.py")
    if not os.path.exists(path):
        print(f"  SKIP   copula: {path} not found")
        return None

    legacy = _load_module(path, "legacy_copula")
    actuals, marginals, _, _, _ = load_dataset(ISO, "sites", None, mmap=True)
    a = np.asarray(actuals[:N_SITES, :N_FIT])
    m = np.asarray(marginals[:N_SITES, :, :N_FIT])

    old = legacy.GaussianCopulaModel()
    old.fit(a, m)
    old_pred = old.multi_step_pred(m[:, :, :N_PRED])

    new = GaussianCopulaModel()
    new.fit(a, m, progress=False)
    new_pred = new.multi_step_pred(m[:, :, :N_PRED], progress=False)

    ok = report("Sigma", old.sigma, new.sigma)
    ok &= report("aggregated quantiles", old_pred, new_pred)
    return ok


def check_conformal(legacy_pycp: str) -> bool | None:
    if not os.path.isdir(legacy_pycp):
        print(f"  SKIP   conformal: {legacy_pycp} not found")
        return None

    sys.path.insert(0, legacy_pycp)
    try:
        from conformal.cqr import CQRConformalizer as LegacyCQR
        from conformal.kernel import KernelConformalizer as LegacyKernel
    except Exception as e:
        print(f"  SKIP   conformal: cannot import legacy modules ({e})")
        return None
    finally:
        sys.path.remove(legacy_pycp)

    actuals, marginals, capacity, cov, _ = load_dataset(
        ISO, "system", ["solarity", "time", "historical"])
    cap = float(capacity[0])
    y = actuals[0] / cap
    Q = marginals[0] / cap
    X = np.transpose(cov[0], (1, 0))

    m = y > 0
    y, Q, X = y[m], Q[:, m], X[m, :]
    y_fit, Q_fit, X_fit = y[:600], Q[:, :600], X[:600]
    Q_te, X_te = Q[:, 600:640], X[600:640]

    ok = True
    for name, Legacy, New, kw in (
        ("CQR", LegacyCQR, None, {}),
        ("CACP-kernel", LegacyKernel, None, {"kernel": "rbf", "gamma": 1.0}),
    ):
        old = Legacy(value_bounds=(0.0, 1.0), **kw)
        old.fit(y_fit, Q_fit, past_cov=X_fit if kw else None)
        old_pred = old.batch_forecast(Q_te, future_cov=X_te if kw else None)

        new = make_conformalizer(
            "cqr" if name == "CQR" else "kernel",
            value_bounds=(0.0, 1.0), **kw)
        new.fit(y_fit, Q_fit, past_cov=X_fit if kw else None)
        new_pred = new.batch_forecast(Q_te, future_cov=X_te if kw else None)

        ok &= report(name, old_pred, new_pred)
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--legacy_copula", default=DEFAULT_LEGACY_COPULA)
    p.add_argument("--legacy_pycp", default=DEFAULT_LEGACY_PYCP)
    args = p.parse_args()

    print("\n[1/2] copula aggregation vs Aggregated_Forecasting/Copula.py")
    r1 = check_copula(args.legacy_copula)
    print("\n[2/2] conformalizers vs PyCP/conformal/")
    r2 = check_conformal(args.legacy_pycp)

    results = [r for r in (r1, r2) if r is not None]
    if not results:
        print("\nNothing to compare (no legacy trees found).")
        return
    print(f"\n{'EQUIVALENT' if all(results) else 'MISMATCH — investigate above'}\n")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
