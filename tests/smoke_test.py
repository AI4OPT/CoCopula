#!/usr/bin/env python3
"""Fast end-to-end wiring check -- safe to run on a login node (~30 s).

Exercises every stage on a deliberately tiny slice: a handful of sites and
hours for the copula, two rolling windows for the calibrator.  It proves the
plumbing works; it says nothing about the published numbers.

    python tests/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_cacp import config
from copula_cacp.calibrate import export_conformalized, export_passthrough
from copula_cacp.copula import GaussianCopulaModel
from copula_cacp.data.loaders import load_dataset
from copula_cacp.evaluate.metrics import picp_aiw, winkler_score

ISO = "MISO"
N_SITES, N_STEPS = 8, 480


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}",
          flush=True)
    if not cond:
        raise AssertionError(name)


def main() -> None:
    t0 = time.time()

    print("\n[1/4] loaders")
    # mmap: the full site tensor is ~5 GB and must never be read whole here.
    actuals, marginals, capacity, cov, ti = load_dataset(
        ISO, "sites", config.DEFAULT_COVARIATES[:1], mmap=True)
    check("sites shapes", actuals.ndim == 2 and marginals.ndim == 3,
          f"{actuals.shape} {marginals.shape}")
    check("time index length", len(ti) == actuals.shape[1], str(len(ti)))
    sa, sm, sc, _, _ = load_dataset(ISO, "system", None)
    check("system is one series", sa.shape[0] == 1, str(sa.shape))
    check("capacity positive", float(sc[0]) > 0, f"{float(sc[0]):.1f}")

    print("\n[2/4] copula aggregation (tiny slice)")
    a = np.asarray(actuals[:N_SITES, :N_STEPS])
    m = np.asarray(marginals[:N_SITES, :, :N_STEPS])
    cop = GaussianCopulaModel()
    cop.fit(a, m, progress=False)
    check("sigma is square", cop.sigma.shape == (N_SITES, N_SITES), str(cop.sigma.shape))
    check("sigma diagonal ~ 1", np.allclose(np.diag(cop.sigma), 1.0, atol=1e-6))
    check("sigma symmetric", np.allclose(cop.sigma, cop.sigma.T, atol=1e-8))

    pred = cop.multi_step_pred(m[:, :, :48], progress=False)
    check("aggregate shape", pred.shape == (99, 48), str(pred.shape))
    check("quantiles non-decreasing", bool(np.all(np.diff(pred, axis=0) >= -1e-9)))
    site_sum_max = a[:, :48].sum(axis=0).max()
    check("aggregate magnitude sane",
          pred[94].max() >= 0 and pred[94].max() < 10 * max(site_sum_max, 1e-9),
          f"p95max={pred[94].max():.1f} vs sum(actual)max={site_sum_max:.1f}")

    print("\n[3/4] calibration (2 rolling windows, system level)")
    with tempfile.TemporaryDirectory() as td:
        p_pass = os.path.join(td, "pass.npy")
        export_passthrough(ISO, "system", p_pass, os.path.join(td, "pass_t.npy"),
                           verbose=0)
        base = np.load(p_pass)
        check("pass-through shape", base.ndim == 2 and base.shape[1] == 99,
              str(base.shape))

        p_cqr = os.path.join(td, "cqr.npy")
        export_conformalized(ISO, "system", "cqr", p_cqr,
                             os.path.join(td, "cqr_t.npy"),
                             max_windows=2, verbose=0)
        cqr = np.load(p_cqr)
        check("cqr shape", cqr.shape[1] == 99, str(cqr.shape))
        check("cqr widened the interval",
              (cqr[:, 94] - cqr[:, 4]).mean() >= (base[:len(cqr), 94] - base[:len(cqr), 4]).mean() - 1e-9,
              f"{(cqr[:, 94] - cqr[:, 4]).mean():.4f} vs "
              f"{(base[:len(cqr), 94] - base[:len(cqr), 4]).mean():.4f}")

        p_k = os.path.join(td, "kernel.npy")
        export_conformalized(ISO, "system", "kernel", p_k,
                             os.path.join(td, "kernel_t.npy"),
                             covariates=config.DEFAULT_COVARIATES,
                             cov_selection=False, max_windows=2, verbose=0)
        ker = np.load(p_k)
        check("CACP-kernel shape", ker.shape[1] == 99, str(ker.shape))
        check("CACP output monotone", bool(np.all(np.diff(ker, axis=1) >= -1e-9)))
        check("CACP within [0,1]", bool(ker.min() >= 0 and ker.max() <= 1))

    print("\n[4/4] metrics")
    y = np.linspace(0.1, 0.9, 100)
    perfect = np.tile(np.linspace(0, 1, 99), (100, 1))
    picp, aiw = picp_aiw(y, perfect)
    ws = winkler_score(y, perfect)
    check("picp in [0,1]", all(0 <= v <= 1 for v in picp.values()))
    check("aiw positive", all(v > 0 for v in aiw.values()))
    check("winkler >= width", all(ws[c] >= aiw[c] - 1e-9 for c in aiw))

    print(f"\nAll smoke checks passed in {time.time() - t0:.1f}s.\n")


if __name__ == "__main__":
    main()
