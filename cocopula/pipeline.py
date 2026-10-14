"""The two stages over one year of data.

aggregate_fleet : site forecasts -> fleet forecast (99, T)
calibrate       : fleet forecast -> calibrated quantiles of the test hours

Protocol: everything is divided by the fleet capacity and hours with zero
generation are dropped.  July-December is split into consecutive 7-day test
windows; for each window the 7 days before it are the validation week and all
earlier hours the calibration set.  CACP chooses its kernel and gamma on the
validation week (lowest mean Winkler score); CQR and CACP are then refit on
calibration + validation and applied to the test window.
"""
import numpy as np
import pandas as pd

from .conformal import cacp, cqr, no_calibration
from .copula import aggregate, all_pits, fit_sigma, sample_uniforms
from .data import YEAR, hours, load_context, load_sites
from .metrics import mean_winkler

KERNELS = ["rbf", "laplacian"]
GAMMAS = [0.5, 1.0, 2.0]


def aggregate_fleet(data_dir, iso, dependence, n_samples=1000, seed=42):
    """Fleet-level quantiles (99, T) from the site forecasts.

    dependence = "copula":      Gaussian copula; Sigma is fitted on January-June
                                (used for January-July), then refitted each month
                                on all earlier months.
    dependence = "independent": Sigma = I.
    """
    actuals, quantiles = load_sites(data_dir, iso)
    month = hours(iso).month.to_numpy()
    n_sites, T = actuals.shape
    print(f"{iso}: aggregating {n_sites} sites ({dependence})")

    fleet = np.full((99, T), np.nan)
    if dependence == "independent":
        U = sample_uniforms(np.identity(n_sites), n_samples, seed)
        fleet[:] = aggregate(quantiles, U, 0, T)
        return fleet

    u = all_pits(actuals, quantiles)
    schedule = [(6, 1, 7)] + [(m, m + 1, m + 1) for m in range(7, 12)]   # (fit months 1..m, forecast months)
    for fit_until, first, last in schedule:
        sigma = fit_sigma(u[:, month <= fit_until])
        U = sample_uniforms(sigma, n_samples, seed)
        idx = np.flatnonzero((month >= first) & (month <= last))
        fleet[:, idx[0]:idx[-1] + 1] = aggregate(quantiles, U, idx[0], idx[-1] + 1)
        print(f"  Sigma fitted on months 1-{fit_until} -> forecast months {first}-{last}")
    return fleet


def calibrate(iso, fleet, actuals, capacity, calibration=None):
    """Calibrated quantiles of the test hours.

    fleet: (99, T) fleet forecast, actuals: (T,) fleet actuals.
    calibration: None (no calibration), "cqr" or "cacp".
    Returns (P (T_test, 99), y_test (T_test,), test time stamps), as fractions of capacity.
    """
    day = actuals > 0
    y, Q, times = actuals[day] / capacity, fleet[:, day] / capacity, hours(iso)[day]
    X = load_context(iso)[day] if calibration == "cacp" else None

    year = YEAR[iso]
    first, last = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-07-01")
    week = pd.Timedelta(days=7)
    preds, test_times = [], []
    while last <= times[-1]:
        cal = np.asarray((times >= first) & (times < last - week))
        val = np.asarray((times >= last - week) & (times < last))
        test = np.asarray((times >= last) & (times < last + week))
        if test.any() and cal.sum() >= 50:
            fit = cal | val
            if calibration is None:
                P = no_calibration(Q[:, test])
            elif calibration == "cqr":
                P = cqr(y[fit], Q[:, fit], Q[:, test])
            else:
                best, best_ws = None, np.inf          # kernel and gamma chosen on the validation week
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

    t = pd.DatetimeIndex(np.concatenate([tt.values for tt in test_times]))
    y_test = pd.Series(y, index=times).loc[t].to_numpy()
    return np.vstack(preds), y_test, t
