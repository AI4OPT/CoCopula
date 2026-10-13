"""Loading the NREL PERFORM arrays.

Expected layout under DATA_DIR (one folder per ISO):

    DATA_DIR/<ISO>/solar/day-ahead/sites/{actuals,marginals,capacity}.npy
    DATA_DIR/<ISO>/solar/day-ahead/system/{actuals,marginals,capacity}.npy
    DATA_DIR/cov/<iso>_system_{solarity,time,historical}.npy

Shapes: actuals (N, T), marginals (N, 99, T) with the 1%,...,99% quantiles,
capacity (N, 1), context blocks (1, k, T).  T = 8760 hourly steps.
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

YEAR = {"MISO": 2019, "SPP": 2019, "ERCOT": 2018}
CONTEXT = ["solarity", "time", "historical"]


def hours(iso):
    """Hourly time stamps of the year covered by the data of this ISO."""
    year = YEAR[iso]
    return pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")


def _folder(data_dir, iso, level):
    return os.path.join(data_dir, iso, "solar", "day-ahead", level)


def load_sites(data_dir, iso):
    """Site-level actuals (N, T) and quantile forecasts (N, 99, T).

    The forecasts are memory-mapped (about 5 GB for MISO) and read in blocks.
    """
    folder = _folder(data_dir, iso, "sites")
    actuals = np.load(os.path.join(folder, "actuals.npy"))
    quantiles = np.load(os.path.join(folder, "marginals.npy"), mmap_mode="r")
    return actuals, quantiles


def load_system(data_dir, iso):
    """Fleet-level actuals (T,), NREL fleet forecast (99, T) and capacity."""
    folder = _folder(data_dir, iso, "system")
    actuals = np.load(os.path.join(folder, "actuals.npy"))[0].astype(float)
    quantiles = np.load(os.path.join(folder, "marginals.npy"))[0].astype(float)
    capacity = float(np.load(os.path.join(folder, "capacity.npy"))[0, 0])
    return actuals, quantiles, capacity


def load_context(data_dir, iso):
    """Context vectors c_t, shape (T, k), each feature min-max scaled over the year.

    solarity   : normalized position in the solar day
    time       : sine/cosine encodings of hour, day of year and month
    historical : the 24 hourly fleet generation values of the previous day
    """
    blocks = [np.load(os.path.join(data_dir, "cov", f"{iso.lower()}_system_{name}.npy"))[0]
              for name in CONTEXT]
    X = np.concatenate(blocks, axis=0).T
    return MinMaxScaler().fit_transform(X)
