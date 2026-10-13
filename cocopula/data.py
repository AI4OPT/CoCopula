"""Loading the data.

One folder per ISO under DATA_DIR:

    DATA_DIR/<ISO>/site_actuals.npy          (N, T)      realized generation of each site
    DATA_DIR/<ISO>/site_quantiles.npy        (N, 99, T)  1%,...,99% quantile forecasts of each site
    DATA_DIR/<ISO>/fleet_actuals.npy         (T,)        realized fleet generation
    DATA_DIR/<ISO>/fleet_quantiles_nrel.npy  (99, T)     NREL's fleet-level forecast (reference)
    DATA_DIR/<ISO>/fleet_capacity.npy        (1,)        fleet capacity

The context features of CACP are not part of the NREL data; they are stored in
this repository under covariates/<iso>_{solarity,time,historical}.npy.

T = 8760 hours of 2019 (MISO, SPP) or 2018 (ERCOT).
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

YEAR = {"MISO": 2019, "SPP": 2019, "ERCOT": 2018}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COVARIATES = os.path.join(REPO, "covariates")


def hours(iso):
    """Hourly time stamps of the year covered by the data of this ISO."""
    year = YEAR[iso]
    return pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")


def _load(data_dir, iso, name, **kw):
    return np.load(os.path.join(data_dir, iso, name + ".npy"), **kw)


def load_sites(data_dir, iso):
    """Site-level actuals (N, T) and quantile forecasts (N, 99, T).

    The forecasts are memory-mapped (2.6 GB for MISO) and read in blocks.
    """
    return _load(data_dir, iso, "site_actuals"), _load(data_dir, iso, "site_quantiles", mmap_mode="r")


def load_system(data_dir, iso):
    """Fleet-level actuals (T,), NREL fleet forecast (99, T) and capacity."""
    actuals = _load(data_dir, iso, "fleet_actuals").astype(float)
    quantiles = _load(data_dir, iso, "fleet_quantiles_nrel").astype(float)
    capacity = float(_load(data_dir, iso, "fleet_capacity")[0])
    return actuals, quantiles, capacity


def load_context(iso):
    """Context vectors c_t, shape (T, k), each feature min-max scaled over the year.

    solarity   : normalized position in the solar day
    time       : sine/cosine encodings of hour, day of year and month
    historical : the 24 hourly fleet generation values of the previous day
    """
    blocks = [np.load(os.path.join(COVARIATES, f"{iso.lower()}_{name}.npy"))[0]
              for name in ["solarity", "time", "historical"]]
    return MinMaxScaler().fit_transform(np.concatenate(blocks, axis=0).T)
