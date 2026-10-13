"""Download the NREL / ARPA-E PERFORM data and build the arrays used by CoCopula.

    python scripts/download_data.py --iso MISO --data_dir DATA_DIR

The raw HDF5 files are fetched from the public AWS bucket of PERFORM
(https://registry.opendata.aws/arpa-e-perform/, no account needed) into
DATA_DIR/raw/ and converted to DATA_DIR/<ISO>/*.npy:

    site_actuals.npy          (N, T)      hourly mean of the 5-minute site actuals
    site_quantiles.npy        (N, 99, T)  day-ahead site forecasts (1%,...,99% quantiles)
    fleet_actuals.npy         (T,)        hourly mean of the 5-minute fleet (BA) actuals
    fleet_quantiles_nrel.npy  (99, T)     NREL's day-ahead fleet (BA) forecast
    fleet_capacity.npy        (1,)        fleet capacity, from the BA actuals file

The sites are stored in the order used in the paper (sites/<ISO>.txt).
The CACP context features are not NREL data; they ship with the repository in
covariates/.  Download sizes:
MISO ~6 GB, SPP ~1.5 GB, ERCOT ~1.9 GB.
"""
import argparse
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

from cocopula.data import YEAR

BUCKET = "https://arpa-e-perform.s3.amazonaws.com"
SITES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sites")


# ---------------------------------------------------------------- download
def list_keys(prefix):
    """All object keys under a prefix of the public bucket."""
    keys, token = [], None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token)
        root = ET.fromstring(urllib.request.urlopen(url).read())
        keys += [k.text for k in root.iter(ns + "Key")]
        token = root.findtext(ns + "NextContinuationToken")
        if not token:
            return keys


def fetch(key, raw_dir):
    """Download one object (skipped if already present).  Returns the local path."""
    path = os.path.join(raw_dir, key)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(f"{BUCKET}/{urllib.request.quote(key)}", path + ".part")
                os.replace(path + ".part", path)
                break
            except Exception:
                if attempt == 4:
                    raise
    return path


def fetch_all(keys, raw_dir, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda k: fetch(k, raw_dir), keys))


# ---------------------------------------------------------------- reading
def hourly_actuals(path):
    """5-minute actuals of one HDF5 file -> hourly means, shape (8760,)."""
    a = h5py.File(path, "r")["actuals"][:]
    if a.dtype.names:                      # ERCOT site files: structured array
        a = a[a.dtype.names[0]]
    if a.ndim == 2:                        # ERCOT BA file: column 0 is the fleet total
        a = a[:, 0]
    return a.astype(float).reshape(-1, 12).mean(axis=1)


def fleet_capacity(path):
    """Fleet capacity (MW) stored in the BA actuals file."""
    meta = h5py.File(path, "r")["meta"][:]
    if meta.dtype.names:                   # ERCOT: one row per site, sum their AC capacity
        return float(sum(float(row["AC_capacity_MW"]) for row in meta[1:]))
    return float(meta[1])                  # MISO, SPP: ["BA", capacity]


def forecasts(path):
    """Day-ahead quantile forecasts of one HDF5 file, shape (99, 8760)."""
    f = h5py.File(path, "r")["forecasts"][:]
    return f if f.shape[0] == 99 else f.T  # ERCOT's fleet file is stored as (99, T)


def key_name(key):
    """Site name of a site-level file, letters and digits only.

    Actual and forecast files spell some names differently (e.g. "Indy_Solar_I_LLC"
    vs "Indy_Solar_I,_LLC"); comparing letters and digits pairs them uniquely.
    """
    base = os.path.basename(key)
    name = re.sub(r"_solar_(actuals|1?day-ahead_fcst)_\d{4}\.h5$", "", base)
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iso", required=True, choices=["MISO", "SPP", "ERCOT"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = p.parse_args()

    iso, year = args.iso, YEAR[args.iso]
    raw_dir = os.path.join(args.data_dir, "raw")
    out = os.path.join(args.data_dir, iso)
    os.makedirs(out, exist_ok=True)
    root = f"{iso}/{year}/Solar"

    # ---- fleet level (BA)
    print(f"{iso} {year}: downloading fleet-level files ...")
    ba_act = fetch(f"{root}/Actuals/BA_level/BA_solar_actuals_{year}.h5", raw_dir)
    ba_fc = [k for k in list_keys(f"{root}/Forecasts/Day-ahead/BA_level/")
             if k.endswith(".h5") and "Existing" not in k]
    ba_fc = fetch(ba_fc[0], raw_dir)
    fleet_actuals = hourly_actuals(ba_act)
    capacity = fleet_capacity(ba_act)
    np.save(os.path.join(out, "fleet_actuals.npy"), fleet_actuals)
    np.save(os.path.join(out, "fleet_quantiles_nrel.npy"), forecasts(ba_fc))
    np.save(os.path.join(out, "fleet_capacity.npy"), np.array([capacity]))

    # ---- site level, in the paper's site order
    order = [line.strip() for line in open(os.path.join(SITES, f"{iso}.txt")) if line.strip()]
    act_keys = {key_name(k): k for k in list_keys(f"{root}/Actuals/Site_level/") if k.endswith(".h5")}
    fc_keys = {key_name(k): k for k in list_keys(f"{root}/Forecasts/Day-ahead/Site_level/") if k.endswith(".h5")}
    names = [key_name(o + ".h5") for o in order]
    missing = [n for n in names if n not in act_keys or n not in fc_keys]
    if missing:
        raise RuntimeError(f"sites not found in the bucket: {missing[:5]} ...")
    print(f"downloading {len(names)} sites (actuals + forecasts) ...")
    act_files = fetch_all([act_keys[n] for n in names], raw_dir, args.workers)
    fc_files = fetch_all([fc_keys[n] for n in names], raw_dir, args.workers)

    # site_quantiles.npy is written one site at a time (5 GB for MISO)
    T = len(fleet_actuals)
    actuals = np.empty((len(names), T))
    with open(os.path.join(out, "site_quantiles.npy"), "wb") as f:
        np.lib.format.write_array_header_1_0(
            f, {"descr": "<f8", "fortran_order": False, "shape": (len(names), 99, T)})
        for i, (fa, ff) in enumerate(zip(act_files, fc_files)):
            actuals[i] = hourly_actuals(fa)
            f.write(np.ascontiguousarray(forecasts(ff), dtype="<f8").tobytes())
    np.save(os.path.join(out, "site_actuals.npy"), actuals)
    print(f"saved {out}/  ({len(names)} sites, {T} hours, fleet capacity {capacity:.1f} MW)")


if __name__ == "__main__":
    main()
