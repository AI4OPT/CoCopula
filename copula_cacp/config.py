"""Central configuration: data locations, experiment calendar, method naming.

Every path used by the pipeline is resolved here so that moving the repo to
another machine only requires editing this file (or setting the environment
variables it reads).
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Input data (NREL PERFORM, pre-processed to .npy)
# ---------------------------------------------------------------------------
# Expected layout under NREL_ROOT:
#   <ISO>/solar/day-ahead/{system,sites}/actuals.npy      (N, T)
#   <ISO>/solar/day-ahead/{system,sites}/marginals.npy    (N, 99, T)
#   <ISO>/solar/day-ahead/{system,sites}/capacity.npy     (N, 1)
#   cov/<iso>_<level>_<covariate>.npy                     (N, k, T)
NREL_ROOT = os.environ.get(
    "NREL_PERFORM_ROOT",
    "/storage/project/r-phentenryck3-0/shared/NREL_PERFORM/processed",
)
COV_ROOT = os.path.join(NREL_ROOT, "cov")

# ---------------------------------------------------------------------------
# Outputs.  Home quota is small, so results default to scratch.
# ---------------------------------------------------------------------------
RESULTS_ROOT = os.environ.get(
    "COPULA_CACP_RESULTS",
    "/storage/scratch1/1/amoradi30/copula_cacp_results",
)
AGG_DIR = os.path.join(RESULTS_ROOT, "aggregated")   # step 1 output
PRED_DIR = os.path.join(RESULTS_ROOT, "preds")       # step 2 output
TABLE_DIR = os.path.join(RESULTS_ROOT, "tables")     # step 3 output
FIGURE_DIR = os.path.join(RESULTS_ROOT, "figures")   # step 4 output
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")

ISOS = ["MISO", "SPP", "ERCOT"]

# The ERCOT export in the source pipeline is stamped 2018; MISO and SPP are
# 2019.  The arrays are all 8760 hourly steps, so only the calendar labels
# differ.  See README ("Known discrepancies").
ISO_YEAR = {"MISO": 2019, "SPP": 2019, "ERCOT": 2018}


def year(iso: str) -> int:
    return ISO_YEAR[iso.upper()]


def time_bounds(iso: str) -> tuple[str, str]:
    """(start, end) timestamps of the hourly index for an ISO."""
    y = year(iso)
    return f"{y}-01-01", f"{y}-12-31 23:00"


# ---------------------------------------------------------------------------
# Experiment calendar (paper, Appendix B.1)
# ---------------------------------------------------------------------------
# "The first six months of observations serve as the initial calibration period
#  and for learning cross-site correlations.  Aggregated forecasts are
#  recalibrated daily, while correlation structures are updated monthly.  The
#  most recent week of data before test day is reserved for validation."
COPULA_TRAIN_MONTHS = 6      # months used for the first correlation fit
TEST_START_MONTH = 7         # first test month (July) => six months of history
TEST_WINDOW_DAYS = 1         # daily recalibration
VAL_DAYS = 7                 # validation week for hyperparameter selection


def first_train_end(iso: str) -> str:
    """Timestamp where calibration stops and the rolling test period begins."""
    return f"{year(iso)}-{TEST_START_MONTH:02d}-01"


def train_start(iso: str) -> str:
    return f"{year(iso)}-01-01"


# ---------------------------------------------------------------------------
# Context vector (paper, Section 2.2)
# ---------------------------------------------------------------------------
# (1) lagged historical generation -> "historical" (24 lags, previous day)
# (2-4) hour / day-of-week / month embeddings -> "time"
# (5) normalized representation of the solar day -> "solarity"
DEFAULT_COVARIATES = ["solarity", "time", "historical"]

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
# A run is identified by (source, method):
#   source "nrel"   -> NREL's own system-level quantile forecast
#   source "copula" -> Gaussian-copula aggregation of the site-level forecasts
#   method "none"   -> uncalibrated pass-through
#   method "cqr"    -> conformalized quantile regression
#   method kernel/knn/kmeans -> CACP weighting variants
SOURCES = ["nrel", "copula"]
METHODS = ["none", "cqr", "kernel", "knn", "kmeans"]

# CACP as reported in the paper uses the RBF-kernel weighting of Eq. (7).
CACP_METHOD = "kernel"

# The six rows of Table 1, in paper order.
PAPER_METHODS = [
    ("nrel", "none", "NREL"),
    ("nrel", "cqr", "NREL+CQR"),
    ("nrel", CACP_METHOD, "NREL+CACP"),
    ("copula", "none", "Copula"),
    ("copula", "cqr", "Copula+CQR"),
    ("copula", CACP_METHOD, "Copula+CACP"),
]

COVERAGE_LEVELS = [0.9, 0.8, 0.7, 0.6]


def run_tag(source: str, method: str) -> str:
    return f"{source}_{method}"


def pred_path(iso: str, source: str, method: str) -> str:
    return os.path.join(PRED_DIR, iso.upper(), f"{run_tag(source, method)}.npy")


def times_path(iso: str, source: str, method: str) -> str:
    return os.path.join(PRED_DIR, iso.upper(), f"{run_tag(source, method)}_times.npy")


def copula_path(iso: str) -> str:
    """Aggregated system-level quantiles produced by step 1."""
    return os.path.join(AGG_DIR, f"{iso.lower()}_solar_copula.npy")


def sigma_path(iso: str) -> str:
    """Per-month dependence matrices produced by step 1 (diagnostic)."""
    return os.path.join(AGG_DIR, f"{iso.lower()}_sigma.npz")
