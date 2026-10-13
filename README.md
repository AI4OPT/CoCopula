<div align="center">

# CoCopula

### Conformal Calibration and Copula Aggregation for Reliable Renewable Energy Forecasting

**Alireza Moradi · Mathieu Tanneau · Reza Zandehshahvar · Pascal Van Hentenryck**

NSF AI Institute for Advances in Optimization (AI4OPT), Georgia Institute of Technology


</div>

---

## Overview

System operators often receive **probabilistic forecasts for individual sites** (solar or wind
farms) but have **no forecasting model for the fleet as a whole**. CoCopula builds a reliable
fleet-level forecast from the site-level forecasts alone, in two stages:

```
 site-level quantile        ┌──────────────────────┐   fleet-level   ┌──────────────────────────┐   calibrated fleet
 forecasts (N sites)  ───►  │ 1. Copula aggregation │ ──► forecast ──► │ 2. Context-aware conformal│ ──► forecast
                            │    (cross-site        │                 │    calibration (CACP)     │   (99 quantiles)
                            │     dependence)       │                 │                           │
                            └──────────────────────┘                  └──────────────────────────┘
```

1. **Aggregation.** A Gaussian copula captures the dependence between the forecast errors of the
   sites. Monte Carlo samples from the copula are mapped through each site's predictive
   distribution and summed, giving a predictive distribution for the total fleet generation.
2. **Calibration.** Context-Aware Conformal Prediction (CACP) corrects the aggregated
   intervals using past forecast errors, weighting each past hour by how similar its context
   (solar position, time of day/year, previous-day generation) is to the hour being forecast.

## Repository structure

The code is a handful of flat, self-contained files:

| File | Content |
|---|---|
| `data.py` | Loading the site-level and fleet-level arrays and the context features |
| `copula.py` | PIT values, Gaussian-copula fit, Monte Carlo aggregation |
| `conformal.py` | CQR and CACP calibration |
| `metrics.py` | PICP, AIW and Winkler score |
| `run_aggregation.py` | **Command:** site forecasts → fleet forecast |
| `run_calibration.py` | **Command:** fleet forecast → calibrated forecast + metrics |

## Installation

```bash
git clone https://github.com/AI4OPT/CoCopula.git
cd CoCopula
pip install -r requirements.txt        # numpy, scipy, pandas, scikit-learn
```

## Data

The experiments use the day-ahead solar forecasts of the
[ARPA-E PERFORM dataset](https://data.openei.org/submissions/5772) (NREL) for MISO and SPP (2019)
and ERCOT (2018). The scripts expect the forecasts as NumPy arrays in the following layout:

```
DATA_DIR/
├── MISO/solar/day-ahead/
│   ├── sites/    actuals.npy (N, T)   marginals.npy (N, 99, T)   capacity.npy (N, 1)
│   └── system/   actuals.npy (1, T)   marginals.npy (1, 99, T)   capacity.npy (1, 1)
├── SPP/...   ERCOT/...
└── cov/
    └── <iso>_system_{solarity,time,historical}.npy        (1, k, T)
```

* `marginals` holds the 1%, 2%, …, 99% quantiles of each hour (T = 8760 hours).
* `sites` are the individual solar sites (751 in MISO, 172 in SPP, 226 in ERCOT);
  `system` is the fleet total and NREL's own fleet-level forecast (used only as a reference).
* `cov` holds the context features of CACP: the normalized solar day (`solarity`), sine/cosine
  encodings of hour, day of year and month (`time`), and the 24 hourly fleet generation values of
  the previous day (`historical`).

## Usage

Each command runs one method on one ISO. Nothing is written inside the repository; all outputs go
to the paths you pass.

### 1. Aggregate the site forecasts

```bash
python run_aggregation.py --data_dir DATA_DIR --iso MISO --dependence copula      --out out/miso_copula.npy
python run_aggregation.py --data_dir DATA_DIR --iso MISO --dependence independent --out out/miso_independent.npy
```

| Option | Meaning |
|---|---|
| `--dependence copula` | Gaussian copula, refitted monthly on all earlier data |
| `--dependence independent` | Sites treated as independent (Σ = I) |
| `--n_samples`, `--seed` | Monte Carlo sample size (default 1000) and seed (default 42) |

Output: a `(99, 8760)` array of fleet-level quantiles.

### 2. Calibrate and evaluate

```bash
python run_calibration.py --data_dir DATA_DIR --iso MISO --forecast out/miso_copula.npy --method cacp --out_dir out/
```

| Option | Meaning |
|---|---|
| `--forecast` | `nrel` (NREL's fleet forecast) or a file produced by `run_aggregation.py` |
| `--method` | `none` (no calibration), `cqr`, or `cacp` |

The command prints PICP, AIW and Winkler score at the 90%, 80%, 70% and 60% levels and saves

```
out/miso_copula_cacp_quantiles.npy   calibrated quantiles of every test hour, (T_test, 99)
out/miso_copula_cacp_times.npy       their time stamps
out/miso_copula_cacp_metrics.csv     PICP / AIW / WS per target coverage
```

### Methods of the paper

| Method in the paper | `--forecast` | `--method` |
|---|---|---|
| NREL † | `nrel` | `none` |
| NREL+CQR † | `nrel` | `cqr` |
| NREL+CACP † | `nrel` | `cacp` |
| Ind | `<iso>_independent.npy` | `none` |
| Copula | `<iso>_copula.npy` | `none` |
| Copula+CQR | `<iso>_copula.npy` | `cqr` |
| **CoCopula** | `<iso>_copula.npy` | `cacp` |

† NREL's own fleet-level forecasts, a reference that is not available in the setting studied.

## Experimental protocol

* **Normalization.** Generation and forecasts are divided by the fleet capacity; hours with zero
  generation (night) are excluded.
* **Aggregation.** Σ is first fitted on January–June (used for January–July) and refitted at the
  start of every later month on all earlier data. PIT values use piecewise-linear CDFs through the
  99 quantiles, clipped to [10⁻⁴, 0.999]; 1000 Monte Carlo samples.
* **Calibration.** July–December is split into consecutive 7-day test windows. For each window,
  the preceding 7 days form the validation set and all earlier hours the calibration set.
  CACP selects its kernel (RBF or Laplacian) and bandwidth γ ∈ {0.5, 1, 2} on the validation week
  by the lowest mean Winkler score; the calibrator is then refitted on calibration + validation
  data and applied to the test window.
* **Metrics.** PICP, AIW and Winkler score are computed within each test window and averaged over
  the windows.

## Citation

If you use this code, please cite:

```bibtex
@article{moradi2026copula,
  title={Copula-based aggregation and context-aware conformal prediction for reliable renewable energy forecasting},
  author={Moradi, Alireza and Tanneau, Mathieu and Zandehshahvar, Reza and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2602.02583},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE).
