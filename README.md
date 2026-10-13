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

```
CoCopula/
├── cocopula/                 the method
│   ├── data.py               loading the arrays and the context features
│   ├── copula.py             PIT values, Gaussian-copula fit, Monte Carlo aggregation
│   ├── conformal.py          CQR and CACP calibration
│   └── metrics.py            PICP, AIW and Winkler score
├── scripts/                  the three commands
│   ├── download_data.py      download the PERFORM data and build the arrays
│   ├── run_aggregation.py    site forecasts -> fleet forecast
│   └── run_calibration.py    fleet forecast -> calibrated forecast + metrics
├── covariates/               context features of CACP (not part of the NREL data)
├── sites/                    site order used in the paper, one list per ISO
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/AI4OPT/CoCopula.git
cd CoCopula
pip install -r requirements.txt        # numpy, scipy, pandas, scikit-learn, h5py
```

## Data

**Source.** NREL / ARPA-E **PERFORM** dataset ([documentation](https://github.com/PERFORM-Forecasts/documentation),
[AWS Open Data](https://registry.opendata.aws/arpa-e-perform/)): day-ahead solar forecasts and actuals for
MISO and SPP (2019) and ERCOT (2018). The data is public; no account is needed.

**Download.** One command per ISO downloads the raw files and builds the arrays used by the code:

```bash
python scripts/download_data.py --iso MISO  --data_dir DATA_DIR   # ~6 GB
python scripts/download_data.py --iso SPP   --data_dir DATA_DIR   # ~1.5 GB
python scripts/download_data.py --iso ERCOT --data_dir DATA_DIR   # ~1.9 GB
```

This writes `DATA_DIR/<ISO>/{site_actuals, site_quantiles, fleet_actuals, fleet_quantiles_nrel, fleet_capacity}.npy`
(raw files are cached in `DATA_DIR/raw/`).

**Included in this repository.** `covariates/` holds the CACP context features (not part of PERFORM) and
`sites/` the site order used in the paper.

## Usage

Each command runs one method on one ISO. Nothing is written inside the repository; all outputs go
to the paths you pass.

### 1. Aggregate the site forecasts

```bash
python scripts/run_aggregation.py --data_dir DATA_DIR --iso MISO --dependence copula      --out out/miso_copula.npy
python scripts/run_aggregation.py --data_dir DATA_DIR --iso MISO --dependence independent --out out/miso_independent.npy
```

| Option | Meaning |
|---|---|
| `--dependence copula` | Gaussian copula, refitted monthly on all earlier data |
| `--dependence independent` | Sites treated as independent (Σ = I) |
| `--n_samples`, `--seed` | Monte Carlo sample size (default 1000) and seed (default 42) |

Output: a `(99, 8760)` array of fleet-level quantiles.

### 2. Calibrate and evaluate

```bash
python scripts/run_calibration.py --data_dir DATA_DIR --iso MISO --forecast out/miso_copula.npy --method cacp --out_dir out/
```

| Option | Meaning |
|---|---|
| `--forecast` | `nrel` (NREL's fleet forecast) or a file produced by `scripts/run_aggregation.py` |
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
