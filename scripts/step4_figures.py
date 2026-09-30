#!/usr/bin/env python3
"""Step 4 -- the paper's figures.

  fig2a  coverage-sharpness trade-off across calibration strategies
  fig2b  hourly conditional coverage at the 80% target, with the average
         site-level coverage drawn behind it as bars
  fig4   a representative day-ahead forecast, Copula vs Copula+CACP at 5-95%

Colours are optimised for deuteranopia (see PALETTE); every series also
carries its own marker, so identity never rests on colour alone.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_cacp import config
from copula_cacp.data.loaders import load_dataset
from copula_cacp.evaluate.metrics import coverage_width_curve, level_indices
from copula_cacp.evaluate.results import load_aligned

# --- series styling -------------------------------------------------------
# Deuteranope-optimised palette.  Red-green discrimination is unavailable, so
# the series separate along the blue-yellow axis and by lightness rather than
# by hue alone.  Chosen by maximising the worst pairwise OKLab dE under a
# Machado et al. (2009) deuteranope simulation, subject to the lightness band,
# the chroma floor and a 3:1 contrast minimum: worst pair dE=15.4 across all
# six and 16.6 among the four drawn in fig 2b, against a target of 8 and the
# previous Okabe-Ito set's worst pair of 7.6 (its green/pink collision).
# Check it with tools/preview_palette.py.
PALETTE = {
    # The four fig-2b series take the most separable 4-subset (dE=16.6).
    "Copula+CACP": "#474ac0",   # the proposed method: deep indigo
    "Copula":      "#bd6996",   # rose
    "NREL":        "#a94008",   # brown
    "NREL+CACP":   "#847ffc",   # light periwinkle
    # The CQR pair appears only in fig 2a.
    "Copula+CQR":  "#dd7002",   # orange
    "NREL+CQR":    "#803668",   # plum
}
MARKERS = {
    "Copula+CACP": "o", "Copula": "*", "NREL": "s",
    "NREL+CACP": "v", "NREL+CQR": "P", "Copula+CQR": "D",
}
ORDER = ["Copula+CACP", "Copula", "NREL", "NREL+CACP", "NREL+CQR", "Copula+CQR"]

# Figure 2a shows all six calibration strategies; Figure 2b in the paper shows
# only these four (the CQR curves are not in it).
FIG2B_METHODS = ["Copula+CACP", "Copula", "NREL", "NREL+CACP"]
# ... over the solar day, 5:00-16:00 CST.
FIG2B_HOURS = (5, 16)

GRID = dict(linestyle="--", alpha=0.35, linewidth=0.8)
# Sized for a two-column paper layout, where each panel is reduced to about
# half a page width: axis labels FS, ticks and legend FS-4.
LW, MS, FS = 2.4, 9.0, 22
# Figures 2a and 2b sit side by side in the paper, so they share a canvas size
# and type scale.  The corner differs because the empty region does: 2a fills
# from the bottom-left up, 2b from the top down, so a single placement would
# cover curves in one of them.
FIGSIZE = (8, 8)
LEGEND = dict(fontsize=FS - 4, frameon=True, framealpha=0.9,
              edgecolor="none", ncol=1)
LEGEND_LOC = {"fig2a": "upper left", "fig2b": "lower right", "fig4": "upper left"}


def _style(ax) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FS - 4)


TZ_LABEL = {"Etc/GMT+6": "CST", "America/Chicago": "Chicago"}


def _tz_label(tz: str | None) -> str:
    if not tz:
        return "index"
    return TZ_LABEL.get(tz, tz.split("/")[-1].replace("_", " "))


def _local_hours(times: pd.DatetimeIndex, tz: str | None) -> np.ndarray:
    """Hour of day, optionally converted from UTC to a local zone."""
    if not tz:
        return times.hour.to_numpy()
    idx = times.tz_localize("UTC") if times.tz is None else times
    return idx.tz_convert(tz).hour.to_numpy()


# ---------------------------------------------------------------- figure 2a
def fig2a(iso: str, out_dir: str, allow_missing: bool) -> str:
    y, preds, _ = load_aligned(iso, require_all=not allow_missing)
    levels = np.round(np.arange(0.50, 0.99, 0.02), 2)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label in [m for m in ORDER if m in preds]:
        cov, wid = coverage_width_curve(y, preds[label], levels)
        ax.plot(cov * 100, wid * 100,
                marker=MARKERS[label], markersize=MS, linewidth=LW,
                color=PALETTE[label], label=label, alpha=0.95)

    ax.set_xlabel("Coverage Probability (%)", fontsize=FS)
    ax.set_ylabel("Average Interval Width (%)", fontsize=FS)
    ax.legend(loc=LEGEND_LOC["fig2a"], **LEGEND)
    _style(ax)
    fig.tight_layout()

    out = os.path.join(out_dir, f"fig2a_{iso.lower()}_coverage_sharpness.pdf")
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 2b
def _site_level_coverage(iso: str, hours_wanted, level: float, tz: str | None,
                         site_chunk: int = 32) -> dict[int, float]:
    """Average site-level coverage per hour, pooled over all sites and days.

    The site tensor is several GB, so it is memory-mapped and walked in blocks
    of `site_chunk` sites; only the two quantile rows we need are read.
    """
    actuals, marginals, capacity, _, time_index = load_dataset(
        iso, "sites", None, mmap=True)
    cap = capacity.reshape(-1).astype(float)

    li, ui = level_indices(level)
    hours = _local_hours(time_index, tz)
    in_test = np.asarray(time_index >= pd.Timestamp(config.first_train_end(iso)))

    hour_masks = {h: (hours == h) & in_test for h in hours_wanted}
    covered = {h: 0 for h in hours_wanted}
    total = {h: 0 for h in hours_wanted}

    n_sites = actuals.shape[0]
    for s0 in range(0, n_sites, site_chunk):
        s1 = min(s0 + site_chunk, n_sites)
        c = cap[s0:s1].reshape(-1, 1)
        y = np.asarray(actuals[s0:s1]) / c                  # (n, T)
        lo = np.asarray(marginals[s0:s1, li, :]) / c        # (n, T)
        hi = np.asarray(marginals[s0:s1, ui, :]) / c
        ok = (y >= lo) & (y <= hi)
        day = y > 0                                          # ignore dark sites
        for h, tm in hour_masks.items():
            if not tm.any():
                continue
            d = day[:, tm]
            covered[h] += int((ok[:, tm] & d).sum())
            total[h] += int(d.sum())

    return {int(h): covered[h] / total[h] for h in hours_wanted if total[h] > 0}


def fig2b(iso: str, out_dir: str, level: float, tz: str | None,
          allow_missing: bool, with_sites: bool = True,
          hour_min: int | None = None, hour_max: int | None = None,
          methods: list[str] | None = None) -> str:
    methods = methods or FIG2B_METHODS
    y, preds, times = load_aligned(iso, require_all=not allow_missing)
    preds = {k: v for k, v in preds.items() if k in methods}
    hours = _local_hours(times, tz)
    hour_vals = sorted(set(hours.tolist()))
    if hour_min is not None:
        hour_vals = [h for h in hour_vals if h >= hour_min]
    if hour_max is not None:
        hour_vals = [h for h in hour_vals if h <= hour_max]

    li, ui = level_indices(level)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if with_sites:
        site_cov = _site_level_coverage(iso, hour_vals, level, tz)
        if site_cov:
            ax.bar(list(site_cov.keys()), list(site_cov.values()),
                   width=0.6, color="#6b7280", alpha=0.20, zorder=1,
                   label="Site-level Coverage")

    for label in [m for m in ORDER if m in preds]:
        pred = preds[label]
        cov = []
        for h in hour_vals:
            m = hours == h
            cov.append(float(((y[m] >= pred[m, li]) & (y[m] <= pred[m, ui])).mean()))
        ax.plot(hour_vals, cov, marker=MARKERS[label], markersize=MS,
                linewidth=LW, color=PALETTE[label], label=label, zorder=3)

    ax.axhline(level, linestyle="--", color="black", linewidth=1.5,
               zorder=2, label=f"Target {int(level * 100)}%")
    ax.set_xlabel(f"Hour ({_tz_label(tz)})", fontsize=FS)
    ax.set_ylabel("Coverage Probability", fontsize=FS)
    ax.set_xticks(hour_vals)
    ax.set_xticklabels([f"{h}:00" for h in hour_vals], rotation=45)
    ax.legend(loc=LEGEND_LOC["fig2b"], **LEGEND)
    _style(ax)
    fig.tight_layout()

    out = os.path.join(out_dir, f"fig2b_{iso.lower()}_hourly_coverage.pdf")
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------- figure 4
def fig4(iso: str, out_dir: str, day: str, tz: str | None,
         allow_missing: bool) -> str:
    y, preds, times = load_aligned(iso, require_all=not allow_missing)
    idx = times.tz_localize("UTC").tz_convert(tz) if (tz and times.tz is None) else times

    day_ts = pd.Timestamp(day)
    if tz:
        day_ts = day_ts.tz_localize(tz)
    m = (idx >= day_ts) & (idx < day_ts + pd.Timedelta(days=1))
    if m.sum() == 0:
        raise SystemExit(f"No test points on {day} for {iso}. "
                         f"Test period runs {idx[0]} -> {idx[-1]}.")

    li, ui = 4, 94          # the 5% and 95% quantiles
    x = idx[m]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y[m], color="black", linestyle="--", linewidth=LW, label="Actual")

    for label, color in (("Copula", PALETTE["Copula"]),
                         ("Copula+CACP", PALETTE["Copula+CACP"])):
        if label not in preds:
            continue
        p = preds[label]
        ax.fill_between(x, p[m, li], p[m, ui], color=color, alpha=0.28,
                        linewidth=0, label=f"{label} (5%-95%)")
        ax.plot(x, p[m, li], color=color, linewidth=1.0, alpha=0.9)
        ax.plot(x, p[m, ui], color=color, linewidth=1.0, alpha=0.9)

    ax.set_xlabel(f"Time ({_tz_label(tz)})", fontsize=FS)
    ax.set_ylabel("Power Generation (p.u.)", fontsize=FS)
    ax.legend(loc=LEGEND_LOC["fig4"], **LEGEND)
    fig.autofmt_xdate(rotation=45)
    _style(ax)
    fig.tight_layout()

    out = os.path.join(out_dir, f"fig4_{iso.lower()}_{day}_day_ahead.pdf")
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iso", default="MISO")
    p.add_argument("--figures", nargs="+", default=["2a", "2b", "4"],
                   choices=["2a", "2b", "4"])
    p.add_argument("--level", type=float, default=0.8, help="Target level for fig 2b.")
    p.add_argument("--day", default=None,
                   help="Day for fig 4 (YYYY-MM-DD); defaults to the first test day.")
    # The published figures label the axis "CST" and use a fixed UTC-6 shift,
    # not a DST-aware zone -- America/Chicago moves the summer hours by one.
    p.add_argument("--tz", default="Etc/GMT+6",
                   help="Zone for hour axes ('Etc/GMT+6' = fixed CST, as in the "
                        "paper; 'America/Chicago' for DST-aware; '' for raw).")
    p.add_argument("--hour_min", type=int, default=FIG2B_HOURS[0],
                   help="First hour shown in fig 2b.")
    p.add_argument("--hour_max", type=int, default=FIG2B_HOURS[1],
                   help="Last hour shown in fig 2b.")
    p.add_argument("--fig2b_methods", nargs="+", default=FIG2B_METHODS,
                   choices=ORDER, help="Curves drawn in fig 2b.")
    p.add_argument("--no_site_bars", action="store_true")
    p.add_argument("--out_dir", default=config.FIGURE_DIR)
    p.add_argument("--allow_missing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iso = args.iso.upper()
    tz = args.tz or None
    os.makedirs(args.out_dir, exist_ok=True)

    if "2a" in args.figures:
        print("[OK]", fig2a(iso, args.out_dir, args.allow_missing), flush=True)
    if "2b" in args.figures:
        print("[OK]", fig2b(iso, args.out_dir, args.level, tz,
                            args.allow_missing, not args.no_site_bars,
                            args.hour_min, args.hour_max,
                            args.fig2b_methods), flush=True)
    if "4" in args.figures:
        day = args.day or f"{config.year(iso)}-{config.TEST_START_MONTH:02d}-01"
        print("[OK]", fig4(iso, args.out_dir, day, tz, args.allow_missing), flush=True)


if __name__ == "__main__":
    main()
