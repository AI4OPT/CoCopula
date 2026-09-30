#!/usr/bin/env python3
"""Step 3 -- Table 1: PICP / AIW / Winkler across ISOs and coverage levels.

Writes a tidy CSV, a per-ISO wide CSV, and a LaTeX table in the layout used in
the paper.  Also prints the two aggregate claims from Section 3: the Winkler
reduction of Copula+CACP relative to raw NREL and to uncalibrated Copula.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_cacp import config
from copula_cacp.evaluate.metrics import picp_aiw, winkler_score
from copula_cacp.evaluate.results import load_aligned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--isos", nargs="+", default=config.ISOS)
    p.add_argument("--out_dir", default=config.TABLE_DIR)
    p.add_argument("--allow_missing", action="store_true",
                   help="Skip (source, method) cells that have not been run yet.")
    return p.parse_args()


def collect(isos, allow_missing: bool) -> pd.DataFrame:
    rows = []
    for iso in isos:
        iso = iso.upper()
        y, preds, times = load_aligned(iso, require_all=not allow_missing)
        print(f"[INFO] {iso}: {len(y)} aligned test points "
              f"({times[0]} -> {times[-1]})", flush=True)
        for label, pred in preds.items():
            picp, aiw = picp_aiw(y, pred, config.COVERAGE_LEVELS)
            ws = winkler_score(y, pred, config.COVERAGE_LEVELS)
            for c in config.COVERAGE_LEVELS:
                rows.append({"ISO": iso, "Method": label, "Level": c,
                             "PICP": picp[c], "AIW": aiw[c], "WS": ws[c]})
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame) -> str:
    levels = config.COVERAGE_LEVELS
    order = [lab for _, _, lab in config.PAPER_METHODS]

    head = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{System-Level Probabilistic Forecasting Performance Across ISOs}",
        r"\label{tab:main}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll" + "ccc" * len(levels) + "}",
        r"\toprule",
        "ISO & Method & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{$(1-\alpha) = {int(c * 100)}\%$}}" for c in levels
        ) + r" \\",
        " & & " + " & ".join(
            r"PICP $\uparrow$ & AIW $\downarrow$ & WS $\downarrow$" for _ in levels
        ) + r" \\",
        r"\midrule",
    ]

    body = []
    for iso in df["ISO"].unique():
        sub = df[df["ISO"] == iso]
        methods = [m for m in order if m in set(sub["Method"])]
        # bold the best Winkler score per level within this ISO
        best_ws = {c: sub[sub["Level"] == c]["WS"].min() for c in levels}
        for k, m in enumerate(methods):
            cells = []
            for c in levels:
                r = sub[(sub["Method"] == m) & (sub["Level"] == c)]
                if r.empty:
                    cells += ["--", "--", "--"]
                    continue
                r = r.iloc[0]
                ws = f"{r['WS']:.4f}"
                if np.isclose(r["WS"], best_ws[c]):
                    ws = rf"\textbf{{{ws}}}"
                cells += [f"{r['PICP']:.4f}", f"{r['AIW']:.4f}", ws]
            first = rf"\multirow{{{len(methods)}}}{{*}}{{{iso}}}" if k == 0 else ""
            body.append(f"{first} & {m} & " + " & ".join(cells) + r" \\")
        body.append(r"\midrule")
    if body and body[-1] == r"\midrule":
        body[-1] = r"\bottomrule"

    tail = [r"\end{tabular}}", r"\end{table}"]
    return "\n".join(head + body + tail)


def summarise(df: pd.DataFrame) -> None:
    """The two headline numbers quoted in Section 3."""
    piv = df.pivot_table(index=["ISO", "Level"], columns="Method", values="WS")
    for ref in ("NREL", "Copula"):
        if "Copula+CACP" not in piv.columns or ref not in piv.columns:
            continue
        red = (1 - piv["Copula+CACP"] / piv[ref]) * 100
        print(f"[SUMMARY] Winkler reduction of Copula+CACP vs {ref}: "
              f"{red.mean():.1f}% (mean over ISOs x levels)", flush=True)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = collect(args.isos, args.allow_missing)

    tidy = os.path.join(args.out_dir, "table1_tidy.csv")
    df.to_csv(tidy, index=False)

    wide = df.pivot_table(index=["ISO", "Method"], columns="Level",
                          values=["PICP", "AIW", "WS"])
    wide_path = os.path.join(args.out_dir, "table1_wide.csv")
    wide.to_csv(wide_path)

    tex = os.path.join(args.out_dir, "table1.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(to_latex(df) + "\n")

    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    summarise(df)
    print(f"\n[OK] {tidy}\n[OK] {wide_path}\n[OK] {tex}", flush=True)


if __name__ == "__main__":
    main()
