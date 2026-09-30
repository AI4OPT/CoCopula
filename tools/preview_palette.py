#!/usr/bin/env python3
"""Render the figure palette as normal vision sees it and as a deuteranope does.

Two panels of swatches plus a strip of overlapping lines in the style the
figures actually use, so the separation can be judged on the marks rather than
on the hex values.

    python tools/preview_palette.py --out /path/palette_check.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.step4_figures import MARKERS, ORDER, PALETTE

# Machado, Oliveira & Fernandes (2009), severity 1.0, applied in linear RGB.
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
}


def _s2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(hex_color: str, kind: str | None) -> tuple:
    rgb = np.array([int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    if kind is None:
        return tuple(rgb)
    lin = _s2lin(rgb)
    out = np.clip(np.array(MACHADO[kind]) @ lin, 0, 1)
    return tuple(_lin2s(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="palette_check.png")
    ap.add_argument("--kinds", nargs="+", default=[None, "deutan"])
    args = ap.parse_args()

    kinds = [None if k in (None, "none") else k for k in args.kinds]
    labels = {None: "normal vision", "deutan": "deuteranopia", "protan": "protanopia"}

    fig, axes = plt.subplots(2, len(kinds), figsize=(6 * len(kinds), 7),
                             gridspec_kw={"height_ratios": [1, 1.4]})
    axes = np.atleast_2d(axes)
    if len(kinds) == 1:
        axes = axes.reshape(2, 1)

    x = np.linspace(0, 10, 200)
    for col, kind in enumerate(kinds):
        # swatches
        ax = axes[0, col]
        for i, name in enumerate(ORDER):
            ax.barh(len(ORDER) - i, 1, color=simulate(PALETTE[name], kind),
                    height=0.75)
            ax.text(1.05, len(ORDER) - i, name, va="center", fontsize=12)
        ax.set_xlim(0, 2.6)
        ax.set_ylim(0.3, len(ORDER) + 0.8)
        ax.set_title(labels[kind], fontsize=15)
        ax.axis("off")

        # overlapping lines, as the figures draw them
        ax = axes[1, col]
        for i, name in enumerate(ORDER):
            ax.plot(x, np.sin(x + i * 0.45) + i * 0.32,
                    color=simulate(PALETTE[name], kind),
                    marker=MARKERS[name], markevery=25, markersize=8, linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
