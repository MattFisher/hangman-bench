#!/usr/bin/env python3
"""Render the dictionary-sensitivity figure from analysis/rescore_grid_summary.tsv.

The same 300 pilot trajectories re-scored under four dictionaries: magnitudes
move (hardest for the best model), but the lines never cross — model ranking
is dictionary-invariant, which is the thesis-A robustness result.

Usage:
  uv run analysis/plot_dictionary_grid.py
"""

from __future__ import annotations

import argparse
import csv
import pathlib

from figstyle import (
    FIGURES_DIR,
    INK_MUTED,
    MODEL_COLORS,
    SURFACE,
    finish,
    new_axes,
    save,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Largest dictionary first; the shipped default second.
DICTIONARIES = [
    ("en_GB_70", "en_GB tier-70\n112k words"),
    ("en_GB_50", "en_GB tier-50\nshipped, 61k"),
    ("en_US_50", "en_US tier-50\n61k words"),
    ("en_GB_50_25pct", "en_GB 25% sample\n15k words"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=str(REPO_ROOT / "analysis" / "rescore_grid_summary.tsv"),
    )
    parser.add_argument("--out", default=str(FIGURES_DIR / "dictionary_grid.png"))
    args = parser.parse_args()

    rates: dict[str, dict[str, float]] = {}
    with open(args.summary) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rates.setdefault(row["model"], {})[row["dictionary"]] = float(row["dominated_rate"])

    fig, ax = new_axes(figsize=(7.6, 4.8))
    xs = range(len(DICTIONARIES))
    for model, (label, color) in MODEL_COLORS.items():
        ys = [rates[model][key] for key, _ in DICTIONARIES]
        ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3, label=label)
        ax.scatter(xs, ys, s=44, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.text(
            len(DICTIONARIES) - 1 + 0.1,
            ys[-1],
            label,
            color=color,
            fontsize=9,
            va="center",
        )

    ax.set_xlim(-0.3, len(DICTIONARIES) - 1 + 1.1)
    ax.set_ylim(0, 0.32)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label in DICTIONARIES])
    ax.set_ylabel("dominated-guess rate (pooled per guess)", color=INK_MUTED, fontsize=9)
    finish(
        ax,
        "Model ranking is invariant to the scoring dictionary",
        "same 300 pilot trajectories re-scored under each dictionary — "
        "magnitudes move, the ordering never crosses",
    )
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor="#52514e")

    save(fig, pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
