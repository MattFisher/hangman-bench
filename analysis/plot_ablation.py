#!/usr/bin/env python3
"""Render the prompt-ablation figure from analysis/ablation_summary.tsv.

One line per model across the three prompt variants, all run under the fixed
nudge. Flat lines are the finding: the belief-eliciting prompt does not
collapse dominated_rate, so the metric measures capability, not compliance.
The pilot arm is excluded — it ran under the old nudge, a different condition.

Usage:
  uv run analysis/plot_ablation.py
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
ARMS = ["frequency", "neutral", "belief"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", default=str(REPO_ROOT / "analysis" / "ablation_summary.tsv")
    )
    parser.add_argument("--out", default=str(FIGURES_DIR / "ablation_dominated.png"))
    args = parser.parse_args()

    rates: dict[str, dict[str, float]] = {}
    with open(args.summary) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["arm"] in ARMS:
                rates.setdefault(row["model"], {})[row["arm"]] = float(
                    row["dominated_rate"]
                )

    fig, ax = new_axes(figsize=(7.2, 4.8))
    xs = range(len(ARMS))
    for model, (label, color) in MODEL_COLORS.items():
        ys = [rates[model][arm] for arm in ARMS]
        ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3, label=label)
        ax.scatter(
            xs, ys, s=44, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=4
        )
        ax.text(
            len(ARMS) - 1 + 0.08,
            ys[-1],
            label,
            color=color,
            fontsize=9,
            va="center",
        )

    ax.set_xlim(-0.35, len(ARMS) - 1 + 1.05)
    ax.set_ylim(0, 0.28)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [
            "frequency\n(original prompt)",
            "neutral\n(no advice)",
            "belief\n(reason from candidates)",
        ]
    )
    ax.set_ylabel(
        "dominated-guess rate (pooled per guess)", color=INK_MUTED, fontsize=9
    )
    finish(
        ax,
        "The prompt does not drive the dead guesses",
        "same 100 words per arm, fixed nudge; no pairwise comparison reaches "
        "significance (exact McNemar / sign tests)",
    )
    ax.legend(loc="center left", fontsize=8.5, frameon=False, labelcolor="#52514e")

    save(fig, pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
