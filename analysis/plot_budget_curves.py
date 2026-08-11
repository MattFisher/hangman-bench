#!/usr/bin/env python3
"""Render the win-vs-budget curves figure from the committed TSVs.

Reads analysis/budget_curves.tsv (curves derived from the unlimited run) and
analysis/budget_curves_validation.tsv (actual constrained runs), so the figure
regenerates from committed data without the logs.

Usage:
  uv run analysis/plot_budget_curves.py [--out analysis/figures/budget_curves.png]
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill): first three categorical slots, validated
# all-pairs on the light surface; baselines wear ink, not series hues.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = {
    "anthropic/claude-sonnet-5": ("claude-sonnet-5", "#2a78d6"),
    "openai/gpt-5-nano": ("gpt-5-nano", "#eb6834"),
    "openai/gpt-4o": ("gpt-4o", "#1baf7a"),
}
REFERENCES = {
    "oracle(greedy)": ("greedy oracle (uniform prior)", INK_SECONDARY, (0, (4, 2))),
    "frequency(etaoin)": ("frequency player (etaoin…)", INK_MUTED, (0, (1, 2))),
}


def read_curves(path: pathlib.Path) -> dict[str, list[float]]:
    curves = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            budgets = [k for k in row if k.startswith("b")]
            curves[row["agent"]] = [
                float(row[f"b{b}"]) for b in range(1, len(budgets) + 1)
            ]
    return curves


def read_validation(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves", default=str(REPO_ROOT / "analysis" / "budget_curves.tsv")
    )
    parser.add_argument(
        "--validation",
        default=str(REPO_ROOT / "analysis" / "budget_validation.tsv"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "analysis" / "figures" / "budget_curves.png"),
    )
    args = parser.parse_args()

    curves = read_curves(pathlib.Path(args.curves))
    validation = read_validation(pathlib.Path(args.validation))
    budgets = list(range(1, max(len(v) for v in curves.values()) + 1))

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Reference policies: ink, dashed/dotted, drawn first so models sit on top.
    for agent, (label, color, dashes) in REFERENCES.items():
        if agent in curves:
            ax.plot(
                budgets,
                curves[agent],
                drawstyle="steps-post",
                color=color,
                linewidth=1.6,
                linestyle=dashes,
                label=label,
                zorder=2,
            )

    for agent, (label, color) in SERIES.items():
        if agent in curves:
            ax.plot(
                budgets,
                curves[agent],
                drawstyle="steps-post",
                color=color,
                linewidth=2.0,
                label=label,
                zorder=3,
            )

    # Actual constrained runs: same hue as their model, white surface ring.
    for row in validation:
        model = row["model"]
        if model not in SERIES:
            continue
        _, color = SERIES[model]
        ax.scatter(
            [int(row["budget"])],
            [int(row["actual_wins"]) / int(row["n_words"])],
            s=44,
            color=color,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=4,
        )
    if validation:
        ax.scatter(
            [],
            [],
            s=44,
            color=INK_SECONDARY,
            edgecolors=SURFACE,
            linewidths=1.5,
            label="actual constrained run",
        )

    # The old default cap.
    ax.axvline(10, color=GRID, linewidth=1.2, zorder=1)
    ax.text(
        10.2,
        0.04,
        "old default cap (b=10)",
        color=INK_MUTED,
        fontsize=8.5,
        va="bottom",
    )

    # Direct labels, placed where the curves separate. Short names here; the
    # legend carries the full ones.
    anchors = {
        "oracle(greedy)": (6.8, 0.845, "right", "greedy oracle"),
        "anthropic/claude-sonnet-5": (7.2, 0.945, "right", "claude-sonnet-5"),
        "openai/gpt-5-nano": (10.4, 0.80, "left", "gpt-5-nano"),
        "openai/gpt-4o": (14.6, 0.72, "left", "gpt-4o"),
        "frequency(etaoin)": (17.6, 0.42, "left", "frequency player"),
    }
    for agent, (x, y, ha, label) in anchors.items():
        if agent not in curves:
            continue
        _, color = SERIES.get(agent, (None, None))
        if color is None:
            _, color, _ = REFERENCES[agent]
        ax.text(x, y, label, color=color, fontsize=9, ha=ha)

    ax.set_xlim(1, max(budgets))
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 26])
    ax.set_xlabel("wrong-guess budget b", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("win rate", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Win rate vs wrong-guess budget — 100 words, derived from one unlimited run",
        color=INK,
        fontsize=11.5,
        loc="left",
        pad=24,
    )
    ax.text(
        0,
        1.015,
        "dots: real constrained runs at b=4 and b=10 — models beat their own "
        "derived curve under scarcity",
        transform=ax.transAxes,
        color=INK_MUTED,
        fontsize=8.5,
        va="bottom",
    )

    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    ax.legend(
        loc="lower right",
        fontsize=8.5,
        frameon=False,
        labelcolor=INK_SECONDARY,
        handlelength=2.4,
    )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
