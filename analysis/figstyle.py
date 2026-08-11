"""Shared style for analysis figures.

The conventions (documented for future agents in CLAUDE.md, "Plotting
results"): figures regenerate from committed TSVs alone, land in
analysis/figures/, and every figure draws from this module so the same model
keeps the same hue everywhere. The categorical hues are the first three slots
of a CVD-validated palette (all-pairs, light surface); reference policies
wear neutral ink with distinct line styles so identity never rides on colour
alone.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt

FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Fixed model-to-hue assignment, used by every figure. Colour follows the
# entity: never reassign these, and take new series from the next slot of the
# validated palette (slot 4 is #eda100) rather than inventing a hue.
MODEL_COLORS: dict[str, tuple[str, str]] = {
    "anthropic/claude-sonnet-5": ("claude-sonnet-5", "#2a78d6"),
    "openai/gpt-5-nano": ("gpt-5-nano", "#eb6834"),
    "openai/gpt-4o": ("gpt-4o", "#1baf7a"),
}

# Reference policies: label, ink colour, dash pattern.
REFERENCE_STYLES: dict[str, tuple[str, str, tuple]] = {
    "oracle(greedy)": ("greedy oracle (uniform prior)", INK_SECONDARY, (0, (4, 2))),
    "frequency(etaoin)": ("frequency player (etaoin…)", INK_MUTED, (0, (1, 2))),
}


def new_axes(figsize: tuple[float, float] = (8.4, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def finish(ax, title: str, subtitle: str | None = None) -> None:
    """Recessive chrome: hairline grid, muted ticks, no top/right spines."""
    ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=24)
    if subtitle:
        ax.text(
            0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            color=INK_MUTED,
            fontsize=8.5,
            va="bottom",
        )
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def save(fig, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")
