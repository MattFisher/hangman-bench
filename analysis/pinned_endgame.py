#!/usr/bin/env python3
r"""Miss rate once the dictionary has pinned the word.

Dominated misses concentrate in the endgame, and most of them happen when the
consistent candidate set has already collapsed to a single word. At that point
the game is no longer a decision under uncertainty: the board plus the excluded
letters uniquely determine a common English word, and the only thing left to do
is name it. The conditional miss rate in that regime is a direct measure of
whether the model can retrieve (or verify) a word from a pattern constraint,
separated from the sequential-decision part of the task.

Reads a per-guess TSV as written by ``analysis/pilot_oracle.py`` (one row per
guess, booleans as 0/1) and writes one row per model:

- ``pinned_guesses``     counted guesses made with exactly ``--max-candidates``
                         or fewer candidates remaining (default 1: the word is
                         determined).
- ``pinned_misses``      those guesses that were wrong. With one candidate
                         every miss is also a dominated miss; with more, a
                         miss can still be a legitimate gamble.
- ``pinned_miss_rate``   pinned_misses / pinned_guesses.
- ``dominated``          all dominated misses for the model.
- ``dominated_pinned``   dominated misses made in the pinned regime.
- ``dominated_pinned_share``  dominated_pinned / dominated: how much of the
                         headline metric is retrieval failure at the endgame.
- ``words_pinned_miss``  distinct words with at least one pinned miss.

Usage:
  uv run analysis/pinned_endgame.py \\
      --input analysis/pilot_per_guess.tsv --out analysis/pinned_endgame.tsv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


@dataclass
class ModelTally:
    scored: int = 0
    dominated: int = 0
    dominated_pinned: int = 0
    pinned_guesses: int = 0
    pinned_misses: int = 0
    words_pinned_miss: set[str] = field(default_factory=set)


def tally(rows: list[dict[str, str]], max_candidates: int) -> dict[str, ModelTally]:
    out: dict[str, ModelTally] = defaultdict(ModelTally)
    for row in rows:
        if _flag(row["invalid"]) or _flag(row["repeat"]):
            continue
        t = out[row["model"]]
        t.scored += 1
        candidates = int(row["candidates_before"])
        pinned = 0 < candidates <= max_candidates
        dominated = _flag(row["dominated_miss"])
        hit = _flag(row["hit"])
        if dominated:
            t.dominated += 1
            if pinned:
                t.dominated_pinned += 1
        if pinned:
            t.pinned_guesses += 1
            if not hit:
                t.pinned_misses += 1
                t.words_pinned_miss.add(row["word"])
    return dict(out)


def write_summary(tallies: dict[str, ModelTally], out: pathlib.Path) -> None:
    fields = [
        "model",
        "scored",
        "pinned_guesses",
        "pinned_misses",
        "pinned_miss_rate",
        "dominated",
        "dominated_pinned",
        "dominated_pinned_share",
        "words_pinned_miss",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for model in sorted(tallies):
            t = tallies[model]
            writer.writerow(
                {
                    "model": model,
                    "scored": t.scored,
                    "pinned_guesses": t.pinned_guesses,
                    "pinned_misses": t.pinned_misses,
                    "pinned_miss_rate": f"{t.pinned_misses / t.pinned_guesses:.4f}"
                    if t.pinned_guesses
                    else "",
                    "dominated": t.dominated,
                    "dominated_pinned": t.dominated_pinned,
                    "dominated_pinned_share": f"{t.dominated_pinned / t.dominated:.4f}"
                    if t.dominated
                    else "",
                    "words_pinned_miss": len(t.words_pinned_miss),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Miss rate once the dictionary has pinned the word."
    )
    parser.add_argument("--input", type=pathlib.Path, default="analysis/pilot_per_guess.tsv")
    parser.add_argument("--out", type=pathlib.Path, default="analysis/pinned_endgame.tsv")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1,
        help="Largest candidate set that counts as pinned (default 1: the word is determined).",
    )
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    tallies = tally(rows, args.max_candidates)
    write_summary(tallies, args.out)

    for model in sorted(tallies):
        t = tallies[model]
        rate = t.pinned_misses / t.pinned_guesses if t.pinned_guesses else float("nan")
        share = t.dominated_pinned / t.dominated if t.dominated else float("nan")
        print(
            f"{model}: pinned miss rate {rate:.2f} ({t.pinned_misses}/{t.pinned_guesses}); "
            f"{share:.0%} of {t.dominated} dominated misses were at the pinned endgame; "
            f"{len(t.words_pinned_miss)} words"
        )


if __name__ == "__main__":
    main()
