#!/usr/bin/env python3
r"""Build the pinned-board set for the static probe.

Two sources, written to one TSV that ``hangman_bench/pinned_probe`` reads:

- ``pilot``       the exact states the pilot models faced with one candidate
                  left, reconstructed from ``analysis/pilot_per_guess.tsv``.
                  Each row records which models faced that state and which
                  of them missed there, so the static answer can be paired
                  with the in-game guess on the same board.
- ``dictionary``  fresh boards sampled from the oracle dictionary by playing
                  the greedy reference policy until the word is pinned.
                  Stratified evenly by length and, when wordfreq is
                  installed, by zipf band within length. Seeded on the sorted
                  dictionary contents, so a rebuild that reorders the file
                  does not change the draw. A public seed is a dev set, not a
                  contamination defence: for headline numbers run the task
                  with ``-T seed=None`` so it draws and records a fresh one.

Usage:
  uv run analysis/pinned_boards.py --n 300 --seed 0 --out analysis/pinned_boards.tsv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict
from collections.abc import Callable

from hangman_bench.oracle import load_dictionary_index, resolve_wordlist
from hangman_bench.probe import TSV_FIELDS, PinnedBoard, sample_pinned_boards

# Bands as used in RESEARCH_NOTES.md section 4.
BANDS = ("common", "moderate", "rare")
EXTRA_FIELDS = ["faced_by", "missed_by"]


def zipf_band_function() -> Callable[[str], str] | None:
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None

    def band(word: str) -> str:
        z = zipf_frequency(word, "en")
        if z > 4.5:
            return "common"
        if z < 3.5:
            return "rare"
        return "moderate"

    return band


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


def pilot_states(per_guess: pathlib.Path) -> list[tuple[PinnedBoard, list[str], list[str]]]:
    """Distinct pinned states from the per-guess table, with who faced and missed them.

    Walks each game in step order, tracking the letters that reached the game,
    and captures the state before every counted guess made with exactly one
    candidate remaining. The excluded letters are the guessed letters absent
    from the board: hangman reveals every occurrence of a hit, so a guessed
    letter not on the board is one not in the word.
    """
    with per_guess.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    games: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        games[(row["model"], row["sample_id"])].append(row)

    faced: dict[PinnedBoard, set[str]] = defaultdict(set)
    missed: dict[PinnedBoard, set[str]] = defaultdict(set)
    for (model, _), steps in games.items():
        steps.sort(key=lambda r: int(r["step"]))
        guessed: set[str] = set()
        for row in steps:
            counted = not (_flag(row["invalid"]) or _flag(row["repeat"]))
            if counted and int(row["candidates_before"]) == 1:
                board = row["board_before"]
                revealed = {ch for ch in board if ch != "."}
                state = PinnedBoard(
                    word=row["word"],
                    board=board,
                    excluded="".join(sorted(guessed - revealed)),
                    source="pilot",
                )
                faced[state].add(model)
                if not _flag(row["hit"]):
                    missed[state].add(model)
            if counted:
                guessed.add(row["letter"])
    return [
        (state, sorted(faced[state]), sorted(missed[state]))
        for state in sorted(faced, key=lambda s: (s.word, s.board, s.excluded))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pinned-board set for the probe.")
    parser.add_argument("--per-guess", type=pathlib.Path, default="analysis/pilot_per_guess.tsv")
    parser.add_argument("--out", type=pathlib.Path, default="analysis/pinned_boards.tsv")
    parser.add_argument("--n", type=int, default=300, help="Dictionary boards to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-wrong", type=int, default=0)
    parser.add_argument("--wordlist", default=None)
    parser.add_argument(
        "--band",
        action="store_true",
        default=True,
        help="Stratify dictionary boards by zipf band (needs wordfreq; default on).",
    )
    parser.add_argument("--no-band", dest="band", action="store_false")
    args = parser.parse_args()

    index = load_dictionary_index(str(resolve_wordlist(args.wordlist)))
    band_of = zipf_band_function() if args.band else None
    if args.band and band_of is None:
        print("wordfreq not installed; sampling without frequency bands", file=sys.stderr)

    rows: list[dict[str, str]] = []
    pilot = pilot_states(args.per_guess) if args.per_guess.is_file() else []
    for state, faced_by, missed_by in pilot:
        band = band_of(state.word) if band_of else ""
        rows.append(
            {
                "word": state.word,
                "board": state.board,
                "excluded": state.excluded,
                "source": "pilot",
                "zipf_band": band,
                "faced_by": ",".join(faced_by),
                "missed_by": ",".join(missed_by),
            }
        )

    fresh = sample_pinned_boards(
        index,
        n=args.n,
        seed=args.seed,
        extra_wrong=args.extra_wrong,
        band_of=band_of,
        bands=BANDS if band_of else (),
    )
    for b in fresh:
        rows.append(
            {
                "word": b.word,
                "board": b.board,
                "excluded": b.excluded,
                "source": "dictionary",
                "zipf_band": b.zipf_band,
                "faced_by": "",
                "missed_by": "",
            }
        )

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS + EXTRA_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    n_pilot = len(pilot)
    print(f"{n_pilot} pilot states, {len(fresh)} dictionary boards -> {args.out}")
    if band_of:
        for source in ("pilot", "dictionary"):
            counts: dict[str, int] = defaultdict(int)
            for r in rows:
                if r["source"] == source:
                    counts[r["zipf_band"]] += 1
            print(f"  {source} by band: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
