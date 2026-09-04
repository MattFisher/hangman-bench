#!/usr/bin/env python3
r"""Mine game transcripts for the candidate words a model verbalises.

Models sometimes list the words they think remain ("Possible words: gawky,
happy, ...") in the text around a tool call. Those lists are a free reading
of the model's belief state, and the oracle knows the true candidate set at
the same moment, so each list can be scored for precision (how many named
words were actually consistent) and, when the true set is small, recall.

This is the zero-cost check the plan asks for before a verbalised-posterior
probe is built (RESEARCH_NOTES.md, section 7, stage 1). It needs the raw
Inspect logs, which are not committed; point ``--logs`` at them.

A mention is any alphabetic token of the target's length that is in the
reference dictionary. That over-counts (a model discussing "letter" mid-game
on a six-letter word registers a mention) and under-counts words the
dictionary lacks; both errors are visible in the per-step output.

Usage:
  uv run analysis/mine_verbalised.py --logs logs/pilot --out analysis/verbalised
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pilot_oracle import find_logs

from hangman_bench.oracle import (
    consistent_candidates,
    load_dictionary_index,
    resolve_wordlist,
    reveal,
)

RECALL_MAX_CANDIDATES = 5


@dataclass
class StepMentions:
    model: str
    word: str
    step: int
    board: str
    n_candidates: int
    mentions: list[str]
    consistent: list[str]
    target_mentioned: bool
    next_guess: str | None
    next_dominated: bool
    # Fraction of the true candidates named, only when the true set is small
    # enough that naming them all is a fair ask.
    recall: float | None

    @property
    def precision(self) -> float | None:
        return len(self.consistent) / len(self.mentions) if self.mentions else None


def recall_of(mentions: Sequence[str], candidates: Sequence[str]) -> float | None:
    if not (0 < len(candidates) <= RECALL_MAX_CANDIDATES):
        return None
    return len(set(candidates) & set(mentions)) / len(candidates)


def mine_text(text: str, length: int, dictionary: set[str]) -> list[str]:
    """Distinct dictionary words of ``length`` mentioned in ``text``, in order."""
    seen: list[str] = []
    for token in re.findall(r"[^\W\d_]+", text.lower()):
        if len(token) == length and token in dictionary and token not in seen:
            seen.append(token)
    return seen


@dataclass
class _Walk:
    guessed: list[str] = field(default_factory=list)
    step: int = 0


def _message_text(message: object) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else ""


def _tool_letters(message: object) -> list[str]:
    letters: list[str] = []
    calls: list[Any] = list(getattr(message, "tool_calls", None) or [])
    for call in calls:
        if getattr(call, "function", None) != "hangman_guess":
            continue
        arguments: Any = getattr(call, "arguments", None) or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        letter: Any = cast(dict[str, Any], arguments).get("letter")
        if letter is not None:
            letters.append(str(letter).strip().lower())
    return letters


def mine_log(log_path: pathlib.Path, index: dict[int, list[str]]) -> list[StepMentions]:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    model = log.eval.model or "unknown"
    out: list[StepMentions] = []
    for sample in log.samples or []:
        metadata = sample.metadata or {}
        word = str(metadata.get("word") or "").lower()
        if not word:
            continue
        dictionary = index.get(len(word), [])
        if word not in dictionary:
            dictionary = [*dictionary, word]
        dict_set = set(dictionary)
        walk = _Walk()
        for message in sample.messages or []:
            if getattr(message, "role", None) != "assistant":
                continue
            board = reveal(word, walk.guessed)
            candidates = consistent_candidates(board, walk.guessed, dictionary)
            mentions = mine_text(_message_text(message), len(word), dict_set)
            letters = _tool_letters(message)
            next_guess = letters[0] if letters else None
            if mentions:
                cand_set = set(candidates)
                dominated = bool(
                    next_guess
                    and len(next_guess) == 1
                    and next_guess not in walk.guessed
                    and candidates
                    and not any(next_guess in c for c in candidates)
                )
                out.append(
                    StepMentions(
                        model=model,
                        word=word,
                        step=walk.step,
                        board=board,
                        n_candidates=len(candidates),
                        mentions=mentions,
                        consistent=[m for m in mentions if m in cand_set],
                        target_mentioned=word in mentions,
                        next_guess=next_guess,
                        next_dominated=dominated,
                        recall=recall_of(mentions, candidates),
                    )
                )
            for letter in letters:
                if len(letter) == 1 and letter.isalpha() and letter not in walk.guessed:
                    walk.guessed.append(letter)
                walk.step += 1
    return out


def write_steps(steps: Sequence[StepMentions], path: pathlib.Path) -> None:
    fields = [
        "model",
        "word",
        "step",
        "board",
        "n_candidates",
        "n_mentioned",
        "n_consistent",
        "precision",
        "recall",
        "target_mentioned",
        "next_guess",
        "next_dominated",
        "mentions",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for s in steps:
            writer.writerow(
                {
                    "model": s.model,
                    "word": s.word,
                    "step": s.step,
                    "board": s.board,
                    "n_candidates": s.n_candidates,
                    "n_mentioned": len(s.mentions),
                    "n_consistent": len(s.consistent),
                    "precision": "" if s.precision is None else f"{s.precision:.3f}",
                    "recall": "" if s.recall is None else f"{s.recall:.3f}",
                    "target_mentioned": int(s.target_mentioned),
                    "next_guess": s.next_guess or "",
                    "next_dominated": int(s.next_dominated),
                    "mentions": " ".join(s.mentions),
                }
            )


def summarise(steps: Sequence[StepMentions]) -> list[dict[str, object]]:
    by_model: dict[str, list[StepMentions]] = defaultdict(list)
    for s in steps:
        by_model[s.model].append(s)
    rows: list[dict[str, object]] = []
    for model in sorted(by_model):
        ss = by_model[model]
        precisions = [s.precision for s in ss if s.precision is not None]
        pinned = [s for s in ss if s.n_candidates == 1]
        rows.append(
            {
                "model": model,
                "steps_with_mentions": len(ss),
                "mean_mentions": round(sum(len(s.mentions) for s in ss) / len(ss), 2),
                "mean_precision": round(sum(precisions) / len(precisions), 3) if precisions else "",
                "pinned_steps_with_mentions": len(pinned),
                "pinned_target_named": sum(s.target_mentioned for s in pinned),
                "named_target_then_dead_guess": sum(
                    1 for s in pinned if s.target_mentioned and s.next_dominated
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine transcripts for verbalised candidates.")
    parser.add_argument("--logs", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default="analysis/verbalised")
    parser.add_argument("--wordlist", default=None)
    args = parser.parse_args()

    index = load_dictionary_index(str(resolve_wordlist(args.wordlist)))
    steps: list[StepMentions] = []
    for log_path in find_logs(args.logs):
        steps.extend(mine_log(log_path, index))

    write_steps(steps, args.out.with_name(args.out.name + "_per_step.tsv"))
    rows = summarise(steps)
    with args.out.with_name(args.out.name + "_summary.tsv").open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
