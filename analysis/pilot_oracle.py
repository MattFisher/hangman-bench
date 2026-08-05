#!/usr/bin/env python3
"""
Pilot: score Hangman trajectories against an oracle instead of only win/loss.

The benchmark's headline metric is a win rate, and with a generous wrong-guess
budget that metric saturates (gpt-5-nano reaches 0.93). This script measures
*how* a game was played: how often the agent made a provably wrong move, and
how far its guesses fell below the best available move at that point.

Two modes:

  from-logs   Read Inspect .eval logs, recover each game's guess sequence from
              the recorded hangman_guess tool calls, and score it.

  simulate    Run reference agents of known quality through the same scorer.
              Use this to calibrate what the metrics look like for optimal,
              mediocre, and sloppy play before spending money on real models.

Usage:
  uv run analysis/pilot_oracle.py from-logs \\
      --logs logs/ \\
      --wordlist analysis/wordlist.txt \\
      --out analysis/pilot

  uv run analysis/pilot_oracle.py simulate \\
      --wordlist analysis/wordlist.txt \\
      --out analysis/pilot_sim
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics
import sys
import zlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle import (  # noqa: E402
    ALPHABET,
    CHOOSERS,
    TrajectoryReport,
    by_length,
    choose_max_hit_probability,
    consistent_candidates,
    hit_probabilities,
    load_wordlist,
    replay_trajectory,
    reveal,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# English letter frequency order, the strategy a casual player uses.
FREQUENCY_ORDER = "etaoinshrdlcumwfgypbvkjxqz"


# --------------------------------------------------------------------------
# Recovering trajectories from Inspect logs
# --------------------------------------------------------------------------


def extract_trajectories(
    log_path: pathlib.Path,
) -> List[Tuple[str, str, str, List[str]]]:
    """Recover (model, sample_id, word, guesses) from one .eval log.

    The guess sequence comes from hangman_guess tool calls, which record what
    the agent actually submitted. It must not be taken from the scorer's
    ``guessed_letters``, which drops repeats and malformed guesses.

    The eval also records raw submissions in the score's ``attempts``. That is
    used as a fallback for logs stored without full message history; tool calls
    stay primary because they are present in every log, including those written
    before ``attempts`` existed.
    """
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    model = log.eval.model or "unknown"
    out: List[Tuple[str, str, str, List[str]]] = []

    for sample in log.samples or []:
        metadata = sample.metadata or {}
        word = metadata.get("word") or str(sample.target or "")
        if isinstance(word, list):
            word = word[0] if word else ""
        if not word:
            continue

        guesses: List[str] = []
        for message in sample.messages or []:
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = getattr(tool_call, "function", None)
                if name != "hangman_guess":
                    continue
                arguments = getattr(tool_call, "arguments", None) or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        continue
                letter = arguments.get("letter")
                if letter is not None:
                    guesses.append(str(letter))

        if not guesses:
            guesses = _attempts_from_scores(sample)

        out.append((model, str(sample.id), str(word).lower(), guesses))

    return out


def _attempts_from_scores(sample: object) -> List[str]:
    """Raw submissions recorded by the scorer, for logs without message history."""
    scores = getattr(sample, "scores", None) or {}
    for score in scores.values():
        attempts = (getattr(score, "metadata", None) or {}).get("attempts")
        if attempts:
            return [str(a) for a in attempts]
    return []


def find_logs(path: pathlib.Path) -> List[pathlib.Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.eval"))


# --------------------------------------------------------------------------
# Reference agents, for calibration
# --------------------------------------------------------------------------

Agent = Callable[[str, Sequence[str], int], List[str]]


def _play(
    word: str,
    dictionary: Sequence[str],
    max_wrong: int,
    pick: Callable[[List[str], List[str], List[str]], Optional[str]],
) -> List[str]:
    """Drive an agent to completion, returning the guess sequence it emitted."""
    guessed: List[str] = []
    emitted: List[str] = []
    wrong = 0
    # A repeat is ignored by the game, so cap emissions to avoid spinning
    # forever on an agent that only ever repeats itself.
    budget = (max_wrong + len(ALPHABET)) * 2

    while wrong < max_wrong and not set(word) <= set(guessed):
        if len(emitted) >= budget:
            break
        board = reveal(word, guessed)
        candidates = consistent_candidates(board, guessed, dictionary)
        letter = pick(board, guessed, candidates)
        if letter is None:
            break
        emitted.append(letter)
        if letter in guessed:
            # The tool ignores repeats and the agent plays on; mirror that
            # rather than ending the game.
            continue
        guessed.append(letter)
        if letter not in word:
            wrong += 1
    return emitted


def agent_optimal(word: str, dictionary: Sequence[str], max_wrong: int) -> List[str]:
    """Plays the max-hit-probability move every turn. Regret zero by design."""

    def pick(board: str, guessed: List[str], candidates: List[str]) -> Optional[str]:
        return choose_max_hit_probability(candidates, frozenset(guessed))

    return _play(word, dictionary, max_wrong, pick)


def agent_frequency(word: str, dictionary: Sequence[str], max_wrong: int) -> List[str]:
    """Fixed English letter-frequency order, ignoring evidence entirely.

    This is the strategy the eval's own system prompt suggests ("common letter
    frequencies"), so it is the relevant baseline for prompt-following play.
    """

    def pick(board: str, guessed: List[str], candidates: List[str]) -> Optional[str]:
        for letter in FREQUENCY_ORDER:
            if letter not in guessed:
                return letter
        return None

    return _play(word, dictionary, max_wrong, pick)


def agent_sloppy(word: str, dictionary: Sequence[str], max_wrong: int) -> List[str]:
    """Mostly frequency order, but sometimes repeats or picks a dead letter.

    Exists to prove the error metrics fire. If the harness cannot detect this
    agent, it cannot detect a real model doing the same thing.
    """
    # Seed from a stable hash: Python randomises str hashes per process, so
    # hash(word) would give different results on every run.
    rng = random.Random(zlib.crc32(word.encode("utf-8")))

    def pick(board: str, guessed: List[str], candidates: List[str]) -> Optional[str]:
        roll = rng.random()
        if roll < 0.15 and guessed:
            return rng.choice(guessed)  # repeat
        if roll < 0.30 and candidates:
            probs = hit_probabilities(candidates, frozenset(guessed))
            dead = [
                letter
                for letter in ALPHABET
                if letter not in guessed and probs.get(letter, 0.0) == 0.0
            ]
            if dead:
                return rng.choice(dead)  # provably dead letter
        for letter in FREQUENCY_ORDER:
            if letter not in guessed:
                return letter
        return None

    return _play(word, dictionary, max_wrong, pick)


AGENTS: Dict[str, Agent] = {
    "optimal": agent_optimal,
    "frequency": agent_frequency,
    "sloppy": agent_sloppy,
}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def write_per_guess(reports: Sequence[TrajectoryReport], path: pathlib.Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "model",
                "sample_id",
                "word",
                "step",
                "letter",
                "board_before",
                "candidates_before",
                "wrong_before",
                "invalid",
                "repeat",
                "dominated_miss",
                "hit",
                "hit_prob",
                "best_hit_prob",
                "hit_prob_regret",
                "optimal_letter",
                "is_optimal",
            ]
        )
        for report in reports:
            for step in report.steps:
                writer.writerow(
                    [
                        report.model,
                        report.sample_id,
                        report.word,
                        step.step,
                        step.letter,
                        step.board_before,
                        step.candidates_before,
                        step.wrong_before,
                        int(step.invalid),
                        int(step.repeat),
                        int(step.dominated_miss),
                        int(step.hit),
                        f"{step.hit_prob:.4f}",
                        f"{step.best_hit_prob:.4f}",
                        f"{step.hit_prob_regret:.4f}",
                        step.optimal_letter or "",
                        int(step.is_optimal),
                    ]
                )


def summarise(reports: Sequence[TrajectoryReport]) -> List[Dict[str, object]]:
    """Aggregate per model. Rates are per scored guess, not per game."""
    by_model: Dict[str, List[TrajectoryReport]] = {}
    for report in reports:
        by_model.setdefault(report.model, []).append(report)

    rows: List[Dict[str, object]] = []
    for model, group in sorted(by_model.items()):
        scored = sum(r.n_scored for r in group)
        emitted = sum(len(r.steps) for r in group)
        rows.append(
            {
                "model": model,
                "games": len(group),
                "win_rate": statistics.mean(1.0 if r.won else 0.0 for r in group),
                "guesses_emitted": emitted,
                "guesses_scored": scored,
                "repeat_rate": (sum(r.n_repeat for r in group) / emitted)
                if emitted
                else 0.0,
                "invalid_rate": (sum(r.n_invalid for r in group) / emitted)
                if emitted
                else 0.0,
                "dominated_rate": (sum(r.n_dominated for r in group) / scored)
                if scored
                else 0.0,
                "suboptimal_rate": (sum(r.n_suboptimal for r in group) / scored)
                if scored
                else 0.0,
                "mean_hit_prob_regret": statistics.mean(
                    r.mean_hit_prob_regret for r in group
                ),
                "mean_wrong_guesses": statistics.mean(r.wrong_guesses for r in group),
                "mean_oracle_wrong": statistics.mean(
                    r.oracle_wrong_guesses for r in group
                ),
                "mean_wrong_regret": statistics.mean(
                    r.wrong_guess_regret for r in group
                ),
            }
        )
    return rows


def write_summary(rows: Sequence[Dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
            )


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        print("No trajectories found.")
        return
    header = (
        f"{'model':<24} {'games':>5} {'win':>6} {'repeat':>7} {'domin':>7} "
        f"{'subopt':>7} {'regret':>7} {'wrong':>6} {'oracle':>7} {'excess':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['model']):<24} "
            f"{row['games']:>5} "
            f"{row['win_rate']:>6.2f} "
            f"{row['repeat_rate']:>7.3f} "
            f"{row['dominated_rate']:>7.3f} "
            f"{row['suboptimal_rate']:>7.3f} "
            f"{row['mean_hit_prob_regret']:>7.3f} "
            f"{row['mean_wrong_guesses']:>6.2f} "
            f"{row['mean_oracle_wrong']:>7.2f} "
            f"{row['mean_wrong_regret']:>7.2f}"
        )
    print()
    print(
        "repeat/domin/subopt are rates per guess; regret is mean shortfall in "
        "hit probability;\nexcess is wrong guesses above the oracle on the same words."
    )


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def load_dataset_words() -> List[Tuple[str, str]]:
    """(word, difficulty) from the benchmark's own dataset."""
    import importlib.util

    path = REPO_ROOT / "src" / "hangman_bench" / "datasets.py"
    spec = importlib.util.spec_from_file_location("hangman_datasets", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [(entry.word.lower(), entry.difficulty) for entry in module.ENGLISH_WORDS]


def restrict_dictionary(
    dictionary: Sequence[str], words: Iterable[str]
) -> Dict[int, List[str]]:
    """Index the dictionary by length; replay only ever needs one length."""
    index = by_length(dictionary)
    for word in words:
        index.setdefault(len(word), [])
    return index


def run_from_logs(args: argparse.Namespace) -> int:
    dictionary = load_wordlist(pathlib.Path(args.wordlist))
    chooser = CHOOSERS[args.strategy]
    logs = find_logs(pathlib.Path(args.logs))
    if not logs:
        print(f"No .eval logs found under {args.logs}", file=sys.stderr)
        return 1

    index = by_length(dictionary)
    reports: List[TrajectoryReport] = []
    for log_path in logs:
        for model, sample_id, word, guesses in extract_trajectories(log_path):
            reports.append(
                replay_trajectory(
                    word=word,
                    raw_guesses=guesses,
                    dictionary=index.get(len(word), []),
                    chooser=chooser,
                    max_wrong=args.max_guesses,
                    sample_id=sample_id,
                    model=model,
                )
            )

    emit(reports, pathlib.Path(args.out))
    return 0


def run_simulate(args: argparse.Namespace) -> int:
    dictionary = load_wordlist(pathlib.Path(args.wordlist))
    chooser = CHOOSERS[args.strategy]
    words = [word for word, _ in load_dataset_words()]
    if args.limit:
        words = words[: args.limit]
    index = by_length(dictionary)

    reports: List[TrajectoryReport] = []
    for name, agent in AGENTS.items():
        for word in words:
            pool = index.get(len(word), [])
            # The agent needs a dictionary it can actually win with, but the
            # report is scored against the real one so a missing target still
            # shows up as a data-quality warning.
            playable = pool if word in pool else pool + [word]
            guesses = agent(word, playable, args.max_guesses)
            reports.append(
                replay_trajectory(
                    word=word,
                    raw_guesses=guesses,
                    dictionary=pool,
                    chooser=chooser,
                    max_wrong=args.max_guesses,
                    sample_id=word,
                    model=name,
                )
            )

    emit(reports, pathlib.Path(args.out))
    return 0


def emit(reports: Sequence[TrajectoryReport], out_prefix: pathlib.Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_guess = out_prefix.with_name(out_prefix.name + "_per_guess.tsv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.tsv")

    write_per_guess(reports, per_guess)
    rows = summarise(reports)
    write_summary(rows, summary_path)
    print_summary(rows)

    missing = [r for r in reports if not r.target_in_dictionary]
    if missing:
        print()
        print(
            f"Note: {len(missing)}/{len(reports)} target words were absent from the "
            f"dictionary and were injected so the oracle stays defined. "
            f"Examples: {', '.join(sorted({r.word for r in missing})[:8])}"
        )
    print()
    print(f"Wrote {per_guess}")
    print(f"Wrote {summary_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--wordlist",
            default=str(REPO_ROOT / "analysis" / "wordlist.txt"),
            help="Dictionary defining the oracle's belief state.",
        )
        p.add_argument(
            "--strategy",
            choices=sorted(CHOOSERS),
            default="max_hit_prob",
            help="Reference policy the agent is scored against.",
        )
        p.add_argument("--max-guesses", type=int, default=10)
        p.add_argument("--out", default=str(REPO_ROOT / "analysis" / "pilot"))

    p_logs = sub.add_parser("from-logs", help="Score Inspect .eval logs.")
    p_logs.add_argument("--logs", required=True, help="A .eval file or a directory.")
    add_common(p_logs)
    p_logs.set_defaults(func=run_from_logs)

    p_sim = sub.add_parser("simulate", help="Score reference agents.")
    p_sim.add_argument("--limit", type=int, default=0)
    add_common(p_sim)
    p_sim.set_defaults(func=run_simulate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
