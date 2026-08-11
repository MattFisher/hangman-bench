#!/usr/bin/env python3
"""Paired comparison of experiment arms that played the same words.

Each arm is a directory of Inspect .eval logs (one or more models). Every
model plays the same word list in every arm, so comparisons are paired by
word: exact McNemar on win/loss, an exact sign test on per-word dominated
counts, and paired mean deltas. Statistics are exact binomial, stdlib only.

Usage:
  uv run analysis/compare_arms.py \\
      frequency=logs/ablation/frequency \\
      neutral=logs/ablation/neutral \\
      belief=logs/ablation/belief \\
      --baseline frequency --out analysis/ablation
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import Counter
from math import comb
from statistics import mean
from typing import Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pilot_oracle import extract_trajectories, find_logs  # noqa: E402

from hangman_bench.oracle import (  # noqa: E402
    TrajectoryReport,
    load_dictionary_index,
    replay_trajectory,
    resolve_wordlist,
)

ArmGames = Dict[str, Dict[str, TrajectoryReport]]  # model -> word -> report


def load_arm(path: pathlib.Path, index) -> ArmGames:
    out: ArmGames = {}
    sources: Dict[tuple, pathlib.Path] = {}
    for log_path in find_logs(path):
        for game in extract_trajectories(log_path):
            key = (game.model, game.word)
            if key in sources:
                raise ValueError(
                    f"Duplicate trajectory for {game.model} on {game.word!r}: "
                    f"{sources[key]} and {log_path}. An arm must contain exactly "
                    f"one run per model; point at a single run's logs."
                )
            sources[key] = log_path
            report = replay_trajectory(
                word=game.word,
                raw_guesses=game.guesses,
                dictionary=index.get(len(game.word), []),
                max_wrong=game.max_guesses,
                sample_id=game.sample_id,
                model=game.model,
            )
            report.recorded_won = game.recorded_won
            out.setdefault(game.model, {})[game.word] = report
    return out


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test at p=0.5."""
    if n == 0:
        return 1.0
    p_k = comb(n, k) / 2**n
    return min(
        1.0,
        sum(comb(n, i) / 2**n for i in range(n + 1) if comb(n, i) / 2**n <= p_k + 1e-12),
    )


def arm_row(arm: str, model: str, reports) -> dict:
    scored = sum(r.n_scored for r in reports)
    emitted = sum(len(r.steps) for r in reports)
    repeat_letters = Counter(s.letter for r in reports for s in r.steps if s.repeat)
    return {
        "arm": arm,
        "model": model,
        "games": len(reports),
        "win_rate": mean(1.0 if r.final_won else 0.0 for r in reports),
        "dominated_rate": sum(r.n_dominated for r in reports) / scored if scored else 0.0,
        "hit_prob_regret": mean(r.mean_hit_prob_regret for r in reports),
        "excess_wrong": mean(r.excess_wrong_guesses for r in reports),
        "repeats": sum(r.n_repeat for r in reports),
        "guesses_emitted": emitted,
        "repeat_letters": " ".join(f"{k}:{v}" for k, v in repeat_letters.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", help="name=path/to/logs pairs")
    parser.add_argument("--baseline", help="Arm name the paired stats compare against.")
    parser.add_argument("--wordlist", default=None)
    parser.add_argument("--out", default=None, help="Prefix for <out>_summary.tsv")
    args = parser.parse_args()

    index = load_dictionary_index(str(resolve_wordlist(args.wordlist)))
    arms: Dict[str, ArmGames] = {}
    for pair in args.arms:
        name, _, path = pair.partition("=")
        if not path:
            parser.error(f"Expected name=path, got {pair!r}")
        arms[name] = load_arm(pathlib.Path(path), index)
    if args.baseline and args.baseline not in arms:
        parser.error(f"--baseline {args.baseline!r} is not one of {sorted(arms)}")

    models = sorted({m for arm in arms.values() for m in arm})
    rows = []
    header = (
        f"{'model':<26} {'arm':<18} {'n':>3} {'win':>5} {'domin':>6} "
        f"{'regret':>7} {'excess':>7} {'repeats':>8}"
    )
    print(header)
    print("-" * len(header))
    for model in models:
        for arm_name, arm in arms.items():
            games = arm.get(model)
            if not games:
                continue
            row = arm_row(arm_name, model, list(games.values()))
            rows.append(row)
            print(
                f"{model.split('/')[-1]:<26} {arm_name:<18} {row['games']:>3} "
                f"{row['win_rate']:>5.2f} {row['dominated_rate']:>6.3f} "
                f"{row['hit_prob_regret']:>7.3f} {row['excess_wrong']:>7.2f} "
                f"{row['repeats']:>3}/{row['guesses_emitted']:<4}"
                + (f" [{row['repeat_letters']}]" if row["repeats"] else "")
            )
        print()

    if args.baseline:
        print(f"== Paired vs {args.baseline} (exact McNemar / sign test, by word) ==")
        for model in models:
            base = arms[args.baseline].get(model)
            if not base:
                continue
            for arm_name, arm in arms.items():
                if arm_name == args.baseline:
                    continue
                comp = arm.get(model)
                if not comp:
                    continue
                words = sorted(set(base) & set(comp))
                unpaired = (set(base) | set(comp)) - set(words)
                if unpaired:
                    # An incomplete arm drops words from the paired population,
                    # and missingness likely correlates with difficulty. Say so
                    # rather than silently shrinking the test.
                    print(
                        f"  note: {len(unpaired)} unpaired words dropped from "
                        f"{args.baseline} vs {arm_name} for {model}: "
                        f"{', '.join(sorted(unpaired)[:5])}" + ("…" if len(unpaired) > 5 else "")
                    )
                b_only = sum(1 for w in words if base[w].final_won and not comp[w].final_won)
                c_only = sum(1 for w in words if comp[w].final_won and not base[w].final_won)
                p_win = binom_two_sided(min(b_only, c_only), b_only + c_only)
                b_more = sum(1 for w in words if base[w].n_dominated > comp[w].n_dominated)
                c_more = sum(1 for w in words if comp[w].n_dominated > base[w].n_dominated)
                p_dom = binom_two_sided(min(b_more, c_more), b_more + c_more)
                delta = mean(comp[w].n_dominated - base[w].n_dominated for w in words)
                print(
                    f"{model.split('/')[-1]:<26} vs {arm_name:<18} "
                    f"wins {b_only}:{c_only} (p={p_win:.3f}) | "
                    f"dominated/word {delta:+.2f}, "
                    f"{args.baseline}-more:{b_more} {arm_name}-more:{c_more} "
                    f"(p={p_dom:.3f})"
                )

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        path = out.with_name(out.name + "_summary.tsv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
                )
        print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
