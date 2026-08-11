#!/usr/bin/env python3
r"""Win-vs-budget curves from a single high-budget run, with validation.

A game won with w wrong guesses would have been won at any budget > w, so one
run at a high budget yields the whole win-vs-budget curve below its cap by
reading off the empirical CDF of wrong-guesses-needed. With guesses restricted
to the language's declared alphabet, a budget of |alphabet| cannot be
exhausted (guessing every letter reveals the word first), so a run at that
budget — 26 for English — is effectively unlimited and the distribution is
uncensored except for games that never finish (message limit; reported
separately as DNF).

The derived curve assumes budget-invariant play: the model is told its
budget, so it could in principle play differently under scarcity. --validate
compares the derived prediction at budget b against a real constrained run at
b, paired by word with an exact McNemar test. Agreement validates the cheap
single-run protocol; disagreement is a finding about budget-conditional play.

Reference curves: the greedy oracle (its wrong-guess count per word is exact)
and the fixed-order frequency agent, both computed from the dictionary.

Usage:
  uv run analysis/budget_curves.py --unlimited logs/budget/unlimited \\
      --validate 4=logs/budget/b4 --validate 10=logs/ablation/frequency \\
      --out analysis/budget
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from compare_arms import binom_two_sided, load_arm
from pilot_oracle import agent_frequency

from hangman_bench.oracle import (
    ALPHABET,
    load_dictionary_index,
    replay_trajectory,
    resolve_wordlist,
)

# The "unlimited" budget is the alphabet size: with guesses restricted to the
# language's declared alphabet, at most |alphabet| - len(set(word)) can ever
# be wrong, so a budget of |alphabet| cannot be exhausted. 26 for English.
MAX_BUDGET = len(ALPHABET)


def wrong_needed(games) -> dict[str, int | None]:
    """Per word: wrong guesses needed to win, or None if never finished."""
    return {w: (r.wrong_guesses if r.final_won else None) for w, r in games.items()}


def win_at(needed: dict[str, int | None], budget: int) -> float:
    return sum(1 for v in needed.values() if v is not None and v < budget) / len(needed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlimited", required=True, help="Log dir of the b=26 run.")
    parser.add_argument(
        "--validate",
        action="append",
        default=[],
        metavar="B=PATH",
        help="Constrained run at budget B to test the derived prediction against.",
    )
    parser.add_argument("--wordlist", default=None)
    parser.add_argument("--out", default=None, help="Prefix for <out>_curves.tsv")
    args = parser.parse_args()

    index = load_dictionary_index(str(resolve_wordlist(args.wordlist)))
    arm = load_arm(pathlib.Path(args.unlimited), index)
    models = sorted(arm)

    # Reference agents play exactly the words the loaded arm played, so the
    # curves stay comparable when the run was filtered or partial.
    word_sets = {model: set(games) for model, games in arm.items()}
    words_union = set.union(*word_sets.values())
    words = sorted(set.intersection(*word_sets.values()))
    if len(words) != len(words_union):
        print(
            f"note: word sets differ across models; using the {len(words)} words "
            f"common to all models (dropping {len(words_union) - len(words)})."
        )
    oracle_needed: dict[str, int | None] = {}
    frequency_needed: dict[str, int | None] = {}
    for word in words:
        pool = index.get(len(word), [])
        playable = pool if word in pool else [*pool, word]
        report = replay_trajectory(
            word=word,
            raw_guesses=agent_frequency(word, playable, MAX_BUDGET),
            dictionary=pool,
            max_wrong=MAX_BUDGET,
            sample_id=word,
        )
        frequency_needed[word] = report.wrong_guesses if report.final_won else None
        # replay_trajectory computes the greedy oracle's count as a side effect
        oracle_needed[word] = report.oracle_wrong_guesses

    curves: dict[str, dict[str, int | None]] = {
        "oracle(greedy)": oracle_needed,
        "frequency(etaoin)": frequency_needed,
    }
    for model in models:
        needed = wrong_needed(arm[model])
        curves[model] = {w: needed[w] for w in words}

    budgets = list(range(1, MAX_BUDGET + 1))
    print("win rate at wrong-guess budget b (derived from the unlimited run):")
    print(f"{'agent':<28}" + "".join(f"b={b:<5}" for b in [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 26]))
    shown = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 26]
    for agent, needed in curves.items():
        row = "".join(f"{win_at(needed, b):<7.2f}" for b in shown)
        dnf = sum(1 for v in needed.values() if v is None)
        print(f"{agent.split('/')[-1]:<28}{row}" + (f" DNF={dnf}" if dnf else ""))

    print()
    print("wrong-guesses-needed distribution (finished games):")
    for agent, needed in curves.items():
        done = sorted(v for v in needed.values() if v is not None)
        if not done:
            continue
        print(
            f"{agent.split('/')[-1]:<28} n={len(done):<4} "
            f"median={done[len(done) // 2]:<3} p90={done[int(len(done) * 0.9)]:<3} "
            f"max={done[-1]}"
        )

    validation_rows = []
    for spec in args.validate:
        b_str, _, path = spec.partition("=")
        budget = int(b_str)
        actual_arm = load_arm(pathlib.Path(path), index)
        print()
        print(
            f"== Budget-invariance check at b={budget}: derived prediction vs "
            f"actual constrained run ({path}) =="
        )
        for model in models:
            actual = actual_arm.get(model)
            if not actual:
                continue
            needed = curves[model]
            words_both = sorted(set(needed) & set(actual))
            predicted = {w: needed[w] is not None and needed[w] < budget for w in words_both}
            real = {w: actual[w].final_won for w in words_both}
            pred_only = sum(1 for w in words_both if predicted[w] and not real[w])
            real_only = sum(1 for w in words_both if real[w] and not predicted[w])
            p = binom_two_sided(min(pred_only, real_only), pred_only + real_only)
            validation_rows.append(
                {
                    "budget": budget,
                    "model": model,
                    "n_words": len(words_both),
                    "predicted_wins": sum(predicted.values()),
                    "actual_wins": sum(real.values()),
                    "pred_only": pred_only,
                    "actual_only": real_only,
                    "p_mcnemar": f"{p:.4f}",
                }
            )
            print(
                f"{model.split('/')[-1]:<28} predicted {sum(predicted.values())}/100"
                f" actual {sum(real.values())}/100 | discordant "
                f"pred-only:{pred_only} actual-only:{real_only} (p={p:.3f})"
            )

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        path = out.with_name(out.name + "_curves.tsv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["agent"] + [f"b{b}" for b in budgets] + ["dnf"])
            for agent, needed in curves.items():
                writer.writerow(
                    [agent]
                    + [f"{win_at(needed, b):.4f}" for b in budgets]
                    + [sum(1 for v in needed.values() if v is None)]
                )
        print(f"\nWrote {path}")
        if validation_rows:
            vpath = out.with_name(out.name + "_validation.tsv")
            with vpath.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(validation_rows[0].keys()),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(validation_rows)
            print(f"Wrote {vpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
