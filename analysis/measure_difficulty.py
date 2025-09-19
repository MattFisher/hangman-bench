#!/usr/bin/env python3
"""
Objective difficulty measurement for Hangman words independent of the
original simulation. Produces a TSV report with multiple metrics per word.

Metrics per word:
- wrong_freq_raw: wrong guesses using a raw letter-frequency heuristic
  (counts duplicated letters within words), with pruning by wrong guesses.
- wrong_coverage: wrong guesses using a coverage heuristic (counts unique
  word-incidence of letters), with pruning by wrong guesses.
- wrong_info_gain: wrong guesses using an information-gain heuristic (counts
  unique word-incidence of letters), with pruning by wrong guesses.
- rare_score: sum over unique letters of -log(p(letter | length)), where p is
  the fraction of same-length dictionary words that contain that letter
  (binary incidence across words).
- dup_factor: len(word) / len(unique_letters)
- structural_score: rare_score / dup_factor

Usage:
  uv run analysis/measure_difficulty.py \
    --datasets src/hangman_bench/datasets.py \
    --wordlist analysis/wordlist.txt \
    --output analysis/difficulty_report.tsv

This script uses only the Python standard library.
"""

import argparse
import importlib.util
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

ALPHABET = [chr(c) for c in range(ord("a"), ord("z") + 1)]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class SolverResult:
    wrong_guesses: int
    total_guesses: int


def load_dataset_words(datasets_path: pathlib.Path) -> List[str]:
    """Load ENGLISH_WORDS from datasets.py via file path to avoid package imports."""
    spec = importlib.util.spec_from_file_location(
        "hangman_datasets", str(datasets_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {datasets_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    english_words = getattr(module, "ENGLISH_WORDS", None)
    if english_words is None:
        raise AttributeError("ENGLISH_WORDS not found in datasets module")
    words: List[str] = []
    for we in english_words:
        w = getattr(we, "word", None)
        if isinstance(w, str):
            words.append(w.lower())
    return words


def load_wordlist(path: pathlib.Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip().lower() for line in f if line.strip()]


def build_length_index(words: List[str]) -> Dict[int, List[str]]:
    idx: Dict[int, List[str]] = {}
    for w in words:
        idx.setdefault(len(w), []).append(w)
    return idx


def filter_candidates(
    board: str, wrong_guesses: List[str], dictionary: List[str]
) -> List[str]:
    # Build regex from board: '.' means any letter
    pattern = re.compile("^" + re.escape(board).replace("\\.", ".") + "$")
    wrong_set = set(wrong_guesses)
    return [w for w in dictionary if pattern.fullmatch(w) and not (set(w) & wrong_set)]


def best_move_freq_raw(
    board: str, wrong_guesses: List[str], dictionary: List[str]
) -> Optional[str]:
    letters_already_found = {c for c in board if c != "."}
    excluded = letters_already_found.union(wrong_guesses)
    counts: Dict[str, int] = {c: 0 for c in ALPHABET if c not in excluded}
    if not counts:
        return None
    for word in dictionary:
        for ch in word:
            if ch in counts:
                counts[ch] += 1
    max_count = max(counts.values(), default=0)
    if max_count <= 0:
        return None
    # Argmax with alphabetical tie-break
    return min([ch for ch, cnt in counts.items() if cnt == max_count])


def best_move_coverage(
    board: str, wrong_guesses: List[str], dictionary: List[str]
) -> Optional[str]:
    letters_already_found = {c for c in board if c != "."}
    excluded = letters_already_found.union(wrong_guesses)
    counts: Dict[str, int] = {c: 0 for c in ALPHABET if c not in excluded}
    if not counts:
        return None
    for word in dictionary:
        s = set(word)
        for ch in s:
            if ch in counts:
                counts[ch] += 1
    max_count = max(counts.values(), default=0)
    if max_count <= 0:
        return None
    # Argmax with alphabetical tie-break
    return min([ch for ch, cnt in counts.items() if cnt == max_count])


def best_move_info_gain(
    board: str, wrong_guesses: List[str], dictionary: List[str]
) -> Optional[str]:
    """Choose the letter that minimizes expected remaining candidate size.

    For each letter l not yet used, partition the dictionary by the mask of
    positions where l appears in a word (empty tuple means a miss). The
    expected remaining size after guessing l is:
        E[|S|] = sum_m p(m) * |S_m| = sum_m (|S_m|/N) * |S_m| = (1/N) * sum_m |S_m|^2
    We choose the letter minimizing sum_m |S_m|^2 (N is constant per letter).
    Ties break alphabetically.
    """
    letters_already_found = {c for c in board if c != "."}
    excluded = letters_already_found.union(wrong_guesses)
    candidates = [c for c in ALPHABET if c not in excluded]
    if not candidates:
        return None

    # Precompute masks per word for speed? Compute per letter to keep memory small.
    best_letter: Optional[str] = None
    best_score: Optional[int] = None

    for letter in candidates:
        # Partition counts by mask (tuple of indices where the letter occurs)
        part_counts: dict[tuple[int, ...], int] = {}
        for w in dictionary:
            mask_list = [i for i, ch in enumerate(w) if ch == letter]
            mask = tuple(mask_list)  # empty tuple means miss
            part_counts[mask] = part_counts.get(mask, 0) + 1

        # Score = sum(count^2) (proportional to expected remaining size)
        score = 0
        for cnt in part_counts.values():
            score += cnt * cnt

        if (
            best_score is None
            or score < best_score
            or (score == best_score and letter < (best_letter or "z"))
        ):
            best_score = score
            best_letter = letter

    return best_letter


def make_move(
    target_word: str, board_list: List[str], guess: str, wrong_guesses: List[str]
) -> Tuple[str, List[str]]:
    correct = False
    for i, ch in enumerate(target_word):
        if ch == guess:
            board_list[i] = guess
            correct = True
    if not correct:
        wrong_guesses.append(guess)
    return ("".join(board_list), wrong_guesses)


def solve_with_strategy(
    target_word: str, dictionary_all: List[str], chooser
) -> SolverResult:
    board = "." * len(target_word)
    wrong_guesses: List[str] = []
    total_guesses = 0
    dictionary = [w for w in dictionary_all if len(w) == len(target_word)]

    while board != target_word:
        if total_guesses != 0:
            dictionary = filter_candidates(board, wrong_guesses, dictionary)
        guess = chooser(board, wrong_guesses, dictionary)
        if guess is None:
            # Fallback: first remaining letter
            used = set(wrong_guesses) | {c for c in board if c != "."}
            remaining = [c for c in ALPHABET if c not in used]
            if not remaining:
                break
            guess = remaining[0]
        board_list = list(board)
        board, wrong_guesses = make_move(target_word, board_list, guess, wrong_guesses)
        total_guesses += 1

    return SolverResult(wrong_guesses=len(wrong_guesses), total_guesses=total_guesses)


def precompute_letter_incidence(
    length_words: Dict[int, List[str]],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for L, words in length_words.items():
        denom = max(1, len(words))
        counts = {c: 0 for c in ALPHABET}
        for w in words:
            s = set(w)
            for c in s:
                if c in counts:
                    counts[c] += 1
        out[L] = {c: counts[c] / denom for c in ALPHABET}
    return out


def structural_scores(
    word: str, p_by_len: Dict[int, Dict[str, float]]
) -> Tuple[float, float, float]:
    L = len(word)
    uniq = set(word)
    pmap = p_by_len.get(L, {})
    rare = 0.0
    for c in uniq:
        p = pmap.get(c, 1e-9)
        p = max(p, 1e-9)
        rare += -math.log(p)
    dup_factor = L / max(1, len(uniq))
    structural = rare / dup_factor
    return rare, dup_factor, structural


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Objective difficulty metrics for Hangman words"
    )
    parser.add_argument(
        "--datasets",
        default=str(REPO_ROOT / "src/hangman_bench/datasets.py"),
        help="Path to datasets.py",
    )
    parser.add_argument(
        "--wordlist",
        default=str(REPO_ROOT / "analysis/wordlist.txt"),
        help="Path to dictionary wordlist",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "analysis/difficulty_report.tsv"),
        help="Path to output TSV",
    )
    args = parser.parse_args(argv)

    datasets_path = pathlib.Path(args.datasets)
    dict_path = pathlib.Path(args.wordlist)
    out_path = pathlib.Path(args.output)

    dataset_words = load_dataset_words(datasets_path)
    dictionary_all = load_wordlist(dict_path)
    length_index = build_length_index(dictionary_all)
    p_letter_len = precompute_letter_incidence(length_index)

    # Compute metrics per word
    rows: List[List[str]] = []
    for w in dataset_words:
        L = len(w)
        # Skip words not present in dictionary length index; still compute structural using length bin
        dict_L = length_index.get(L, [])
        # Solvers
        wrong_freq_raw = None
        wrong_coverage = None
        wrong_info_gain = None
        if dict_L:
            res_freq = solve_with_strategy(w, dict_L, best_move_freq_raw)
            res_cov = solve_with_strategy(w, dict_L, best_move_coverage)
            res_inf = solve_with_strategy(w, dict_L, best_move_info_gain)
            wrong_freq_raw = res_freq.wrong_guesses
            wrong_coverage = res_cov.wrong_guesses
            wrong_info_gain = res_inf.wrong_guesses
        # Structural
        rare, dup, structural = structural_scores(w, p_letter_len)
        rows.append(
            [
                w,
                str(L),
                str(wrong_freq_raw) if wrong_freq_raw is not None else "",
                str(wrong_coverage) if wrong_coverage is not None else "",
                str(wrong_info_gain) if wrong_info_gain is not None else "",
                f"{rare:.3f}",
                f"{dup:.3f}",
                f"{structural:.3f}",
            ]
        )

    # Write TSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "word",
                "length",
                "wrong_freq_raw",
                "wrong_coverage",
                "wrong_info_gain",
                "rare_score",
                "dup_factor",
                "structural_score",
            ]
        )
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(
        "Columns: word, length, wrong_freq_raw, wrong_coverage, wrong_info_gain, rare_score, dup_factor, structural_score"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
