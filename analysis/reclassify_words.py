#!/usr/bin/env python3
"""
Reclassify words in src/hangman_bench/datasets.py based on
mean_wrong_guesses from SimulationData_parsed.tsv.

By default, words are divided into 5 quantile-based bins and assigned to
[v_easy, easy, medium, hard, v_hard] in ascending order of mean wrong guesses.

Usage examples:
  # Default: use 5 quantile bins over the means of words present in datasets.py
  python scripts/reclassify_words.py --tsv SimulationData_parsed.tsv

  # Use custom floating-point cut thresholds (must provide exactly 4 values)
  python scripts/reclassify_words.py --tsv SimulationData_parsed.tsv \
      --cuts 0.5,1.5,2.5,3.5

  # Emit a Python snippet you can paste back into datasets.py
  python scripts/reclassify_words.py --emit-snippet scripts/reclassified_snippet.py

Outputs:
- A TSV (default: reclassified_words.tsv) with columns:
    word, mean_wrong_guesses, old_difficulty, new_difficulty, change
- Optional Python snippet file containing an `ENGLISH_WORDS_RECLASSIFIED` list.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Repository paths (used by CLI defaults and import fallback inside main())
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

DIFFICULTY_ORDER: List[str] = ["v_easy", "easy", "medium", "hard", "v_hard"]


@dataclass
class ReclassResult:
    word: str
    mean_wrong_guesses: float
    old_difficulty: Optional[str]
    new_difficulty: str

    @property
    def change(self) -> str:
        if self.old_difficulty is None:
            return "new"
        return "same" if self.old_difficulty == self.new_difficulty else "changed"


def read_means_from_tsv(tsv_path: pathlib.Path) -> Dict[str, float]:
    """Read word -> mean_wrong_guesses from a TSV file.

    Expects header with columns: word, wrong_guesses, mean_wrong_guesses
    """
    means: Dict[str, float] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Basic validation of expected columns
        if reader.fieldnames is None or "word" not in reader.fieldnames or "mean_wrong_guesses" not in reader.fieldnames:
            raise ValueError(
                f"Input TSV must contain columns 'word' and 'mean_wrong_guesses' (got: {reader.fieldnames})"
            )
        for row in reader:
            w = (row.get("word") or "").strip()
            if not w:
                continue
            try:
                m = float(row.get("mean_wrong_guesses") or "nan")
            except ValueError:
                m = math.nan
            if not math.isnan(m):
                means[w.lower()] = m
    return means


def load_current_words(datasets_path: pathlib.Path) -> Dict[str, str]:
    """Load ENGLISH_WORDS from datasets.py directly via file path.

    Returns a mapping word(lowercased) -> difficulty.
    """
    spec = importlib.util.spec_from_file_location("hangman_datasets", str(datasets_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {datasets_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    english_words = getattr(module, "ENGLISH_WORDS", None)
    if english_words is None:
        raise AttributeError("ENGLISH_WORDS not found in datasets module")
    words: Dict[str, str] = {}
    for we in english_words:
        word = getattr(we, "word", None)
        difficulty = getattr(we, "difficulty", None)
        if isinstance(word, str) and isinstance(difficulty, str):
            words[word.lower()] = difficulty
    return words


def compute_quantile_thresholds(values: Sequence[float], bins: int = 5) -> List[float]:
    """Compute (bins-1) quantile thresholds for the provided values.

    Returns a sorted list of length (bins-1) containing interior thresholds.
    Uses a simple linear interpolation percentile approach to avoid numpy dependency.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if not values:
        raise ValueError("No values provided to compute thresholds")

    sorted_vals = sorted(values)

    def percentile(p: float) -> float:
        # p in [0, 100]
        if p <= 0:
            return sorted_vals[0]
        if p >= 100:
            return sorted_vals[-1]
        rank = (p / 100) * (len(sorted_vals) - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return sorted_vals[lo]
        frac = rank - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    step = 100.0 / bins
    thresholds = [percentile(step * k) for k in range(1, bins)]
    # Ensure monotonic non-decreasing thresholds
    for i in range(1, len(thresholds)):
        if thresholds[i] < thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1]
    return thresholds


def classify_by_thresholds(value: float, thresholds: Sequence[float], labels: Sequence[str]) -> str:
    """Classify a numeric value into one of len(labels) bins based on thresholds.

    thresholds are interior cut points sorted ascending: [t1, t2, ..., t_{k-1}]
    We assign:
      value <= t1           -> labels[0]
      t1 < value <= t2      -> labels[1]
      ...
      value > t_{k-1}       -> labels[k]
    """
    import bisect

    # Use bisect_left so equality goes to the lower bin, matching the
    # documented mapping above.
    idx = bisect.bisect_left(thresholds, value)
    idx = max(0, min(idx, len(labels) - 1))
    return labels[idx]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reclassify words by mean_wrong_guesses")
    parser.add_argument(
        "--tsv",
        default=str(REPO_ROOT / "SimulationData_parsed.tsv"),
        help="Path to SimulationData_parsed.tsv (default: repo root)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "reclassified_words.tsv"),
        help="Path to write the reclassification report TSV",
    )
    parser.add_argument(
        "--cuts",
        default=None,
        help=(
            "Comma-separated custom thresholds (exactly 4 floats for 5 bins). "
            "Example: --cuts 0.5,1.5,2.5,3.5"
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Number of quantile bins to use when --cuts is not provided (default: 5)",
    )
    parser.add_argument(
        "--emit-snippet",
        default=None,
        help=(
            "Optional path to write a Python snippet with ENGLISH_WORDS_RECLASSIFIED. "
            "If omitted, no snippet is written."
        ),
    )

    args = parser.parse_args(argv)

    tsv_path = pathlib.Path(args.tsv)
    out_path = pathlib.Path(args.output)

    # Load datasets.py directly to avoid package __init__ side-effects
    datasets_py = REPO_ROOT / "src" / "hangman_bench" / "datasets.py"
    if not datasets_py.exists():
        print(f"datasets.py not found at {datasets_py}", file=sys.stderr)
        return 2
    current_words: Dict[str, str] = load_current_words(datasets_py)

    # Read means and filter to words in datasets
    means_map_all = read_means_from_tsv(tsv_path)
    missing: List[str] = []
    present_means: Dict[str, float] = {}
    for w in current_words:
        if w in means_map_all:
            present_means[w] = means_map_all[w]
        else:
            missing.append(w)

    if not present_means:
        print("No words from datasets.py were found in the TSV. Check input paths.", file=sys.stderr)
        return 2

    # Determine thresholds
    if args.cuts:
        try:
            thresholds = [float(x.strip()) for x in args.cuts.split(",") if x.strip()]
        except ValueError:
            print("--cuts must be a comma-separated list of floats", file=sys.stderr)
            return 2
        if len(thresholds) != len(DIFFICULTY_ORDER) - 1:
            print(
                f"--cuts must provide exactly {len(DIFFICULTY_ORDER)-1} thresholds; got {len(thresholds)}",
                file=sys.stderr,
            )
            return 2
        thresholds = sorted(thresholds)
    else:
        thresholds = compute_quantile_thresholds(list(present_means.values()), bins=args.bins)

    # Build results
    results: List[ReclassResult] = []
    for w, m in sorted(present_means.items()):
        new_diff = classify_by_thresholds(m, thresholds, DIFFICULTY_ORDER)
        old_diff = current_words.get(w)
        results.append(ReclassResult(word=w, mean_wrong_guesses=m, old_difficulty=old_diff, new_difficulty=new_diff))

    # Write report TSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["word", "mean_wrong_guesses", "old_difficulty", "new_difficulty", "change"])
        for r in results:
            writer.writerow([r.word, f"{r.mean_wrong_guesses:.3f}", r.old_difficulty or "", r.new_difficulty, r.change])

    # Console summary
    from collections import Counter

    counts = Counter(r.new_difficulty for r in results)
    changed = sum(1 for r in results if r.change == "changed")
    print("Reclassification complete:")
    print(f"  Words processed: {len(results)}")
    if missing:
        print(f"  Words missing from TSV: {len(missing)} (e.g., {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''})")
    print("  New difficulty counts:")
    for label in DIFFICULTY_ORDER:
        print(f"    {label:7s}: {counts.get(label, 0)}")
    print(f"  Changed assignments: {changed}")
    print(f"  Report written to: {out_path}")
    print(f"  Thresholds used: {', '.join(f'{t:.3f}' for t in thresholds)}")

    # Optional snippet
    if args.emit_snippet:
        snippet_path = pathlib.Path(args.emit_snippet)
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        # Sort by new difficulty order then alphabetically
        order_index = {d: i for i, d in enumerate(DIFFICULTY_ORDER)}
        sorted_entries = sorted(results, key=lambda r: (order_index[r.new_difficulty], r.word))
        with snippet_path.open("w", encoding="utf-8") as sf:
            sf.write("# Auto-generated by scripts/reclassify_words.py\n")
            sf.write("from hangman_bench.datasets import WordEntry\n\n")
            sf.write("ENGLISH_WORDS_RECLASSIFIED = [\n")
            for r in sorted_entries:
                sf.write(f"    WordEntry(\"{r.word}\", \"{r.new_difficulty}\"),\n")
            sf.write("]\n")
        print(f"  Snippet written to: {snippet_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
