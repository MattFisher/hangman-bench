#!/usr/bin/env python3
"""
Bin words into difficulty tiers based on a chosen metric column from
analysis/difficulty_report.tsv (default: wrong_coverage).

- Computes (bins-1) quantile thresholds over available numeric metric values
  and assigns labels: [v_easy, easy, medium, hard, v_hard].
- Writes a TSV with columns: word, metric, label
- Optionally emits a Python snippet ENGLISH_WORDS_RECLASSIFIED for pasting
  into datasets.py

Usage:
  uv run analysis/bin_difficulty.py \
    --input analysis/difficulty_report.tsv \
    --metric wrong_coverage \
    --output analysis/difficulty_binned.tsv \
    --emit-snippet analysis/reclassified_from_coverage.py
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
from typing import Dict, List, Optional, Sequence

LABELS = ["v_easy", "easy", "medium", "hard", "v_hard"]


def read_metric(input_path: pathlib.Path, metric: str, fallback_metric: Optional[str]) -> Dict[str, float]:
    """Read word -> metric value. If metric empty and fallback provided, try fallback."""
    out: Dict[str, float] = {}
    with input_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames is None or 'word' not in reader.fieldnames:
            raise ValueError("Input TSV must include a 'word' column")
        for row in reader:
            w = (row.get('word') or '').strip().lower()
            if not w:
                continue
            val_str = (row.get(metric) or '').strip()
            v: Optional[float] = None
            if val_str:
                try:
                    v = float(val_str)
                except ValueError:
                    v = None
            if v is None and fallback_metric:
                fb_str = (row.get(fallback_metric) or '').strip()
                if fb_str:
                    try:
                        v = float(fb_str)
                    except ValueError:
                        v = None
            if v is not None and not math.isnan(v):
                out[w] = v
    if not out:
        raise ValueError(f"No numeric values found for metric '{metric}'")
    return out


def compute_quantile_thresholds(values: List[float], bins: int) -> List[float]:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    s = sorted(values)
    if not s:
        return []
    def percentile(pct: float) -> float:
        if pct <= 0:
            return s[0]
        if pct >= 100:
            return s[-1]
        rank = (pct / 100.0) * (len(s) - 1)
        lo = int(rank)
        hi = min(len(s) - 1, lo + 1)
        if lo == hi:
            return s[lo]
        frac = rank - lo
        return s[lo] * (1 - frac) + s[hi] * frac
    step = 100.0 / bins
    cuts = [percentile(step * k) for k in range(1, bins)]
    # Ensure non-decreasing
    for i in range(1, len(cuts)):
        if cuts[i] < cuts[i-1]:
            cuts[i] = cuts[i-1]
    return cuts


def classify(value: float, thresholds: List[float], labels: List[str]) -> str:
    # Use left-bisect so equality stays in the lower bin
    import bisect
    idx = bisect.bisect_left(thresholds, value)
    idx = max(0, min(idx, len(labels) - 1))
    return labels[idx]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bin words into difficulty tiers using quantiles")
    parser.add_argument("--input", default="analysis/difficulty_report.tsv", help="Path to difficulty report TSV")
    parser.add_argument("--metric", default="wrong_coverage", help="Metric column to use (default: wrong_coverage)")
    parser.add_argument("--fallback-metric", default="wrong_freq_raw", help="Fallback metric column if primary is missing")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins (default: 5)")
    parser.add_argument("--output", default="analysis/difficulty_binned.tsv", help="Path to write binned TSV")
    parser.add_argument("--emit-snippet", default=None, help="Optional path to write ENGLISH_WORDS_RECLASSIFIED snippet")
    args = parser.parse_args(argv)

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)

    metric_map = read_metric(in_path, args.metric, args.fallback_metric)
    values = list(metric_map.values())
    thresholds = compute_quantile_thresholds(values, bins=args.bins)

    # Build labeled rows
    rows: List[List[str]] = []
    labeled: List[tuple[str, str, float]] = []
    for w, v in sorted(metric_map.items()):
        label = classify(v, thresholds, LABELS)
        rows.append([w, f"{v:.3f}", label])
        labeled.append((w, label, v))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["word", args.metric, "label"])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Thresholds used (interior cuts): " + ", ".join(f"{t:.3f}" for t in thresholds))

    if args.emit_snippet:
        snippet_path = pathlib.Path(args.emit_snippet)
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        # Sort by label order then alphabetically
        order_index = {lab: i for i, lab in enumerate(LABELS)}
        labeled.sort(key=lambda x: (order_index[x[1]], x[0]))
        with snippet_path.open('w', encoding='utf-8') as sf:
            sf.write("# Auto-generated by analysis/bin_difficulty.py\n")
            sf.write("from hangman_bench.datasets import WordEntry\n\n")
            sf.write("ENGLISH_WORDS_RECLASSIFIED = [\n")
            for w, label, _ in labeled:
                sf.write(f"    WordEntry(\"{w}\", \"{label}\"),\n")
            sf.write("]\n")
        print(f"Snippet written to: {snippet_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
