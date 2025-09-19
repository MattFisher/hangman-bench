#!/usr/bin/env python3
"""
Extract a unique, lowercase wordlist (one word per line) from the first column
of a TSV file like analysis/SimulationData_parsed.tsv.

It prefers the header column named "word" if available; otherwise it falls
back to taking the first column.

Usage:
  uv run analysis/extract_wordlist.py \
    --input analysis/SimulationData_parsed.tsv \
    --output analysis/wordlist.txt
"""
from __future__ import annotations

import argparse
import csv
from typing import List


def read_words(input_path: str) -> List[str]:
    words: List[str] = []
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        # Try DictReader first to honor header names
        sniffer = csv.Sniffer()
        # We know it's TSV, but DictReader needs fieldnames; we manually set delimiter
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and "word" in reader.fieldnames:
            for row in reader:
                w = (row.get("word") or "").strip().lower()
                if w:
                    words.append(w)
            return words

    # Fallback: standard reader, first column only
    with open(input_path, "r", encoding="utf-8", newline="") as f2:
        reader2 = csv.reader(f2, delimiter="\t")
        first = True
        for row in reader2:
            if not row:
                continue
            if first and row[0].strip().lower() == "word":
                first = False
                continue
            first = False
            w = row[0].strip().lower()
            if w:
                words.append(w)
    return words


def write_unique(words: List[str], output_path: str) -> None:
    seen = set()
    with open(output_path, "w", encoding="utf-8") as out:
        for w in words:
            if w not in seen:
                seen.add(w)
                out.write(w + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract wordlist from TSV first column")
    parser.add_argument("--input", required=True, help="Path to TSV file (expects a 'word' header)")
    parser.add_argument("--output", required=True, help="Path to write word list (one word per line)")
    args = parser.parse_args()

    words = read_words(args.input)
    write_unique(words, args.output)

    print(f"Wrote {len(set(words))} unique words to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
