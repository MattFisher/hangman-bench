#!/usr/bin/env python3
"""
Ingest SimulationData.txt and export a tab-separated file with one line per word,
including the list of wrong guesses and the mean number of wrong guesses.

Usage:
  python scripts/ingest_simulation.py --input SimulationData.txt --output SimulationData_parsed.tsv

Notes:
- The input file is large (~50MB). This script reads it into memory once and uses
  a regex to extract pairs of the form {"word", {n1, n2, ..., nk}}. This is acceptable
  for this size. If needed, a streaming parser can be implemented later.
- Output is TSV with columns: word, wrong_guesses, mean_wrong_guesses
  where wrong_guesses is a comma-separated list inside square brackets.
- If the --input file does not exist, the script will download the source ZIP from
  the Wolfram Library link and extract SimulationData.txt automatically to that path.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from statistics import mean
from typing import Iterable, List, Tuple

# Regex to capture entries like {"word", {1, 2, 3}}
# DOTALL to allow numbers list across multiple lines
ENTRY_REGEX = re.compile(r"\{\s*\"([^\"]+)\"\s*,\s*\{([^}]*)\}\s*\}", re.DOTALL)

# Source ZIP containing SimulationData.txt
SIMULATION_ZIP_URL = (
    "https://library.wolfram.com/infocenter/MathSource/7635/SimulationData.zip?file_id=7257"
)


def download_and_extract_simulation_data(dest_path: str) -> str:
    """Download SimulationData.zip and extract SimulationData.txt to dest_path.

    Args:
        dest_path: Desired output path of SimulationData.txt

    Returns:
        The path to the extracted SimulationData.txt (same as dest_path)
    """
    # Ensure destination directory exists
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "SimulationData.zip")
        urllib.request.urlretrieve(SIMULATION_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find SimulationData.txt within extracted contents
        candidate = None
        for root, _dirs, files in os.walk(tmpdir):
            if "SimulationData.txt" in files:
                candidate = os.path.join(root, "SimulationData.txt")
                break

        if candidate is None:
            raise FileNotFoundError(
                "SimulationData.txt not found inside downloaded ZIP archive"
            )

        shutil.copyfile(candidate, dest_path)
        return dest_path


def parse_simulation_data(text: str) -> Iterable[Tuple[str, List[int]]]:
    """Parse the SimulationData.txt content.

    Args:
        text: Full text content of SimulationData.txt

    Yields:
        Tuples of (word, list_of_wrong_guess_counts)
    """
    for match in ENTRY_REGEX.finditer(text):
        word = match.group(1)
        numbers_blob = match.group(2)
        # Split by commas and parse integers, ignoring whitespace and newlines
        nums: List[int] = []
        for part in numbers_blob.split(','):
            s = part.strip()
            if not s:
                continue
            # Some files may contain trailing comments or braces; be defensive
            # Keep only leading numeric portion (optional minus not expected but handled)
            m = re.match(r"^(-?\d+)", s)
            if m:
                try:
                    nums.append(int(m.group(1)))
                except ValueError:
                    continue
        yield word, nums


def write_tsv(rows: Iterable[Tuple[str, List[int]]], out_path: str) -> None:
    """Write parsed rows to a TSV file with mean.

    Columns: word, wrong_guesses, mean_wrong_guesses
    wrong_guesses is a comma-separated list inside square brackets.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["word", "wrong_guesses", "mean_wrong_guesses"])
        for word, nums in rows:
            m = float(mean(nums)) if nums else 0.0
            # Format list similar to the original (comma + space)
            nums_str = f"[{', '.join(str(n) for n in nums)}]"
            writer.writerow([word, nums_str, f"{m:.3f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SimulationData.txt and export TSV per word with mean.")
    parser.add_argument("--input", required=True, help="Path to SimulationData.txt")
    parser.add_argument("--output", default="SimulationData_parsed.tsv", help="Path to output TSV file")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        print(f"Input file not found at {in_path}. Downloading and extracting...")
        download_and_extract_simulation_data(in_path)

    # Read entire file (approx 50MB)
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    rows = list(parse_simulation_data(text))
    write_tsv(rows, out_path)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
