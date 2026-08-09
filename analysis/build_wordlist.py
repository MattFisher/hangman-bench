#!/usr/bin/env python3
"""
Build the oracle's dictionary from SCOWL.

The dictionary defines the oracle's belief state, so every per-guess metric
depends on it. It therefore needs a stated, licensed, reproducible source
rather than an arbitrary list. SCOWL provides word lists split by size tier
and by dialect, with proper names, abbreviations and capitalised forms kept
in separate files so a clean lexicon is the default.

One dialect is built at a time and written to its own file, so other dialects
can be added later without disturbing the one in use.

Usage:
  uv run analysis/build_wordlist.py --dialect en_GB
  uv run analysis/build_wordlist.py --dialect en_US --tier 60
  uv run analysis/build_wordlist.py --dialect en_GB --with-frequencies

SCOWL is licensed under an MIT-like licence that requires its copyright
notice to accompany copies, so the notice is written alongside the wordlist.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from typing import Dict, List, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "hangman_bench" / "data"

SCOWL_VERSION = "2020.12.07"
SCOWL_URL = f"https://downloads.sourceforge.net/wordlist/scowl-{SCOWL_VERSION}.tar.gz"

# SCOWL splits its lists by dialect. 'english' is the dialect-neutral core;
# each dialect file adds only the spellings specific to it.
#
# British is published in two spelling conventions: 'british' uses -ise,
# 'british_z' uses -ize (Oxford spelling). They are alternatives, not
# complements — merging them would put 228 words in the dictionary under both
# spellings, and a guesser facing 'reali_e' would have to pick s or z on no
# information. Pick one.
DIALECTS: Dict[str, List[str]] = {
    "en_GB": ["english-words", "british-words"],
    "en_GB_oxendict": ["english-words", "british_z-words"],
    "en_US": ["english-words", "american-words"],
    "en_AU": ["english-words", "australian-words"],
    "en_CA": ["english-words", "canadian-words"],
}

# SCOWL tiers are cumulative: a size-50 list means every tier up to 50.
TIERS = [10, 20, 35, 40, 50, 55, 60, 70, 80, 95]

# Hangman on a one or two letter word is degenerate, and the short entries in
# SCOWL are mostly single letters and initialisms (cs, kw, ls, rs, ts).
DEFAULT_MIN_LENGTH = 3


def download_scowl(dest: pathlib.Path) -> pathlib.Path:
    """Fetch and unpack SCOWL, returning the extracted directory."""
    dest.mkdir(parents=True, exist_ok=True)
    extracted = dest / f"scowl-{SCOWL_VERSION}"
    if extracted.is_dir():
        return extracted

    print(f"Downloading SCOWL {SCOWL_VERSION} ...", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        archive = pathlib.Path(tmp) / "scowl.tar.gz"
        urllib.request.urlretrieve(SCOWL_URL, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest)
    if not extracted.is_dir():
        raise RuntimeError(f"Expected {extracted} after extracting {SCOWL_URL}")
    return extracted


def collect_words(
    scowl_dir: pathlib.Path, prefixes: Sequence[str], tier: int, min_length: int
) -> List[str]:
    """Words from the given SCOWL lists, up to and including ``tier``.

    Only the ``*-words.*`` files are read. Proper names, abbreviations,
    capitalised forms and contractions live in sibling files and are left out.
    """
    final = scowl_dir / "final"
    if not final.is_dir():
        raise FileNotFoundError(f"No final/ directory under {scowl_dir}")

    words: set[str] = set()
    used: List[str] = []
    for prefix in prefixes:
        for size in (t for t in TIERS if t <= tier):
            path = final / f"{prefix}.{size}"
            if not path.is_file():
                continue
            used.append(path.name)
            # SCOWL files are Latin-1; a few entries carry accents and are
            # dropped by the isalpha/ASCII check below.
            with path.open("r", encoding="latin-1") as handle:
                for line in handle:
                    word = line.strip().lower()
                    if len(word) >= min_length and word.isalpha() and word.isascii():
                        words.add(word)

    print(f"Read {len(used)} SCOWL files: {', '.join(sorted(used))}", file=sys.stderr)
    return sorted(words)


def write_wordlist(words: Sequence[str], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for word in words:
            handle.write(word + "\n")


def write_frequencies(words: Sequence[str], path: pathlib.Path) -> None:
    """Emit word<TAB>frequency using wordfreq.

    Precomputing means the package needs no runtime dependency on wordfreq;
    it is only required to regenerate this file.
    """
    try:
        from wordfreq import word_frequency
    except ImportError:
        raise SystemExit(
            "wordfreq is required for --with-frequencies. "
            "Install it with: uv pip install wordfreq"
        )

    missing = 0
    with path.open("w", encoding="utf-8") as handle:
        for word in words:
            freq = word_frequency(word, "en")
            if freq == 0.0:
                missing += 1
            handle.write(f"{word}\t{freq:.6e}\n")
    print(
        f"{missing}/{len(words)} words have zero frequency and need smoothing "
        f"when used as a prior",
        file=sys.stderr,
    )


def copy_licence(scowl_dir: pathlib.Path, dest: pathlib.Path) -> None:
    """SCOWL's licence requires its copyright notice to travel with copies."""
    source = scowl_dir / "Copyright"
    if not source.is_file():
        raise FileNotFoundError(f"No Copyright file in {scowl_dir}")
    shutil.copyfile(source, dest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dialect",
        choices=sorted(DIALECTS),
        default="en_GB",
        help="Which SCOWL dialect to build (default: en_GB).",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=TIERS,
        default=50,
        help="Largest SCOWL size to include; tiers are cumulative (default: 50).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help=f"Drop shorter words (default: {DEFAULT_MIN_LENGTH}).",
    )
    parser.add_argument(
        "--scowl-dir",
        default=None,
        help="Existing SCOWL checkout; downloaded to a cache directory if omitted.",
    )
    parser.add_argument("--output", default=None, help="Output wordlist path.")
    parser.add_argument(
        "--with-frequencies",
        action="store_true",
        help="Also write <stem>_freq.tsv from wordfreq, for a weighted prior.",
    )
    args = parser.parse_args(argv)

    if args.scowl_dir:
        scowl_dir = pathlib.Path(args.scowl_dir)
    else:
        scowl_dir = download_scowl(REPO_ROOT / ".scowl-cache")

    words = collect_words(scowl_dir, DIALECTS[args.dialect], args.tier, args.min_length)
    output = (
        pathlib.Path(args.output)
        if args.output
        else DATA_DIR / f"wordlist_{args.dialect}.txt"
    )
    write_wordlist(words, output)
    print(f"Wrote {len(words)} words to {output}")

    copy_licence(scowl_dir, DATA_DIR / "SCOWL-Copyright")
    print(f"Wrote SCOWL copyright notice to {DATA_DIR / 'SCOWL-Copyright'}")

    if args.with_frequencies:
        freq_path = output.with_name(output.stem + "_freq.tsv")
        write_frequencies(words, freq_path)
        print(f"Wrote frequencies to {freq_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
