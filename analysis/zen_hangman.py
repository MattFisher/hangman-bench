#!/usr/bin/env python3
"""
Python port of [zen-hangman.rb](https://gist.github.com/Dan-Q/7910309) by Dan Q,
from blog post [The Hardest Hangman](https://danq.me/2013/12/15/hangman/)

Original dictionary was downloaded from https://www.curlewcommunications.uk/wordlist.html

One improvement over the original:
- When filtering the candidate dictionary, also prune words that contain any
  known wrong guesses.

Behavior summary:
- Load words from a plain text wordlist (whitespace-separated), keep only
  lowercase ASCII alphabetic words of length N.
- For each target word, simulate an "optimal" player that repeatedly:
  - Filters the candidate dictionary to words matching the current board
    (e.g., "c.t..") and not containing any wrong guesses.
  - Chooses the next guess as the letter with the highest raw occurrence
    count across all candidate words (duplicates within a single word count
    multiple times), excluding letters already revealed and previously-wrong.
  - Updates the board (if correct) or wrong guesses (if not).
- Outputs a table of Word, Guesses, Wrong, sorted by most wrong guesses,
  then most total guesses, then alphabetically by the word (like the Ruby).

Usage:
  python analysis/zen_hangman.py --wordlist wordlist.txt --num-letters 6

This script uses only the Python standard library.
"""

import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple

ALPHABET = [chr(c) for c in range(ord("a"), ord("z") + 1)]


@dataclass
class GameResult:
    word: str
    num_guesses: int
    wrong_guesses: int


def load_words(path: str, num_letters: int) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tokens = f.read().split()
    words: List[str] = []
    for token in tokens:
        w = token.strip().lower()
        if len(w) == num_letters and re.fullmatch(r"[a-z]+", w):
            words.append(w)
    # Deduplicate while preserving order
    seen = set()
    unique_words: List[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    return unique_words


def best_move_for(
    board: str, wrong_guesses: List[str], dictionary: List[str]
) -> str | None:
    letters_already_found = {c for c in board if c != "."}
    excluded = letters_already_found.union(wrong_guesses)
    counts = {c: 0 for c in ALPHABET if c not in excluded}

    if not counts:
        return None

    for word in dictionary:
        for ch in word:
            if ch in counts:
                counts[ch] += 1

    # Pick the letter with the greatest count; break ties alphabetically
    # by selecting the smallest letter among those with max count.
    max_count = max(counts.values(), default=0)
    if max_count <= 0:
        return None
    candidates = [ch for ch, cnt in counts.items() if cnt == max_count]
    return min(candidates) if candidates else None


def make_move(
    target_word: str, board: List[str], guess: str, wrong_guesses: List[str]
) -> Tuple[str, List[str]]:
    correct = False
    for i, ch in enumerate(target_word):
        if ch == guess:
            board[i] = guess
            correct = True
    if not correct:
        wrong_guesses.append(guess)
    return ("".join(board), wrong_guesses)


def result_for(
    target_word: str, board: str, dictionary_all: List[str], debug: bool = False
) -> GameResult:
    wrong_guesses: List[str] = []
    num_guesses = 0
    dictionary = list(dictionary_all)  # clone per original ruby behavior

    while board != target_word:
        # After first guess, filter by board and wrong guesses, like the ruby (which only
        # filters by board). We also prune by wrong guesses.
        if num_guesses != 0:
            # Build regex from board where '.' matches any letter
            # e.g., board 'c.t..' => r'^c.t..$'
            pattern = re.compile("^" + re.escape(board).replace("\.", ".") + "$")
            wrong_set = set(wrong_guesses)
            dictionary = [
                w
                for w in dictionary
                if pattern.fullmatch(w) and not (set(w) & wrong_set)
            ]

        if debug:
            print(" > Considering my best move...")

        guess = best_move_for(board, wrong_guesses, dictionary)
        if guess is None:
            # Fallback: pick first remaining alphabet letter not already used
            used = set(wrong_guesses) | {c for c in board if c != "."}
            remaining = [c for c in ALPHABET if c not in used]
            if not remaining:
                # No moves possible; break to avoid infinite loop
                break
            guess = remaining[0]

        if debug:
            print(f" > I guess '{guess}'")

        # Apply guess
        board_list = list(board)
        board, wrong_guesses = make_move(target_word, board_list, guess, wrong_guesses)
        num_guesses += 1

    return GameResult(
        word=target_word, num_guesses=num_guesses, wrong_guesses=len(wrong_guesses)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zen Hangman (Python) with pruning by wrong guesses"
    )
    parser.add_argument(
        "--wordlist", default="wordlist.txt", help="Path to word list file"
    )
    parser.add_argument(
        "--num-letters", type=int, required=True, help="Word length to consider"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug output"
    )
    args = parser.parse_args()

    words = load_words(args.wordlist, args.num_letters)
    print(f"There are {len(words)} words to play.")

    results: List[GameResult] = []
    for target in words:
        if args.debug:
            print(f"Target word is {target}")
        board = "." * args.num_letters
        results.append(result_for(target, board, words, debug=args.debug))

    # Sort: reverse by wrong_guesses, then reverse by num_guesses, then alphabetically by word
    results.sort(key=lambda r: (-r.wrong_guesses, -r.num_guesses, r.word))

    word_col_width = max(4, args.num_letters)
    print(f"{'Word'.rjust(word_col_width)} {'Guesses'.rjust(9)} {'Wrong'.rjust(7)}")
    for r in results:
        print(
            f"{r.word.rjust(word_col_width)} {str(r.num_guesses).rjust(9)} {str(r.wrong_guesses).rjust(7)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
