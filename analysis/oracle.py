#!/usr/bin/env python3
"""
Oracle replay for Hangman trajectories.

Hangman is one of the few agentic tasks where, at every step, both the exact
posterior over the hidden word and the optimal next action are computable from
a dictionary. This module replays a recorded sequence of guesses and scores
each individual guess against that posterior, rather than only scoring the
final win/loss.

The metrics fall into two families.

Provable errors (no judgement required):
- invalid:        the guess was not a single alphabetic character.
- repeat:         the letter had already been guessed, so the guess cannot
                  change the belief state. Strictly wasted.
- dominated_miss: a fresh letter that appears in ZERO consistent candidate
                  words. Guaranteed to cost a life for zero information.

Quality relative to optimal play:
- hit_prob_regret: (best available hit probability) - (hit probability of the
                  letter actually guessed), under a uniform posterior over the
                  consistent candidate set.
- wrong_guess_regret: wrong guesses taken by the agent minus wrong guesses an
                  oracle solver takes on the same word and dictionary.

This module uses only the standard library, matching the other analysis
scripts.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ALPHABET = [chr(c) for c in range(ord("a"), ord("z") + 1)]

# A chooser takes (candidates, already_guessed) and returns the next letter.
Chooser = Callable[[Sequence[str], frozenset], Optional[str]]


# --------------------------------------------------------------------------
# Belief state
# --------------------------------------------------------------------------


def reveal(word: str, guessed: Iterable[str]) -> str:
    """Board as seen by the player: revealed letters, '.' for unknown."""
    guessed_set = set(guessed)
    return "".join(ch if ch in guessed_set else "." for ch in word)


def consistent_candidates(
    board: str, guessed: Iterable[str], dictionary: Sequence[str]
) -> List[str]:
    """Words consistent with everything the player has been told.

    A word is consistent iff, for every letter the player has guessed, the set
    of positions where that letter occurs in the word is exactly the set of
    positions where it is revealed on the board. This is stricter than pattern
    matching: a guessed letter cannot hide in an unrevealed position, because
    hangman reveals *all* of its occurrences at once.

    ``analysis/measure_difficulty.py:filter_candidates`` builds a regex in
    which '.' also matches the guessed letter, so it admits words such as
    'aaaaaa' for the board '.a.a.a'. Those words are not reachable states.
    """
    guessed_set = set(guessed)
    length = len(board)

    revealed_positions: Dict[str, set] = {letter: set() for letter in guessed_set}
    for i, ch in enumerate(board):
        if ch != ".":
            revealed_positions.setdefault(ch, set()).add(i)

    out: List[str] = []
    for word in dictionary:
        if len(word) != length:
            continue
        for letter in guessed_set:
            positions = {i for i, ch in enumerate(word) if ch == letter}
            if positions != revealed_positions.get(letter, set()):
                break
        else:
            out.append(word)
    return out


def hit_probabilities(
    candidates: Sequence[str], guessed: frozenset
) -> Dict[str, float]:
    """P(letter occurs in the hidden word) under a uniform posterior.

    Only letters not yet guessed are scored; guessing a letter already guessed
    is a wasted move rather than a probabilistic one.
    """
    total = len(candidates)
    if total == 0:
        return {}
    counts: Dict[str, int] = {}
    for word in candidates:
        for letter in set(word):
            if letter not in guessed:
                counts[letter] = counts.get(letter, 0) + 1
    return {letter: count / total for letter, count in counts.items()}


# --------------------------------------------------------------------------
# Choosers (reference policies)
# --------------------------------------------------------------------------


def choose_max_hit_probability(
    candidates: Sequence[str], guessed: frozenset
) -> Optional[str]:
    """Greedy: the letter most likely to appear. Ties break alphabetically."""
    probs = hit_probabilities(candidates, guessed)
    if not probs:
        return None
    best = max(probs.values())
    return min(letter for letter, p in probs.items() if p == best)


def choose_min_expected_candidates(
    candidates: Sequence[str], guessed: frozenset
) -> Optional[str]:
    """Information-gain: minimise expected size of the surviving candidate set.

    Guessing a letter partitions the candidates by the mask of positions where
    it occurs. Expected remaining size is (1/N) * sum_m |S_m|^2, so minimising
    sum of squared partition sizes is equivalent. Ties break alphabetically.
    """
    if not candidates:
        return None
    available = [
        letter
        for letter in ALPHABET
        if letter not in guessed and any(letter in word for word in candidates)
    ]
    if not available:
        return None

    best_letter: Optional[str] = None
    best_score: Optional[int] = None
    for letter in available:
        partitions: Dict[Tuple[int, ...], int] = {}
        for word in candidates:
            mask = tuple(i for i, ch in enumerate(word) if ch == letter)
            partitions[mask] = partitions.get(mask, 0) + 1
        score = sum(count * count for count in partitions.values())
        if best_score is None or score < best_score:
            best_score, best_letter = score, letter
    return best_letter


CHOOSERS: Dict[str, Chooser] = {
    "max_hit_prob": choose_max_hit_probability,
    "info_gain": choose_min_expected_candidates,
}


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GuessRecord:
    """One guess, scored against the belief state that preceded it."""

    step: int
    letter: str
    board_before: str
    candidates_before: int
    wrong_before: int

    invalid: bool
    repeat: bool
    dominated_miss: bool

    hit: bool
    hit_prob: float
    best_hit_prob: float
    optimal_letter: Optional[str]

    @property
    def counted(self) -> bool:
        """Whether this guess reached the game as a state-changing move."""
        return not (self.invalid or self.repeat)

    @property
    def hit_prob_regret(self) -> float:
        if not self.counted:
            return 0.0
        return max(0.0, self.best_hit_prob - self.hit_prob)

    @property
    def is_optimal(self) -> bool:
        return self.counted and self.hit_prob >= self.best_hit_prob


@dataclass
class TrajectoryReport:
    """Scored replay of one game."""

    word: str
    sample_id: str
    model: str
    steps: List[GuessRecord] = field(default_factory=list)
    won: bool = False
    wrong_guesses: int = 0
    oracle_wrong_guesses: int = 0
    target_in_dictionary: bool = True

    @property
    def n_invalid(self) -> int:
        return sum(1 for s in self.steps if s.invalid)

    @property
    def n_repeat(self) -> int:
        return sum(1 for s in self.steps if s.repeat)

    @property
    def n_dominated(self) -> int:
        return sum(1 for s in self.steps if s.dominated_miss)

    @property
    def n_scored(self) -> int:
        return sum(1 for s in self.steps if s.counted)

    @property
    def n_suboptimal(self) -> int:
        return sum(1 for s in self.steps if s.counted and not s.is_optimal)

    @property
    def mean_hit_prob_regret(self) -> float:
        scored = [s for s in self.steps if s.counted]
        if not scored:
            return 0.0
        return sum(s.hit_prob_regret for s in scored) / len(scored)

    @property
    def wrong_guess_regret(self) -> int:
        return self.wrong_guesses - self.oracle_wrong_guesses


def oracle_play(
    word: str,
    dictionary: Sequence[str],
    chooser: Chooser,
    max_wrong: int,
) -> int:
    """Wrong guesses an oracle solver needs for ``word``. Reference point."""
    guessed: set = set()
    wrong = 0
    while wrong < max_wrong and not set(word) <= guessed:
        board = reveal(word, guessed)
        candidates = consistent_candidates(board, guessed, dictionary)
        letter = chooser(candidates, frozenset(guessed))
        if letter is None:
            remaining = [c for c in ALPHABET if c not in guessed]
            if not remaining:
                break
            letter = remaining[0]
        guessed.add(letter)
        if letter not in word:
            wrong += 1
    return wrong


def replay_trajectory(
    word: str,
    raw_guesses: Sequence[str],
    dictionary: Sequence[str],
    chooser: Chooser = choose_max_hit_probability,
    max_wrong: int = 10,
    sample_id: str = "",
    model: str = "",
) -> TrajectoryReport:
    """Score every guess in ``raw_guesses`` against the belief state.

    ``raw_guesses`` is the sequence as the agent actually emitted it, including
    repeats and malformed guesses. Those are invisible in the eval's stored
    ``guessed_letters`` (``GameState.guess`` returns early on a repeat), which
    is why the trajectory should be recovered from tool calls.
    """
    word = word.lower()

    # An oracle cannot reason about a word absent from its dictionary: the
    # candidate set would empty out mid-game. Flag it and continue.
    target_in_dictionary = word in set(dictionary)
    working_dictionary = list(dictionary)
    if not target_in_dictionary:
        working_dictionary.append(word)

    report = TrajectoryReport(
        word=word,
        sample_id=sample_id,
        model=model,
        target_in_dictionary=target_in_dictionary,
        oracle_wrong_guesses=oracle_play(word, working_dictionary, chooser, max_wrong),
    )

    guessed: List[str] = []
    wrong = 0

    for step, raw in enumerate(raw_guesses):
        letter = (raw or "").strip().lower()
        invalid = len(letter) != 1 or not letter.isalpha()
        repeat = (not invalid) and letter in guessed

        board = reveal(word, guessed)
        candidates = consistent_candidates(board, guessed, working_dictionary)
        probs = hit_probabilities(candidates, frozenset(guessed))
        best_letter = chooser(candidates, frozenset(guessed))
        best_prob = max(probs.values()) if probs else 0.0
        hit_prob = probs.get(letter, 0.0) if not invalid else 0.0

        # A fresh letter in zero candidate words is a guaranteed miss for zero
        # information: provably the wrong move, given this dictionary.
        dominated_miss = (
            not invalid and not repeat and len(candidates) > 0 and hit_prob == 0.0
        )

        hit = (not invalid) and (not repeat) and letter in word

        report.steps.append(
            GuessRecord(
                step=step,
                letter=letter,
                board_before=board,
                candidates_before=len(candidates),
                wrong_before=wrong,
                invalid=invalid,
                repeat=repeat,
                dominated_miss=dominated_miss,
                hit=hit,
                hit_prob=hit_prob,
                best_hit_prob=best_prob,
                optimal_letter=best_letter,
            )
        )

        if invalid or repeat:
            continue

        guessed.append(letter)
        if letter not in word:
            wrong += 1

        if set(word) <= set(guessed):
            report.won = True
            break
        if wrong >= max_wrong:
            break

    report.wrong_guesses = wrong
    return report


# --------------------------------------------------------------------------
# Dictionary loading
# --------------------------------------------------------------------------


def load_wordlist(path: pathlib.Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip().lower() for line in handle if line.strip().isalpha()]


def by_length(dictionary: Sequence[str]) -> Dict[int, List[str]]:
    index: Dict[int, List[str]] = {}
    for word in dictionary:
        index.setdefault(len(word), []).append(word)
    return index
