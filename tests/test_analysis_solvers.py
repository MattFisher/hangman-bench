"""Tests for the candidate filtering used by the difficulty solvers."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

MEASURE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "analysis" / "measure_difficulty.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("measure_difficulty", MEASURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_difficulty"] = module
    spec.loader.exec_module(module)
    return module


measure_difficulty = _load_module()
filter_candidates = measure_difficulty.filter_candidates
solve_with_strategy = measure_difficulty.solve_with_strategy
best_move_coverage = measure_difficulty.best_move_coverage


class TestFilterCandidates:
    def test_excludes_words_hiding_an_extra_occurrence(self) -> None:
        """Guessing 'a' reveals every 'a', so 'aaaaaa' cannot match '.a.a.a'.

        A plain regex over the board admits it, because '.' also matches the
        guessed letter. That state is unreachable in a real game.
        """
        dictionary = ["cabana", "aaaaaa", "banana", "madama"]
        candidates = filter_candidates(".a.a.a", [], dictionary)
        assert "aaaaaa" not in candidates
        assert "banana" in candidates
        assert "cabana" in candidates

    def test_excludes_words_containing_a_known_absent_letter(self) -> None:
        dictionary = ["cat", "cot", "dog"]
        assert filter_candidates("...", ["a"], dictionary) == ["cot", "dog"]

    def test_keeps_words_matching_revealed_positions(self) -> None:
        dictionary = ["cat", "cot", "cut", "dog"]
        assert filter_candidates("c.t", [], dictionary) == ["cat", "cot", "cut"]

    def test_respects_word_length(self) -> None:
        dictionary = ["cat", "cats", "scatter"]
        assert filter_candidates("...", [], dictionary) == ["cat"]

    def test_revealed_letter_must_occupy_exactly_its_positions(self) -> None:
        """With 'd' and 'e' guessed, 'deed' would show an 'e' at index 2 too."""
        dictionary = ["deed", "dead"]
        assert filter_candidates("de.d", [], dictionary) == ["dead"]


class TestSolverConvergence:
    def test_solver_converges_on_a_word_in_its_dictionary(self) -> None:
        dictionary = ["cat", "cot", "cut", "dog", "cog"]
        result = solve_with_strategy("cat", dictionary, best_move_coverage)
        assert result.wrong_guesses <= 2
        assert result.total_guesses > 0
