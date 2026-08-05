"""Tests for the oracle replay harness in analysis/oracle.py."""

from __future__ import annotations

import pathlib
import sys

import pytest

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from oracle import (  # noqa: E402
    choose_max_hit_probability,
    consistent_candidates,
    hit_probabilities,
    oracle_play,
    replay_trajectory,
    reveal,
)


class TestBeliefState:
    def test_reveal_shows_only_guessed_letters(self) -> None:
        assert reveal("banana", ["a"]) == ".a.a.a"
        assert reveal("banana", ["a", "n"]) == ".anana"
        assert reveal("banana", []) == "......"

    def test_candidates_exclude_words_hiding_a_guessed_letter(self) -> None:
        """Guessing a letter reveals every occurrence, so none can remain hidden.

        The regex filter in measure_difficulty.py admits 'aaaaaa' for the board
        '.a.a.a' because '.' also matches 'a'. That state is unreachable.
        """
        dictionary = ["cabana", "aaaaaa", "banana", "madama"]
        candidates = consistent_candidates(".a.a.a", ["a"], dictionary)
        assert "aaaaaa" not in candidates
        assert "banana" in candidates
        assert "cabana" in candidates

    def test_candidates_exclude_words_containing_a_missed_letter(self) -> None:
        dictionary = ["cat", "cot", "dog"]
        assert consistent_candidates("...", ["z"], dictionary) == ["cat", "cot", "dog"]
        assert consistent_candidates("...", ["a"], dictionary) == ["cot", "dog"]

    def test_candidates_respect_word_length(self) -> None:
        dictionary = ["cat", "cats", "scatter"]
        assert consistent_candidates("...", [], dictionary) == ["cat"]

    def test_hit_probabilities_are_incidence_over_candidates(self) -> None:
        probs = hit_probabilities(["cat", "cot", "cut"], frozenset())
        assert probs["c"] == pytest.approx(1.0)
        assert probs["t"] == pytest.approx(1.0)
        assert probs["a"] == pytest.approx(1 / 3)

    def test_hit_probabilities_skip_guessed_letters(self) -> None:
        probs = hit_probabilities(["cat", "cot"], frozenset({"c"}))
        assert "c" not in probs

    def test_max_hit_probability_breaks_ties_alphabetically(self) -> None:
        assert choose_max_hit_probability(["cat", "cot", "cut"], frozenset()) == "c"


class TestReplay:
    def test_optimal_play_has_no_regret_and_no_errors(self) -> None:
        dictionary = ["cat", "cot", "cut", "dog", "cog"]
        report = replay_trajectory(
            word="cat",
            raw_guesses=["c", "t", "a"],
            dictionary=dictionary,
            max_wrong=6,
        )
        assert report.won
        assert report.n_dominated == 0
        assert report.n_repeat == 0
        assert report.mean_hit_prob_regret == pytest.approx(0.0)

    def test_repeat_is_flagged_and_does_not_advance_the_game(self) -> None:
        report = replay_trajectory(
            word="cat",
            raw_guesses=["c", "c", "a", "t"],
            dictionary=["cat", "cot"],
            max_wrong=6,
        )
        assert report.n_repeat == 1
        assert report.steps[1].repeat
        assert not report.steps[1].counted
        assert report.won

    def test_invalid_guess_is_flagged(self) -> None:
        report = replay_trajectory(
            word="cat",
            raw_guesses=["ab", "", "3", "c", "a", "t"],
            dictionary=["cat"],
            max_wrong=6,
        )
        assert report.n_invalid == 3
        assert report.won

    def test_dominated_miss_detects_a_provably_dead_letter(self) -> None:
        """Once 'o' misses, only 'cat' survives, so 'u' cannot be in the word."""
        dictionary = ["cat", "cot"]
        report = replay_trajectory(
            word="cat",
            raw_guesses=["c", "o", "u"],
            dictionary=dictionary,
            max_wrong=6,
        )
        assert not report.steps[0].dominated_miss
        assert not report.steps[1].dominated_miss  # 'o' was live at the time
        assert report.steps[2].candidates_before == 1
        assert report.steps[2].dominated_miss

    def test_a_live_letter_is_never_dominated(self) -> None:
        report = replay_trajectory(
            word="cat",
            raw_guesses=["a"],
            dictionary=["cat", "cot"],
            max_wrong=6,
        )
        assert not report.steps[0].dominated_miss

    def test_game_stops_at_the_wrong_guess_budget(self) -> None:
        report = replay_trajectory(
            word="cat",
            raw_guesses=list("zyxwv"),
            dictionary=["cat"],
            max_wrong=3,
        )
        assert report.wrong_guesses == 3
        assert not report.won
        assert len(report.steps) == 3

    def test_target_absent_from_dictionary_is_flagged_not_fatal(self) -> None:
        report = replay_trajectory(
            word="zzz",
            raw_guesses=["z"],
            dictionary=["cat", "cot"],
            max_wrong=6,
        )
        assert not report.target_in_dictionary
        assert report.won

    def test_wrong_guess_regret_is_relative_to_the_oracle(self) -> None:
        dictionary = ["cat", "cot", "cut"]
        report = replay_trajectory(
            word="cat",
            raw_guesses=["z", "q", "c", "a"],
            dictionary=dictionary,
            max_wrong=6,
        )
        expected = oracle_play(
            "cat", dictionary, choose_max_hit_probability, max_wrong=6
        )
        assert report.oracle_wrong_guesses == expected
        assert report.wrong_guess_regret == report.wrong_guesses - expected


class TestLogIngestion:
    """The guess sequence must come from tool calls, not from the store.

    GameState.guess returns early on a repeat, so repeated letters never reach
    guessed_letters and would be invisible to any store-based analysis.
    """

    def test_repeats_are_recovered_from_tool_calls(self) -> None:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        from hangman_bench.hangman import hangman
        from pilot_oracle import extract_trajectories

        def guess(letter: str) -> ModelOutput:
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="hangman_guess",
                tool_arguments={"letter": letter},
            )

        # 'a' is repeated; the store will show it once, the log twice.
        emitted = ["a", "a", "e", "p", "l"]
        log = inspect_eval(
            tasks=hangman(difficulty="v_easy", max_guesses=6, shuffle=False),
            model=get_model(
                "mockllm/model", custom_outputs=[guess(x) for x in emitted]
            ),
            limit=1,
        )[0]

        assert log.status == "success"
        assert log.location is not None

        trajectories = extract_trajectories(pathlib.Path(log.location))
        assert len(trajectories) == 1
        _model, _sample_id, word, guesses = trajectories[0]
        assert word == "apple"
        assert guesses == emitted

        report = replay_trajectory(
            word=word,
            raw_guesses=guesses,
            dictionary=["apple", "ample", "adobe"],
            max_wrong=6,
        )
        assert report.n_repeat == 1
        assert report.won
