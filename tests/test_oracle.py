"""Tests for the oracle replay harness in analysis/oracle.py."""

from __future__ import annotations

import pathlib
import sys

import pytest

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from hangman_bench.oracle import (  # noqa: E402
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

    def test_non_ascii_guess_is_invalid_not_dominated(self) -> None:
        report = replay_trajectory(
            word="cat",
            raw_guesses=["é", "c", "a", "t"],
            dictionary=["cat", "cot"],
            max_wrong=6,
        )
        assert report.steps[0].invalid
        assert not report.steps[0].dominated_miss
        assert report.n_invalid == 1
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

    def test_excess_wrong_guesses_is_relative_to_the_oracle(self) -> None:
        dictionary = ["cat", "cot", "cut"]
        report = replay_trajectory(
            word="cat",
            raw_guesses=["z", "q", "c", "a"],
            dictionary=dictionary,
            max_wrong=6,
        )
        expected = oracle_play("cat", dictionary, choose_max_hit_probability, max_wrong=6)
        assert report.oracle_wrong_guesses == expected
        assert report.excess_wrong_guesses == report.wrong_guesses - expected


class TestLogIngestion:
    """The guess sequence must come from tool calls, not from the store.

    GameState.guess returns early on a repeat, so repeated letters never reach
    guessed_letters and would be invisible to any store-based analysis.
    """

    def test_repeats_are_recovered_from_tool_calls(self) -> None:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model
        from pilot_oracle import extract_trajectories

        from hangman_bench.hangman import hangman

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
            model=get_model("mockllm/model", custom_outputs=[guess(x) for x in emitted]),
            limit=1,
        )[0]

        assert log.status == "success"
        assert log.location is not None

        trajectories = extract_trajectories(pathlib.Path(log.location))
        assert len(trajectories) == 1
        game = trajectories[0]
        assert game.word == "apple"
        assert game.guesses == emitted
        assert game.max_guesses == 6

        report = replay_trajectory(
            word=game.word,
            raw_guesses=game.guesses,
            dictionary=["apple", "ample", "adobe"],
            max_wrong=game.max_guesses,
        )
        assert report.n_repeat == 1
        assert report.won


class TestReferenceAgents:
    """Reference agents must be deterministic across processes.

    Seeding from hash(word) would not be: Python randomises str hashes per
    process, so calibration numbers would drift between runs.
    """

    def test_sloppy_agent_is_reproducible(self) -> None:
        from pilot_oracle import agent_sloppy

        dictionary = ["cat", "cot", "cut", "cog", "dog", "log"]
        first = agent_sloppy("cat", dictionary, 6)
        second = agent_sloppy("cat", dictionary, 6)
        assert first == second

    def test_optimal_agent_makes_no_dominated_moves(self) -> None:
        from pilot_oracle import agent_optimal

        dictionary = ["cat", "cot", "cut", "cog", "dog", "log"]
        guesses = agent_optimal("cat", dictionary, 6)
        report = replay_trajectory(
            word="cat", raw_guesses=guesses, dictionary=dictionary, max_wrong=6
        )
        assert report.n_dominated == 0
        assert report.mean_hit_prob_regret == pytest.approx(0.0)

    def test_frequency_agent_ignores_evidence(self) -> None:
        """Fixed etaoin order should make provably dead guesses."""
        from pilot_oracle import agent_frequency

        dictionary = ["cat", "cot", "cut", "cog", "dog", "log"]
        guesses = agent_frequency("cat", dictionary, 6)
        report = replay_trajectory(
            word="cat", raw_guesses=guesses, dictionary=dictionary, max_wrong=6
        )
        assert report.n_dominated > 0


class TestOracleScorer:
    """The oracle metrics must be available as an Inspect scorer.

    Being a real Scorer is what allows `inspect score <log>` to add oracle
    metrics to logs that were produced before the scorer existed.
    """

    @staticmethod
    def _run(letters: list[str], **task_kwargs: object):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        from hangman_bench.hangman import hangman

        outputs = [
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="hangman_guess",
                tool_arguments={"letter": letter},
            )
            for letter in letters
        ]
        return inspect_eval(
            tasks=hangman(difficulty="v_easy", max_guesses=6, shuffle=False, **task_kwargs),
            model=get_model("mockllm/model", custom_outputs=outputs),
            limit=1,
        )[0]

    def test_both_scorers_run_and_report_metrics(self) -> None:
        log = self._run(["a", "e", "p", "l"])
        assert log.status == "success"
        assert log.samples is not None

        scores = log.samples[0].scores
        assert "game_scorer" in scores
        assert "oracle_scorer" in scores

        value = scores["oracle_scorer"].value
        assert isinstance(value, dict)
        for key in (
            "excess_wrong_guesses",
            "hit_prob_regret",
            "dominated_rate",
            "suboptimal_rate",
            "repeat_rate",
            "invalid_rate",
        ):
            assert key in value

    def test_scorer_counts_repeats_and_invalid_guesses(self) -> None:
        log = self._run(["a", "a", "zz", "e", "p", "l"])
        assert log.samples is not None
        metadata = log.samples[0].scores["oracle_scorer"].metadata

        assert metadata["num_repeat"] == 1
        assert metadata["num_invalid"] == 1
        assert metadata["guesses_emitted"] == 6
        assert len(metadata["per_guess"]) == 6

    def test_oracle_can_be_disabled(self) -> None:
        log = self._run(["a", "e", "p", "l"], oracle=False)
        assert log.samples is not None
        assert "oracle_scorer" not in log.samples[0].scores

    def test_scorer_can_be_applied_to_an_existing_log(self) -> None:
        """A log scored without the oracle can have it added afterwards."""
        from inspect_ai import score
        from inspect_ai.log import read_eval_log

        from hangman_bench.hangman import oracle_scorer

        log = self._run(["e", "t", "a", "o", "p", "l"], oracle=False)
        assert log.location is not None
        original = read_eval_log(log.location)
        assert "oracle_scorer" not in (original.samples or [])[0].scores

        rescored = score(original, oracle_scorer(), action="append", display="none")
        assert rescored.samples is not None
        scores = rescored.samples[0].scores
        # The original score is preserved and the oracle metrics are added.
        assert "game_scorer" in scores
        assert isinstance(scores["oracle_scorer"].value, dict)

    def test_unknown_strategy_fails_at_construction(self) -> None:
        from hangman_bench.hangman import oracle_scorer

        with pytest.raises(ValueError, match="Unknown strategy"):
            oracle_scorer(strategy="nonsense")

    def test_missing_wordlist_fails_at_construction(self) -> None:
        from hangman_bench.hangman import oracle_scorer

        with pytest.raises(FileNotFoundError, match="Wordlist not found"):
            oracle_scorer(wordlist="/nonexistent/words.txt")


class TestReviewFixes:
    """Regressions for three issues raised in review of the oracle harness."""

    def test_reference_letter_and_its_probability_agree(self) -> None:
        """Under info_gain, the chooser's own move must not score as suboptimal.

        best_hit_prob previously came from the maximum hit probability while
        optimal_letter came from the configured chooser, so info_gain — which
        trades hit probability for a better partition — contradicted itself.
        """
        from hangman_bench.oracle import CHOOSERS, choose_min_expected_candidates

        dictionary = ["cat", "cot", "cut", "cog", "dog", "log", "bat", "bag"]
        reference = choose_min_expected_candidates(dictionary, frozenset())
        assert reference is not None

        report = replay_trajectory(
            word="cat",
            raw_guesses=[reference],
            dictionary=dictionary,
            chooser=CHOOSERS["info_gain"],
            max_wrong=6,
        )
        step = report.steps[0]
        assert step.optimal_letter == reference
        assert step.is_optimal
        assert step.hit_prob_regret == pytest.approx(0.0)

    def test_max_hit_prob_scoring_is_unchanged(self) -> None:
        """The default strategy must behave exactly as before the fix."""
        dictionary = ["cat", "cot", "cut", "cog", "dog", "log"]
        report = replay_trajectory(
            word="cat", raw_guesses=["z"], dictionary=dictionary, max_wrong=6
        )
        step = report.steps[0]
        probs = hit_probabilities(dictionary, frozenset())
        assert step.best_hit_prob == pytest.approx(max(probs.values()))

    def test_recorded_outcome_overrides_the_replay(self) -> None:
        report = replay_trajectory(
            word="apple",
            raw_guesses=["a", "e"],
            dictionary=["apple", "ample"],
            max_wrong=6,
        )
        assert not report.won
        assert not report.final_won
        report.recorded_won = True
        assert report.final_won
        assert not report.won  # the replay's own verdict is left intact


class TestLogMetadataIsHonoured:
    """The replay must use the limit and outcome the game was actually played under."""

    @staticmethod
    def _run(letters, **kwargs):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        from hangman_bench.hangman import hangman

        outputs = [
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="hangman_guess",
                tool_arguments={"letter": letter},
            )
            for letter in letters
        ]
        return inspect_eval(
            tasks=hangman(difficulty="v_easy", shuffle=False, **kwargs),
            model=get_model("mockllm/model", custom_outputs=outputs),
            limit=1,
        )[0]

    def test_per_sample_max_guesses_is_used(self) -> None:
        """A log run at max_guesses=15 must not be replayed at the CLI default."""
        from pilot_oracle import extract_trajectories

        # 12 wrong guesses then a win: legal only under the logged limit of 15.
        letters = list("bcdfghijkmnq") + ["a", "p", "l", "e"]
        log = self._run(letters, max_guesses=15)
        assert log.location is not None

        games = extract_trajectories(pathlib.Path(log.location), default_max_guesses=10)
        assert len(games) == 1
        game = games[0]
        assert game.max_guesses == 15

        report = replay_trajectory(
            word=game.word,
            raw_guesses=game.guesses,
            dictionary=["apple", "ample", "adobe"],
            max_wrong=game.max_guesses,
        )
        assert report.won
        assert report.wrong_guesses == 12

    def test_default_is_only_a_fallback(self) -> None:
        """Metadata wins; the CLI value applies only when the log omits it."""
        from pilot_oracle import extract_trajectories

        log = self._run(["a", "e", "p", "l"], max_guesses=6)
        assert log.location is not None
        games = extract_trajectories(pathlib.Path(log.location), default_max_guesses=99)
        assert games[0].max_guesses == 6

    def test_word_submission_outcome_is_preserved(self) -> None:
        """A win by submitting the full word must not be replayed as a loss."""
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model
        from pilot_oracle import extract_trajectories

        from hangman_bench.hangman import hangman

        def guess(letter: str) -> ModelOutput:
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="hangman_guess",
                tool_arguments={"letter": letter},
            )

        log = inspect_eval(
            tasks=hangman(
                difficulty="v_easy",
                max_guesses=6,
                shuffle=False,
                allow_word_guesses=True,
            ),
            model=get_model(
                "mockllm/model",
                custom_outputs=[
                    guess("a"),
                    guess("e"),
                    ModelOutput.for_tool_call(
                        model="mockllm/model",
                        tool_name="submit",
                        tool_arguments={"answer": "apple"},
                    ),
                ],
            ),
            limit=1,
        )[0]

        assert log.samples is not None
        sample = log.samples[0]
        assert sample.scores["game_scorer"].value == "C"
        # The scorer's own view must agree with the eval.
        assert sample.scores["oracle_scorer"].metadata["won"] is True

        # And so must the batch reader, which only sees the letter guesses.
        assert log.location is not None
        game = extract_trajectories(pathlib.Path(log.location))[0]
        assert game.guesses == ["a", "e"]
        assert game.recorded_won is True

        report = replay_trajectory(
            word=game.word,
            raw_guesses=game.guesses,
            dictionary=["apple", "ample"],
            max_wrong=game.max_guesses,
        )
        report.recorded_won = game.recorded_won
        assert not report.won  # letters alone do not complete the word
        assert report.final_won  # but the game was genuinely won
