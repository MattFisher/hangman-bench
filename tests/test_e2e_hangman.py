"""End-to-end tests for hangman benchmark with mock models."""

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from hangman_bench.datasets import ENGLISH_WORDS
from hangman_bench.hangman import hangman, _calculate_message_limit


def create_letter_guess(letter: str) -> ModelOutput:
    """Helper function to create a ModelOutput for a hangman letter guess."""
    return ModelOutput.for_tool_call(
        model="mockllm/model",
        tool_name="hangman_guess",
        tool_arguments={"letter": letter},
    )


class TestHangmanE2E:
    """End-to-end tests for hangman benchmark using mock models."""

    def test_hangman_win_easy_word(self) -> None:
        """Test hangman win scenario with easy word and optimal guesses."""
        # Mock model outputs for winning "apple" (difficulty 1)
        # Strategy: guess common vowels first, then consonants
        mock_outputs = [
            create_letter_guess("a"),
            create_letter_guess("e"),
            create_letter_guess("p"),
            create_letter_guess("l"),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty="v_easy",
                max_guesses=6,
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            limit=1,  # Test just one word
        )[0]

        assert log.status == "success"
        assert log.results is not None

        # Check that the model won the game
        scores = {}
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                scores[f"{score.name}.{metric_name}"] = metric.value

        # Should have overall accuracy of 1.0 (won)
        assert "game_scorer.all" in scores
        assert scores["game_scorer.all"] == 1.0

        # Should have difficulty-v_easy accuracy of 1.0
        assert "game_scorer.v_easy" in scores
        assert scores["game_scorer.v_easy"] == 1.0

    def test_hangman_loss_hard_word(self) -> None:
        """Test hangman loss scenario with hard word and poor guesses."""
        # Mock model outputs for losing on a difficult word
        # Strategy: guess uncommon letters that likely won't be in the word
        mock_outputs = [
            create_letter_guess("w"),
            create_letter_guess("q"),
            create_letter_guess("x"),
            create_letter_guess("j"),
            create_letter_guess("k"),
            # Model should lose after 5 guesses
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty="v_hard",  # Very hard words
                max_guesses=5,  # Limited guesses
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            sample_id="buzzard",
        )[0]

        assert log.status == "success"
        assert log.results is not None

        scores = {}
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                scores[f"{score.name}.{metric_name}"] = metric.value

        # Should have overall accuracy of 0.0 (lost)
        assert "game_scorer.all" in scores
        assert scores["game_scorer.all"] == 0.0

        # Should have difficulty-v_hard accuracy of 0.0
        assert "game_scorer.v_hard" in scores
        assert scores["game_scorer.v_hard"] == 0.0

    def test_hangman_mixed_results(self) -> None:
        """Test hangman with mixed win/loss results."""
        # Mock outputs for multiple games with mixed results
        # This will test multiple samples if we increase the limit
        mock_outputs = [
            # First game - winning strategy
            create_letter_guess("a"),
            create_letter_guess("p"),
            create_letter_guess("l"),
            create_letter_guess("e"),
            # Second game - losing strategy
            create_letter_guess("z"),
            create_letter_guess("q"),
            create_letter_guess("x"),
            create_letter_guess("j"),
            create_letter_guess("k"),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                max_guesses=5,
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            sample_id=["apple", "happy"],
            max_samples=1,  # prevent parallel execution
        )[0]

        assert log.status == "success"
        assert log.results is not None

        scores = {}
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                scores[f"{score.name}.{metric_name}"] = metric.value

        # Should have overall accuracy between 0.0 and 1.0 (mixed results)
        assert "game_scorer.all" in scores
        assert 0.0 <= scores["game_scorer.all"] <= 1.0

        # Should have difficulty-v_easy specific scores
        assert "game_scorer.v_easy" in scores

    def test_hangman_word_guess_allowed_win(self) -> None:
        """Test hangman with word guessing allowed - early correct word guess."""
        mock_outputs = [
            create_letter_guess("a"),
            create_letter_guess("e"),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "apple"},
            ),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                allow_word_guesses=True,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            sample_id="apple",
        )[0]

        assert log.status == "success"
        assert log.results is not None

        scores = {}
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                scores[f"{score.name}.{metric_name}"] = metric.value

        # Should win with early word guess
        assert "game_scorer.all" in scores
        assert scores["game_scorer.all"] == 1.0

    def test_hangman_word_guess_allowed_wrong_word(self) -> None:
        """Test hangman with word guessing allowed - wrong word guess."""
        mock_outputs = [
            create_letter_guess("a"),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "wrong"},
            ),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty="v_easy",
                allow_word_guesses=True,
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            limit=1,
        )[0]

        assert log.status == "success"
        assert log.results is not None

        scores = {}
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                scores[f"{score.name}.{metric_name}"] = metric.value

        # Should lose with wrong word guess
        assert "game_scorer.all" in scores
        assert scores["game_scorer.all"] == 0.0

    def test_hangman_incomplete_game(self) -> None:
        """Test hangman when game doesn't complete (model stops early)."""
        # Enough refusals to actually reach the message limit. Too few and the
        # mock runs out of outputs first, which errors the sample instead of
        # exercising the limit.
        mock_outputs = [
            create_letter_guess("a"),
            create_letter_guess("e"),
            # Model stops without completing the game properly
        ] + [ModelOutput.from_content(model="mockllm/model", content="I give up")] * 30

        max_guesses = 4
        # The task derives its limit from the longest word in the whole dataset,
        # not from the sample under test, so this must not hardcode a word:
        # 4 * (longest word + max_guesses) + NUM_ALLOWABLE_EXTRA_MESSAGES
        longest_word = max(len(entry.word) for entry in ENGLISH_WORDS)
        expected_limit = _calculate_message_limit(longest_word, max_guesses)
        assert expected_limit == 61

        log = eval(
            tasks=hangman(
                max_guesses=max_guesses,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            sample_id="apple",
        )[0]

        assert log.samples is not None
        messages = log.samples[0].messages

        # The solver should have nudged the model to keep playing. Which role
        # ends the transcript depends on where the limit happens to bite, so
        # assert the nudge was issued rather than that it came last.
        assert any(
            "Continue the game by calling the hangman_guess tool" in (m.text or "")
            for m in messages
        )
        assert log.samples[0].limit is not None
        assert log.samples[0].limit.type == "message"
        assert log.samples[0].limit.limit == expected_limit

        assert log.status == "success"
        assert log.results is not None

        scores = {}
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                scores[f"{score.name}.{metric_name}"] = metric.value

        # Incomplete game should result in loss (score = 0.0)
        assert "game_scorer.all" in scores
        assert scores["game_scorer.all"] == 0.0

    def test_hangman_invalid_language(self) -> None:
        """Test that invalid language raises ValueError."""
        with pytest.raises(
            ValueError, match="Language 'invalid_language' not supported"
        ):
            hangman(language="invalid_language")

    def test_hangman_task_parameters(self) -> None:
        """Test that task parameters are correctly set."""
        # Test with specific parameters
        task = hangman(
            language="english",
            difficulty="medium",
            max_guesses=8,
            allow_word_guesses=True,
            shuffle=False,
        )

        assert task is not None
        assert len(task.dataset) > 0

        # All samples should have difficulty 3
        for sample in task.dataset:
            metadata = sample.metadata or {}
            assert metadata["difficulty"] == "medium"
            assert metadata["max_guesses"] == 8
            assert metadata["language"] == "english"
            assert metadata["allow_word_guesses"] is True

    def test_hangman_scoring_metrics(self) -> None:
        """Test that scoring includes expected metrics."""
        mock_outputs = [
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "apple"},
                content="Quick win",
            ),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty=None,  # Mixed difficulties
                allow_word_guesses=True,
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            limit=1,
        )[0]

        assert log.status == "success"
        assert log.results is not None

        # Check that we have grouped scoring by difficulty
        score_names = [score.name for score in log.results.scores]

        # Should have game_scorer
        assert "game_scorer" in score_names

        # Check available metrics
        all_metrics = {}
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                all_metrics[f"{score.name}.{metric_name}"] = metric.value

        # Should have overall accuracy ("all" group)
        assert "game_scorer.all" in all_metrics

        # Should have difficulty-specific metrics
        difficulty_metrics = [
            name
            for name in all_metrics.keys()
            if name.startswith("game_scorer.")
            and name.split(".")[1] not in ("all", "stderr")
        ]
        assert len(difficulty_metrics) > 0

        # Should have stderr metric
        assert "game_scorer.stderr" in all_metrics


class TestMalformedGuessHandling:
    """A malformed guess must not abort the sample.

    Before this was handled, the ValueError raised inside GameState.guess
    propagated out of the tool and errored the whole sample, so the game was
    dropped from the results rather than being scored.
    """

    def test_malformed_guess_does_not_error_the_sample(self) -> None:
        mock_outputs = [
            create_letter_guess("a"),
            create_letter_guess("ab"),  # malformed
            create_letter_guess("e"),
            create_letter_guess("p"),
            create_letter_guess("l"),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty="v_easy",
                max_guesses=6,
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            limit=1,
        )[0]

        assert log.status == "success"
        assert log.samples is not None
        sample = log.samples[0]
        assert sample.error is None

        metadata = sample.scores["game_scorer"].metadata
        assert metadata["num_invalid_guesses"] == 1
        assert "ab" in metadata["attempts"]
        # The malformed guess must not have cost a life or entered the game.
        assert "ab" not in metadata["guessed_letters"]

    def test_repeated_guess_is_recorded_but_costs_no_life(self) -> None:
        mock_outputs = [
            create_letter_guess("a"),
            create_letter_guess("a"),  # repeat
            create_letter_guess("e"),
            create_letter_guess("p"),
            create_letter_guess("l"),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty="v_easy",
                max_guesses=6,
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            limit=1,
        )[0]

        assert log.status == "success"
        assert log.samples is not None
        metadata = log.samples[0].scores["game_scorer"].metadata

        assert metadata["num_repeated_guesses"] == 1
        assert metadata["attempts"].count("a") == 2
        assert metadata["guessed_letters"].count("a") == 1
        assert metadata["remaining_guesses"] == 6  # no wrong guesses at all
