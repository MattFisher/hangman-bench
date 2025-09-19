"""End-to-end tests for hangman benchmark with mock models."""

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from hangman_bench.hangman import hangman, NUM_ALLOWABLE_EXTRA_MESSAGES


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
                difficulty=1,
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

        # Should have difficulty-1 accuracy of 1.0
        assert "game_scorer.1" in scores
        assert scores["game_scorer.1"] == 1.0

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
                difficulty=5,  # Very hard words
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

        # Should have difficulty-5 accuracy of 0.0
        assert "game_scorer.5" in scores
        assert scores["game_scorer.5"] == 0.0

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

        # Should have difficulty-1 specific scores
        assert "game_scorer.1" in scores

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
                difficulty=1,
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
        mock_outputs = [
            create_letter_guess("a"),
            create_letter_guess("e"),
            # Model stops without completing the game properly
        ] + [ModelOutput.from_content(model="mockllm/model", content="I give up")] * 10

        max_guesses = 4

        log = eval(
            tasks=hangman(
                max_guesses=max_guesses,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            sample_id="apple",
        )[0]

        # Game should be terminated after len("butterfly") + max_guesses + NUM_ALLOWABLE_EXTRA_MESSAGES
        expected_limit = len("butterfly") + max_guesses + NUM_ALLOWABLE_EXTRA_MESSAGES
        assert expected_limit == 18

        assert log.samples is not None
        assert log.samples[0].messages[-1].role == "user"
        assert (
            "Continue by calling hangman_guess('a')" in log.samples[0].messages[-1].text
        )
        assert log.samples[0].limit is not None
        assert log.samples[0].limit.type == "message"
        assert log.samples[0].limit.limit == expected_limit
        # assert log.messages[-1].content == "Game terminated after 10 messages"

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

    def test_hangman_invalid_difficulty(self) -> None:
        """Test that invalid difficulty raises ValueError."""
        with pytest.raises(ValueError, match="Difficulty must be between 1 and 5"):
            hangman(difficulty=6)

        with pytest.raises(ValueError, match="Difficulty must be between 1 and 5"):
            hangman(difficulty=0)

    def test_hangman_task_parameters(self) -> None:
        """Test that task parameters are correctly set."""
        # Test with specific parameters
        task = hangman(
            language="english",
            difficulty=3,
            max_guesses=8,
            allow_word_guesses=True,
            shuffle=False,
        )

        assert task is not None
        assert len(task.dataset) > 0

        # All samples should have difficulty 3
        for sample in task.dataset:
            metadata = sample.metadata or {}
            assert metadata["difficulty"] == 3
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
            if name.startswith("game_scorer.") and name.split(".")[1].isdigit()
        ]
        assert len(difficulty_metrics) > 0

        # Should have stderr metric
        assert "game_scorer.stderr" in all_metrics
