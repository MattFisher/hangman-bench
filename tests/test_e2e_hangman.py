"""End-to-end tests for hangman benchmark with mock models."""

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from hangman_bench import hangman


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
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "GG"},
            ),
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
            create_letter_guess("z"),
            create_letter_guess("q"),
            create_letter_guess("x"),
            create_letter_guess("j"),
            create_letter_guess("k"),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "GG"},
                content="Game lost!",
            ),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty=5,  # Very hard words
                max_guesses=5,  # Limited guesses
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

        # Should have overall accuracy of 0.0 (lost)
        assert "game_scorer.all" in scores
        assert scores["game_scorer.all"] == 0.0

        # Should have difficulty-5 accuracy of 0.0
        assert "game_scorer.5" in scores
        assert scores["game_scorer.5"] == 0.0

    def test_hangman_mixed_results_medium_difficulty(self) -> None:
        """Test hangman with mixed win/loss results on medium difficulty."""
        # Mock outputs for multiple games with mixed results
        # This will test multiple samples if we increase the limit
        mock_outputs = [
            # First game - winning strategy
            ModelOutput.from_content(
                model="mockllm/model",
                content='Starting with vowel: hangman_guess("e")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='Another vowel: hangman_guess("a")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='Common consonant: hangman_guess("r")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='Another common one: hangman_guess("t")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='Trying: hangman_guess("n")',
            ),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "GG"},
                content="Game complete",
            ),
            # Second game - losing strategy
            ModelOutput.from_content(
                model="mockllm/model",
                content='Bad guess: hangman_guess("z")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='Another bad guess: hangman_guess("q")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='Still guessing poorly: hangman_guess("x")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='More bad guesses: hangman_guess("j")',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='Final bad guess: hangman_guess("k")',
            ),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "GG"},
                content="Lost",
            ),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty=3,
                max_guesses=5,
                shuffle=False,
            ),
            model=get_model("mockllm/model", custom_outputs=mock_outputs),
            limit=2,  # Test two games
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

        # Should have difficulty-3 specific scores
        assert "game_scorer.3" in scores

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
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "GG"},
            ),
        ]

        log = eval(
            tasks=hangman(
                language="english",
                difficulty=2,
                max_guesses=10,
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
