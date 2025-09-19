import pytest
from hangman_bench import hangman
from hangman_bench.datasets import (
    Language,
    WordEntry,
    get_words_by_difficulty,
    get_words_by_language,
)


def test_get_words_by_language():
    words = get_words_by_language(Language.ENGLISH)
    assert len(words) > 0
    assert all(isinstance(word, WordEntry) for word in words)


def test_get_words_by_difficulty():
    easy_words = get_words_by_difficulty(Language.ENGLISH, "v_easy")
    assert len(easy_words) > 0
    assert all(word.difficulty == "v_easy" for word in easy_words)


def test_task_creation_with_defaults():
    task = hangman()
    assert task is not None
    assert len(task.dataset) > 0


def test_task_creation_with_difficulty():
    task = hangman(difficulty="medium")
    assert len(task.dataset) > 0
    for sample in task.dataset:
        metadata = sample.metadata or {}
        assert metadata["difficulty"] == "medium"


def test_hangman_task_creation_with_parameters():
    task = hangman(language="english", difficulty="easy", max_guesses=6)
    assert task is not None
    assert len(task.dataset) > 0


def test_hangman_invalid_language():
    with pytest.raises(ValueError, match="Language .* not supported"):
        hangman(language="invalid")


def test_hangman_invalid_difficulty():
    with pytest.raises(ValueError, match="Difficulty must be one of"):
        hangman(difficulty=6)  # Invalid difficulty
    with pytest.raises(ValueError, match="Difficulty must be one of"):
        hangman(difficulty="shmedium")  # Invalid difficulty


def test_dataset_structure():
    """Test that the dataset has the expected structure."""
    task = hangman(language="english", difficulty="v_easy")

    assert len(task.dataset) > 0

    for sample in task.dataset:
        # Check sample has required fields
        assert sample.input is not None
        assert sample.target is not None
        assert sample.metadata is not None

        # Check metadata structure
        metadata = sample.metadata
        assert "word" in metadata
        assert "difficulty" in metadata
        assert "language" in metadata
        assert "max_guesses" in metadata

        # Validate metadata values
        assert isinstance(metadata["word"], str)
        assert len(metadata["word"]) > 0
        assert metadata["difficulty"] == "v_easy"
        assert metadata["language"] == "english"
        assert isinstance(metadata["max_guesses"], int)
        assert metadata["max_guesses"] > 0
