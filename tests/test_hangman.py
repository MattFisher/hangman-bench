import pytest
from hangman_eval import hangman
from hangman_eval.datasets import Language, WordEntry, get_words_by_difficulty, get_words_by_language

def test_get_words_by_language():
    words = get_words_by_language(Language.ENGLISH)
    assert len(words) > 0
    assert all(isinstance(word, WordEntry) for word in words)

def test_get_words_by_difficulty():
    easy_words = get_words_by_difficulty(Language.ENGLISH, 1)
    assert len(easy_words) > 0
    assert all(word.difficulty == 1 for word in easy_words)

def test_hangman_task_creation():
    task = hangman(language="english", difficulty=2, max_guesses=6)
    assert task is not None
    assert len(task.dataset) > 0

def test_hangman_invalid_language():
    with pytest.raises(ValueError):
        hangman(language="invalid_language")

def test_hangman_invalid_difficulty():
    with pytest.raises(ValueError):
        hangman(difficulty=6)  # Invalid difficulty
    with pytest.raises(ValueError):
        hangman(difficulty=0)   # Invalid difficulty
