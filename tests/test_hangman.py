import pytest

from hangman_bench import hangman
from hangman_bench.datasets import (
    Language,
    WordEntry,
    get_alphabet,
    get_words_by_difficulty,
    get_words_by_language,
)
from hangman_bench.hangman import GameState, _build_system_message


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
    with pytest.raises(ValueError, match=r"Language .* not supported"):
        hangman(language="invalid")


def test_hangman_invalid_difficulty():
    with pytest.raises(ValueError, match="Difficulty must be one of"):
        hangman(difficulty=6)  # Invalid difficulty
    with pytest.raises(ValueError, match="Difficulty must be one of"):
        hangman(difficulty="shmedium")  # Invalid difficulty


def test_strategy_hint_frequency_reproduces_original_prompt():
    """The default variant must be byte-identical to the pre-parameter prompt.

    The pilot runs used the original wording; if the default drifts, later
    runs are no longer comparable with them.
    """
    original = (
        """
    You are playing a game of Hangman. Try to guess the word one letter at a time.
    The current state shows '_' for unguessed letters and reveals correctly guessed letters.
    You have limited guesses before losing. Make smart guesses based on common letter frequencies
    and word patterns in the specified language, and the result of previous guesses.

    Guess one letter at a time using the hangman_guess tool like `hangman_guess("a")`.
    The tool will return:
    - current_state: The word with guessed letters revealed and unguessed letters as '_'
    - remaining_guesses: Number of guesses remaining
    - incorrect_guesses: List of incorrect letters guessed
    - game_over: Whether the game has ended
    - won: Whether the player won (only valid if game_over is True)
    - language: The language of the current word

    Adapt your strategy based on the language.
    """
        """
    You must guess all the letters in the word to win the game.
    You can ONLY win by guessing all the letters in the word, one at a time.
    The game will end automatically when it is over.
    """
    )
    assert _build_system_message(False, "frequency") == original


def test_strategy_hint_variants_differ_only_in_strategy_sentence():
    frequency = _build_system_message(False, "frequency")
    neutral = _build_system_message(False, "neutral")
    belief = _build_system_message(False, "belief")

    assert "common letter frequencies" in frequency
    assert "common letter frequencies" not in neutral
    assert "common letter frequencies" not in belief
    assert "consistent with the" in belief

    # Everything outside the strategy sentence is shared.
    for message in (frequency, neutral, belief):
        assert message.startswith("\n    You are playing a game of Hangman.")
        assert "You have limited guesses before losing." in message
        assert "Adapt your strategy based on the language." in message
        assert "You must guess all the letters in the word to win the game." in message


def test_strategy_hint_variants_create_tasks():
    for variant in ("frequency", "neutral", "belief"):
        task = hangman(strategy_hint=variant)
        assert task is not None


def test_hangman_invalid_strategy_hint():
    with pytest.raises(ValueError, match="Unknown strategy_hint"):
        hangman(strategy_hint="etaoin")


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


class TestGuessRecording:
    """Repeats and malformed guesses must stay visible in the game state.

    GameState.guess returns early on a repeat and rejects malformed input, so
    neither reaches guessed_letters. The attempts list records what was
    actually submitted.
    """

    def test_attempts_records_every_submission(self):
        state = GameState.start("apple", max_guesses=6)
        for letter in ["a", "a", "zz", "p"]:
            state.attempts.append(letter)
        assert state.attempts == ["a", "a", "zz", "p"]

    def test_repeated_attempts_counts_only_duplicates(self):
        state = GameState.start("apple", max_guesses=6)
        state.attempts.extend(["a", "e", "a", "e", "a"])
        assert state.repeated_attempts == ["a", "e", "a"]

    def test_repeated_attempts_ignores_malformed_input(self):
        state = GameState.start("apple", max_guesses=6)
        state.attempts.extend(["ab", "ab"])
        assert state.repeated_attempts == []

    def test_invalid_attempts_flags_non_letters(self):
        state = GameState.start("apple", max_guesses=6)
        state.attempts.extend(["a", "ab", "", "3", "!", "e"])
        assert state.invalid_attempts == ["ab", "", "3", "!"]

    def test_attempts_are_normalised_before_comparison(self):
        state = GameState.start("apple", max_guesses=6)
        state.attempts.extend(["A", " a ", "a"])
        assert state.repeated_attempts == ["a", "a"]
        assert state.invalid_attempts == []

    def test_guess_still_rejects_malformed_letters(self):
        state = GameState.start("apple", max_guesses=6)
        with pytest.raises(ValueError, match="single letter"):
            state.guess("ab")

    def test_letters_outside_the_alphabet_are_invalid(self):
        # str.isalpha() would accept these; a letter outside the language's
        # declared alphabet can never be revealed, so it must not reach the
        # game and cost a life.
        state = GameState.start("apple", max_guesses=6)
        with pytest.raises(ValueError, match="single letter"):
            state.guess("é")
        state.attempts.extend(["é", "ß", "λ"])
        assert state.invalid_attempts == ["é", "ß", "λ"]
        assert state.remaining_guesses == 6

    def test_alphabet_sized_budget_is_unreachable(self):
        # The unlimited-budget protocol rests on this: guessing every letter
        # of the alphabet completes any word before |alphabet| wrong guesses
        # accrue.
        alphabet = get_alphabet(Language.ENGLISH)
        state = GameState.start("apple", max_guesses=len(alphabet))
        for letter in alphabet:
            state.guess(letter)
            if state.game_over:
                break
        assert state.won
        assert state.remaining_guesses > 0

    def test_shipped_wordlist_fits_declared_alphabet(self):
        # The alphabet is declared, not derived; this is the automatic
        # cross-check that the shipped dictionary stays within it.
        from hangman_bench.oracle import DEFAULT_WORDLIST, load_wordlist

        alphabet = set(get_alphabet(Language.ENGLISH))
        stray = {ch for word in load_wordlist(DEFAULT_WORDLIST) for ch in word} - alphabet
        assert not stray

    def test_guess_ignores_repeats_without_costing_a_life(self):
        state = GameState.start("apple", max_guesses=6)
        state.guess("z")
        assert state.remaining_guesses == 5
        state.guess("z")
        assert state.remaining_guesses == 5
        assert state.guessed_letters == ["z"]
