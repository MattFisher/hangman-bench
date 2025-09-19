"""
Word datasets for the hangman game evaluation.

Each dataset contains words with their difficulty ratings specifically for hangman gameplay.
Difficulty is rated on a scale from 1-5:
1: Very Easy - Common words with many vowels and repeated letters
2: Easy - Common words with standard letter distributions
3: Medium - Less common words or those with less common letters
4: Hard - Words with uncommon letter patterns or rare letters
5: Very Hard - Words with rare letters, unusual patterns, or misleading letter distributions
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal


class Language(str, Enum):
    """Supported languages for hangman words."""

    ENGLISH = "english"
    # Add more languages as needed, e.g.:
    # SPANISH = "spanish"
    # FRENCH = "french"
    # GERMAN = "german"


Difficulty = Literal["v_easy", "easy", "medium", "hard", "v_hard"]


@dataclass
class WordEntry:
    """A word entry in the hangman dataset."""

    word: str
    difficulty: Difficulty


# English word dataset with difficulty ratings for hangman
ENGLISH_WORDS = [
    # Difficulty v_easy (Very Easy)
    WordEntry("apple", "v_easy"),
    WordEntry("happy", "v_easy"),
    WordEntry("banana", "v_easy"),
    WordEntry("letter", "v_easy"),
    WordEntry("summer", "v_easy"),
    WordEntry("yellow", "v_easy"),
    WordEntry("people", "v_easy"),
    WordEntry("window", "v_easy"),
    WordEntry("school", "v_easy"),
    WordEntry("family", "v_easy"),
    WordEntry("coffee", "v_easy"),
    WordEntry("cookie", "v_easy"),
    WordEntry("paper", "v_easy"),
    WordEntry("puppy", "v_easy"),
    WordEntry("kitten", "v_easy"),
    WordEntry("butter", "v_easy"),
    WordEntry("pizza", "v_easy"),
    WordEntry("puzzle", "v_easy"),
    WordEntry("bubble", "v_easy"),
    WordEntry("butterfly", "v_easy"),
    # Difficulty easy (Easy)
    WordEntry("garden", "easy"),
    WordEntry("market", "easy"),
    WordEntry("pocket", "easy"),
    WordEntry("rocket", "easy"),
    WordEntry("guitar", "easy"),
    WordEntry("jacket", "easy"),
    WordEntry("pencil", "easy"),
    WordEntry("rabbit", "easy"),
    WordEntry("turtle", "easy"),
    WordEntry("monkey", "easy"),
    WordEntry("orange", "easy"),
    WordEntry("purple", "easy"),
    WordEntry("silver", "easy"),
    WordEntry("winter", "easy"),
    WordEntry("zebra", "easy"),
    WordEntry("bridge", "easy"),
    WordEntry("planet", "easy"),
    WordEntry("forest", "easy"),
    WordEntry("castle", "easy"),
    WordEntry("river", "easy"),
    # Difficulty medium (Medium)
    WordEntry("bicycle", "medium"),
    WordEntry("camera", "medium"),
    WordEntry("diamond", "medium"),
    WordEntry("elephant", "medium"),
    WordEntry("fishing", "medium"),
    WordEntry("giraffe", "medium"),
    WordEntry("holiday", "medium"),
    WordEntry("jungle", "medium"),
    WordEntry("kangaroo", "medium"),
    WordEntry("ladder", "medium"),
    WordEntry("mountain", "medium"),
    WordEntry("notebook", "medium"),
    WordEntry("octopus", "medium"),
    WordEntry("penguin", "medium"),
    WordEntry("question", "medium"),
    WordEntry("volcano", "medium"),
    WordEntry("postcard", "medium"),
    WordEntry("harvest", "medium"),
    WordEntry("compass", "medium"),
    WordEntry("lantern", "medium"),
    # Difficulty hard (Hard)
    WordEntry("awkward", "hard"),
    WordEntry("bagpipes", "hard"),
    WordEntry("croquet", "hard"),
    WordEntry("dwarves", "hard"),
    WordEntry("espionage", "hard"),
    WordEntry("flopping", "hard"),
    WordEntry("gazebo", "hard"),
    WordEntry("haiku", "hard"),
    WordEntry("ivory", "hard"),
    WordEntry("jukebox", "hard"),
    WordEntry("kayak", "hard"),
    WordEntry("lymph", "hard"),
    WordEntry("matrix", "hard"),
    WordEntry("zephyr", "hard"),
    WordEntry("jigsaw", "hard"),
    WordEntry("whiskey", "hard"),
    WordEntry("quibble", "hard"),
    WordEntry("pyjamas", "hard"),
    WordEntry("jackpot", "hard"),
    WordEntry("oxidize", "hard"),
    # Difficulty v_hard (Very Hard)
    WordEntry("buzzard", "v_hard"),
    WordEntry("crypt", "v_hard"),
    WordEntry("duplex", "v_hard"),
    WordEntry("foxglove", "v_hard"),
    WordEntry("galvanize", "v_hard"),
    WordEntry("hyphen", "v_hard"),
    WordEntry("ivy", "v_hard"),
    WordEntry("jinx", "v_hard"),
    WordEntry("kiosk", "v_hard"),
    WordEntry("larynx", "v_hard"),
    WordEntry("marquis", "v_hard"),
    WordEntry("nymph", "v_hard"),
    WordEntry("onyx", "v_hard"),
    WordEntry("pixel", "v_hard"),
    WordEntry("quartz", "v_hard"),
    WordEntry("rhythm", "v_hard"),
    WordEntry("sphinx", "v_hard"),
    WordEntry("topaz", "v_hard"),
    WordEntry("vortex", "v_hard"),
    WordEntry("zodiac", "v_hard"),
]

# Map of language to word list
LANGUAGE_WORDS = {
    Language.ENGLISH: ENGLISH_WORDS,
}


def get_words_by_language(language: Language) -> List[WordEntry]:
    """Get the word dataset for a specific language.

    Args:
        language: The language to get words for

    Returns:
        List of WordEntry objects for the specified language

    Raises:
        ValueError: If the language is not supported
    """
    if language not in LANGUAGE_WORDS:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages: {', '.join([lang.value for lang in LANGUAGE_WORDS.keys()])}"
        )
    return LANGUAGE_WORDS[language].copy()


def get_words_by_difficulty(language: Language, difficulty: Difficulty) -> List[WordEntry]:
    """Get words of a specific difficulty level for a language.

    Args:
        language: The language to get words for
        difficulty: Difficulty level label (v_easy, easy, medium, hard, v_hard)

    Returns:
        List of WordEntry objects matching the criteria
    """
    allowed: tuple[Difficulty, ...] = ("v_easy", "easy", "medium", "hard", "v_hard")
    if difficulty not in allowed:
        allowed_str = ", ".join(allowed)
        raise ValueError(f"Difficulty must be one of: {allowed_str}")

    words = get_words_by_language(language)
    return [word for word in words if word.difficulty == difficulty]
