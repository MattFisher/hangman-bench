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
from typing import List


class Language(str, Enum):
    """Supported languages for hangman words."""

    ENGLISH = "english"
    # Add more languages as needed, e.g.:
    # SPANISH = "spanish"
    # FRENCH = "french"
    # GERMAN = "german"


@dataclass
class WordEntry:
    """A word entry in the hangman dataset."""

    word: str
    difficulty: int  # 1-5 scale


# English word dataset with difficulty ratings for hangman
ENGLISH_WORDS = [
    # Difficulty 1 (Very Easy)
    WordEntry("apple", 1),
    WordEntry("happy", 1),
    WordEntry("banana", 1),
    WordEntry("letter", 1),
    WordEntry("summer", 1),
    WordEntry("yellow", 1),
    WordEntry("people", 1),
    WordEntry("window", 1),
    WordEntry("school", 1),
    WordEntry("family", 1),
    WordEntry("coffee", 1),
    WordEntry("cookie", 1),
    WordEntry("paper", 1),
    WordEntry("puppy", 1),
    WordEntry("kitten", 1),
    WordEntry("butter", 1),
    WordEntry("pizza", 1),
    WordEntry("puzzle", 1),
    WordEntry("bubble", 1),
    WordEntry("butterfly", 1),

    # Difficulty 2 (Easy)
    WordEntry("garden", 2),
    WordEntry("market", 2),
    WordEntry("pocket", 2),
    WordEntry("rocket", 2),
    WordEntry("guitar", 2),
    WordEntry("jacket", 2),
    WordEntry("pencil", 2),
    WordEntry("rabbit", 2),
    WordEntry("turtle", 2),
    WordEntry("monkey", 2),
    WordEntry("orange", 2),
    WordEntry("purple", 2),
    WordEntry("silver", 2),
    WordEntry("summer", 2),
    WordEntry("winter", 2),
    WordEntry("yellow", 2),
    WordEntry("zebra", 2),

    # Difficulty 3 (Medium)
    WordEntry("bicycle", 3),
    WordEntry("camera", 3),
    WordEntry("diamond", 3),
    WordEntry("elephant", 3),
    WordEntry("fishing", 3),
    WordEntry("giraffe", 3),
    WordEntry("holiday", 3),
    WordEntry("jungle", 3),
    WordEntry("kangaroo", 3),
    WordEntry("ladder", 3),
    WordEntry("mountain", 3),
    WordEntry("notebook", 3),
    WordEntry("octopus", 3),
    WordEntry("penguin", 3),
    WordEntry("question", 3),

    # Difficulty 4 (Hard)
    WordEntry("awkward", 4),
    WordEntry("bagpipes", 4),
    WordEntry("croquet", 4),
    WordEntry("dwarves", 4),
    WordEntry("espionage", 4),
    WordEntry("flopping", 4),
    WordEntry("gazebo", 4),
    WordEntry("haiku", 4),
    WordEntry("ivory", 4),
    WordEntry("jukebox", 4),
    WordEntry("kayak", 4),
    WordEntry("lymph", 4),
    WordEntry("matrix", 4),

    # Difficulty 5 (Very Hard)
    WordEntry("buzzard", 5),
    WordEntry("crypt", 5),
    WordEntry("duplex", 5),
    WordEntry("espionage", 5),
    WordEntry("foxglove", 5),
    WordEntry("galvanize", 5),
    WordEntry("hyphen", 5),
    WordEntry("ivy", 5),
    WordEntry("jinx", 5),
    WordEntry("kiosk", 5),
    WordEntry("larynx", 5),
    WordEntry("marquis", 5),
    WordEntry("nymph", 5),
    WordEntry("onyx", 5),
    WordEntry("pixel", 5),
    WordEntry("quartz", 5),
    WordEntry("rhythm", 5),
    WordEntry("sphinx", 5),
    WordEntry("topaz", 5),
    WordEntry("unknown", 5),
    WordEntry("vortex", 5),
    WordEntry("wizard", 5),
    WordEntry("xylophone", 5),
    WordEntry("yacht", 5),
    WordEntry("zodiac", 5),
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
            f"Unsupported language: {language}. Supported languages: {', '.join([l.value for l in LANGUAGE_WORDS.keys()])}"
        )
    return LANGUAGE_WORDS[language].copy()


def get_words_by_difficulty(language: Language, difficulty: int) -> List[WordEntry]:
    """Get words of a specific difficulty level for a language.

    Args:
        language: The language to get words for
        difficulty: Difficulty level (1-5)

    Returns:
        List of WordEntry objects matching the criteria
    """
    if not 1 <= difficulty <= 5:
        raise ValueError("Difficulty must be between 1 and 5")

    words = get_words_by_language(language)
    return [word for word in words if word.difficulty == difficulty]
