import json
from dataclasses import dataclass, field
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import AgentState, AgentSubmit, as_solver, react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    grouped,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import StoreModel, store_as
from pydantic import Field

from hangman_bench.datasets import (
    Difficulty,
    Language,
    get_alphabet,
    get_words_by_difficulty,
    get_words_by_language,
)
from hangman_bench.oracle import (
    CHOOSERS,
    load_dictionary_index,
    replay_trajectory,
    resolve_wordlist,
)

DEFAULT_MAX_GUESSES = 10
DEFAULT_LANGUAGE = Language.ENGLISH
DEFAULT_ALPHABET = get_alphabet(DEFAULT_LANGUAGE)
NUM_ALLOWABLE_EXTRA_MESSAGES = 5  # Extra messages beyond word length + max guesses


@task
def hangman(
    language: str = DEFAULT_LANGUAGE.value,
    difficulty: Difficulty | None = None,
    max_guesses: int = DEFAULT_MAX_GUESSES,
    shuffle: bool = True,
    allow_word_guesses: bool = False,
    strategy_hint: str = "frequency",
    oracle: bool = True,
    oracle_wordlist: str | None = None,
) -> Task:
    """Evaluate an agent's ability to play Hangman

    Args:
        language: The language to use for the words (default: english)
        difficulty: Specific difficulty label to use (v_easy, easy, medium, hard, v_hard), or None for mixed difficulties
        max_guesses: Maximum number of incorrect guesses allowed
        shuffle: Whether to shuffle the words before playing
        allow_word_guesses: Whether to allow the agent to guess the entire word
        strategy_hint: Strategy sentence in the system prompt — one of
            frequency (the original wording), neutral (no advice), or belief
            (reason from the consistent candidate set)
        oracle: Whether to also score each guess against optimal play
        oracle_wordlist: Dictionary for the oracle scorer, defaulting to the
            wordlist shipped with the package

    Returns:
        Task configured for the hangman game with specified parameters
    """
    try:
        lang_enum = Language(language.lower())
    except ValueError:
        supported = ", ".join([lang.value for lang in Language])
        raise ValueError(f"Language '{language}' not supported. Supported languages: {supported}")

    # Get words based on language and optional difficulty
    if difficulty is not None:
        word_entries = get_words_by_difficulty(lang_enum, difficulty)
    else:
        word_entries = get_words_by_language(lang_enum)

    longest_word_length = max(len(entry.word) for entry in word_entries)

    # Create samples
    samples = []
    for entry in word_entries:
        samples.append(
            Sample(
                id=entry.word,
                input=(
                    f"You are playing a game of Hangman in {lang_enum.value}. "
                    f"Try to guess the word one letter at a time. "
                    f"The current state shows '_' for unguessed letters and reveals correctly guessed letters. "
                    f"You lose the game after {max_guesses} incorrect guesses. "
                    f"Make smart guesses based on common letter frequencies and word patterns in {lang_enum.value}."
                ),
                target=[entry.word],
                metadata={
                    "word": entry.word,
                    "max_guesses": max_guesses,
                    "difficulty": entry.difficulty,
                    "language": lang_enum.value,
                    "allow_word_guesses": allow_word_guesses,
                },
            )
        )

    dataset = MemoryDataset(samples)
    if shuffle:
        dataset.shuffle()

    scorers: list[Scorer] = [game_scorer()]
    if oracle:
        scorers.append(oracle_scorer(wordlist=oracle_wordlist))

    return Task(
        dataset=dataset,
        solver=hangman_player(allow_word_guesses=allow_word_guesses, strategy_hint=strategy_hint),
        setup=game_initialiser(),
        scorer=scorers,
        message_limit=_calculate_message_limit(longest_word_length, max_guesses),
    )


def _calculate_message_limit(word_length: int, max_guesses: int) -> int:
    # Models sometimes respond with commentary, then need a "continue" message,
    # and then call the tool and get the tool response. So we allow 4 messages per guess.
    return (word_length + max_guesses) * 4 + NUM_ALLOWABLE_EXTRA_MESSAGES


def _normalise(letter: str) -> str:
    return (letter or "").strip().lower()


def _is_valid_letter(letter: str, alphabet: str = DEFAULT_ALPHABET) -> bool:
    # Only letters of the language's declared alphabet are valid: any other
    # character can never be revealed, so counting it as a miss would let the
    # wrong-guess count reach |alphabet| and break the unlimited-budget
    # protocol (a budget of |alphabet| is otherwise impossible to exhaust).
    return len(letter) == 1 and letter in alphabet


@dataclass
class GameState:
    word: str
    guessed_letters: list[str]
    remaining_guesses: int
    game_over: bool = False
    won: bool = False
    # Every guess as submitted, including repeats and malformed input. Repeats
    # and invalid guesses never reach guessed_letters, so without this record
    # they are invisible to any analysis of how the game was played.
    attempts: list[str] = field(default_factory=list)
    # The language's declared letter inventory; guesses outside it are invalid.
    alphabet: str = DEFAULT_ALPHABET

    @staticmethod
    def start(
        word: str,
        max_guesses: int = DEFAULT_MAX_GUESSES,
        alphabet: str = DEFAULT_ALPHABET,
    ) -> "GameState":
        return GameState(
            word=word.lower(),
            guessed_letters=[],
            remaining_guesses=max_guesses,
            alphabet=alphabet,
        )

    @property
    def invalid_attempts(self) -> list[str]:
        """Submissions that were not a single letter of the alphabet."""
        return [a for a in self.attempts if not _is_valid_letter(_normalise(a), self.alphabet)]

    @property
    def repeated_attempts(self) -> list[str]:
        """Valid letters submitted more than once, in the order repeated."""
        seen: set[str] = set()
        repeats: list[str] = []
        for raw in self.attempts:
            letter = _normalise(raw)
            if not _is_valid_letter(letter, self.alphabet):
                continue
            if letter in seen:
                repeats.append(letter)
            else:
                seen.add(letter)
        return repeats

    @property
    def current_state(self) -> str:
        """Returns the current state of the word with unguessed letters as '_'"""
        return " ".join(letter if letter in self.guessed_letters else "_" for letter in self.word)

    @property
    def incorrect_guesses(self) -> list[str]:
        """Returns list of incorrect guesses"""
        return sorted(list(set(self.guessed_letters) - set(self.word)))

    def guess(self, letter: str) -> "GameState":
        """Process a letter guess and return the new game state"""
        if self.game_over:
            return self

        letter = _normalise(letter)
        if not _is_valid_letter(letter, self.alphabet):
            raise ValueError("Guess must be a single letter of the game's alphabet")

        if letter in self.guessed_letters:
            return self

        self.guessed_letters.append(letter)

        if letter not in self.word:
            self.remaining_guesses -= 1

        # Check win condition
        if all(letter in self.guessed_letters for letter in self.word):
            self.game_over = True
            self.won = True

        # Check lose condition
        if self.remaining_guesses <= 0:
            self.game_over = True

        return self


class HangmanStore(StoreModel):
    """Typed interface to the per-sample store."""

    game_state: GameState | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@tool(parallel=False)
def hangman_guess() -> Tool:
    """Tool for guessing letters in the hangman game"""

    async def execute(letter: str) -> str:
        """Submit a letter guess for the current hangman game.

        Args:
            letter (str): A single letter to guess

        Returns:
            A formatted string containing the game state with:
            - Current word progress
            - Remaining guesses
            - Incorrect guesses made so far
            - Game status (ongoing, won, or lost)
        """
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        metadata = hstore.metadata or {}

        if game_state is None:
            raise RuntimeError(
                "No game in progress. The game must be started by the evaluation setup."
            )

        # Record the raw submission before validating, so malformed and
        # repeated guesses are still measurable.
        game_state.attempts.append(letter)

        normalised = _normalise(letter)
        if not _is_valid_letter(normalised, game_state.alphabet):
            # A ToolError is reported back to the model, which can then correct
            # itself. Letting the ValueError escape would abort the sample and
            # drop the game from the results entirely.
            language = metadata.get("language", DEFAULT_LANGUAGE.value)
            raise ToolError(
                f"'{letter}' is not a single letter of the {language} alphabet. "
                f"Guess exactly one letter, for example hangman_guess('a')."
            )

        # A repeat leaves the game unchanged, and silently returning the same
        # board is the precondition for repeat spirals: the model sees no
        # signal that anything went wrong. Say so, the way a human setter
        # would. Lives are never charged for repeats — they buy no evidence.
        already_guessed = normalised in game_state.guessed_letters

        if not game_state.game_over:
            game_state.guess(normalised)  # Updates the game state

        # Format the result as a readable string
        result_lines = []
        if already_guessed:
            result_lines.append(
                f"You already guessed '{normalised}'. Guessing it again has no effect."
            )
        result_lines += [
            f"Word: {game_state.current_state}",
            f"Remaining guesses: {game_state.remaining_guesses}",
            f"Incorrect guesses: {', '.join(game_state.incorrect_guesses) if game_state.incorrect_guesses else 'none'}",
        ]

        if game_state.game_over:
            if game_state.won:
                result_lines.append("Status: WON! You guessed the word!")
            else:
                result_lines.append(f"Status: LOST! The word was '{game_state.word}'")
        else:
            result_lines.append("Status: Game continues")

        result_lines.append(f"Language: {metadata.get('language', DEFAULT_LANGUAGE.value)}")

        return "\n".join(result_lines)

    return execute


# The strategy sentence in the system prompt is an experimental variable: the
# default instructs frequency play, which is itself a failure mode the oracle
# penalises (RESEARCH_NOTES.md, external review 2026-08-09). "frequency" must
# reproduce the original prompt byte-for-byte so runs remain comparable; a
# test pins this.
STRATEGY_HINTS: dict[str, str] = {
    "frequency": (
        " Make smart guesses based on common letter frequencies\n"
        "    and word patterns in the specified language, and the result of previous guesses."
    ),
    "neutral": "",
    "belief": (
        " Before each guess, consider which words are consistent with the\n"
        "    revealed pattern and the letters already ruled out, and guess a letter\n"
        "    likely to appear in one of those words."
    ),
}


def _build_system_message(allow_word_guesses: bool, strategy_hint: str) -> str:
    """System prompt for the player, with the strategy sentence as a variable."""
    if strategy_hint not in STRATEGY_HINTS:
        raise ValueError(
            f"Unknown strategy_hint '{strategy_hint}'. "
            f"Choose from: {', '.join(sorted(STRATEGY_HINTS))}"
        )
    base = f"""
    You are playing a game of Hangman. Try to guess the word one letter at a time.
    The current state shows '_' for unguessed letters and reveals correctly guessed letters.
    You have limited guesses before losing.{STRATEGY_HINTS[strategy_hint]}

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
    if allow_word_guesses:
        return (
            base
            + """
    You can also guess the entire word at any point before running out of guesses,
    by submitting the word as a single string using `submit("word")`. This will end the game.
    The game will end automatically when it is over.
    """
        )
    return (
        base
        + """
    You must guess all the letters in the word to win the game.
    You can ONLY win by guessing all the letters in the word, one at a time.
    The game will end automatically when it is over.
    """
    )


@solver
def hangman_player(allow_word_guesses: bool = False, strategy_hint: str = "frequency") -> Solver:
    """Solver that uses the hangman_guess tool to play hangman"""
    final_system_message = _build_system_message(allow_word_guesses, strategy_hint)

    async def on_continue(state: AgentState) -> bool | str:
        # Stop automatically when game is over; otherwise, urge model to keep using tools
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        if game_state is None or game_state.game_over:
            return False
        # If the last response was a tool call, return True
        if state.output.message.tool_calls:
            return True
        # No concrete example letter here: models follow one literally,
        # guessing 'a' every turn (see RESEARCH_NOTES.md, pilot 2026-08-07).
        guidance = "Continue the game by calling the hangman_guess tool with your next letter."
        if allow_word_guesses:
            guidance += " If you know the full word, submit it with the submit tool."
        return guidance

    return as_solver(
        react(
            prompt=final_system_message,
            tools=[hangman_guess()],
            on_continue=on_continue,
            submit=AgentSubmit(answer_only=True) if allow_word_guesses else False,
        )
    )


@solver
def game_initialiser() -> Solver:
    """Initialise the game, and store the game state in the store"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}
        word = metadata.get("word", None)
        if not word:
            raise RuntimeError("No word provided in metadata")

        max_guesses = metadata.get("max_guesses", DEFAULT_MAX_GUESSES)
        language = metadata.get("language", DEFAULT_LANGUAGE.value)
        difficulty = metadata.get("difficulty", "medium")
        allow_word_guesses = metadata.get("allow_word_guesses", False)

        hangman_game = GameState.start(
            word=word,
            max_guesses=max_guesses,
            alphabet=get_alphabet(Language(language)),
        )

        # Store game state and metadata using a typed store model
        hstore = store_as(HangmanStore)
        hstore.game_state = hangman_game
        hstore.metadata = {
            "language": language,
            "difficulty": difficulty,
            "allow_word_guesses": allow_word_guesses,
        }

        state.user_prompt.text = (
            f"Let's play hangman in {language}! You have {max_guesses} guesses.\n"
            f"The word is {' '.join(['_'] * len(word))}.\n"
        )
        return state

    return solve


def _guesses_from_messages(messages: list[Any]) -> list[str]:
    """Raw letters submitted to hangman_guess, in order.

    Tool calls record what the agent actually sent, including repeats and
    malformed input that never reach guessed_letters.
    """
    guesses: list[str] = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if getattr(tool_call, "function", None) != "hangman_guess":
                continue
            arguments = getattr(tool_call, "arguments", None) or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            letter = arguments.get("letter")
            if letter is not None:
                guesses.append(str(letter))
    return guesses


@scorer(
    metrics={
        "excess_wrong_guesses": [mean(), stderr()],
        "hit_prob_regret": [mean(), stderr()],
        "dominated_rate": [mean(), stderr()],
        "suboptimal_rate": [mean(), stderr()],
        "repeat_rate": [mean()],
        "invalid_rate": [mean()],
    }
)
def oracle_scorer(
    wordlist: str | None = None,
    strategy: str = "max_hit_prob",
) -> Scorer:
    """Score how the game was played, not just whether it was won.

    Hangman admits an exactly computable posterior over the hidden word and an
    optimal next move at every step, so each guess can be compared against the
    best available one. A model can win with a generous wrong-guess budget while
    ignoring the evidence entirely; these metrics separate the two.

    Reported per game:
      excess_wrong_guesses  wrong guesses above what an oracle solver needs
      hit_prob_regret     mean shortfall in hit probability per scored guess
      dominated_rate      guesses of a letter in zero consistent candidates
      suboptimal_rate     guesses below the best available hit probability
      repeat_rate         letters guessed more than once
      invalid_rate        submissions that were not a single letter

    Args:
        wordlist: Dictionary defining the belief state. Defaults to the
            wordlist shipped with the package.
        strategy: Reference policy, one of max_hit_prob or info_gain.

    Returns:
        Scorer producing a dict of oracle metrics per game.
    """
    if strategy not in CHOOSERS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {', '.join(sorted(CHOOSERS))}"
        )
    # Resolve eagerly so a bad path fails at task construction, not mid-run.
    wordlist_path = str(resolve_wordlist(wordlist))
    chooser = CHOOSERS[strategy]

    async def score(state: TaskState, target: Target) -> Score:
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        metadata: dict[str, Any] = state.metadata or {}

        word = (metadata.get("word") or (game_state.word if game_state else "")).lower()
        if not word:
            raise RuntimeError("No word found in sample metadata or game state")
        max_guesses = metadata.get("max_guesses", DEFAULT_MAX_GUESSES)

        # Tool calls are present in every log; attempts only in those written
        # after the eval began recording it.
        guesses = _guesses_from_messages(state.messages)
        if not guesses and game_state is not None:
            guesses = list(game_state.attempts)

        index = load_dictionary_index(wordlist_path)
        report = replay_trajectory(
            word=word,
            raw_guesses=guesses,
            dictionary=index.get(len(word), []),
            chooser=chooser,
            max_wrong=max_guesses,
            sample_id=str(state.sample_id),
            model=str(state.model),
            alphabet=get_alphabet(Language(metadata.get("language", DEFAULT_LANGUAGE.value))),
        )

        # A game won by submitting the full word ends before every letter is
        # revealed, so replaying the letter guesses alone would call it a loss.
        # Recover the real outcome the same way game_scorer does.
        if metadata.get("allow_word_guesses") and game_state and not game_state.game_over:
            submitted = state.output.completion
            report.recorded_won = submitted == word
        elif game_state is not None:
            report.recorded_won = game_state.won

        emitted = len(report.steps)
        scored = report.n_scored
        value = {
            "excess_wrong_guesses": float(report.excess_wrong_guesses),
            "hit_prob_regret": report.mean_hit_prob_regret,
            "dominated_rate": (report.n_dominated / scored) if scored else 0.0,
            "suboptimal_rate": (report.n_suboptimal / scored) if scored else 0.0,
            "repeat_rate": (report.n_repeat / emitted) if emitted else 0.0,
            "invalid_rate": (report.n_invalid / emitted) if emitted else 0.0,
        }

        return Score(
            value=value,
            answer="".join(s.letter for s in report.steps if s.counted),
            explanation=(
                f"Word: {word}. Guesses: {emitted} emitted, {scored} scored. "
                f"Wrong: {report.wrong_guesses} vs oracle {report.oracle_wrong_guesses}. "
                f"Dominated: {report.n_dominated}. Repeats: {report.n_repeat}. "
                f"Invalid: {report.n_invalid}."
            ),
            metadata={
                "won": report.final_won,
                "difficulty": metadata.get("difficulty"),
                "wrong_guesses": report.wrong_guesses,
                "oracle_wrong_guesses": report.oracle_wrong_guesses,
                "num_dominated": report.n_dominated,
                "num_repeat": report.n_repeat,
                "num_invalid": report.n_invalid,
                "num_suboptimal": report.n_suboptimal,
                "guesses_emitted": emitted,
                "guesses_scored": scored,
                "target_in_dictionary": report.target_in_dictionary,
                "strategy": strategy,
                "per_guess": [
                    {
                        "letter": s.letter,
                        "board_before": s.board_before,
                        "candidates_before": s.candidates_before,
                        "hit": s.hit,
                        "hit_prob": round(s.hit_prob, 4),
                        "best_hit_prob": round(s.best_hit_prob, 4),
                        "optimal_letter": s.optimal_letter,
                        "dominated_miss": s.dominated_miss,
                        "repeat": s.repeat,
                        "invalid": s.invalid,
                    }
                    for s in report.steps
                ],
            },
        )

    return score


@scorer(
    metrics=[
        grouped(accuracy(), group_key="difficulty"),
        stderr(),
    ]
)
def game_scorer() -> Scorer:
    """Score the hangman game based on whether the player won or not"""

    async def score(state: TaskState, target: Target) -> Score:
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        metadata: dict[str, Any] = hstore.metadata or {}
        language = metadata.get("language", DEFAULT_LANGUAGE.value)
        difficulty = metadata.get("difficulty", 3)

        if not game_state:
            raise RuntimeError("No game state found in store")

        allow_word_guesses = metadata.get("allow_word_guesses", False)
        if allow_word_guesses:
            # If word guesses are allowed and the game is not over, the agent guessed early
            if not game_state.game_over:
                guessed_word = state.output.completion
                explanation = (
                    f"Early guess. Word: {game_state.word}. Language: {language}. "
                    f"Difficulty: {difficulty}. "
                    f"Guessed word: {guessed_word}. "
                    f"Guessed letters: {game_state.guessed_letters}. "
                    f"Final word state: {game_state.current_state}. "
                    f"Remaining guesses: {game_state.remaining_guesses}. "
                )
                return Score(
                    value=CORRECT if guessed_word == game_state.word else INCORRECT,
                    answer=guessed_word,
                    explanation=explanation,
                    metadata={
                        "won": game_state.won,
                        "language": language,
                        "difficulty": difficulty,
                        "allow_word_guesses": allow_word_guesses,
                        "guessed_word": guessed_word,
                        "guessed_letters": game_state.guessed_letters,
                        "final_word_state": game_state.current_state,
                        "remaining_guesses": game_state.remaining_guesses,
                        "incorrect_guesses": game_state.incorrect_guesses,
                        "num_incorrect_guesses": len(game_state.incorrect_guesses),
                        "attempts": game_state.attempts,
                        "num_repeated_guesses": len(game_state.repeated_attempts),
                        "num_invalid_guesses": len(game_state.invalid_attempts),
                    },
                )

        if not game_state.game_over:
            return Score(
                value=INCORRECT,
                answer=game_state.current_state,
                explanation="The game did not complete.",
                metadata={
                    "won": game_state.won,
                    "language": language,
                    "difficulty": difficulty,
                    "allow_word_guesses": allow_word_guesses,
                    "guessed_letters": game_state.guessed_letters,
                    "final_word_state": game_state.current_state,
                    "remaining_guesses": game_state.remaining_guesses,
                    "incorrect_guesses": game_state.incorrect_guesses,
                    "num_incorrect_guesses": len(game_state.incorrect_guesses),
                    "attempts": game_state.attempts,
                    "num_repeated_guesses": len(game_state.repeated_attempts),
                    "num_invalid_guesses": len(game_state.invalid_attempts),
                },
            )

        explanation = (
            f"Game ended. Word: {game_state.word}. Language: {language}. "
            f"Difficulty: {difficulty}. "
            f"Won: {game_state.won}. "
            f"Guessed letters: {game_state.guessed_letters}. "
            f"Final word state: {game_state.current_state}. "
            f"Remaining guesses: {game_state.remaining_guesses}. "
        )

        return Score(
            value=CORRECT if game_state.won else INCORRECT,
            answer=game_state.current_state,
            explanation=explanation,
            metadata={
                "won": game_state.won,
                "language": language,
                "difficulty": difficulty,
                "allow_word_guesses": allow_word_guesses,
                "guessed_letters": game_state.guessed_letters,
                "final_word_state": game_state.current_state,
                "remaining_guesses": game_state.remaining_guesses,
                "incorrect_guesses": game_state.incorrect_guesses,
                "num_incorrect_guesses": len(game_state.incorrect_guesses),
                "attempts": game_state.attempts,
                "num_repeated_guesses": len(game_state.repeated_attempts),
                "num_invalid_guesses": len(game_state.invalid_attempts),
            },
        )

    return score
