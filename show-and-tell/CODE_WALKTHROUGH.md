# Code Walkthrough Guide

## Quick Navigation

**Main Implementation**: `src/hangman_bench/hangman.py` (435 lines)

- Lines 119-174: `GameState` class
- Lines 176-181: `HangmanStore` typed store
- Lines 183-233: `hangman_guess()` tool
- Lines 236-299: `hangman_player()` solver
- Lines 302-337: `game_initialiser()` setup
- Lines 340-434: `game_scorer()` scorer

**Dataset**: `src/hangman_bench/datasets.py` (192 lines)

- Lines 31-37: `WordEntry` dataclass
- Lines 40-146: `ENGLISH_WORDS` list (100 words)
- Lines 154-191: Helper functions

---

## Walkthrough Flow

### 1. Start with GameState (Lines 119-174)

**Show this first** - it's the heart of the game logic:

```python
@dataclass
class GameState:
    word: str
    guessed_letters: list[str]
    remaining_guesses: int
    game_over: bool = False
    won: bool = False
```

**Key points:**

- Simple, immutable state
- No Inspect-specific code (pure game logic)
- Properties for derived state (`current_state`, `incorrect_guesses`)

**Demo the `guess()` method** (Lines 147-173):

```python
def guess(self, letter: str) -> "GameState":
    # Validation
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError("Guess must be a single letter")
    
    # Update state
    self.guessed_letters.append(letter)
    if letter not in self.word:
        self.remaining_guesses -= 1
    
    # Check win/loss
    if all(letter in self.guessed_letters for letter in self.word):
        self.game_over = True
        self.won = True
    
    return self
```

---

### 2. Show the Store Pattern (Lines 176-181)

**This is the key Inspect pattern:**

```python
class HangmanStore(StoreModel):
    """Typed interface to the per-sample store."""
    game_state: GameState | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Explain:**

- `StoreModel` from Inspect provides per-sample storage
- Type-safe access to state
- Persists across tool calls within a sample
- Each sample gets its own isolated store

---

### 3. Walk Through the Tool (Lines 183-233)

**This is where the magic happens:**

```python
@tool(parallel=False)  # ← Sequential guessing only
def hangman_guess() -> Tool:
    async def execute(letter: str) -> str:
        # 1. Access the store
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        
        # 2. Validate game exists
        if game_state is None:
            raise RuntimeError("No game in progress")
        
        # 3. Process the guess
        if not game_state.game_over:
            game_state.guess(letter)  # Mutates state!
        
        # 4. Format feedback
        result_lines = [
            f"Word: {game_state.current_state}",
            f"Remaining guesses: {game_state.remaining_guesses}",
            f"Incorrect guesses: {', '.join(game_state.incorrect_guesses)}",
        ]
        
        if game_state.game_over:
            if game_state.won:
                result_lines.append("Status: WON!")
            else:
                result_lines.append(f"Status: LOST! The word was '{game_state.word}'")
        
        return "\n".join(result_lines)
    
    return execute
```

**Key points:**

- `parallel=False` prevents concurrent guesses
- `store_as(HangmanStore)` gets typed store access
- State mutation happens here
- Structured feedback guides model strategy
- Reveals word on loss (important for learning!)

---

### 4. Explain the Solver (Lines 236-299)

**This is the agent configuration:**

```python
@solver
def hangman_player(allow_word_guesses: bool = False) -> Solver:
    # System message (lines 239-275)
    SYSTEM_MESSAGE = """
    You are playing Hangman. Guess one letter at a time using hangman_guess("a").
    Make smart guesses based on letter frequencies and word patterns.
    """
    
    # Custom continuation logic (lines 278-290)
    async def on_continue(state: AgentState) -> bool | str:
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        
        # Auto-stop when game ends
        if game_state is None or game_state.game_over:
            return False
        
        # If last response was a tool call, continue
        if state.output.message.tool_calls:
            return True
        
        # Otherwise, provide guidance
        guidance = "Continue by calling hangman_guess('a')..."
        return guidance
    
    # Return configured agent (lines 292-299)
    return as_solver(
        react(
            prompt=final_system_message,
            tools=[hangman_guess()],
            on_continue=on_continue,
            submit=AgentSubmit(answer_only=True) if allow_word_guesses else False,
        )
    )
```

**Key points:**

- `react` agent with tools
- `on_continue` provides graceful degradation
- Returns `False` to stop, `True` to continue, or `str` for guidance
- Optional `submit` for early word guesses

---

### 5. Setup Solver (Lines 302-337)

**Runs before the agent:**

```python
@solver
def game_initialiser() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # 1. Extract metadata
        word = state.metadata.get("word")
        max_guesses = state.metadata.get("max_guesses", 10)
        
        # 2. Create game state
        hangman_game = GameState.start(word=word, max_guesses=max_guesses)
        
        # 3. Store it
        hstore = store_as(HangmanStore)
        hstore.game_state = hangman_game
        hstore.metadata = {
            "language": language,
            "difficulty": difficulty,
            "allow_word_guesses": allow_word_guesses,
        }
        
        # 4. Set initial prompt
        state.user_prompt.text = (
            f"Let's play hangman! You have {max_guesses} guesses.\n"
            f"The word is {' '.join(['_'] * len(word))}.\n"
        )
        
        return state
    
    return solve
```

**Key points:**

- Initializes store before agent runs
- Extracts metadata from sample
- Sets up initial user prompt
- Returns modified `TaskState`

---

### 6. Scorer (Lines 340-434)

**Evaluates the final result:**

```python
@scorer(
    metrics=[
        grouped(accuracy(), group_key="difficulty"),  # ← Group by difficulty!
        stderr(),
    ]
)
def game_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        
        # Handle early word guess (if allowed)
        if allow_word_guesses and not game_state.game_over:
            guessed_word = state.output.completion
            return Score(
                value=CORRECT if guessed_word == game_state.word else INCORRECT,
                answer=guessed_word,
                explanation=f"Early guess: {guessed_word}",
                metadata={...}  # Rich metadata
            )
        
        # Handle incomplete game
        if not game_state.game_over:
            return Score(
                value=INCORRECT,
                explanation="Game did not complete"
            )
        
        # Normal completion
        return Score(
            value=CORRECT if game_state.won else INCORRECT,
            answer=game_state.current_state,
            explanation=f"Game ended. Won: {game_state.won}",
            metadata={
                "won": game_state.won,
                "guessed_letters": game_state.guessed_letters,
                "incorrect_guesses": game_state.incorrect_guesses,
                "num_incorrect_guesses": len(game_state.incorrect_guesses),
                # ... more metadata
            }
        )
    
    return score
```

**Key points:**

- `grouped(accuracy(), group_key="difficulty")` breaks down results by tier
- Handles three cases: early guess, incomplete, normal
- Rich metadata for post-analysis
- Stores full game trajectory

---

### 7. Task Function (Lines 40-110)

**Ties everything together:**

```python
@task
def hangman(
    language: str = "english",
    difficulty: Difficulty | None = None,
    max_guesses: int = 10,
    shuffle: bool = True,
    allow_word_guesses: bool = False,
) -> Task:
    # 1. Get words
    word_entries = get_words_by_difficulty(lang_enum, difficulty) if difficulty else get_words_by_language(lang_enum)
    
    # 2. Create samples
    samples = []
    for entry in word_entries:
        samples.append(
            Sample(
                id=entry.word,
                input="You are playing Hangman...",
                target=[entry.word],
                metadata={
                    "word": entry.word,
                    "max_guesses": max_guesses,
                    "difficulty": entry.difficulty,
                    "language": lang_enum.value,
                    "allow_word_guesses": allow_word_guesses,
                }
            )
        )
    
    # 3. Create dataset
    dataset = MemoryDataset(samples)
    if shuffle:
        dataset.shuffle()
    
    # 4. Return configured task
    return Task(
        dataset=dataset,
        solver=hangman_player(allow_word_guesses=allow_word_guesses),
        setup=game_initialiser(),
        scorer=game_scorer(),
        message_limit=_calculate_message_limit(longest_word_length, max_guesses),
    )
```

**Key points:**

- Configurable parameters
- Metadata flows through to setup/scorer
- `setup` runs before `solver`
- Message limit prevents runaway conversations

---

## Message Limit Calculation (Lines 113-116)

**Important detail:**

```python
def _calculate_message_limit(word_length: int, max_guesses: int) -> int:
    # Models sometimes respond with commentary, then need a "continue" message,
    # and then call the tool and get the tool response. So we allow 4 messages per guess.
    return (word_length + max_guesses) * 4 + NUM_ALLOWABLE_EXTRA_MESSAGES
```

**Why 4x?**

1. Model commentary
2. "Continue" prompt from `on_continue`
3. Tool call
4. Tool response

Plus buffer for initial setup messages.

---

## Demo Script

### Option 1: Trace a Single Game

1. Show `GameState` initialization
2. Step through first guess: "e"
   - Tool receives "e"
   - `game_state.guess("e")` updates state
   - Returns formatted feedback
3. Show how store persists across calls
4. Show `on_continue` triggering
5. Show final scoring

### Option 2: Compare Strategies

Run `measure_difficulty.py` on "apple":

```bash
uv run python -c "
from analysis.measure_difficulty import *
from analysis.zen_hangman import *

word = 'apple'
dictionary = load_wordlist('analysis/wordlist.txt')
dict_5 = [w for w in dictionary if len(w) == 5]

# Frequency solver
result_freq = solve_with_strategy(word, dict_5, best_move_freq_raw)
print(f'Frequency: {result_freq.wrong_guesses} wrong')

# Coverage solver  
result_cov = solve_with_strategy(word, dict_5, best_move_coverage)
print(f'Coverage: {result_cov.wrong_guesses} wrong')

# Info gain solver
result_info = solve_with_strategy(word, dict_5, best_move_info_gain)
print(f'InfoGain: {result_info.wrong_guesses} wrong')
"
```

### Option 3: Live Eval

```bash
# Run 5 words with verbose logging
inspect eval hangman_bench/hangman --limit 5 --model openai/gpt-4o-mini --log-level debug
```

---

## Key Patterns to Emphasize

### 1. Store Pattern

```python
# Setup
hstore = store_as(HangmanStore)
hstore.game_state = GameState.start(...)

# Tool
hstore = store_as(HangmanStore)
hstore.game_state.guess(letter)

# Scorer
hstore = store_as(HangmanStore)
if hstore.game_state.won:
    return CORRECT
```

### 2. Tool Feedback Design

- Structured output (not just "correct" or "wrong")
- Enough info for strategic decisions
- Clear game state representation

### 3. Graceful Degradation

- `on_continue` provides guidance
- Message limits prevent infinite loops
- Handle edge cases (incomplete games, early guesses)

### 4. Rich Metadata

- Capture everything for analysis
- Grouped metrics by difficulty
- Full game trajectory in metadata

---

## Questions to Prepare For

**Q: Why not use a simpler state management approach?**
A: Store pattern scales to complex multi-turn interactions. Clean separation between game logic and Inspect integration.

**Q: What if a model guesses the same letter twice?**
A: `GameState.guess()` checks `if letter in self.guessed_letters` and returns early. Tool feedback shows already-guessed letters.

**Q: How do you handle models that don't use tools?**
A: `on_continue` provides gentle nudges. Message limit eventually stops them. We accept some failures.

**Q: Why `parallel=False` on the tool?**
A: Guessing multiple letters simultaneously doesn't make sense for Hangman. Sequential only.

**Q: Can you add more games?**
A: Yes! This pattern works for any turn-based game. Just change `GameState` and tool logic.

---

## Files to Have Open

1. `src/hangman_bench/hangman.py` - Main implementation
2. `src/hangman_bench/datasets.py` - Word list
3. `analysis/difficulty_report.tsv` - Metrics comparison
4. `analysis/measure_difficulty.py` - Solver implementations
5. `analysis/README.md` - Analysis documentation

---

## Timing Guide

- **GameState**: 3 minutes
- **Store pattern**: 2 minutes
- **Tool implementation**: 5 minutes
- **Solver & on_continue**: 5 minutes
- **Setup & Scorer**: 3 minutes
- **Task function**: 2 minutes
- **Total**: ~20 minutes for full walkthrough

Keep it interactive - ask if they want to dive deeper into any part!
