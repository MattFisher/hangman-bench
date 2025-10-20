# Hangman Bench: Show and Tell

## Overview

A presentation about building a game-based eval using Inspect's tool features, and the surprising research journey into word difficulty.

---

## Part 1: Building Games with Tools (10-15 min)

### The Original Goal

- **Demonstrate Inspect's new tool capabilities** by implementing a complete game
- Hangman is perfect: simple rules, clear win/loss, requires strategic thinking
- Shows how to build **stateful, interactive evaluations**

### Architecture Deep Dive

#### Core Components (`src/hangman_bench/hangman.py`)

**1. Game State Management (Lines 119-174)**

```python
@dataclass
class GameState:
    word: str
    guessed_letters: list[str]
    remaining_guesses: int
    game_over: bool = False
    won: bool = False
```

- Immutable game state stored per-sample using `store_as(HangmanStore)`
- Clean separation: state management vs. game logic
- Properties for derived state (`current_state`, `incorrect_guesses`)

**2. Tool Implementation (Lines 183-233)**

```python
@tool(parallel=False)
def hangman_guess() -> Tool:
    async def execute(letter: str) -> str:
        # Access per-sample store
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        
        # Process guess and update state
        game_state.guess(letter)
        
        # Return formatted feedback
        return formatted_game_state
```

- `parallel=False` ensures sequential guessing
- Tool returns **structured feedback** to guide the model
- Store pattern enables stateful interactions across tool calls

**3. Solver Pattern (Lines 236-299)**

```python
@solver
def hangman_player(allow_word_guesses: bool = False) -> Solver:
    async def on_continue(state: AgentState) -> bool | str:
        # Auto-stop when game ends
        if game_state.game_over:
            return False
        # Provide guidance if model doesn't use tools
        return "Continue by calling hangman_guess('a')..."
    
    return as_solver(
        react(
            prompt=SYSTEM_MESSAGE,
            tools=[hangman_guess()],
            on_continue=on_continue,
        )
    )
```

- Uses `react` agent with custom `on_continue` logic
- Automatically stops when game ends
- Provides gentle nudges if model forgets to use tools
- Optional `submit()` for early word guesses

**4. Setup and Scoring (Lines 302-434)**

```python
@solver
def game_initialiser() -> Solver:
    # Initialize game state in store before agent runs
    hstore = store_as(HangmanStore)
    hstore.game_state = GameState.start(word, max_guesses)

@scorer(metrics=[grouped(accuracy(), group_key="difficulty")])
def game_scorer() -> Scorer:
    # Score based on final game state
    return Score(
        value=CORRECT if game_state.won else INCORRECT,
        metadata={...}  # Rich metadata for analysis
    )
```

- Setup runs before solver, initializes state
- Scorer uses grouped metrics to break down by difficulty
- Metadata captures full game trajectory

### Key Patterns for Game Evals

1. **Typed Store Models** - Type-safe per-sample state
2. **Tool Feedback Design** - Return enough info for strategic play
3. **Message Limits** - Calculate based on game complexity: `(word_length + max_guesses) * 4`
4. **Graceful Degradation** - Handle models that don't follow instructions
5. **Rich Metadata** - Capture everything for post-analysis

### Dataset Structure (`src/hangman_bench/datasets.py`)

```python
@dataclass
class WordEntry:
    word: str
    difficulty: Difficulty  # v_easy, easy, medium, hard, v_hard

ENGLISH_WORDS = [
    WordEntry("apple", "v_easy"),
    WordEntry("rhythm", "v_hard"),
    # ... 100 total words
]
```

- 20 words per difficulty tier
- Initially classified by intuition (Windsurf AI)
- Supports multiple languages (extensible)

---

## Part 2: The "Apple" Problem (15-20 min)

### The Question

**How do we objectively measure word difficulty in Hangman?**

Initial dataset was classified by intuition, but is "apple" really v_easy?

### The Research Journey

#### Attempt 1: Use Existing Simulation Data

- Found Wolfram's 2010 simulation: 90,000+ words played by random weighted solver
- Blog post: ["25 Best Hangman Words"](https://blog.wolfram.com/2010/08/13/25-best-hangman-words/)
- **Problem**: Random solver doesn't match optimal play
  - "apple" rated as **hard** (6.67 mean wrong guesses)
  - Doesn't align with intuition

#### Attempt 2: Build Deterministic Solvers (`analysis/`)

**Three different strategies implemented:**

1. **Raw Frequency Solver** (`zen_hangman.py`)
   - Port of Dan Q's "Hardest Hangman" heuristic
   - Counts raw letter occurrences (duplicates count multiple times)
   - Filters candidates by known positions + wrong guesses
   - **Apple result**: 1 wrong guess (easy!)

2. **Coverage Solver** (`measure_difficulty.py`)
   - Counts unique word incidence (duplicates ignored)
   - Maximizes probability of hitting *any* candidate word
   - **Apple result**: 5 wrong guesses (medium)

3. **Information Gain Solver** (`measure_difficulty.py`)
   - Minimizes expected remaining candidate set size
   - Uses position masks to partition candidates
   - **Apple result**: 1 wrong guess (easy!)

#### The Apple Paradox

From `analysis/difficulty_report.tsv`:

```
word   wrong_freq_raw  wrong_coverage  wrong_info_gain
apple       1               5               1
happy       8               9               9
banana      2               4               11
```

**Why the variance?**

- Different solvers optimize different objectives
- Coverage solver: "p" appears in many words but often in wrong positions
- Frequency/InfoGain: "a", "e" are so common they hit quickly
- **No single "correct" difficulty metric**

#### Other Surprising Results

From `analysis/reclassified_words.tsv`:

```
word      old_difficulty  new_difficulty  mean_wrong_guesses
apple     v_easy          hard            6.667
happy     v_easy          v_hard          8.716
puppy     v_easy          v_hard          9.888
awkward   hard            easy            5.300
rhythm    v_hard          medium          5.540
```

- Words with repeated letters (happy, puppy) are **harder** than expected
- "Awkward" looks scary but has common letters
- "Rhythm" (no vowels!) is medium difficulty for smart solvers

### Why We Kept the Original Dataset

From `analysis/README.md`:
> Results vary greatly depending on the solvers used, so the words have not currently been reclassified from the original dataset

**Reasons:**

1. **Solver-dependent**: Different strategies give wildly different rankings
2. **Human intuition matters**: Eval users expect "apple" to be easy
3. **Eval stability**: Don't want to change ground truth mid-research
4. **It's still interesting**: The variance itself is valuable data

### What We Built

Complete analysis pipeline in `analysis/`:

- `ingest_simulation.py` - Parse Wolfram data
- `extract_wordlist.py` - Build dictionary (90k+ words)
- `zen_hangman.py` - Frequency-based solver
- `measure_difficulty.py` - Multi-strategy difficulty measurement
- `bin_difficulty.py` - Quantile-based reclassification
- `cost_estimation.py` - Project eval costs from logs

All reproducible with `uv run analysis/...`

---

## Part 3: It's Actually a Good Eval (5-10 min)

### Why Hangman Tests Interesting Capabilities

1. **Strategic Tool Use**
   - Must use tools repeatedly with different arguments
   - Need to adapt strategy based on feedback
   - Tests planning and state tracking

2. **Language Understanding**
   - Letter frequency knowledge
   - Word pattern recognition
   - Vocabulary breadth

3. **Probabilistic Reasoning**
   - Bayesian updates from each guess
   - Risk/reward tradeoffs
   - Candidate set pruning

4. **Instruction Following**
   - Use tools correctly
   - Don't repeat guesses
   - Stop when game ends

### Eval Features

- **100 words** across 5 difficulty tiers
- **Grouped scoring** by difficulty level
- **Rich metadata** for analysis (guessed letters, wrong guesses, etc.)
- **Configurable**: max guesses, allow word guesses, language
- **Cost estimation** utilities included

### Usage

```bash
# Basic eval
inspect eval hangman_bench/hangman --model openai/gpt-4o

# Limit samples for quick testing
inspect eval hangman_bench/hangman --limit 10

# Specific difficulty
inspect eval hangman_bench/hangman -T difficulty=v_hard

# Allow early word guesses
inspect eval hangman_bench/hangman -T allow-word-guesses=True
```

---

## Key Takeaways

### For Building Game Evals

1. **Store pattern is powerful** - Clean state management across tool calls
2. **Tool feedback matters** - Design output to guide model strategy
3. **Message limits are critical** - Calculate based on game complexity
4. **Graceful degradation** - Handle models that don't follow instructions

### For Difficulty Analysis

1. **Intuition ≠ Objective Difficulty** - "Apple" is hard for some solvers
2. **Solver strategy matters** - Different objectives yield different rankings
3. **Multiple metrics are valuable** - No single "correct" measure
4. **Keep it reproducible** - All analysis scripts included

### For Eval Design

1. **Games make great evals** - Clear objectives, natural tool use
2. **Start with demonstration** - Then discover it's useful research
3. **Document the journey** - The "apple problem" is part of the story
4. **Provide analysis tools** - Cost estimation, difficulty measurement

---

## Demo Ideas

### Code Walkthrough

1. Show `GameState` and store pattern
2. Walk through a tool call execution
3. Explain `on_continue` logic
4. Show scorer metadata

### Live Analysis

1. Run `measure_difficulty.py` on a few words
2. Compare solver strategies on "apple"
3. Show how different metrics rank words
4. Explore `difficulty_report.tsv`

### Quick Eval

```bash
# Run 5 words to show it working
inspect eval hangman_bench/hangman --limit 5 --model openai/gpt-4o-mini
```

---

## Resources

- **Repo**: `/Users/matt/Developer/inspect_ai/hangman-bench`
- **Main code**: `src/hangman_bench/hangman.py` (435 lines)
- **Analysis**: `analysis/` directory with full pipeline
- **Wolfram blog**: <https://blog.wolfram.com/2010/08/13/25-best-hangman-words/>
- **Dan Q's solver**: <https://danq.me/2013/12/15/hangman/>

## Questions to Anticipate

**Q: Why not use the reclassified difficulties?**
A: Results vary by solver strategy. We kept the intuitive labels for stability and because the variance itself is interesting.

**Q: How long does a full eval take?**
A: ~100 samples, depends on model. Use `cost_estimation.py` to project from small runs.

**Q: Can you add more languages?**
A: Yes! Just add to `datasets.py` and extend `Language` enum.

**Q: What if models don't use tools?**
A: `on_continue` provides guidance. Message limit eventually stops runaway cases.

**Q: Is "apple" really hard?**
A: Depends on your solver! That's the whole point. 🍎
