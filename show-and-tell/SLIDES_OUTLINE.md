---
marp: true
theme: 
paginate: true
---
<!-- markdownlint-disable-file MD025 -->

# Hangman Bench

**Building Game Evals with Inspect Tools**
*An interesting eval that started as a tool demonstration*

---

# What is Hangman Bench?

- Eval for testing AI models playing Hangman
- Built using Inspect framework's tool features
- 100 words across 5 difficulty levels
- Started as a **tool demonstration**, became a **research project**

---

# The Original Goal

**Demonstrate Inspect's Tool Capabilities**

- How to build stateful, interactive evaluations
- How to implement games with tools
- Pattern for multi-turn tool interactions
- Clean state management across tool calls

---

# Architecture Overview

```
┌─────────────────────────────────────┐
│  @task hangman()                    │
│  - Creates samples from dataset     │
│  - Configures solver & scorer       │
└──────────┬──────────────────────────┘
           │
┌──────────▼──────────────────────────┐
│  @solver game_initialiser()         │
│  - Sets up GameState in store       │
└──────────┬──────────────────────────┘
           │
┌──────────▼──────────────────────────┐
│  @solver hangman_player()           │
│  - react agent with tools           │
│  - on_continue logic                │
└──────────┬──────────────────────────┘
           │
┌──────────▼──────────────────────────┐
│  @tool hangman_guess()              │
│  - Processes letter guesses         │
│  - Updates GameState                │
│  - Returns formatted feedback       │
└──────────┬──────────────────────────┘
           │
┌──────────▼──────────────────────────┐
│  @scorer game_scorer()              │
│  - Checks win/loss                  │
│  - Grouped by difficulty            │
│  - Rich metadata                    │
└─────────────────────────────────────┘
```

---

# Key Pattern - Typed Store

**Problem**: Need to maintain game state across tool calls

**Solution**: Typed store model

```python
class HangmanStore(StoreModel):
    game_state: GameState | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

# In tool:
hstore = store_as(HangmanStore)
game_state = hstore.game_state
game_state.guess(letter)  # Mutates state
```

- Type-safe per-sample state
- Persists across tool calls
- Clean separation of concerns

---

# Key Pattern - Tool Feedback

**Design tool output to guide model strategy**

```python
async def execute(letter: str) -> str:
    game_state.guess(letter)
    
    return "\n".join([
        f"Word: {game_state.current_state}",      # a p p l e
        f"Remaining guesses: {game_state.remaining_guesses}",
        f"Incorrect guesses: {', '.join(game_state.incorrect_guesses)}",
        f"Status: {'WON!' if game_state.won else 'Game continues'}"
    ])
```

Structured feedback helps models:

- Track progress
- Avoid repeated guesses
- Adapt strategy

---

# Key Pattern - on_continue

**Problem**: Models sometimes forget to use tools or don't stop

**Solution**: Custom continuation logic

```python
async def on_continue(state: AgentState) -> bool | str:
    game_state = store_as(HangmanStore).game_state
    
    # Auto-stop when game ends
    if game_state.game_over:
        return False
    
    # Encourage tool use
    if not state.output.message.tool_calls:
        return "Continue by calling hangman_guess('a')..."
    
    return True
```

---

# Key Pattern - Message Limits

**Calculate based on game complexity**

```python
def _calculate_message_limit(word_length: int, max_guesses: int) -> int:
    # Models sometimes need multiple messages per guess:
    # 1. Commentary
    # 2. "Continue" prompt
    # 3. Tool call
    # 4. Tool response
    return (word_length + max_guesses) * 4 + EXTRA_BUFFER
```

Prevents runaway conversations while allowing flexibility

---

# The Dataset

**100 words, 5 difficulty tiers (20 each)**

```python
ENGLISH_WORDS = [
    WordEntry("apple", "v_easy"),
    WordEntry("happy", "v_easy"),
    WordEntry("awkward", "hard"),
    WordEntry("rhythm", "v_hard"),
    # ... 96 more
]
```

Initially classified by **intuition** (via Windsurf AI)

But is "apple" really very easy? 🤔

---

# The Apple Problem 🍎

**Question**: How do we objectively measure word difficulty?

We went down a research rabbit hole...

---

# Attempt 1 - Wolfram Simulation

Found existing data: Wolfram's 2010 simulation

- 90,000+ words
- Random weighted solver
- Mean wrong guesses per word

**Result for "apple"**: 6.67 wrong guesses (HARD!)

**Problem**: Random solver ≠ optimal play

---

## Attempt 2 - Build Our Own Solvers

Implemented **3 deterministic strategies**:

1. **Raw Frequency** - Count letter occurrences (with duplicates)
2. **Coverage** - Count unique word incidence (no duplicates)  
3. **Information Gain** - Minimize expected remaining candidates

All filter by known positions + wrong guesses

---

# The Apple Paradox

**Same word, different solvers, wildly different results:**

| Word   | Freq | Coverage | InfoGain | Intuition |
|--------|------|----------|----------|-----------|
| apple  | 1    | **5**    | 1        | v_easy    |
| happy  | 8    | 9        | 9        | v_easy    |
| banana | 2    | 4        | 11       | v_easy    |
| rhythm | 6    | 6        | 6        | v_hard    |

**Why?** Coverage solver: "p" appears in many words but often in wrong positions

---

# More Surprises

**Words that got harder:**

- `apple`: v_easy → hard (6.67 wrong)
- `happy`: v_easy → v_hard (8.72 wrong)
- `puppy`: v_easy → v_hard (9.89 wrong)

**Words that got easier:**

- `awkward`: hard → easy (5.30 wrong)
- `rhythm`: v_hard → medium (5.54 wrong)

**Pattern**: Repeated letters make words harder!

---

# Why We Kept Original Labels

**Decision**: Don't reclassify the dataset

**Reasons**:

1. Results vary by solver strategy (no "correct" answer)
2. Human intuition matters for eval usability
3. Eval stability (don't change ground truth)
4. The variance itself is valuable data

**All analysis code included** for reproducibility

---

# What We Built (Analysis)

Complete analysis pipeline in `analysis/`:

- `ingest_simulation.py` - Parse Wolfram data
- `extract_wordlist.py` - Build 90k word dictionary
- `zen_hangman.py` - Frequency solver (Dan Q's algorithm)
- `measure_difficulty.py` - Multi-strategy measurement
- `bin_difficulty.py` - Quantile-based reclassification
- `cost_estimation.py` - Project eval costs

All reproducible with `uv run analysis/...`

---

# It's Actually a Good Eval

**Tests multiple capabilities:**

1. **Strategic Tool Use** - Multi-turn planning with feedback
2. **Language Understanding** - Letter frequency, word patterns
3. **Probabilistic Reasoning** - Bayesian updates, candidate pruning
4. **Instruction Following** - Correct tool usage, no repeats

**Features:**

- Grouped scoring by difficulty
- Rich metadata (all guesses, game trajectory)
- Configurable (max guesses, languages, word guesses)
- Cost estimation utilities

---

# Usage Examples

```bash
# Basic eval
inspect eval hangman_bench/hangman --model openai/gpt-4o

# Quick test
inspect eval hangman_bench/hangman --limit 10

# Specific difficulty
inspect eval hangman_bench/hangman -T difficulty=v_hard

# Allow early word guesses
inspect eval hangman_bench/hangman -T allow-word-guesses=True
```

Published as `hangman-bench` on PyPI

---

# Key Takeaways - Building Games

**Patterns for game-based evals:**

1. ✅ **Typed store models** - Clean state management
2. ✅ **Tool feedback design** - Guide model strategy
3. ✅ **Message limits** - Calculate from game complexity
4. ✅ **Graceful degradation** - Handle non-compliant models
5. ✅ **Rich metadata** - Capture full trajectory

**Games make excellent evals** - Clear objectives, natural tool use

---

# Key Takeaways - Research

**Lessons from the "apple problem":**

1. 🍎 **Intuition ≠ Objective Difficulty**
2. 📊 **Solver strategy matters** - Different objectives, different rankings
3. 🔢 **Multiple metrics are valuable** - No single "correct" measure
4. 📝 **Document the journey** - Failed experiments are interesting
5. 🔬 **Make it reproducible** - Include all analysis code

---

# Demo Time

**Options:**

1. Code walkthrough (GameState, tool, on_continue)
2. Run analysis on "apple" vs "happy"
3. Quick eval (5 words)
4. Explore difficulty metrics

---

# Questions?

**Resources:**

- Repo: `hangman-bench` on GitHub
- PyPI: `pip install hangman-bench`
- Wolfram blog: "25 Best Hangman Words"
- Dan Q's solver: "The Hardest Hangman"

**Is "apple" really hard?**
*Depends on your solver!* 🍎

---

## Backup Slides

### Backup: Code - GameState

```python
@dataclass
class GameState:
    word: str
    guessed_letters: list[str]
    remaining_guesses: int
    game_over: bool = False
    won: bool = False

    @property
    def current_state(self) -> str:
        """Returns word with unguessed letters as '_'"""
        return " ".join(
            letter if letter in self.guessed_letters else "_" 
            for letter in self.word
        )

    def guess(self, letter: str) -> "GameState":
        """Process a letter guess and return new state"""
        if letter not in self.word:
            self.remaining_guesses -= 1
        
        self.guessed_letters.append(letter)
        
        # Check win/loss conditions
        if all(l in self.guessed_letters for l in self.word):
            self.game_over = True
            self.won = True
        elif self.remaining_guesses <= 0:
            self.game_over = True
        
        return self
```

### Backup: Solver Strategies Explained

**Raw Frequency:**

```python
# Count all letter occurrences (duplicates count)
for word in candidates:
    for letter in word:
        counts[letter] += 1
# Choose letter with max count
```

**Coverage:**

```python
# Count unique word incidence
for word in candidates:
    unique_letters = set(word)
    for letter in unique_letters:
        counts[letter] += 1
# Choose letter appearing in most words
```

**Information Gain:**

```python
# Partition by position masks
for letter in candidates:
    for word in dictionary:
        mask = tuple(i for i, c in enumerate(word) if c == letter)
        partitions[mask].append(word)
    
    # Score = sum of squared partition sizes
    score = sum(len(p)**2 for p in partitions.values())

# Choose letter minimizing expected remaining size
```

### Backup: Difficulty Metrics

From `measure_difficulty.py`, we compute:

- `wrong_freq_raw` - Wrong guesses (frequency solver)
- `wrong_coverage` - Wrong guesses (coverage solver)
- `wrong_info_gain` - Wrong guesses (info gain solver)
- `rare_score` - Sum of -log(p(letter|length))
- `dup_factor` - len(word) / len(unique_letters)
- `structural_score` - rare_score / dup_factor

All metrics tell different stories!

### Backup: Cost Estimation

```python
from hangman_bench.analysis.cost_estimation import (
    summarise_eval_logs,
    project_costs,
    ModelPricing
)

# Load eval logs
logs = [read_eval_log("path/to/log.json")]

# Summarize usage
summaries = summarise_eval_logs(logs)

# Project costs for full 100-word eval
pricings = [
    ModelPricing("openai", "gpt-4o", 2.50, 10.00),
    ModelPricing("anthropic", "claude-3-5-sonnet", 3.00, 15.00),
]

estimates = project_costs(summaries, target_games=100, pricings=pricings)
```
