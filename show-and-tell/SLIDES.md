---
marp: true
theme: default
class:
  - invert
paginate: true
---
<!-- markdownlint-disable-file MD025 no-space-in-emphasis no-inline-html first-line-heading -->

<style>
section {
  padding: 40px 40px;  /* Reduced from default ~70px */
}
ul {
  margin: 0.5em 0;
}
ul li {
  margin: 0.2em 0;
}
</style>
<style scoped>
section {
  padding: 50px 70px;
}
</style>

# Hangman Bench

**Building Game Evals with Inspect Tools**

*An interesting eval that started as a tool demonstration*

```
  +---+
  |   |
  O   |
 /|\  |
 / \  |
      |
=========
_ p p _ e
```

---

# What is Hangman Bench?

- Eval for testing AI models playing Hangman
- Built using Inspect framework's tool features
- 100 words across 5 difficulty levels
- Started as a **tool demonstration**, became a **research project**
- Hypothesis: Models are better at working with whole words than letter-by-letter :strawberry:

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
┌──────────▼──────────────────────────┐
│  @solver game_initialiser()         │
│  - Sets up GameState in store       │
└──────────┬──────────────────────────┘
┌──────────▼──────────────────────────┐
│  @solver hangman_player()           │
│  - react agent with tools           │
│  - on_continue logic                │
└──────────┬──────────────────────────┘
┌──────────▼──────────────────────────┐
│  @tool hangman_guess()              │
│  - Processes letter guesses         │
│  - Updates GameState                │
│  - Returns formatted feedback       │
└──────────┬──────────────────────────┘
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
        f"Word: {game_state.current_state}",      # a _ _ l e
        f"Remaining guesses: {game_state.remaining_guesses}",
        f"Incorrect guesses: {', '.join(game_state.incorrect_guesses)}",
        f"Status: {'WON!' if game_state.won else 'Game continues'}"
    ])
```

Structured feedback helps models track progress

Could also provide feedback as a JSON string (tool results are strings, not structured data)

---

# Key Pattern - on_continue

We don't want to rely on the model calling the `submit` tool to end the interaction after the game is already over, so we use `on_continue` to look at the game state and stop when the game is over.

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

## Calculate based on game complexity

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
<style scoped>
ul {
  margin: 0.4em 0;
}
ul li {
  line-height: 1.0em;
}
</style>

# Model Behavior: Commentary Problem

**Observation**: Many models narrate before acting

```text
Model: "I'll start by guessing the most common letter 'e'"
User:  "Continue by calling hangman_guess('a') (replace 'a' with your next letter)."
Model: [calls hangman_guess("e")]
```

**Impact**:

- Roughly **doubles the number of turns** taken
- Why we use `4x` multiplier in message limits
- Not a failure - just chattier behavior

**Mitigation**:

- `on_continue` encourages direct tool use
- Message limits prevent runaway
- Accept it as part of model personality

---

# Optional Feature: Whole-Word Guesses

**Standard mode**: Must guess all letters one-by-one

**With `allow-word-guesses=True`**: Can submit early

```python
# Model can call submit("apple") instead of guessing remaining letters
return as_solver(
    react(
        tools=[hangman_guess()],
        submit=AgentSubmit(answer_only=True) if allow_word_guesses else False,
    )
)
```

---

# Performance Impact: Word Guesses

**Key Finding**: Performance **improves** with word guesses enabled

**Why?**

- Comparing model capabilities (letter-by-letter vs. word guesses)
- In theory, if you know the word you can guess all the letters
- Not the case in practice!
- Hypothesis: Models are better at working with whole words than letter-by-letter
- 'How many R's are there in "strawberry"?' :strawberry:

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

I went down a research rabbit hole...

---

# Attempt 1 - Wolfram Simulation

Found existing data: Wolfram's 2010 simulation

- 90,000+ words
- Weighted ordering of letters with random number generator
- Mean wrong guesses per word

**Result for "apple"**: 6.67 wrong guesses (HARD!)

**Problem**: This solver ≠ optimal play

---

## Attempt 2 - Find existing implementations online and port to python

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

## Solver Strategies Explained

### Raw Frequency

```python
# Count all letter occurrences (duplicates count)
for word in candidates:
    for letter in word:
        counts[letter] += 1
# Choose letter with max count
```

### Coverage

```python
# Count unique word incidence
for word in candidates:
    unique_letters = set(word)
    for letter in unique_letters:
        counts[letter] += 1
# Choose letter appearing in most words
```

---

## Solver Strategies Explained

### Information Gain

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

---

<style scoped>
section {
  padding: 40px 40px;  /* top/bottom left/right */
}
table {
  font-size: 0.45em;
}
h3 {
  font-size: 1.0em;
  margin-bottom: 0.4em;
}
td:nth-child(4), th:nth-child(4) {
  min-width: 80px;
}
</style>

| Turn | Model: gpt-4o-mini; Sample: banana | Guess | Game State |
|------|------------|-------|------------|
| 1 | Since the word is six letters long, I will start by guessing the letter "e". The letter "e" is the most common letter in the English language and could help reveal some letters in the word. Let's make that guess: `hangman_guess("e")`. | `a` | `_ a _ a _ a` |
| 2 | The current state of the word is "*a* a _ a", which means there are two 'a's in the word. Next, I will guess the letter "r," as it is also a common consonant in English and might help narrow down the word further. I'll make that guess: `hangman_guess("r")`. | `r` | `_ a _ a _ a` |
| 3 | The letter "r" was incorrect. The current state of the word is still "*a* a _ a". Next, I will guess the letter "c." It's a commonly used consonant and might fit within the word structure. I'll make that guess: `hangman_guess("c")`. | `c` | `_ a _ a _ a` |
| 4 | The letter "c" was also incorrect. The word still remains "*a* a _ a". Next, I will guess the letter "t," as it's a common consonant that might fit well in this structure. I will make the guess: `hangman_guess("t")`. | `t` | `_ a _ a _ a` |
| 5 | The letter "t" was incorrect as well. The state of the word remains "*a* a _ a". Next, I will guess the letter "s." It is quite common in English and should help narrow down the possibilities. I'll make the guess: `hangman_guess("s")`. | `s` | `_ a _ a _ a` |
| 6 | The letter "s" was also incorrect. The word remains "*a* a _ a". At this point, I will guess the letter "h." It is another common consonant and could potentially fit in the word. I'll make that guess: `hangman_guess("h")`. | `h` | `_ a _ a _ a` |
| 7 | The letter "h" was incorrect as well. The word is still "*a* a _ a". Next, I will guess the letter "l." It's a common consonant that could fit well in the pattern. I'll make that guess: `hangman_guess("l")`. | `l` | `_ a _ a _ a` |
| 8 | The letter "l" was also incorrect. The word remains "*a* a _ a". Next, I'll guess the letter "n." It's a common consonant that might fit in the word structure. I'll make that guess: `hangman_guess("n")`. | `n` | `_ a n a n a` |
| 9 | The current state of the word is "_ a n a n a". This tells us that the letters "n" have been revealed, providing more insight into the word. Now, I will guess the letter "p," as it might fit before the first "a" in the pattern. I'll make that guess: `hangman_guess("p")`. | `p` | `_ a n a n a` |
| 10 | The letter "p" was incorrect. The word remains "_ a n a n a". Next, I will guess the letter "m." It could fit well in the first position. I'll make that guess: `hangman_guess("m")`. | `m` | `_ a n a n a` |
| 11 | The letter "m" was also incorrect. The word continues to be "_ a n a n a". At this stage, I will guess the letter "g". It might fit well in the first position of the word. I'll make that guess: `hangman_guess("g")`. | `g` | `_ a n a n a` |
| 12 | The letter "g" was incorrect as well. The word still is "_ a n a n a". With only one guess remaining, I will make a strategic guess by trying the letter "f." It’s less common, but it could fit into the word. I'll make that guess: `hangman_guess("f")`. | `f` | `_ a n a n a` Game over

---
<style scoped>
section {
  font-size: 2.0em;
}
</style>

# Key quotes

## gpt-4o-mini

Banana
> The letter 'k' was not in the word. The current pattern remains _ a n a n a.
> At this point, I have very few guesses left. I suspect that the word might be "banana".
> Let's submit the guess for the whole word: submit("banana").

Bagpipes (with word guesses)
> The letter 'Y' is not in the word, which leaves me with 3 remaining guesses. The word state remains "_ a _ p _ p e s".
> Given the current state, it is clear that the word is "happiness." Since I have already provided incorrect guesses for 'H' and an additional letter, I'll now submit the entire word "happiness."
> Let’s submit that as my final guess!

---

### Model: gpt-4o; Sample: banana

<style scoped>
section {
  padding: 20px 40px;  /* top/bottom left/right */
}
table {
  font-size: 0.6em;
}
h3 {
  font-size: 1.0em;
  margin-bottom: 0.4em;
}
td:nth-child(4), th:nth-child(4) {
  min-width: 120px;
}
</style>

| Turn | Commentary | Guess |  State |
|------|------------|-------|------------|
| 1 | Let's start by guessing a common letter in English. The letter "e" is the most frequently used letter, so it's a good starting point. I'll make that guess now. | `e` | `_ _ _ _ _ _` |
| 2 | The letter "e" is not in the word. Let's try another common letter. The letter "a" is also frequently used in English words, so I'll guess "a" next. | `a` | `_ a _ a _ a` |
| 3 | Great! The letter "a" appears three times in the word: _ a _ a _ a. Based on this pattern, I'll guess the letter "r" next, which is also common and might fit into the pattern. | `r` | `_ a _ a _ a` |
| 4 | The letter "r" is not in the word. Let's try another common letter, "s", which might fit into the pattern as well. | `s` | `_ a _ a _ a` |
| 5 | The letter "s" is also not in the word. Let's try "l" next, as it is a common consonant and might fit the pattern of _ a _ a _ a. | `l` | `_ a _ a _ a` |
| 6 | The letter "l" is not in the word either. Let's try the letter "t" next, which is common and might fit into the pattern. | `t` | `_ a _ a _ a` |
| 7 | The letter "t" is not in the word. Let's try the letter "m" this time, as it might work in the pattern _ a _ a _ a. | `m` | `_ a _ a _ a` |
| 8 | The letter "m" is not in the word either. Let's try the letter "c" next, which might fit this pattern. | `c` | `_ a _ a _ a` |
| 9 | The letter "c" is not in the word. Let's try the letter "n", which is a common letter and might fit into our pattern. | `n` | `_ a n a n a` |
| 10 | The letter "n" fits! The current word is _ a n a n a. Now, let's try the letter "b", which might help us complete this word. | `b` | `b a n a n a` |

---
<!-->
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

---

### Backup: Difficulty Metrics

From `measure_difficulty.py`, we compute:

- `wrong_freq_raw` - Wrong guesses (frequency solver)
- `wrong_coverage` - Wrong guesses (coverage solver)
- `wrong_info_gain` - Wrong guesses (info gain solver)
- `rare_score` - Sum of -log(p(letter|length))
- `dup_factor` - len(word) / len(unique_letters)
- `structural_score` - rare_score / dup_factor

All metrics tell different stories!

---

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

---
-->