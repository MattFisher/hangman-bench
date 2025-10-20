# Demo Script for Hangman Bench Show-and-Tell

## Pre-Demo Setup

### Terminal Setup

Open 2 terminal windows:

1. **Terminal 1**: For running evals
2. **Terminal 2**: For analysis scripts

### Files to Have Open in IDE

1. `src/hangman_bench/hangman.py` (main code)
2. `src/hangman_bench/datasets.py` (word list)
3. `analysis/difficulty_report.tsv` (metrics)
4. `analysis/README.md` (analysis docs)

---

## Demo 1: Quick Eval Run (5 min)

**Goal**: Show the eval in action

### Terminal 1

```bash
cd /Users/matt/Developer/inspect_ai/hangman-bench

# Run 5 words to show it working
uv run inspect eval hangman_bench/hangman --limit 5 --model openai/gpt-4o-mini

# While it runs, explain:
# - Creating samples from dataset
# - Each sample is one word
# - Model uses hangman_guess() tool repeatedly
# - Grouped scoring by difficulty
```

### What to Point Out

- Watch for tool calls in the output
- See how models adapt strategy based on feedback
- Note the grouped metrics at the end
- Show rich metadata in logs

### Expected Output

```
[Sample 1/5] apple
  Tool: hangman_guess("e")
  Tool: hangman_guess("a")
  ...
  Result: CORRECT

Metrics:
  accuracy/v_easy: 1.0
  accuracy/easy: 0.8
  ...
```

---

## Demo 2: The Apple Problem (10 min)

**Goal**: Show why "apple" difficulty is controversial

### Show the Dataset Classification

```bash
# In IDE, show datasets.py line 42
WordEntry("apple", "v_easy"),  # ← Intuition says "very easy"
```

### Show Wolfram Simulation Result

```bash
# Terminal 2
cd /Users/matt/Developer/inspect_ai/hangman-bench

# Show simulation data
grep "^apple" analysis/SimulationData_parsed.tsv | head -1
```

**Expected output:**

```
apple [list of wrong guesses] 6.667
```

**Explain**: Random weighted solver thinks it's HARD (6.67 wrong guesses)

### Run Our Solvers on "Apple"

Create a quick test script:

```bash
# Terminal 2
cat > /tmp/test_apple.py << 'EOF'
import sys
sys.path.insert(0, '/Users/matt/Developer/inspect_ai/hangman-bench')

from analysis.measure_difficulty import (
    load_wordlist, build_length_index,
    solve_with_strategy, best_move_freq_raw,
    best_move_coverage, best_move_info_gain
)

word = "apple"
dictionary = load_wordlist("analysis/wordlist.txt")
length_idx = build_length_index(dictionary)
dict_5 = length_idx[5]

print(f"Testing word: {word}")
print(f"Dictionary size (5-letter words): {len(dict_5)}")
print()

# Frequency solver
result_freq = solve_with_strategy(word, dict_5, best_move_freq_raw)
print(f"Frequency solver:     {result_freq.wrong_guesses} wrong guesses")

# Coverage solver
result_cov = solve_with_strategy(word, dict_5, best_move_coverage)
print(f"Coverage solver:      {result_cov.wrong_guesses} wrong guesses")

# Info gain solver
result_info = solve_with_strategy(word, dict_5, best_move_info_gain)
print(f"Info gain solver:     {result_info.wrong_guesses} wrong guesses")

print()
print("Why the difference?")
print("- Frequency/InfoGain: 'a' and 'e' are so common they hit quickly")
print("- Coverage: 'p' appears in many words but often in wrong positions")
EOF

uv run python /tmp/test_apple.py
```

**Expected output:**

```
Testing word: apple
Dictionary size (5-letter words): 8636

Frequency solver:     1 wrong guesses
Coverage solver:      5 wrong guesses
Info gain solver:     1 wrong guesses

Why the difference?
- Frequency/InfoGain: 'a' and 'e' are so common they hit quickly
- Coverage: 'p' appears in many words but often in wrong positions
```

### Show the Full Report

```bash
# Terminal 2
# Show apple in the difficulty report
grep "^apple" analysis/difficulty_report.tsv
```

**Expected output:**

```
apple 5 1 5 1 4.981 1.250 3.985
```

**Explain columns:**

- word: apple
- length: 5
- wrong_freq_raw: 1
- wrong_coverage: 5
- wrong_info_gain: 1
- rare_score: 4.981
- dup_factor: 1.250 (5 letters / 4 unique)
- structural_score: 3.985

---

## Demo 3: Compare Multiple Words (5 min)

**Goal**: Show the pattern across different words

### Create Comparison Script

```bash
# Terminal 2
cat > /tmp/compare_words.py << 'EOF'
import sys
sys.path.insert(0, '/Users/matt/Developer/inspect_ai/hangman-bench')

from analysis.measure_difficulty import (
    load_wordlist, build_length_index,
    solve_with_strategy, best_move_freq_raw,
    best_move_coverage, best_move_info_gain
)

words = ["apple", "happy", "puppy", "rhythm", "awkward"]
dictionary = load_wordlist("analysis/wordlist.txt")
length_idx = build_length_index(dictionary)

print(f"{'Word':<10} {'Freq':<6} {'Cover':<6} {'Info':<6} {'Intuition'}")
print("-" * 50)

intuitions = {
    "apple": "v_easy",
    "happy": "v_easy", 
    "puppy": "v_easy",
    "rhythm": "v_hard",
    "awkward": "hard"
}

for word in words:
    dict_n = length_idx.get(len(word), [])
    if not dict_n:
        continue
    
    freq = solve_with_strategy(word, dict_n, best_move_freq_raw)
    cov = solve_with_strategy(word, dict_n, best_move_coverage)
    info = solve_with_strategy(word, dict_n, best_move_info_gain)
    
    print(f"{word:<10} {freq.wrong_guesses:<6} {cov.wrong_guesses:<6} {info.wrong_guesses:<6} {intuitions[word]}")

print()
print("Observations:")
print("- Words with repeated letters (happy, puppy) are harder")
print("- 'Awkward' looks scary but has common letters")
print("- 'Rhythm' (no vowels!) is medium for smart solvers")
EOF

uv run python /tmp/compare_words.py
```

**Expected output:**

```
Word       Freq   Cover  Info   Intuition
--------------------------------------------------
apple      1      5      1      v_easy
happy      8      9      9      v_easy
puppy      9      11     19     v_easy
rhythm     6      6      6      v_hard
awkward    3      3      3      hard

Observations:
- Words with repeated letters (happy, puppy) are harder
- 'Awkward' looks scary but has common letters
- 'Rhythm' (no vowels!) is medium for smart solvers
```

### Show Reclassified Data

```bash
# Terminal 2
# Show how words would be reclassified
head -20 analysis/reclassified_words.tsv | column -t -s $'\t'
```

**Point out interesting cases:**

- apple: v_easy → hard
- happy: v_easy → v_hard
- puppy: v_easy → v_hard
- awkward: hard → easy

---

## Demo 4: Code Walkthrough (15 min)

**Goal**: Show key implementation patterns

### 1. GameState (3 min)

```python
# In IDE: src/hangman_bench/hangman.py lines 119-174

# Show the dataclass
@dataclass
class GameState:
    word: str
    guessed_letters: list[str]
    remaining_guesses: int
    game_over: bool = False
    won: bool = False

# Show the guess method
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

**Key points:**

- Pure game logic, no Inspect code
- Immutable state pattern
- Clear win/loss conditions

### 2. Store Pattern (2 min)

```python
# Lines 176-181

class HangmanStore(StoreModel):
    """Typed interface to the per-sample store."""
    game_state: GameState | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Explain:**

- Per-sample storage
- Type-safe access
- Persists across tool calls

### 3. Tool Implementation (5 min)

```python
# Lines 183-233

@tool(parallel=False)
def hangman_guess() -> Tool:
    async def execute(letter: str) -> str:
        # Access store
        hstore = store_as(HangmanStore)
        game_state = hstore.game_state
        
        # Process guess
        if not game_state.game_over:
            game_state.guess(letter)
        
        # Format feedback
        result_lines = [
            f"Word: {game_state.current_state}",
            f"Remaining guesses: {game_state.remaining_guesses}",
            f"Incorrect guesses: {', '.join(game_state.incorrect_guesses)}",
        ]
        
        return "\n".join(result_lines)
    
    return execute
```

**Key points:**

- `parallel=False` for sequential guessing
- Structured feedback guides strategy
- State mutation happens here

### 4. on_continue Logic (3 min)

```python
# Lines 278-290

async def on_continue(state: AgentState) -> bool | str:
    hstore = store_as(HangmanStore)
    game_state = hstore.game_state
    
    # Auto-stop when game ends
    if game_state is None or game_state.game_over:
        return False
    
    # Continue if last response was a tool call
    if state.output.message.tool_calls:
        return True
    
    # Otherwise provide guidance
    guidance = "Continue by calling hangman_guess('a')..."
    return guidance
```

**Key points:**

- Automatic stopping
- Graceful degradation
- Returns bool or string

### 5. Message Limit (2 min)

```python
# Lines 113-116

def _calculate_message_limit(word_length: int, max_guesses: int) -> int:
    # 4 messages per guess: commentary, continue, tool call, tool response
    return (word_length + max_guesses) * 4 + NUM_ALLOWABLE_EXTRA_MESSAGES
```

**Explain why 4x multiplier**

---

## Demo 5: Analysis Pipeline (5 min)

**Goal**: Show the reproducible analysis

### Show the Pipeline

```bash
# Terminal 2

# 1. Ingest simulation data (if not already done)
ls -lh analysis/SimulationData_parsed.tsv

# 2. Extract wordlist
head -20 analysis/wordlist.txt

# 3. Show difficulty report
head -20 analysis/difficulty_report.tsv | column -t -s $'\t'

# 4. Show binned difficulties
head -20 analysis/difficulty_binned.tsv | column -t -s $'\t'
```

### Explain Each Step

1. **Ingest**: Parse Wolfram's simulation output
2. **Extract**: Build dictionary from simulation words
3. **Measure**: Run 3 solvers on each dataset word
4. **Bin**: Classify into tiers by quantiles

### Show It's Reproducible

```bash
# Terminal 2

# Re-run difficulty measurement on a subset
uv run python -c "
import sys
sys.path.insert(0, '/Users/matt/Developer/inspect_ai/hangman-bench')

from analysis.measure_difficulty import *

# Just test a few words
test_words = ['apple', 'happy', 'rhythm']
dictionary = load_wordlist('analysis/wordlist.txt')
length_idx = build_length_index(dictionary)

for word in test_words:
    dict_n = length_idx.get(len(word), [])
    freq = solve_with_strategy(word, dict_n, best_move_freq_raw)
    cov = solve_with_strategy(word, dict_n, best_move_coverage)
    print(f'{word}: freq={freq.wrong_guesses}, cov={cov.wrong_guesses}')
"
```

---

## Demo 6: Cost Estimation (3 min)

**Goal**: Show practical eval planning

### Show Cost Estimation Code

```python
# In IDE: analysis/cost_estimation.py

# Show the key functions:
# - summarise_eval_log()
# - scale_usage()
# - estimate_cost()
# - project_costs()
```

### Example Usage

```bash
# Terminal 2
uv run python -c "
from inspect_ai.log import read_eval_log
from hangman_bench.analysis.cost_estimation import *

# If you have a log file from earlier eval
# log = read_eval_log('logs/some_eval.json')
# summary = summarise_eval_log(log)
# print(f'Samples: {summary.samples}')
# print(f'Avg tokens per sample: {summary.mean_total_tokens:.0f}')

# Project to full 100 words
# scaled = scale_usage(summary, 100)
# print(f'Projected for 100 words: {scaled.total_tokens} tokens')

print('Cost estimation utilities available in analysis/cost_estimation.py')
print('Use after running evals to project costs for full benchmark')
"
```

---

## Backup Demos (If Time Permits)

### Backup 1: Run Zen Hangman

```bash
# Terminal 2
uv run analysis/zen_hangman.py --wordlist analysis/wordlist.txt --num-letters 5 | head -20
```

Shows the frequency solver in action on 5-letter words.

### Backup 2: Show Different Difficulty Levels

```bash
# Terminal 1
# Run only v_hard words
uv run inspect eval hangman_bench/hangman -T difficulty=v_hard --limit 5
```

### Backup 3: Allow Word Guesses

```bash
# Terminal 1
# Let models guess the full word early
uv run inspect eval hangman_bench/hangman -T allow-word-guesses=True --limit 5
```

---

## Troubleshooting

### If Eval Fails

- Check API keys in `.env`
- Try `--model openai/gpt-4o-mini` (cheaper, faster)
- Use `--limit 1` for quick test

### If Analysis Scripts Fail

- Make sure you're in repo root
- Check that `analysis/wordlist.txt` exists
- Run `uv sync --dev` to ensure dependencies

### If Demo Runs Long

- Skip Demo 5 (analysis pipeline)
- Shorten Demo 4 (code walkthrough) to just store pattern and tool
- Focus on Demo 2 (apple problem) - that's the most interesting part

---

## Timing Summary

- Demo 1 (Quick Eval): 5 min
- Demo 2 (Apple Problem): 10 min ⭐ **Most important**
- Demo 3 (Compare Words): 5 min
- Demo 4 (Code Walkthrough): 15 min
- Demo 5 (Analysis Pipeline): 5 min
- Demo 6 (Cost Estimation): 3 min
- **Total**: 43 minutes

**Recommended for 30-min slot:**

- Demo 2 (10 min)
- Demo 4 (15 min)
- Demo 1 (5 min)
- Total: 30 min

**Recommended for 45-min slot:**

- All demos except Demo 5
- Total: 38 min + questions

---

## Key Messages to Emphasize

1. **Started as a tool demonstration** → became interesting research
2. **Store pattern is powerful** for stateful interactions
3. **"Apple" is hard** (depending on your solver!)
4. **No single correct difficulty metric** - that's the insight
5. **All analysis is reproducible** - code included
6. **Games make great evals** - clear objectives, natural tool use

---

## Questions to Anticipate

**Q: Why not use the reclassified difficulties?**
A: Results vary by solver. We kept intuitive labels for stability.

**Q: What models work well?**
A: GPT-4o, Claude 3.5 Sonnet work great. Smaller models struggle with tool use.

**Q: Can you add more languages?**
A: Yes! Just extend `datasets.py`. Need word lists and difficulty ratings.

**Q: How long for full eval?**
A: ~100 samples. Use cost estimation on small run to project.

**Q: What's the hardest word?**
A: Depends on solver! "pyjamas" (19 wrong) for coverage, "pizza" (22 wrong) for info gain.
