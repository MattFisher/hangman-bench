# Game Transcript Extraction

Two scripts for extracting game transcripts from Hangman eval logs for use in presentations.

## Scripts

### 1. `extract_game_transcript.py` - Full Transcripts

Extracts complete game transcripts showing all model commentary, tool calls, and tool responses.

**Usage:**

```bash
# Extract first sample
uv run analysis/extract_game_transcript.py --log logs/your-log.eval

# Extract specific sample by word
uv run analysis/extract_game_transcript.py --log logs/your-log.eval --sample apple

# Extract specific sample by index
uv run analysis/extract_game_transcript.py --log logs/your-log.eval --sample 0

# Limit to first N turns
uv run analysis/extract_game_transcript.py --log logs/your-log.eval --sample apple --max-turns 5

# Show all samples
uv run analysis/extract_game_transcript.py --log logs/your-log.eval --all

# Output in markdown format (for slides)
uv run analysis/extract_game_transcript.py --log logs/your-log.eval --sample apple --format markdown

# Show system message
uv run analysis/extract_game_transcript.py --log logs/your-log.eval --sample apple --show-system
```

**Example Output:**

```text
Model: "To start off the game, I will guess the letter 'e'. It is the most common vowel..."
User: Continue by calling hangman_guess('a') (replace 'a' with your next letter).
Model: [calls hangman_guess("e")]
Tool: Word: _ _ _ _ _ _ _
      Status: Game continues
Model: "The letter 'e' was not in the word. I have 9 guesses remaining..."
User: Continue by calling hangman_guess('a') (replace 'a' with your next letter).
Model: [calls hangman_guess("a")]
Tool: Word: a _ _ _ a _ _
      Status: Game continues
```

### 2. `extract_game_snippet.py` - Short Snippets for Slides

Extracts just a few turns showing the commentary pattern, perfect for presentation slides.

### 3. `extract_game_table.py` - Table Format for Slides

Extracts game transcript as a clean table showing Turn | Commentary | Guess | Game State.

**Usage:**

```bash
# Markdown table (default)
uv run analysis/extract_game_table.py --log logs/your-log.eval --sample apple --turns 5

# ASCII table
uv run analysis/extract_game_table.py --log logs/your-log.eval --sample apple --turns 5 --format ascii

# CSV format
uv run analysis/extract_game_table.py --log logs/your-log.eval --sample apple --turns 5 --format csv
```

**Example Output (Markdown):**

```markdown
| Turn | Commentary | Guess | Game State |
|------|------------|-------|------------|
| 1 | To start off the game, I will guess the letter 'e' | `e` | `_ _ _ _ _ _ _` |
| 2 | The letter 'e' was not in the word | `a` | `a _ _ _ a _ _` |
| 3 | The letter 'a' is in the word! | `r` | `a _ _ _ a r _` |
```

**Example Output (ASCII):**

```text
Turn   | Commentary                                         | Guess  | Game State          
-------------------------------------------------------------------------------------------
1      | To start off the game, I will guess the letter 'e' | e      | _ _ _ _ _ _ _       
2      | The letter 'e' was not in the word                 | a      | a _ _ _ a _ _       
3      | The letter 'a' is in the word!                     | r      | a _ _ _ a r _       
```

## `extract_game_snippet.py` Usage

```bash
# Extract 2 turns (default)
uv run analysis/extract_game_snippet.py --log logs/your-log.eval --sample apple

# Extract 3 turns
uv run analysis/extract_game_snippet.py --log logs/your-log.eval --sample apple --turns 3

# Use sample index
uv run analysis/extract_game_snippet.py --log logs/your-log.eval --sample 0 --turns 2
```

**Example Output:**

```text
Model: "To start off the game, I will guess the letter 'e'."
User:  "Continue by calling hangman_guess('a') (replace 'a' with your next letter)."
Model: [calls hangman_guess("e")]
Model: "The letter 'e' was not in the word."
User:  "Continue by calling hangman_guess('a') (replace 'a' with your next letter)."
Model: [calls hangman_guess("a")]
```

## Use Cases

### For Slides

Use `extract_game_snippet.py` with `--turns 2` to show the commentary pattern:

```bash
uv run analysis/extract_game_snippet.py \
  --log logs/2025-09-19T12-41-16+10-00_hangman_bxwgPm2hugB3hdg2LVhgCR.eval \
  --sample 0 \
  --turns 2
```

Copy the output directly into your slide's code block.

### For Analysis

Use `extract_game_transcript.py` with `--max-turns` to see full game flow:

```bash
uv run analysis/extract_game_transcript.py \
  --log logs/2025-09-19T12-41-16+10-00_hangman_bxwgPm2hugB3hdg2LVhgCR.eval \
  --sample apple \
  --max-turns 10 \
  --format markdown
```

### For Documentation

Use `extract_game_transcript.py` with `--all` to document all games:

```bash
uv run analysis/extract_game_transcript.py \
  --log logs/2025-09-19T12-41-16+10-00_hangman_bxwgPm2hugB3hdg2LVhgCR.eval \
  --all \
  --format markdown > game_transcripts.md
```

## Finding Log Files

List available logs:

```bash
ls -lh logs/*.eval
```

Most recent log:

```bash
ls -t logs/*.eval | head -1
```

## Tips

1. **For slides**: Use `extract_game_snippet.py` with 2-3 turns
2. **For debugging**: Use `extract_game_transcript.py` with `--max-turns 10`
3. **For patterns**: Extract multiple samples and compare
4. **For presentations**: Use markdown format for easy copy-paste

## Example Workflow

```bash
# Find latest log
LATEST_LOG=$(ls -t logs/*.eval | head -1)

# Extract snippet for slide
uv run analysis/extract_game_snippet.py --log $LATEST_LOG --sample 0 --turns 2

# Extract full transcript for analysis
uv run analysis/extract_game_transcript.py --log $LATEST_LOG --sample 0 --max-turns 8
```

## Output Formats

### Text Format (default)
Plain text output, good for terminal viewing

### Markdown Format
Wrapped in markdown code blocks, ready for slides or documentation

```bash
--format markdown
```
