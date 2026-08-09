# Hangman Benchmark

A benchmark for testing AI models' ability to play the classic game of Hangman.
Uses the [Inspect framework](https://github.com/UKGovernmentBEIS/inspect_ai) to evaluate the models.

This eval was built as a demonstration of how to enable models to play games using tools within the Inspect framework.

## Installation

### Using pip
```bash
pip install hangman-bench
```

### Using uv (recommended)
```bash
uv add hangman-bench
```

## Usage

```bash
inspect eval hangman_bench/hangman

# To limit the number of samples to 10
inspect eval hangman_bench/hangman --limit=10

# To evaluate against a specific model
inspect eval hangman_bench/hangman --model openai/gpt-4o-mini

# To allow the model to guess the word before guessing all letters
inspect eval hangman_bench/hangman -T allow-word-guesses=True
```

### Scoring

Two scorers run by default:

- `game_scorer` — did the agent win, grouped by difficulty.
- `oracle_scorer` — *how* the game was played, by replaying each guess against
  the exactly computable posterior over the hidden word. Reports the rate of
  provably wrong moves (guessing a letter that appears in no remaining
  candidate word, repeats, malformed input) and how far each guess fell below
  the best available one.

Because `oracle_scorer` is a normal Inspect scorer, its metrics can be added to
logs that were produced without it:

```bash
inspect score <log.eval> \
  --scorer src/hangman_bench/hangman.py@oracle_scorer \
  --action append --overwrite
```

See [analysis/README.md](analysis/README.md) for the metric definitions.

### Task Parameters

- `language`: The language to use for the words (default: "english")
- `difficulty`: Specific difficulty level to use (1-5), or None for mixed difficulties (default: None)
- `max_guesses`: Maximum number of incorrect guesses allowed (default: 10)
- `shuffle`: Whether to shuffle the words before playing (default: True)
- `allow_word_guesses`: Whether to allow the model to guess the word before guessing all letters (default: False)

## Development

### Using uv (recommended)
1. Clone the repository
2. Install with development dependencies:
   ```bash
   uv sync --dev
   ```
3. Run tests:
   ```bash
   uv run pytest
   ```
4. Run evaluations:
   ```bash
   uv run inspect eval hangman_bench/hangman
   ```

### Using pip
1. Clone the repository
2. Install with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## License

MIT
