# Hangman Evaluation

An evaluation for testing AI models' ability to play the classic game of Hangman.

## Installation

```bash
pip install hangman-eval
```

## Usage

```python
from hangman_eval import hangman

# Create a hangman task
task = hangman(
    language="english",  # or another supported language
    difficulty=3,        # 1-5, or None for mixed
    max_guesses=6,       # number of allowed incorrect guesses
    shuffle=True,        # whether to shuffle words
    allow_word_guesses=False  # whether to allow guessing the full word
)

# Use with an inspect-ai solver
result = await task.run(solver=solver)
```

## Development

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
