# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Hangman benchmark for testing AI models' ability to play the classic word game using the [Inspect framework](https://github.com/UKGovernmentBEIS/inspect_ai). It implements a structured evaluation where models must guess letters to uncover words across different languages and difficulty levels.

## Development Commands

### Installation & Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if contributing)
pre-commit install
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov
```

### Code Quality
```bash
# Format and lint code (development dependency)
ruff check
ruff format
```

### Running Evaluations
```bash
# Basic evaluation
inspect eval hangman_bench/hangman

# Limit samples
inspect eval hangman_bench/hangman --limit=10

# Specific model
inspect eval hangman_bench/hangman --model openai/gpt-4o-mini

# Task parameters
inspect eval hangman_bench/hangman -T allow-word-guesses=True
inspect eval hangman_bench/hangman -T difficulty=3
inspect eval hangman_bench/hangman -T max_guesses=8
```

## Architecture

### Core Components
The evaluation is built around three main modules:

- **hangman.py**: Main evaluation logic with `@task`, `@solver`, `@scorer`, and `@tool` decorators
- **datasets.py**: Word datasets with difficulty ratings (1-5 scale) and language support
- **__init__.py**: Package exports

### Key Classes and Functions

**GameState** (`hangman.py:109`): Manages hangman game state including word, guessed letters, remaining guesses, and win/lose conditions.

**hangman()** (`hangman.py:38`): Main task function that creates Inspect evaluation with configurable parameters (language, difficulty, max_guesses, shuffle, allow_word_guesses).

**hangman_guess()** (`hangman.py:166`): Tool function that processes letter guesses and returns game state information.

### Inspect Framework Integration
- Uses `Task` with dataset, solver, setup, and scorer components
- `basic_agent` solver with custom system messages and tool access
- `grouped` scorer by difficulty level with accuracy metrics
- `store()` for maintaining game state across tool calls

### Task Parameters
- `language`: Word language (default: "english")
- `difficulty`: Specific difficulty 1-5, or None for mixed
- `max_guesses`: Maximum incorrect guesses (default: 10)
- `shuffle`: Randomize word order (default: True)
- `allow_word_guesses`: Allow full word guesses (default: False)

## Testing Strategy
Tests cover dataset functions, task creation, and parameter validation. Located in `tests/test_hangman.py` with pytest framework.