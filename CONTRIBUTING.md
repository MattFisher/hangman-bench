# Contributing to Hangman Eval

Thank you for your interest in contributing to Hangman Eval! Here's how you can help:

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. When reporting a bug, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Any relevant error messages
- Your Python version and operating system

## Development Setup

1. Fork the repository and clone it locally
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Include docstrings for all public functions and classes
- Keep lines under 88 characters (Black's default)

## Testing

Run the test suite with:

```bash
pytest
```

We aim for good test coverage. Please add tests for any new functionality.

## Submitting Changes

1. Create a new branch for your changes
2. Make your changes, including tests and documentation
3. Run the test suite to ensure everything passes
4. Submit a pull request with a clear description of your changes

## Code Review Process

- All pull requests require at least one approval from a maintainer
- All CI tests must pass before merging
- Maintainers will review your code and may request changes

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
