# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold from [python-project-template](https://github.com/MattFisher/python-project-template): shared CI, pre-commit stack (ruff, zizmor, mdformat, actionlint, shellcheck), basedpyright strict, this changelog.

### Changed

- Build backend moved from setuptools to hatchling; version is now read from `hangman_bench.__version__`.

## [0.1.0] - unreleased baseline

The state of the project at template adoption:

- Hangman eval for the Inspect framework: `hangman` task, `game_scorer` (win/loss by difficulty) and `oracle_scorer` (per-guess quality against a computable belief state).
- Oracle replay over a packaged SCOWL en-GB dictionary; per-language declared alphabets.
- Analysis pipeline: reference-agent calibration, cross-arm paired comparisons, dictionary-sensitivity grid, win-vs-budget curves, and figures regenerating from committed TSVs.
- Research notes (`RESEARCH_NOTES.md`) recording the pilot, prompt ablation, and budget-curve findings.
