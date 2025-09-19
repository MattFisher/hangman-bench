# Hangman-Bench Analysis

This folder documents the analysis workflow we used to derive objective word difficulty scores and (optionally) reclassify words in the dataset.

Results vary greatly depending on the solvers used, so the words have not currently been reclassified from the original dataset, which was produced by Windsurf.

## Data sources

- Wolfram SimulationData
  - Downloaded by our ingestion script from the Wolfram MathSource Library:
    - https://library.wolfram.com/infocenter/MathSource/7635/SimulationData.zip?file_id=7257
- Reference wordlist for heuristic solvers
  - Curlew Communications wordlist: https://www.curlewcommunications.uk/wordlist.html
  - We also extract a wordlist directly from the parsed simulation output.
- Original Mathematica “Demonstration” notebook (the source of the simulation’s logic)
  - Blog post: [25 Best Hangman Words](https://blog.wolfram.com/2010/08/13/25-best-hangman-words/)
  - Url: <http://demonstrations.wolfram.com/HangmanWordGameForAComputerPlayer/>
  - File: `analysis/Demonstration-Hangman-Word-Game-for-a-Computer-Player-1-0-0-definition.nb`
  - Key functions we reviewed:
    - `FindWords[…]` builds the candidate set using a pattern from known positions and excludes known-wrong letters.
    - `WeightedLetterChoice[…]` picks the next letter randomly, weighted by raw letter counts across the remaining candidate words (excluding letters already revealed).

## What we built

- `scripts/ingest_simulation.py`
  - Parses `SimulationData.txt` into TSV with columns: `word`, `wrong_guesses` (list), `mean_wrong_guesses`.
  - If `SimulationData.txt` isn’t present, it downloads and extracts it from the Wolfram link above.

- `analysis/extract_wordlist.py`
  - Extracts a unique, lowercased wordlist (first column) from the parsed TSV.
  - Output: `analysis/wordlist.txt`.

- `analysis/zen_hangman.py`
  - Python port of Dan Q’s “Hardest Hangman” heuristic with one improvement.
    - Blog: https://danq.me/2013/12/15/hangman/
    - Gist: https://gist.github.com/Dan-Q/7910309
  - Improvement: filters the candidate dictionary by known positions and also prunes words containing any known-wrong letters.
  - Chooses next letter deterministically by raw letter frequency across candidate words (ties broken alphabetically).

- `analysis/measure_difficulty.py`
  - Computes multiple objective metrics per dataset word using a dictionary:
    - `wrong_freq_raw`: wrong guesses using raw letter-frequency solver (duplicates within words counted).
    - `wrong_coverage`: wrong guesses using a coverage solver (counts unique word incidence per word).
    - `wrong_info_gain`: wrong guesses using an information-gain solver that minimizes expected remaining candidate set size by partitioning candidates using position masks of the guessed letter.
    - `rare_score`, `dup_factor`, `structural_score` from letter incidence by word length.
  - Output: `analysis/difficulty_report.tsv`.

- `analysis/bin_difficulty.py`
  - Bins words into difficulty tiers by quantiles of a chosen metric (default `wrong_coverage`; can use `wrong_freq_raw` or `wrong_info_gain`).
  - Outputs:
    - `analysis/difficulty_binned*.tsv`
    - Optional paste-ready snippet: `analysis/reclassified_from_*.py` containing `ENGLISH_WORDS_RECLASSIFIED`.

## Why we didn’t rely on the original simulation means

The Mathematica notebook’s next-letter selection uses weighted randomness by raw letter counts (not maximizing probability of a hit), which introduces variance and can mis-rank intuitive words (e.g., rating “apple” as hard). Our analysis uses deterministic solvers and alternative heuristics, including an information-gain approach, to provide more stable, reproducible difficulty signals.

## Reproducing the analysis

All commands assume repo root and `uv` installed.

1) Ingest Wolfram simulation data to TSV

```bash
uv run analysis/ingest_simulation.py \
  --input analysis/SimulationData.txt \
  --output analysis/SimulationData_parsed.tsv
```

2) Extract a wordlist from the parsed TSV (first column)

```bash
uv run analysis/extract_wordlist.py \
  --input analysis/SimulationData_parsed.tsv \
  --output analysis/wordlist.txt
```

3) Compute objective difficulty metrics

```bash
uv run analysis/measure_difficulty.py \
  --datasets src/hangman_bench/datasets.py \
  --wordlist analysis/wordlist.txt \
  --output analysis/difficulty_report.tsv
```

4) Bin words into tiers by quantiles (choose metric)

- Coverage (proxy for probability-of-any-hit):

```bash
uv run analysis/bin_difficulty.py \
  --input analysis/difficulty_report.tsv \
  --metric wrong_coverage \
  --output analysis/difficulty_binned.tsv \
  --emit-snippet analysis/reclassified_from_coverage.py
```

- Raw frequency (closer to Dan Q’s heuristic):

```bash
uv run analysis/bin_difficulty.py \
  --input analysis/difficulty_report.tsv \
  --metric wrong_freq_raw \
  --output analysis/difficulty_binned_freq.tsv \
  --emit-snippet analysis/reclassified_from_freq.py
```

- Information gain (minimize expected remaining candidates):

```bash
uv run analysis/bin_difficulty.py \
  --input analysis/difficulty_report.tsv \
  --metric wrong_info_gain \
  --output analysis/difficulty_binned_info.tsv \
  --emit-snippet analysis/reclassified_from_info.py
```

## Outputs

- `analysis/SimulationData_parsed.tsv` — parsed simulation means by word
- `analysis/wordlist.txt` — dictionary for the solvers
- `analysis/difficulty_report.tsv` — metrics per dataset word
- `analysis/difficulty_binned*.tsv` — quantile-binned labels
- `analysis/reclassified_from_*.py` — pasteable snippets for `src/hangman_bench/datasets.py`

## Notes and caveats

- Coverage vs Frequency vs Info Gain
  - `wrong_coverage` prioritizes letters that appear in many candidate words (ignores duplicates within words). Often minimizes wrong guesses but doesn’t use positional information.
  - `wrong_freq_raw` counts raw occurrences (duplicates included); simple baseline; can overweight double letters.
  - `wrong_info_gain` minimizes expected remaining candidate set size using position masks; it may incur more wrong guesses but reduce total guesses.
- Dictionary matters
  - Metrics depend on the dictionary for each word length. We use `analysis/wordlist.txt` derived from the simulation data and compatible with the Curlew wordlist.
- Reproducibility
  - All solvers here are deterministic; no weighted randomness.
