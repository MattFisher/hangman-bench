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

## Oracle replay: scoring *how* a game was played

`analysis/oracle.py` and `analysis/pilot_oracle.py` score individual guesses
against the belief state, rather than scoring only the final win or loss.

The motivation is that the headline win rate saturates: `gpt-5-nano` reaches
0.93 with a 10 wrong-guess budget, which is generous enough that a player can
ignore all evidence and still usually win. Hangman is unusual in that the exact
posterior over the hidden word and the best available move are both computable
at every step, so we can measure the gap between winning and playing well.

### Metrics

Provable errors, requiring no judgement:

- `invalid` — the guess was not a single alphabetic character.
- `repeat` — the letter had already been guessed, so the guess cannot change
  the belief state.
- `dominated_miss` — a fresh letter appearing in **zero** consistent candidate
  words: guaranteed to cost a life for no information.

Quality relative to optimal play:

- `hit_prob_regret` — shortfall between the best available hit probability and
  that of the letter actually guessed, under a uniform posterior.
- `wrong_guess_regret` — wrong guesses taken minus wrong guesses an oracle
  solver needs on the same word and dictionary.

### Running it

```bash
# Score real Inspect logs
uv run analysis/pilot_oracle.py from-logs --logs logs/ --out analysis/pilot

# Calibrate against reference agents of known quality
uv run analysis/pilot_oracle.py simulate --out analysis/pilot_sim
```

Both write `<out>_per_guess.tsv` (one row per guess) and `<out>_summary.tsv`
(one row per model).

### Calibration

Three reference agents, 100 dataset words, 10 wrong guesses allowed:

| agent | win | repeat | dominated | subopt | regret | wrong | oracle | excess |
| --------- | ---- | ----- | ----- | ----- | ----- | ---- | ---- | ---- |
| optimal   | 0.98 | 0.000 | 0.000 | 0.000 | 0.000 | 3.92 | 3.92 | 0.00 |
| frequency | 0.13 | 0.000 | 0.373 | 0.760 | 0.448 | 9.64 | 3.92 | 5.72 |
| sloppy    | 0.08 | 0.149 | 0.405 | 0.786 | 0.452 | 9.79 | 3.92 | 5.87 |

`frequency` plays a fixed `etaoin…` order and never conditions on evidence —
which is roughly what the eval's own system prompt asks for ("common letter
frequencies"). It makes a provably dead guess 37% of the time. `sloppy` adds
deliberate repeats to confirm the repeat detector fires. The spread between
these agents is what makes the metrics usable on real models.

### Two correctness notes

- `filter_candidates` in `measure_difficulty.py` is over-inclusive. It builds a
  regex where `.` also matches the guessed letter, so for the board `.a.a.a` it
  admits `aaaaaa` — a state that cannot occur, since guessing `a` reveals every
  `a` at once. `oracle.consistent_candidates` enforces the exact-position
  constraint instead. This affects the difficulty metrics in
  `difficulty_report.tsv`, which have not been regenerated.
- Trajectories must be recovered from `hangman_guess` tool calls, not from the
  scorer's `guessed_letters`. `GameState.guess` returns early on a repeated
  letter, so repeats never reach the store and are invisible to any
  store-based analysis. `tests/test_oracle.py` pins this behaviour.

### Known gaps this surfaced

- A malformed guess (e.g. `"ab"`) raises `ValueError` inside the tool, which
  propagates and **errors the whole sample** rather than being recorded as a
  bad move. Invalid-guess rate is therefore currently unmeasurable, and win
  rates are computed only over samples that survived.
- `dwarves` and `pyjamas` are absent from `wordlist.txt`, so oracle metrics for
  them are computed against an injected entry. This also explains the outlier
  `wrong_coverage` of 16 for `dwarves` in `difficulty_binned.tsv`.

## Notes and caveats

- Coverage vs Frequency vs Info Gain
  - `wrong_coverage` prioritizes letters that appear in many candidate words (ignores duplicates within words). Often minimizes wrong guesses but doesn’t use positional information.
  - `wrong_freq_raw` counts raw occurrences (duplicates included); simple baseline; can overweight double letters.
  - `wrong_info_gain` minimizes expected remaining candidate set size using position masks; it may incur more wrong guesses but reduce total guesses.
- Dictionary matters
  - Metrics depend on the dictionary for each word length. We use `analysis/wordlist.txt` derived from the simulation data and compatible with the Curlew wordlist.
- Reproducibility
  - All solvers here are deterministic; no weighted randomness.
