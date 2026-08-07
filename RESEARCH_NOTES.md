# Research notes

Working notes for turning hangman-bench into a paper. Written to be picked up
cold — by a person or an agent in a different environment — so it states what
exists, what we found, what to do next, and what is still undecided.

**Status:** infrastructure and methodology are built. The central empirical
claim has not yet been tested against a real model. That test is the next
action and is described in full under [The pilot](#the-pilot).

---

## 1. Why there might be a paper here

hangman-bench started as a demonstration of tool use in Inspect. Two properties
turn out to make it more interesting than that.

**The benchmark is saturated at the outcome level.** `gpt-5-nano` scores 0.93.
With a ten wrong-guess budget on five to ten letter words, a player can ignore
the evidence entirely and still usually win. Win rate carries little signal.

**Hangman has computable ground truth.** At every step the exact posterior over
the hidden word, and the best available move, are computable from a dictionary.
Almost no agentic benchmark has this. It means we can score the *process* — was
each guess defensible given what the player had been told — rather than only the
outcome.

The working thesis:

> On a task where ground truth is computable, measuring process rather than
> outcome requires committing to a prior over the hidden word. The obvious
> choices — uniform over a dictionary, or corpus frequency — are measurably
> wrong in opposite directions. We quantify how far conclusions move across
> defensible choices.

The prior-sensitivity result is the contribution, not a caveat to bury.

### Related work to position against

The character-level angle ("how many r's in strawberry") is a crowded field:
The Strawberry Problem (arXiv:2505.14172), CharBench (arXiv:2508.02591), TASE,
SubTokenTest. Do not frame this as a character-level competence paper. Likewise,
"here is another text game" is covered by TextArena and Game Reasoning Arena.
The differentiator is the computable oracle and the treatment of the prior.

---

## 2. What is built

### Merged to `main`

Four measurement bugs, each of which silently distorted a number rather than
failing loudly:

- A malformed guess raised `ValueError` inside the tool, which propagated and
  errored the whole sample — the game was dropped from results rather than
  scored. Now raises `ToolError`, which the model can recover from.
- Repeats and malformed guesses never reached the store, so no analysis could
  see them. `GameState.attempts` now records every submission.
- `filter_candidates` matched the board with a regex in which `.` also matched
  the guessed letter, admitting unreachable states such as `aaaaaa` for
  `.a.a.a`. Fixing it changed a solver metric for 72 of 100 words.
- Difficulty was measured against a dictionary that did not contain all the
  dataset words, so the solver could never converge on them.

### On the working branch

- `src/hangman_bench/oracle.py` — belief-state replay. Computes the consistent
  candidate set and scores each guess.
- `oracle_scorer` in `hangman.py` — a real Inspect scorer, in the task's scorer
  list by default. Returns a dict-valued `Score` that Inspect expands into
  per-key metrics.
- `analysis/pilot_oracle.py` — batch scoring over logs (`from-logs`) and
  calibration against reference agents (`simulate`).
- `analysis/build_wordlist.py` — builds the oracle dictionary from SCOWL.

### Metrics

Provable errors, requiring no judgement:

| metric | meaning |
| --- | --- |
| `invalid_rate` | submission was not a single letter |
| `repeat_rate` | letter already guessed; cannot change the belief state |
| `dominated_rate` | fresh letter appearing in **zero** consistent candidates — guaranteed to cost a life for no information |

Quality against a reference policy:

| metric | meaning |
| --- | --- |
| `hit_prob_regret` | shortfall against the hit probability of the reference policy's own move. A true regret; never negative |
| `excess_wrong_guesses` | wrong guesses taken minus what the reference solver needs |
| `suboptimal_rate` | fraction of guesses below the reference policy's own hit probability |

`excess_wrong_guesses` compares against a *greedy* solver, not a globally
optimal one — maximising per-guess hit probability does not minimise total
wrong guesses. It can be negative. It is a comparison against a strong
baseline, not a bound. Do not call it "regret" in the paper.

Both `hit_prob_regret` and `suboptimal_rate` are measured against the
*configured* reference policy (`--strategy` / `-S strategy=`). Under the
default `max_hit_prob` that is also the best available hit probability; under
`info_gain` the reference deliberately gives up hit probability for a better
partition, so the two differ. State which policy produced a number.

### Calibration

Three reference agents over the 100 dataset words, 10 wrong guesses allowed,
scored against the shipped en-GB dictionary:

| agent | win | repeat | dominated | subopt | hit regret | wrong | oracle | excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimal | 1.00 | 0.000 | 0.000 | 0.000 | 0.000 | 3.75 | 3.75 | 0.00 |
| frequency | 0.13 | 0.000 | 0.383 | 0.755 | 0.461 | 9.64 | 3.75 | 5.89 |
| sloppy | 0.09 | 0.148 | 0.413 | 0.781 | 0.466 | 9.82 | 3.75 | 6.07 |

`frequency` plays a fixed `etaoin…` order and never conditions on evidence —
roughly what the eval's own system prompt asks for. It makes a provably dead
guess 38% of the time. The spread between these agents is what makes the
metrics usable on real models. All three are deterministic.

---

## 3. The pilot

**This is the next action, and everything else is downstream of it.**

The thesis assumes models win while playing badly. That has not been tested.
If real models turn out to play near-optimally, the thesis is wrong and the
interesting paper is a different one. Better to learn that in a weekend than
after three months.

This environment has no model API keys, so the pilot needs to run elsewhere.

### Running it

```bash
git clone https://github.com/MattFisher/hangman-bench
cd hangman-bench
git checkout claude/hangmanbench-paper-scope-hdyz69
uv sync --dev

# Two models, all 100 words, oracle scoring on by default
uv run inspect eval src/hangman_bench/hangman.py@hangman \
  --model openai/gpt-5-nano --limit 100 --log-dir logs/pilot

uv run inspect eval src/hangman_bench/hangman.py@hangman \
  --model anthropic/claude-sonnet-5 --limit 100 --log-dir logs/pilot

# Aggregate across models
uv run analysis/pilot_oracle.py from-logs --logs logs/pilot --out analysis/pilot
```

`oracle_scorer` runs inside the eval, so the metrics are already in the log.
`pilot_oracle.py from-logs` is for cross-model aggregation and per-guess dumps.

To add oracle metrics to a log produced without them:

```bash
uv run inspect score <log.eval> \
  --scorer src/hangman_bench/hangman.py@oracle_scorer \
  --action append --overwrite
```

Pass `--scorer` explicitly. Bare `inspect score <log>` fails for any package:
`scorer_from_spec` falls back to loading from the task file only on
`ValueError`, but `scorer_create` raises `LookupError`, so the fallback never
runs (`inspect_ai/_eval/loader.py`). Worth reporting upstream.

### What decides the outcome

Look at **`dominated_rate`** first. It counts guesses of a letter that appears
in zero remaining candidate words — provably wrong, no judgement involved.

- **Meaningfully above zero** while win rate stays high: the thesis holds.
  Proceed to scale-up.
- **At or near zero**: models are tracking the belief state. The thesis is
  wrong. The interesting question becomes why the benchmark saturates anyway,
  and the paper pivots to the prior-sensitivity and benchmark-construction
  findings, which stand on their own.

Also worth reading on the first run: `repeat_rate` and `invalid_rate` (basic
state-tracking failures), and whether `excess_wrong_guesses` is positive.

### Before trusting the numbers

Sanity-check a handful of trajectories by hand against
`analysis/pilot_sim_per_guess.tsv` format. The oracle has been validated on
synthetic agents and a mocked Inspect log, but **never on a real model
trajectory**. First contact with real data is where harness bugs surface.

---

## 4. Findings so far

Ordered by how much a reviewer would care.

**Per-guess metrics depend heavily on the dictionary.** Holding trajectories
completely fixed and varying only the dictionary used to score them:

| dictionary | avg candidates | dominated_rate | excess_wrong |
| --- | --- | --- | --- |
| full | 7844 | 0.353 | 5.12 |
| 50% | 3927 | 0.418 | 6.00 |
| 25% | 1962 | 0.456 | 6.42 |
| 10% | 787 | 0.505 | 7.20 |
| 2% | 158 | 0.567 | 8.07 |

"Model X makes provably dead guesses 35% of the time" is not a fact about the
model. It is a fact about (model, dictionary). This must be reported as a
sensitivity analysis, not hidden behind a default.

**Frequency weighting differentially helps common words.** Replacing the
uniform posterior with a wordfreq-weighted one, wrong guesses needed:

| band | uniform | weighted | delta |
| --- | --- | --- | --- |
| common (zipf>4.5) | 3.50 | 1.67 | −1.83 |
| moderate | 3.80 | 3.10 | −0.71 |
| rare (zipf<3.5) | 3.77 | 3.63 | −0.14 |

But the words that got *harder* were `puppy`, `kitten`, `turtle`, `monkey`,
`giraffe`, `ladder`, `bagpipes` — concrete nouns of middling frequency. Under
corpus-frequency weighting the oracle is pulled toward commoner words of the
same shape. So corpus frequency is not the right prior either. There are three
distributions, not two: uniform over a dictionary, general corpus frequency,
and whatever distribution people actually draw hangman targets from.

**Human-curated puzzle targets are strongly frequency-skewed.** Using Wordle as
the closest public analogue — 2,315 curated answers versus 10,657 words Wordle
accepts but never uses:

| | median zipf | rare (<3) | common (>4.5) |
| --- | --- | --- | --- |
| Wordle answers (human-curated) | 3.52 | 26.9% | 16.8% |
| Wordle accepted-but-never-answers | 1.39 | 89.6% | 1.3% |

A 2.13 zipf gap, roughly 135×. This replicates the frequency skew on human
curation, independent of any LLM. The Wordle answer/guess split is a citable
precedent for exactly the target-vs-dictionary distinction we need.

**Extreme concreteness looks like an LLM fingerprint.** Brysbaert concreteness
norms (1 abstract, 5 concrete):

| | mean | % highly concrete |
| --- | --- | --- |
| Wordle answers (human-curated) | 3.53 | 36.8% |
| hangman-bench dataset (LLM-generated) | 4.41 | 77.7% |
| SCOWL-50 sample | 3.20 | 27.4% |

Human puzzle-setters barely shift concreteness; the LLM-generated dataset more
than doubles it. Caveat: the norms cover only ~40k common words, so only 4% of
the Wordle accepted-only list is rated — treat that row as suggestive. The
frequency comparison is solid; the concreteness one is not, on its own.

**LLM-authored difficulty labels are uncorrelated with computed difficulty.**
Spearman between the dataset's `v_easy…v_hard` labels and solver-derived
difficulty: `wrong_coverage` −0.008, `wrong_freq_raw` +0.026,
`wrong_info_gain` +0.368. The labels were generated by an LLM (Windsurf).

### A confound in the benchmark itself

The dataset was LLM-generated and the systems under test are LLMs, so targets
and guessers are drawn from correlated priors. An LLM guesser may score well
partly because another LLM chose words that LLMs find natural. This is
independent of everything else and is only fixable by not letting an LLM choose
the targets.

**Do not claim the current 100 words are representative of human hangman
targets.** They are representative of what one LLM produced when asked.

---

## 5. The dictionary

The oracle's dictionary is part of the measurement, so it needs stated,
licensed, reproducible provenance.

**Shipped:** SCOWL 2020.12.07, British English (`-ise`), size tiers ≤50, words
of three or more letters, proper names and abbreviations excluded. 61,460
words. Built by `analysis/build_wordlist.py`, licence notice retained at
`src/hangman_bench/data/SCOWL-Copyright`, attribution in `NOTICE`.

One dialect per file, so en_US / en_AU / en_CA can be added later without
disturbing the one in use:

```bash
uv run analysis/build_wordlist.py --dialect en_US --tier 60
```

The dataset was made consistently en-GB to match: `dwarves`→`dwarfs`,
`whiskey`→`whisky`, `oxidize`→`oxidise`, `galvanize`→`galvanise`. All 100
words are now in the dictionary.

### What this replaced, and why

The previous list came from Wolfram MathSource `SimulationData.zip` (the data
behind the 2010 "25 Best Hangman Words" blog post), via
`analysis/extract_wordlist.py`. It was indefensible on four counts:

- **Licensing.** Wolfram MathSource has no clear redistribution terms, and the
  file had been placed inside the built wheel.
- **Misattributed.** `analysis/README.md` credited the Curlew British English
  list. The file was unambiguously American: `color`, `realize`, `pajamas`,
  `dwarfs` present; no British spellings at all.
- **Content.** A spell-checker list, which exists to *accept* strings. It
  contained initialisms as entries (`cs`, `cw`, `kc`, `kw`, `ls`, `rs`, `ts`),
  lowercased proper nouns, and vocabulary that stops around the late 1990s
  (`email` and `laptop` present; `internet` and `blog` absent).
- **Unstated inclusion criteria.** "Words Wolfram happened to simulate in 2010"
  is not a specification a reviewer can reproduce.

The Wolfram ingestion scripts remain in `analysis/` as a record of the original
difficulty analysis. They are no longer the source of the shipped dictionary.

### Note on British spelling variants

SCOWL publishes British in two conventions: `british-words` (`-ise`) and
`british_z-words` (`-ize`, Oxford spelling). They are alternatives, not
complements. Merging them puts 228 words in the dictionary under both
spellings, and a guesser facing `reali_e` would then have to choose `s` or `z`
on no information — pure orthographic convention, zero inference. We use
`-ise` only.

Note that `en_GB` still contains both endings, correctly: 84 words end `-ise`
(the obligatory class — `advertise`, `exercise`, `surprise`, which are `-ise`
in every dialect) and the alternating class is spelled `-ise` throughout. Only
`prize`/`prise` and `apprize`/`apprise` appear in both spellings, and those are
genuinely different words.

### ESDB — checked, rejected for now

ESDB (formerly SCOWLv2) carries commonness, dialect, variant, POS and
inflection data under the same licence. Not usable yet:

- `scowl.db` and `scowl.txt` are build outputs, not committed. There is no
  downloadable artifact; you would build from a git branch.
- SourceForge's `Rev 1`…`Rev 7.1` entries look like ESDB releases and are
  titled "English Speller Database", but are SCOWLv1 from 2011 — the project
  was renamed and all historical files inherited the new title. The latest
  packaged release is `2020.12.07`, which is SCOWLv1.
- Its README says "ESDB is still a work in progress"; only size 60 is vetted.
- Its commonness data derives from Moby Words II and Brian Kelk's frequency
  classification — both 1990s, bucketed into ~10 coarse levels. It would not
  replace wordfreq for continuous frequency weighting.

Revisit if a packaged release appears. Its POS and inflection data would be
useful for filtering to lemma nouns. AGID, in the same SourceForge project,
already provides that today without ESDB.

---

## 6. Open questions and decisions

**Undecided — needs a call before scale-up:**

- *What prior to treat as primary.* Recommendation: make it an explicit
  parameter (`uniform` | `frequency` | a curated-target prior) and report
  across all of them, rather than picking one and defending it.
- *Where a target distribution could come from.* No source gives "what humans
  pick for hangman". Candidates, best first: published hangman word lists;
  wordlists bundled with open-source hangman implementations; adjacent
  human-curated puzzle targets (Wordle answers, Codenames); psycholinguistic
  norms to *characterise* rather than supply a list. Actual play logs would be
  ideal and are probably unobtainable.
- *Dataset scale and composition.* 100 words is far too few, and they are
  LLM-chosen. Needs thousands, stratified by computed difficulty, with a
  contamination-controlled held-out set — these 100 are on public GitHub.

**Fixed after review** (raised by an automated reviewer on PR #3, all three
confirmed by reproduction before fixing):

- `from-logs` applied one global `--max-guesses` to every replay instead of the
  limit each sample was played under. On a log run at 15, replaying at the
  default 10 turned a genuine win into a loss and truncated the wrong-guess
  count from 12 to 10. The logged value now wins; the flag is a fallback only.
- Games won by submitting the full word were replayed as losses, because
  `extract_trajectories` drops the `submit` action and the letter sequence
  alone never completes the word. Both the batch reader and `oracle_scorer`
  now take the outcome from the recorded score.
- Under `--strategy info_gain`, `optimal_letter` came from the chooser while
  `best_hit_prob` came from the maximum hit probability, so the reference
  policy scored as suboptimal against itself. Both now come from the
  configured policy. The default `max_hit_prob` is unaffected — calibration
  output is byte-identical.

**Known issues:**

- `tests/test_e2e_hangman.py::TestHangmanE2E::test_hangman_incomplete_game`
  fails on `main` and on the working branch. It is stale in three independent
  ways: the `44` constant predates the turn-limit change to four messages per
  guess (now 57), the mock supplies too few outputs to reach the raised limit,
  and it asserts on the final message role, which is incidental. Fixing it
  needs a decision about what the test is meant to pin.
- The registered evaluation report (0.93, `gpt-5-nano`) predates the malformed
  guess fix. Samples that previously errored out and vanished are now scored,
  so it is not comparable. Re-baseline before citing it.
- `analysis/README.md` still describes the Wolfram-derived pipeline as the
  source of the shipped dictionary in places.

---

## 7. Roadmap

1. **Run the pilot.** Nothing else is worth doing first. See section 3.
2. **Add `prior=uniform|frequency` to the oracle**, so the comparison is a
   first-class result rather than a fork in the code. Frequencies are one flag
   away: `build_wordlist.py --with-frequencies` emits `word<TAB>frequency`
   (~295 KB gzipped, no runtime dependency on wordfreq). 8.6% of SCOWL words
   have zero frequency and need a smoothing floor — the floor value is a real
   parameter, since it sets how much mass sits on morphological deadwood.
3. **Re-baseline** `gpt-5-nano` and update the inspect_evals register entry.
4. **De-saturate.** Sweep `max_guesses` over {2,3,4,6,8,10} and plot win rate
   against budget for each model and for the oracle. This turns a saturated
   benchmark into a discriminative one and gives a competence-gap curve.
5. **Scale the dataset.** Thousands of words, stratified by computed
   difficulty, held-out set for contamination control.
6. **Score commitment, if word guesses are used.** `allow_word_guesses=True`
   lets a model end the game by submitting the whole word. That is a *commit*
   action, and it measures something letter guesses cannot: whether the model
   knows that it knows. The oracle can score it exactly — was the submitted
   word still in the consistent candidate set (submitting one the board had
   ruled out is a provable error), how many candidates remained at the moment
   of commit, and does realised success match the 1/|candidates| prior. That
   last one is a calibration measurement and may be a better headline than
   `dominated_rate`. Word guesses are currently out of scope for the primary
   experiment and off by default; the replay no longer mis-scores them, but it
   does not score the commitment itself.

7. **The lexical-access ablation.** Give the model the explicit candidate list
   at each step. If it then plays near-optimally, the bottleneck is
   character-level lexical retrieval; if not, it is sequential decision-making
   under uncertainty. Cheap and decisive, and it turns the character-level
   question into a measured variable instead of an assumption.
8. **Write up.** Aim at a NeurIPS/ICLR evaluation or datasets-and-benchmarks
   workshop rather than a cold arXiv drop: 4–8 pages, real review, still goes
   on arXiv. Note that arXiv cs.* requires endorsement for first-time
   submitters without an institutional affiliation — sort that early.

The LLM-fingerprint findings (difficulty labels are noise; LLM-generated target
distributions are detectably skewed) are a **separate paper** about benchmark
construction. Mention and move on; do not try to fit both.

---

## 8. Caveats to state explicitly in any write-up

- The model is never told which dictionary the oracle uses, so its belief state
  is not the oracle's. Some of what looks like model error is a specification
  mismatch. This applies at the orthographic level too: if the hidden word is
  `realise` and the model guesses `z`, it is penalised for a convention nobody
  communicated.
- The reference solvers are greedy, not optimal. `excess_wrong_guesses` can be
  negative.
- Difficulty labels in the current dataset are LLM-authored and do not track
  computed difficulty. Do not use them as a difficulty axis without saying so.
- The current 100 words are LLM-chosen and are not a sample of human hangman
  targets.
