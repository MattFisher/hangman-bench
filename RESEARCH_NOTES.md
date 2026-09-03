# Research notes

Working notes for turning hangman-bench into a paper. Written to be picked up cold — by a person or an agent in a different environment — so it states what exists, what we found, what to do next, and what is still undecided.

**Status (2026-09-03): refocused.** The benchmark and its measurement pipeline are built and validated (sections 2–5). The pilot, the dictionary grid, the prompt ablation and the budget-curve experiment all ran on 100 words and one seed each (sections 3–4). A review of the whole project on 2026-09-03 concluded that the thesis those experiments were organised around — process metrics that keep discriminating after outcome saturation — is the weakest paper this material supports, and that the two most interesting findings in the repo were being treated as side results. The project now targets those two (section 1); the superseded theses and the reasons they are backburnered are in section 6; the plan, with a decision gate before any expensive scale-up, is section 7. Nothing in sections 2–5 has been re-run; they are the record the new direction rests on.

______________________________________________________________________

## 1. The question

hangman-bench started as a demonstration of tool use in Inspect. One property makes it worth more than that: at every step the exact posterior over the hidden word, and the best available move, are computable from a dictionary. Almost no agentic task has this. It means the belief state a model *should* hold is known exactly, so the benchmark can answer something sharper than "does the model win":

> Where does an LLM's belief state break inside an agentic loop, and what breaks it?

Two results to chase. Both are already visible in the pilot data (section 4), neither is replicated, and each has a cheap experiment that decides whether it is real (section 7).

**Result 1 — the residual failure is retrieval at the pinned endgame.** Most provably dead guesses are made *after* the dictionary has collapsed to a single candidate. At that point there is no decision left, only a word to name, and the models still miss:

| model           | guesses with word pinned | misses | pinned miss rate | share of dominated misses at the pinned endgame |
| --------------- | ------------------------ | ------ | ---------------- | ----------------------------------------------- |
| gpt-4o          | 326                      | 167    | 0.51             | 73%                                             |
| gpt-5-nano      | 230                      | 41     | 0.18             | 59%                                             |
| claude-sonnet-5 | 216                      | 20     | 0.09             | 67%                                             |

(`analysis/pinned_endgame.py` over `analysis/pilot_per_guess.tsv`; output in `analysis/pinned_endgame.tsv`.) gpt-4o, facing `.o..ee` with only `a` and `t` excluded, guesses r, s, l, m, n, d before finding `c`, then guesses `b` on `co..ee`. On `.a.e.o` it guesses n, l, i, c, d, m, p and loses without ever producing `gazebo`. That is not sequential decision-making failing; it is lexical retrieval, or verification, under a pattern constraint. The pinned miss rate is a cleaner headline than `dominated_rate`: it needs no argument about what "dominated" means, it is conditional on a regime a reader can picture, and it isolates the part of the task that the character-level literature studies statically. What this benchmark can add is the decomposition — retrieval (pattern → word), verification (pattern + word → fits?), decision (pattern + candidate list → choose), and whether a belief state exists at all (verbalised posterior) — measured inside a loop where the evidence accumulated turn by turn, and with the oracle supplying the ground truth for every leg.

**Result 2 — play is budget-conditional.** Models told a tight wrong-guess budget win more than the curve derived from their own unlimited play predicts, in five of six cells and tied in the sixth (section 4). If that replicates across seeds and is dose-dependent in the budget, it is a finding about evaluations in general, not about hangman: a model's policy depends on the stakes it is told, so a fixed-budget score is neither a capability estimate nor a ceiling, and the derived curve is a lower bound. Hangman supplies the exactly computable reference curves (greedy oracle, frequency floor) that make the effect attributable to the model rather than the task.

Everything else in the repo — dictionary sensitivity, the prompt ablation, the LLM-fingerprint findings about the dataset — is either robustness material for those two results or a blog post. Section 6 records the theses this replaces and why.

### Related work to position against

The character-level competence papers — The Strawberry Problem (arXiv:2505.14172), CharBench (arXiv:2508.02591), TASE, SubTokenTest — are the nearest neighbours to result 1. Earlier notes said to avoid that framing because the field is crowded; the data says the phenomenon lives there, so the positioning changes instead. Those papers measure static string competence with a single prompt. This work measures where the same competence breaks inside an agentic loop, separates retrieval from verification from decision, and adds two things with no static analogue: whether the model's verbalised belief state matches the computable one, and whether its policy shifts with the budget it is told. Text-game benchmarks (TextArena, Game Reasoning Arena) score outcomes without a computable reference. LLM papers on Wordle and other deduction games are closer than either cluster and must be engaged directly, as the external review noted. For result 2, AISI's inference-scaling cyber post (section 4) is the framing to borrow: report curves, not points.

______________________________________________________________________

## 2. What is built

### Merged to `main`

Four measurement bugs, each of which silently distorted a number rather than failing loudly:

- A malformed guess raised `ValueError` inside the tool, which propagated and errored the whole sample — the game was dropped from results rather than scored. Now raises `ToolError`, which the model can recover from.
- Repeats and malformed guesses never reached the store, so no analysis could see them. `GameState.attempts` now records every submission.
- `filter_candidates` matched the board with a regex in which `.` also matched the guessed letter, admitting unreachable states such as `aaaaaa` for `.a.a.a`. Fixing it changed a solver metric for 72 of 100 words.
- Difficulty was measured against a dictionary that did not contain all the dataset words, so the solver could never converge on them.

### On the working branch

- `src/hangman_bench/oracle.py` — belief-state replay. Computes the consistent candidate set and scores each guess.
- `oracle_scorer` in `hangman.py` — a real Inspect scorer, in the task's scorer list by default. Returns a dict-valued `Score` that Inspect expands into per-key metrics.
- `analysis/pilot_oracle.py` — batch scoring over logs (`from-logs`) and calibration against reference agents (`simulate`).
- `analysis/build_wordlist.py` — builds the oracle dictionary from SCOWL.
- `analysis/pinned_endgame.py` — conditional miss rate once the candidate set is pinned, from a committed per-guess TSV. The number behind result 1.

### Metrics

Provable errors, requiring no judgement:

| metric           | meaning                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `invalid_rate`   | submission was not a single letter                                                                      |
| `repeat_rate`    | letter already guessed; cannot change the belief state                                                  |
| `dominated_rate` | fresh letter appearing in **zero** consistent candidates — guaranteed to cost a life for no information |

Quality against a reference policy:

| metric                 | meaning                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `hit_prob_regret`      | shortfall against the hit probability of the reference policy's own move. A true regret; never negative |
| `excess_wrong_guesses` | wrong guesses taken minus what the reference solver needs                                               |
| `suboptimal_rate`      | fraction of guesses below the reference policy's own hit probability                                    |

`excess_wrong_guesses` compares against a *greedy* solver, not a globally optimal one — maximising per-guess hit probability does not minimise total wrong guesses. It can be negative. It is a comparison against a strong baseline, not a bound. Do not call it "regret" in the paper.

Both `hit_prob_regret` and `suboptimal_rate` are measured against the *configured* reference policy (`--strategy` / `-S strategy=`). Under the default `max_hit_prob` that is also the best available hit probability; under `info_gain` the reference deliberately gives up hit probability for a better partition, so the two differ. State which policy produced a number.

### Calibration

Three reference agents over the 100 dataset words, 10 wrong guesses allowed, scored against the shipped en-GB dictionary:

| agent     | win  | repeat | dominated | subopt | hit regret | wrong | oracle | excess |
| --------- | ---- | ------ | --------- | ------ | ---------- | ----- | ------ | ------ |
| optimal   | 1.00 | 0.000  | 0.000     | 0.000  | 0.000      | 3.75  | 3.75   | 0.00   |
| frequency | 0.13 | 0.000  | 0.383     | 0.755  | 0.461      | 9.64  | 3.75   | 5.89   |
| sloppy    | 0.09 | 0.148  | 0.413     | 0.781  | 0.466      | 9.82  | 3.75   | 6.07   |

`frequency` plays a fixed `etaoin…` order and never conditions on evidence — roughly what the eval's own system prompt asks for. It makes a provably dead guess 38% of the time. The spread between these agents is what makes the metrics usable on real models. All three are deterministic.

______________________________________________________________________

## 3. The pilot

**This is the next action, and everything else is downstream of it.**

The thesis assumes models win while playing badly. That has not been tested. If real models turn out to play near-optimally, the thesis is wrong and the interesting paper is a different one. Better to learn that in a weekend than after three months.

This environment has no model API keys, so the pilot needs to run elsewhere.

### Running it

```bash
git clone https://github.com/MattFisher/hangman-bench
cd hangman-bench
git checkout claude/hangmanbench-paper-scope-hdyz69
# --all-extras pulls in the openai/anthropic provider SDKs, which are
# optional-dependency extras; a bare `uv sync --dev` removes them.
uv sync --dev --all-extras

# Two models, all 100 words, oracle scoring on by default
uv run inspect eval src/hangman_bench/hangman.py@hangman \
  --model openai/gpt-5-nano --limit 100 --log-dir logs/pilot

uv run inspect eval src/hangman_bench/hangman.py@hangman \
  --model anthropic/claude-sonnet-5 --limit 100 --log-dir logs/pilot

# Aggregate across models
uv run analysis/pilot_oracle.py from-logs --logs logs/pilot --out analysis/pilot
```

`oracle_scorer` runs inside the eval, so the metrics are already in the log. `pilot_oracle.py from-logs` is for cross-model aggregation and per-guess dumps.

To add oracle metrics to a log produced without them:

```bash
uv run inspect score <log.eval> \
  --scorer src/hangman_bench/hangman.py@oracle_scorer \
  --action append --overwrite
```

Pass `--scorer` explicitly. Bare `inspect score <log>` fails for any package: `scorer_from_spec` falls back to loading from the task file only on `ValueError`, but `scorer_create` raises `LookupError`, so the fallback never runs (`inspect_ai/_eval/loader.py`). Worth reporting upstream.

### What decides the outcome

Look at **`dominated_rate`** first. It counts guesses of a letter that appears in zero remaining candidate words — provably wrong, no judgement involved.

- **Meaningfully above zero** while win rate stays high: the thesis holds. Proceed to scale-up.
- **At or near zero**: models are tracking the belief state. The thesis is wrong. The interesting question becomes why the benchmark saturates anyway, and the paper pivots to the prior-sensitivity and benchmark-construction findings, which stand on their own.

Also worth reading on the first run: `repeat_rate` and `invalid_rate` (basic state-tracking failures), and whether `excess_wrong_guesses` is positive.

### Before trusting the numbers

Sanity-check a handful of trajectories by hand against `analysis/pilot_sim_per_guess.tsv` format. The oracle has been validated on synthetic agents and a mocked Inspect log, but **never on a real model trajectory**. First contact with real data is where harness bugs surface.

### Result (run 2026-08-07)

Ran as a single `inspect eval-set` (all 100 words, defaults, `--continue-on-fail`) at commit 68493ba, inspect-ai 0.3.132: first gpt-5-nano and claude-sonnet-5, then a third leg re-invoking the same eval-set with gpt-4o added, which resumed cleanly (the completed evals were skipped untouched — worth knowing for scale-up). gpt-4o is there as a 2024-era reference point: it tests whether the metrics track capability across generations, not just between two current models. All three legs played under the pre-fix continue nudge, so they are directly comparable. All completed 100/100 with status success. Logs in `logs/pilot` (gitignored); aggregates committed as `analysis/pilot_summary.tsv` and `analysis/pilot_per_guess.tsv`.

Validation before trusting the numbers, per the plan above:

- The simulate calibration reproduced byte-identically in the new environment.
- Every trajectory was cross-checked step by step against the boards the tool actually returned to the model (a stricter check than eyeballing a handful): 199/200 clean. The one flag is benign — on `happy` the model's final tool call went unanswered because the sample hit its message limit, so the replay scores one trailing repeat the game never processed. One guess in one game; no effect on any conclusion.

Two aggregation conventions are in play: `oracle_scorer` reports the mean of per-game rates (± stderr below); `pilot_oracle.py from-logs` pools per guess. Both are shown; say which one a number is whenever citing it.

| metric                     | gpt-4o        | gpt-5-nano                  | claude-sonnet-5 | optimal | frequency |
| -------------------------- | ------------- | --------------------------- | --------------- | ------- | --------- |
| win rate                   | 0.69          | 0.93                        | 0.99            | 1.00    | 0.13      |
| dominated_rate (per-game)  | 0.179 ± 0.017 | 0.057 ± 0.011               | 0.026 ± 0.006   | 0.000   | 0.383     |
| dominated_rate (pooled)    | 0.201         | 0.072                       | 0.031           | 0.000   | 0.383     |
| suboptimal_rate (per-game) | 0.648 ± 0.015 | 0.529 ± 0.017               | 0.462 ± 0.015   | 0.000   | 0.755     |
| hit_prob_regret (per-game) | 0.297 ± 0.015 | 0.178 ± 0.012               | 0.124 ± 0.007   | 0.000   | 0.461     |
| wrong guesses / game       | 6.45          | 4.24                        | 4.02            | 3.75    | 9.64      |
| excess_wrong_guesses       | +2.70 ± 0.33  | +0.49 ± 0.27                | +0.27 ± 0.23    | 0.00    | +5.89     |
| repeat_rate (pooled)       | 0.003         | 0.021 (artifact, see below) | 0.000           | 0.000   | 0.000     |
| invalid_rate               | 0.000         | 0.000                       | 0.000           | 0.000   | 0.000     |

**The thesis holds, and the metric tracks capability.** For the two current models, win rate sits at the ceiling (0.93 / 0.99; nano's 0.93 matches the registered report, though one of its seven losses is artifact-contested — see the nudge paragraph) while `dominated_rate` stays meaningfully above zero — the eval barely separates them on outcome but separates them cleanly on process (nano makes provably dead guesses at about twice sonnet's rate). The 2024-era reference point extends this into a monotone gradient on every process metric: gpt-4o → nano → sonnet is 0.18 → 0.06 → 0.03 dominated, +2.7 → +0.5 → +0.3 excess wrong. Two consequences for the paper. First, the benchmark was *not* saturated a generation ago (gpt-4o wins 0.69); saturation is a property of current models, and the process metrics keep discriminating after it sets in. Second, current models are far closer to the optimal reference than to the frequency baseline, so the claim to make is "wins while sometimes playing provably wrong", not "wins while playing badly".

**Where the dead guesses live.** For all three models the dominated guesses concentrate in the endgame: 86% (nano) / 97% (sonnet) / 91% (gpt-4o) were made with three or fewer candidates remaining, and 86–87% (all three) while a *certain* letter (hit probability 1.0) was available. The canonical failure: board `.a..i.es`, candidate set = {bagpipes}, and nano guesses c, d, u, m, h on five consecutive turns. The board plus dictionary has already pinned the word; the model cannot retrieve it. This makes the endgame decomposition (section 7) the single most informative next experiment — the failure is concentrated exactly where it would bite, and it holds across a model generation.

**Losses.** nano lost 7: bagpipes, puppy, foxglove, jukebox, zephyr, ivy, happy. sonnet lost 1: puppy — a v_easy label, which says again that the LLM-authored difficulty labels do not track computed difficulty. gpt-4o lost 31, dominated by rare-letter and doubled-letter words (quartz, sphinx, onyx, nymph, lymph, larynx, jinx, kayak, kiosk, haiku, zephyr, zodiac, …). One nuance to keep honest: gpt-4o's win rate *does* fall along the LLM-authored labels (v_easy 0.80, medium 0.95, hard 0.55, v_hard 0.25), even though the labels do not correlate with solver-computed difficulty — plausibly both the labeller and the player are sensitive to word rarity. Worth one line in the benchmark-construction paper, not more.

**A harness artifact the pilot uncovered.** All 21 of nano's repeat guesses — across happy (15), zephyr (3), jukebox (2), jinx (1) — were the letter `a`, emitted after the solver's continue nudge, which read "Continue by calling hangman_guess('a') (replace 'a' with your next letter)." The model follows the literal example instead of substituting, in the worst case narrating the game back to itself ("Proceeding to submit the guess 'a' again as requested") for fifteen consecutive turns on `happy` until the message limit ended the sample. The artifact is model-specific: sonnet emitted zero repeats and gpt-4o's three (g, d, d) are organic slips, not the example letter. As measured, nano's `repeat_rate` is harness-induced instruction-literalism, not spontaneous state-tracking failure. The artifact is **not confined to `repeat_rate`** (external review, 2026-08-09): nano entered the loop on `happy` at 9/10 wrong with 11 candidates and a p=0.45 best move still available, and the message limit then scored the game a loss — so the 0.93 win rate contains one artifact-contested loss, and its match with the registered report partly rests on it. The nudge has since been reworded to name no letter; these three runs all predate the fix, so their numbers are comparable with each other but `repeat_rate` (and nano's exact win rate) should be re-measured before being cited.

Re-measured under the fixed nudge (2026-08-09, the ablation's frequency arm): the example-letter artifact is gone — but nano still spiralled, this time organically. On `flopping` it repeated `g`, a letter it had already *hit*, ten times and lost. So the refined claim is: the old nudge chose the *letter*; the loop tendency is nano's own. `repeat_rate` is now citable as model behaviour: nano 13/976 guesses (10 of them that one spiral), sonnet zero in three arms and two in the belief arm (g and i, out of 957 guesses), gpt-4o ≤ 6 scattered singles per arm.

______________________________________________________________________

## 4. Findings so far

Ordered by how much a reviewer would care.

**Most dead guesses are made after the word is already pinned (2026-09-03).** Re-reading the pilot per-guess table conditional on candidate-set size: when exactly one candidate remains, gpt-4o misses 51% of its guesses, gpt-5-nano 18%, claude-sonnet-5 9% (`analysis/pinned_endgame.tsv`; 326 / 230 / 216 such guesses). Those pinned misses account for 73% / 59% / 67% of each model's dominated misses, and they cluster: gpt-4o's fall on 51 distinct words, sonnet's on 16. This sharpens the endgame-concentration finding below (≤3 candidates) to its limiting case, where the task has no decision component left. It is the basis for result 1 in section 1. Caveats: one seed, the pinned regime is defined by the shipped dictionary's support (a larger dictionary pins later), and the rate is not comparable across budgets for the same composition reason as the other per-guess metrics.

**Per-guess metrics depend heavily on the dictionary.** Holding trajectories completely fixed and varying only the dictionary used to score them:

| dictionary | avg candidates | dominated_rate | excess_wrong |
| ---------- | -------------- | -------------- | ------------ |
| full       | 7844           | 0.353          | 5.12         |
| 50%        | 3927           | 0.418          | 6.00         |
| 25%        | 1962           | 0.456          | 6.42         |
| 10%        | 787            | 0.505          | 7.20         |
| 2%         | 158            | 0.567          | 8.07         |

"Model X makes provably dead guesses 35% of the time" is not a fact about the model. It is a fact about (model, dictionary). This must be reported as a sensitivity analysis, not hidden behind a default.

**Re-scored on the 300 real pilot trajectories (2026-08-09): rankings and structure are dictionary-invariant; magnitudes are not.** The table above used synthetic agents in the 0.35–0.57 dominated regime; the external review asked whether the models' 0.03–0.20 regime behaves the same. Grid: {SCOWL en_GB tier-50 (shipped), en_GB tier-70, deterministic 25% subsample (`crc32(word) % 4 == 0`), en_US tier-50}, trajectories held fixed. Full table in `analysis/rescore_grid_summary.tsv`; what it shows:

- Model ranking (gpt-4o > gpt-5-nano > claude-sonnet-5) is unchanged in every cell, on every metric.
- Endgame concentration survives everywhere: dominated misses with ≤3 candidates stay at 86–97% across cells; with a certain letter available, 73–91%.
- Magnitudes move as the synthetic table predicted, and hardest for the best model: sonnet's dominated_rate triples under the 25% subsample (0.031 → 0.095) and the gpt-4o/sonnet contrast compresses from 6.5× to 3.0×. Dictionary choice changes between-model *contrast*, not order.
- en_US ≈ en_GB to three decimals — consistent with the dialect audit finding zero orthographic dominated misses.
- Caveats: the 25% cell injects 222/300 targets, so it stress-tests the injected-target regime more than it represents a plausible dictionary. And the grid covers the only axis that can move `dominated_rate` at all — by the support argument, the pending wordfreq prior axis can only move the graded metrics (given the smoothing floor preserves support).

**Verdict per the review's decision rule: thesis A is the paper** — process metrics keep discriminating after outcome saturation, and that finding is stable across defensible dictionaries. Prior/dictionary sensitivity becomes the robustness section, with "declare your dictionary" as a reporting requirement: magnitudes remain a fact about (model, dictionary). *(Superseded 2026-09-03: the finding stands, the thesis built on it does not — see section 6.)*

**The prompt does not drive the dead guesses (2026-08-09).** The system prompt instructs frequency play, so `dominated_rate` might have measured compliance rather than capability. Ablation: {frequency (original), neutral (no advice), belief (reason from the consistent candidate set)} × three models × the same 100 words, all under the post-fix nudge; paired by word (`analysis/compare_arms.py`, `analysis/ablation_summary.tsv`). Pooled dominated_rate:

| model           | frequency | neutral | belief |
| --------------- | --------- | ------- | ------ |
| claude-sonnet-5 | 0.037     | 0.032   | 0.051  |
| gpt-5-nano      | 0.077     | 0.065   | 0.073  |
| gpt-4o          | 0.230     | 0.236   | 0.193  |

No comparison reaches significance (all McNemar p ≥ 0.24; the closest sign test is gpt-4o frequency-vs-belief at p = 0.060, −0.49 dominated per word under belief). The belief-eliciting prompt does **not** collapse the metric — the review's conditional resolves in thesis A's favour: dominated_rate measures capability, not instruction compliance. The only directional hints are that belief-prompting weakly helps the weakest model and, if anything, slightly hurts the strongest (sonnet +0.14/word, p = 0.12) — consistent with prompt-following mattering most where the underlying skill is weakest, and too weak to claim without replication.

The identical-prompt arms also calibrate single-run noise: pilot vs the re-run frequency arm (only the nudge wording differs) flips 18 of gpt-4o's 100 word outcomes (12:6, p = 0.24) and moves its win rate 0.69 → 0.63. Word-level flips of this size on one seed are why the review's replication requirement stands before any headline claim.

**Win-vs-budget curves from one unlimited run — and models rise to scarcity (2026-08-10).** The wrong-guess budget of 10 was arbitrary. With guesses restricted to the language's declared alphabet, a budget of |alphabet| cannot be exhausted — guessing every letter reveals the word first — so a run at b = |alphabet| (26 for English) is effectively unlimited and the whole win-vs-budget curve falls out of a single run: win@b is the fraction of games needing fewer than b wrong guesses. Run 2026-08-10, three models × the same 100 words, default prompt, fixed nudge; plus a real constrained run at b=4 and the ablation's frequency arm as a b=10 point (`analysis/budget_curves.py`, `analysis/budget_curves.tsv`; all 600 new trajectories cross-checked clean):

| agent                    | b=2 | b=4 | b=6 | b=10 | b=13 | b=16 | b=20 | median | p90 |
| ------------------------ | --- | --- | --- | ---- | ---- | ---- | ---- | ------ | --- |
| oracle (greedy, uniform) | .12 | .54 | .79 | 1.00 | 1.00 | 1.00 | 1.00 | 3      | 6   |
| claude-sonnet-5          | .13 | .45 | .68 | .96  | 1.00 | 1.00 | 1.00 | 4      | 8   |
| gpt-5-nano               | .15 | .36 | .66 | .87  | .96  | .97  | 1.00 | 5      | 11  |
| gpt-4o                   | .08 | .18 | .39 | .63  | .75  | .87  | .97  | 7      | 17  |
| frequency (etaoin)       | .00 | .00 | .02 | .13  | .28  | .53  | .89  | 15     | 20  |

- **Zero DNFs.** When losing is impossible every game completes, so turns-to-success is fully measured — no censoring. The b=10 cap had been hiding gpt-4o's tail: 37% of its words complete only above b=10, all by b=20. This is the small-scale analogue of AISI's inference-scaling result in cyber evals (≈8% of their tasks solved only at the 50M-token budget): a fixed evaluation budget understates the weaker model most. Repeats and invalid submissions never cost a life, so the one failure mode left at b=26 is a spiral into the message limit — the DNF column; none occurred, and the worst game used 58 of 145 messages. gpt-4o won `larynx` only after guessing the entire alphabet (20 wrong guesses): at full budget even alphabet-exhaustion wins, which is exactly why b=26 is a cap in name only.
- **Play is budget-conditional, in the direction of rising to scarcity.** The derived curve assumes the model plays the same regardless of the budget it is told. It does not: the real constrained runs beat the derived prediction in five of six (model × budget) cells and tie the sixth — at b=4, sonnet won 53 actual vs 45 predicted, nano 43 vs 36, gpt-4o 23 vs 18. Pooled discordant words 39:66, exact p = 0.011 (per-cell p 0.10–1.0; pooling treats words as independent across models, which overstates precision). So the single-run protocol is cheap but not behaviourally neutral: the derived curve is a **lower bound** on constrained performance. Protocol: one unlimited run + one constrained point brackets the curve family at a sixth of the cost of the original sweep plan.
- **At low budgets the models sit at or above the greedy uniform-prior frontier** (b=4 actual: sonnet 53 wins vs the oracle's 54; the pilot-derived curves crossed above it at b ≤ 3). The "oracle" is optimal only relative to (greedy policy, uniform prior) — consistent with the correlated-prior confound, and another reason the `prior=` parameter matters.
- **Per-guess rates must not be compared across budgets.** Dominated guesses concentrate in the endgame, so longer games mechanically accrue more of them; the budget changes the composition of guesses. Curves compare outcomes; process metrics compare like-for-like only within a budget.
- **Framing against the AISI cyber result** (https://www.aisi.gov.uk/blog/evidence-for-inference-scaling-in-ai-cyber-tasks-increased-evaluation-budgets-reveal-higher-success-rates): what transfers is "report curves, not fixed-budget points". What hangman adds is an exactly *computable* reference pair — the greedy uniform-prior baseline (median 3 wrong guesses) and the frequency floor — so a declared share of each curve is attributable to the task rather than the model. The horizontal gap at the 90th percentile — +2 / +5 / +11 wrong guesses for sonnet / nano / gpt-4o — is relative to that declared greedy baseline, **not to an optimum**: the reference maximises per-guess hit probability, which does not minimise wrong guesses, and the models crossing it at low budgets (above) is the proof. Truly optimal play is exactly computable where the findings concentrate — the ≤3-candidate endgame — and is worth adding at scale-up. Honest disanalogies: success here is guaranteed at b=26, so this measures efficiency rather than capability emergence (the emergence analogue, DNF, did not occur); and this budget buys *evidence*, where theirs buys compute. The compute axis (reasoning effort) is orthogonal and pairs with this for a 2D evidence × compute scaling surface at scale-up.

**Frequency weighting differentially helps common words.** Replacing the uniform posterior with a wordfreq-weighted one, wrong guesses needed:

| band              | uniform | weighted | delta |
| ----------------- | ------- | -------- | ----- |
| common (zipf>4.5) | 3.50    | 1.67     | −1.83 |
| moderate          | 3.80    | 3.10     | −0.71 |
| rare (zipf\<3.5)  | 3.77    | 3.63     | −0.14 |

But the words that got *harder* were `puppy`, `kitten`, `turtle`, `monkey`, `giraffe`, `ladder`, `bagpipes` — concrete nouns of middling frequency. Under corpus-frequency weighting the oracle is pulled toward commoner words of the same shape. So corpus frequency is not the right prior either. There are three distributions, not two: uniform over a dictionary, general corpus frequency, and whatever distribution people actually draw hangman targets from.

**Human-curated puzzle targets are strongly frequency-skewed.** Using Wordle as the closest public analogue — 2,315 curated answers versus 10,657 words Wordle accepts but never uses:

|                                   | median zipf | rare (\<3) | common (>4.5) |
| --------------------------------- | ----------- | ---------- | ------------- |
| Wordle answers (human-curated)    | 3.52        | 26.9%      | 16.8%         |
| Wordle accepted-but-never-answers | 1.39        | 89.6%      | 1.3%          |

A 2.13 zipf gap, roughly 135×. This replicates the frequency skew on human curation, independent of any LLM. The Wordle answer/guess split is a citable precedent for exactly the target-vs-dictionary distinction we need.

**Extreme concreteness looks like an LLM fingerprint.** Brysbaert concreteness norms (1 abstract, 5 concrete):

|                                       | mean | % highly concrete |
| ------------------------------------- | ---- | ----------------- |
| Wordle answers (human-curated)        | 3.53 | 36.8%             |
| hangman-bench dataset (LLM-generated) | 4.41 | 77.7%             |
| SCOWL-50 sample                       | 3.20 | 27.4%             |

Human puzzle-setters barely shift concreteness; the LLM-generated dataset more than doubles it. Caveat: the norms cover only ~40k common words, so only 4% of the Wordle accepted-only list is rated — treat that row as suggestive. The frequency comparison is solid; the concreteness one is not, on its own.

**LLM-authored difficulty labels are uncorrelated with computed difficulty.** Spearman between the dataset's `v_easy…v_hard` labels and solver-derived difficulty: `wrong_coverage` −0.008, `wrong_freq_raw` +0.026, `wrong_info_gain` +0.368. The labels were generated by an LLM (Windsurf).

### A confound in the benchmark itself

The dataset was LLM-generated and the systems under test are LLMs, so targets and guessers are drawn from correlated priors. An LLM guesser may score well partly because another LLM chose words that LLMs find natural. This is independent of everything else and is only fixable by not letting an LLM choose the targets.

**Do not claim the current 100 words are representative of human hangman targets.** They are representative of what one LLM produced when asked.

______________________________________________________________________

## 5. The dictionary

The oracle's dictionary is part of the measurement, so it needs stated, licensed, reproducible provenance.

**Shipped:** SCOWL 2020.12.07, British English (`-ise`), size tiers ≤50, words of three or more letters, proper names and abbreviations excluded. 61,460 words. Built by `analysis/build_wordlist.py`, licence notice retained at `src/hangman_bench/data/SCOWL-Copyright`, attribution in `NOTICE`.

One dialect per file, so en_US / en_AU / en_CA can be added later without disturbing the one in use:

```bash
uv run analysis/build_wordlist.py --dialect en_US --tier 60
```

The dataset was made consistently en-GB to match: `dwarves`→`dwarfs`, `whiskey`→`whisky`, `oxidize`→`oxidise`, `galvanize`→`galvanise`. All 100 words are now in the dictionary.

### What this replaced, and why

The previous list came from Wolfram MathSource `SimulationData.zip` (the data behind the 2010 "25 Best Hangman Words" blog post), via `analysis/extract_wordlist.py`. It was indefensible on four counts:

- **Licensing.** Wolfram MathSource has no clear redistribution terms, and the file had been placed inside the built wheel.
- **Misattributed.** `analysis/README.md` credited the Curlew British English list. The file was unambiguously American: `color`, `realize`, `pajamas`, `dwarfs` present; no British spellings at all.
- **Content.** A spell-checker list, which exists to *accept* strings. It contained initialisms as entries (`cs`, `cw`, `kc`, `kw`, `ls`, `rs`, `ts`), lowercased proper nouns, and vocabulary that stops around the late 1990s (`email` and `laptop` present; `internet` and `blog` absent).
- **Unstated inclusion criteria.** "Words Wolfram happened to simulate in 2010" is not a specification a reviewer can reproduce.

The Wolfram ingestion scripts remain in `analysis/` as a record of the original difficulty analysis. They are no longer the source of the shipped dictionary.

### Note on British spelling variants

SCOWL publishes British in two conventions: `british-words` (`-ise`) and `british_z-words` (`-ize`, Oxford spelling). They are alternatives, not complements. Merging them puts 228 words in the dictionary under both spellings, and a guesser facing `reali_e` would then have to choose `s` or `z` on no information — pure orthographic convention, zero inference. We use `-ise` only.

Note that `en_GB` still contains both endings, correctly: 84 words end `-ise` (the obligatory class — `advertise`, `exercise`, `surprise`, which are `-ise` in every dialect) and the alternating class is spelled `-ise` throughout. Only `prize`/`prise` and `apprize`/`apprise` appear in both spellings, and those are genuinely different words.

### ESDB — checked, rejected for now

ESDB (formerly SCOWLv2) carries commonness, dialect, variant, POS and inflection data under the same licence. Not usable yet:

- `scowl.db` and `scowl.txt` are build outputs, not committed. There is no downloadable artifact; you would build from a git branch.
- SourceForge's `Rev 1`…`Rev 7.1` entries look like ESDB releases and are titled "English Speller Database", but are SCOWLv1 from 2011 — the project was renamed and all historical files inherited the new title. The latest packaged release is `2020.12.07`, which is SCOWLv1.
- Its README says "ESDB is still a work in progress"; only size 60 is vetted.
- Its commonness data derives from Moby Words II and Brian Kelk's frequency classification — both 1990s, bucketed into ~10 coarse levels. It would not replace wordfreq for continuous frequency weighting.

Revisit if a packaged release appears. Its POS and inflection data would be useful for filtering to lemma nouns. AGID, in the same SourceForge project, already provides that today without ESDB.

______________________________________________________________________

## 6. Other considered theses

Three framings were carried at various points. Each has evidence in this repo and none is wrong; they are backburnered because none would be a paper a reader outside the project would act on.

**Thesis A — process metrics keep discriminating after outcome saturation.** The organising thesis from the pilot (2026-08-07) through the ablation (2026-08-09). Evidence: `dominated_rate` and `hit_prob_regret` order the three models monotonically while win rate sits at 0.93 / 0.99 for the two current ones; the ordering is dictionary-invariant; it is not prompt compliance. Why backburnered:

- It is expected. That per-step metrics have more resolution than a binary outcome, and that a 2024 model makes more dead guesses than a 2026 one, surprises nobody. The external review called A modest and was right.
- The budget-curve result undercuts its premise. Saturation was an artefact of a generous budget; at b=4 win rate discriminates fine and is an outcome metric. Once curves are reported instead of points, process metrics become a nice-to-have, and the motivation for A evaporates.
- The gradient it rests on confounds vendor, generation and reasoning effort, on one seed, over 100 LLM-chosen words, and there is no external-validity evidence that `dominated_rate` on hangman predicts anything else.

What survives: the metrics, the calibration, and the finding itself as one figure in the robustness section. `suboptimal_rate` stays out of headlines (review disposition, 2026-08-09).

**Thesis B — measuring process requires committing to a prior, and defensible priors disagree.** The original working thesis (section 1 before 2026-08-09). Evidence: the synthetic and real dictionary grids, the wordfreq-weighting table, the Wordle answer/guess split. Why backburnered: the real-trajectory grid showed rankings and structure are invariant and only magnitudes move, which makes it a reporting requirement ("declare your dictionary") rather than a result. The support-vs-weight distinction (dominated misses depend only on the prior's support; graded metrics on its weights) is the one idea from B worth a paragraph in the metrics section of whatever gets written.

**The benchmark-construction paper — LLM-authored difficulty labels are noise; LLM-generated targets are frequency- and concreteness-skewed.** Evidence: Spearman ≈ 0 between labels and computed difficulty; concreteness 4.41 vs 3.53 for Wordle's human-curated answers; the correlated-priors confound. The external review suggested this had the larger audience. Disagreed on 2026-09-03: one generator of unknown identity, 100 words, and "LLM-generated data skews common and concrete" has been reported many times. It is a blog post that takes an afternoon, plus one paragraph in any paper's dataset section. Write it, link it, move on.

**Also considered and folded in rather than dropped:** commit calibration (does the model know that it knows, via `allow_word_guesses`) is now the fourth leg of the decomposition in section 7 rather than a separate headline; "evil hangman" and the memory ablation stay on the later list.

______________________________________________________________________

## 7. Plan

The principle: decide the question before spending on scale. The scaled dataset (thousands of words, held-out tier) was the largest cost item on the old roadmap and buys the least until the two results are confirmed. Two cheap experiments decide them; a gate follows; scale-up is conditional on the gate.

### Stage 1 — decide result 1 without the game loop (near zero API cost)

**The static pinned-board probe.** The pinned regime does not need an agentic harness. Sample boards directly from the dictionary: choose a target, choose a set of guessed letters, keep the (board, excluded letters) pair only if exactly one word in the dictionary is consistent with it. Present each as a single prompt — board, excluded letters, "name the word" — and score exact match against the oracle's one candidate. Also record whether the model's answer is *consistent* with the board at all (a wrong-but-consistent answer is a dictionary-coverage mismatch; an inconsistent answer is a verification failure).

- Several hundred boards, stratified by word length and by wordfreq band, drawn from the dictionary rather than the 100-word dataset so the LLM-chosen-targets confound is gone by construction.
- Same three models as the pilot, plus (this is the point of doing it now) a same-family ladder or a thinking on/off pair on one model, so the reasoning-effort confound is settled before any further gradient claim.
- Reads: if the static probe reproduces the 0.51 / 0.18 / 0.09 gradient, the failure is isolated with no game loop and no scaled dataset, and the agentic runs are for the *loop-dependent* legs only. If the static rate is much lower than the in-game rate, the loop is contributing (turn-by-turn evidence accumulation, context length, or the frequency instruction) and that gap is itself the finding.

**Mine the transcripts already on disk.** nano verbalises candidate lists spontaneously ("Possible words remain: gawky, …"). Before building a probe, extract every such list from the existing logs and score it against the oracle's candidate set (precision, recall, count accuracy). Free, and it says whether the verbalised-posterior probe is measuring something models will do.

### Stage 2 — decide result 2 (small API cost)

**Budget-conditional replication.** Two models (one reasoning, one not), three seeds each, real constrained runs at b ∈ {2, 4, 6, 8}, plus the unlimited run per seed to derive the prediction. Paired by word, McNemar per cell, and a dose-response read across b. Add a reasoning-effort sweep on the reasoning model so the effect can be separated from test-time compute.

- Mechanism check without the composition problem: compare `hit_prob_regret` on the first three guesses only, like-for-like across budgets. If models play tighter early when told the budget is tight, that is the mechanism; if the early game is identical and only the endgame differs, the effect is something else.
- Reads: dose-dependent and replicated → a paper section. Present at one budget only, or gone under replication → a paragraph in the methods ("derived curves are a lower bound") and no more.

### Gate

Run stages 1 and 2, then decide:

- **Both hold** → a workshop paper (NeurIPS/ICLR evaluation or datasets-and-benchmarks track; arXiv after review): result 1 as the headline with the decomposition below, result 2 as the second contribution, thesis A's figure and the dictionary grid as robustness.
- **Only result 1 holds** → a short paper on belief-state decomposition with exact ground truth; result 2 becomes a methods caveat.
- **Only result 2 holds** → it is small; write it up as a note alongside the benchmark contribution and stop.
- **Neither holds** → ship the eval to inspect_evals, write the benchmark-construction blog post, and stop with a clear conscience.

### Stage 3 — conditional on the gate: the decomposition matrix

Only if result 1 holds. Each leg has the oracle as ground truth; the first is stage 1's probe.

| leg          | prompt                                              | measures                                                                    |
| ------------ | --------------------------------------------------- | --------------------------------------------------------------------------- |
| retrieval    | pattern + exclusions → name the word                | can the model produce it                                                    |
| verification | pattern + exclusions + a word → does it fit?        | can the model check it (yes/no, both polarities)                            |
| decision     | pattern + exclusions + candidate list → choose      | given the belief state, does it choose well                                 |
| belief       | mid-game: how many candidates? name up to k         | does a belief state exist; calibration vs oracle                            |
| commit       | `allow_word_guesses=True`: when does it submit?     | was the submission consistent; \|C\| at commit; realised success vs 1/\|C\| |
| tokenisation | spaced vs contiguous board rendering, retrieval leg | is the failure in reading the board                                         |

Plus the **human endgame study** the review proposed (10 people × 10 pinned boards): cheap, and it buys the sentence "models fail exactly where humans do not", which is the sentence that makes result 1 land.

### Stage 3 — conditional on the gate: scale-up prerequisites

Carried over from the external review, unchanged, and still required before any headline number: pinned and recorded generation configs; ≥3 seeds on headline cells; paired statistics clustered by word; the stratified runtime target generator (dev / held-out / fresh-seed tiers, salted hash manifest for the held-out set) in place of a committed word list; raw Inspect logs published with any write-up. `prior=uniform|frequency` on the oracle stays on the list because the decomposition's graded metrics depend on it, but it is no longer on the critical path.

### Hygiene (do alongside stage 1, none blocks anything)

- The system prompt still describes tool-output fields (`current_state`, `game_over`, `won`) that the tool does not return; it returns `Word:` / `Status:` lines. Align one with the other.
- `Sample.input` is dead code: `game_initialiser` overwrites `user_prompt.text`. Remove one.
- `analysis/README.md` still describes the Wolfram pipeline as the source of the shipped dictionary in places.
- Write the benchmark-construction blog post from the evidence already in section 4.

### Known issues

- A sample that ends by message limit can leave the model's final tool call unanswered; the replay counts it as an attempt the game never processed. Observed once in 200 games (`happy`, a trailing repeat). Cosmetic at current scale; worth an explicit rule (score only answered calls) if it recurs.
- The registered evaluation report cannot be updated with the pilot results yet. A registry entry pins a `repository_commit` that a reader can check out and reproduce from, and the pilot ran at 68493ba — on the PR #3 branch, not on `main`. Update the register once that history is on `main`: bump `source.repository_commit`, and replace the single stale gpt-5-nano row with the three-model table from section 3.
- The registered evaluation report (0.93, `gpt-5-nano`) predates the malformed guess fix. Samples that previously errored out and vanished are now scored, so it is not comparable. Re-baseline before citing it.
- `hangman_player`'s `on_continue` nudge previously named an example letter and gpt-5-nano followed it literally (section 3). Fixed; all three pilot runs predate the fix, so `repeat_rate` is citable only from the ablation arms onward.
- The pilot runbook previously said `uv sync --dev`, which removes the provider SDKs; use `uv sync --dev --all-extras`.

### Fixed after review (2026-08-09)

Raised by an automated reviewer on PR #3, all three confirmed by reproduction before fixing:

- `from-logs` applied one global `--max-guesses` to every replay instead of the limit each sample was played under. On a log run at 15, replaying at the default 10 turned a genuine win into a loss and truncated the wrong-guess count from 12 to 10. The logged value now wins; the flag is a fallback only.
- Games won by submitting the full word were replayed as losses, because `extract_trajectories` drops the `submit` action and the letter sequence alone never completes the word. Both the batch reader and `oracle_scorer` now take the outcome from the recorded score.
- Under `--strategy info_gain`, `optimal_letter` came from the chooser while `best_hit_prob` came from the maximum hit probability, so the reference policy scored as suboptimal against itself. Both now come from the configured policy. The default `max_hit_prob` is unaffected — calibration output is byte-identical.

### Open decisions deferred past the gate

- *What prior to treat as primary.* Still: make it an explicit parameter and report across all of them. Matters for the graded metrics in the decomposition, not for the pinned miss rate, which depends only on support.
- *Where a target distribution could come from.* Unchanged list of candidates (published hangman lists, open-source hangman wordlists, Wordle answers, psycholinguistic norms). The stage-1 probe sidesteps it by sampling from the dictionary and stratifying by frequency band.
- *Dataset scale and composition.* Thousands of words, stratified, held-out tier. Only after the gate.

______________________________________________________________________

## 8. Caveats to state explicitly in any write-up

- The model is never told which dictionary the oracle uses, so its belief state is not the oracle's. Some of what looks like model error is a specification mismatch. This applies at the orthographic level too: if the hidden word is `realise` and the model guesses `z`, it is penalised for a convention nobody communicated. Call a dominated miss an "error under a declared reference specification", not a "provable error" simpliciter. Audited in the pilot (2026-08-09): **zero of the 328 dominated misses are dialect-orthographic** — no `z` guesses on the `-ise` boards; `whisky`/`dwarfs` differ from their US forms in length; `pajamas` was already excluded by the board evidence wherever it could have mattered. The caveat is conceptually right and currently empty; it becomes live when a scaled dataset admits `-ise/-ize` verbs, so the wording change costs nothing now and pre-empts that.

- The reference solvers are greedy, not optimal. `excess_wrong_guesses` can be negative.

- Difficulty labels in the current dataset are LLM-authored and do not track computed difficulty. Do not use them as a difficulty axis without saying so.

- The current 100 words are LLM-chosen and are not a sample of human hangman targets.

- `repeat_rate` in the *pilot* is a harness artifact (the continue nudge's literal example letter), not a model property. Re-measured under the fixed nudge on 2026-08-09: the artifact is gone and `repeat_rate` is citable from the ablation arms onward — with the nuance that nano's residual repeats are one organic spiral (`flopping`, g × 10), so report it alongside its concentration, not as a smooth rate.

- The pinned miss rate (section 1) is conditional on the shipped dictionary's support: a larger dictionary pins the word later, so the regime is defined relative to a declared dictionary in exactly the way a dominated miss is. Report the dictionary with it.

## External review (2026-08-09)

Condensed from an advisory review of PR #3 and the pilot. Items are things the notes do *not* already say, or say and then fail to act on.

### The two-theses problem

The notes state Thesis B (prior sensitivity) as the working thesis, but the pilot, PR, and roadmap are organised around Thesis A (process metrics keep discriminating after saturation). These are different papers. A alone is modest — "score the trajectory" is established, and one game is a case study. B is the more general contribution but **has never been tested on real trajectories**: the dictionary-sensitivity table used synthetic agents at dominated ≈ 0.35–0.57, a different regime from the models' 0.03–0.20, and it shows the numbers move, not the conclusions.

**Decisive, zero-API-cost experiment (do first, before the lexical-access ablation):** re-score the 300 pilot trajectories under {SCOWL-50, SCOWL-70, 25% subsample, en_US} × {uniform, wordfreq}. If model ranking and the endgame-concentration finding are invariant → Thesis A is the paper, B is a robustness section. If anything flips → B is the paper, pilot is its example.

**Unexploited structural point:** `dominated_rate` depends only on the *support* of the prior (a zero-coverage letter is dominated under any prior on the candidates); the graded metrics depend on the *weights*. So the headline metric is provably invariant to uniform-vs-frequency and sensitive only to the dictionary. Support-sensitive vs weight-sensitive is the organising idea of the metrics section, whichever thesis wins.

### "Provably wrong" needs reframing

A dominated miss is provable only relative to a dictionary the model was never told about — it conflates irrationality with specification mismatch. This is not hypothetical: four targets were *converted* to en-GB (`oxidise`, `galvanise`, `whisky`, `pyjamas`/`dwarfs`), so a US-prior model can score a "provably dead" guess for a spelling convention. Actions: audit pilot dominated misses for dialect-orthographic cases; reword "provable error" → "error under a declared reference specification"; optionally use a union dictionary for the dominance test only (widening support is conservative).

### The prompt instructs the failure mode

The system prompt says "make smart guesses based on common letter frequencies" — and the notes already observe the `frequency` baseline is "roughly what the eval's own system prompt asks for" without following through: `dominated_rate` may partly measure *compliance*. Prompt ablation (current / neutral / "consider which words fit the pattern") is cheaper than and logically prior to the candidate-list ablation. If dominated collapses under the belief-eliciting prompt, the claim changes entirely.

Adjacent harness defects: the `Sample.input` text is dead code (`game_initialiser` overwrites `user_prompt.text`); the system prompt describes tool-output fields (`current_state`, `game_over`, `won`) that don't match what the tool returns ("Word:", "Status:").

### Nudge artifact is not confined to repeat_rate

nano entered the 'a'-loop on `happy` at 9/10 wrong with 11 candidates left (best move p=0.45) and was killed by message limit → scored a loss. So the 0.93 win rate contains one artifact-contested loss, and "matches the registered report" partly rests on it. Correct the "other metrics unaffected" claim.

### Capability-gradient claim is under-designed

Three models confound vendor × scale × generation × (unrecorded) reasoning effort; one run each; generation config unpinned; monotone-over-3 across six correlated metrics is weak. For scale-up: pin/record configs; replicate ≥1 model ×3 for seed variance; same-family ladder; reasoning-effort sweep (if dominated falls steeply with thinking budget, it's largely a test-time-compute metric and the language changes). Use **paired stats** — all models play the same 100 words; per-word deltas + McNemar, cluster by word (dominated guesses concentrate in few words, e.g. bagpipes). Drop `suboptimal_rate` from headlines: any-epsilon-below-argmax, sits ≈0.46 even for strong play; `hit_prob_regret` subsumes it.

### Retrieval is one of four hypotheses

Endgame dead guesses could be: retrieval failure, verification failure, tokenization (mapping `. a . . i . e s` to a lexeme), or never attempting enumeration because the prompt says frequencies. The planned candidate-list ablation tests verification+choice only. Replace with a matrix: pattern→generate (retrieval); pattern+word→fits? (verification); pattern+list→choose (decision); spaced vs contiguous board (tokenization).

**Human baseline:** nobody enumerates 61k words mid-game, so nonzero dominated is not per se damning; but in the ≤3-candidate endgame humans plausibly hit zero (`.a..i.es` → bagpipes is easy for people). Small study (10 people × 10 endgame boards) buys the sentence "models fail exactly where humans don't".

### Benchmark additions, ranked

1. **Verbalised-posterior probe** (not on roadmap; should be near top): ask the model to state candidate count / name up to k candidates; score count accuracy, precision/recall vs oracle set, calibration. Direct measurement of whether a belief state exists; reframes the project as *belief-state evaluation with computable ground truth*.
2. Commit calibration (roadmap item 6) — agree it may beat dominated_rate as headline.
3. Budget sweep reframed: publish the win-vs-budget *curve* per model against the oracle's curve as the primary outcome-level artifact.
4. Memory ablation: omit the board after turn 1 → clean state-tracking test.
5. Evil hangman (adversarial setter, still computable) — later.

State explicitly that `allow_word_guesses=False` *creates* the endgame regime where the effect lives; report the commit variant alongside.

### Dataset generation: sample from the dictionary at runtime

Generate targets from the oracle's dictionary with a seeded, stratified sampler instead of committing a word list. Be clear what this does and does not buy:

- **A public seed is not a contamination defence.** Dictionary + sampler + seed in a public repo fully determines the list; it's a committed list with one level of indirection. Secrecy comes from the seed being secret or fresh, not from generation happening at runtime.
- What it *does* buy: no words or per-letter solutions ever committed; `target_in_dictionary` holds by construction (which also softens the specification-mismatch caveat — say so); cheap scale to thousands of words.
- **Uniform sampling changes the construct.** Uniform SCOWL-50 draws land on the accepted-but-never-an-answer distribution from the Wordle analysis, not a hangman-target distribution. Stratify by length, frequency band (`--with-frequencies`), and computed difficulty; filter to lemma forms via AGID. The strata spec becomes part of the benchmark specification — a reproducibility claim no hand-curated list can make.

Three tiers:

1. **Public dev set** — fixed published seed. Reproducible, preserves the paired design (all models play identical words). Assume it gets contaminated; it's for development and comparability.
2. **Private held-out set** — same generator, secret seed, never committed; publish a salted hash manifest so the set can be proven later. Headline numbers come from here.
3. **Fresh-seed mode** — `seed=None` draws one and records it in the Inspect log. Reproducible after the fact, no canonical list to memorise; labs compare by sharing a seed.

Implementation: pin the dictionary version and sort the wordlist before sampling (a rebuild or file-order change must not silently alter the draw); derive RNG state from a hash of (sorted wordlist, seed, strata spec), not incidental determinism.

**Contamination as a measurement, not just a threat.** Because ground truth is computable, compare endgame retrieval (the bagpipes regime) on dev-set words vs freshly-sampled words matched on length, frequency band, and computed difficulty — the oracle supplies matched controls for free. A systematic dev-set advantage is a behavioural contamination signal. Few benchmarks can write this paragraph; it turns fresh-seed mode into an instrument.

### Hygiene

- `pilot_per_guess.tsv` publishes complete per-letter solutions for all 100 words — fine for the pilot, fatal for the scaled benchmark. The runtime generator above supersedes committed lists; held-out words are never committed, salted hash manifest only.
- Publish raw Inspect logs, not only derived TSVs (the extraction pipeline has already produced six scoring bugs; readers shouldn't have to trust it).
- Lit sweep gap: prior work on LLMs playing Wordle/deduction games is closer than either cluster currently cited; engage process-reward / trajectory-eval literature explicitly for Thesis A's novelty question.

### The other paper

The quarantined benchmark-construction findings (labels-are-noise; frequency/concreteness skew with the Wordle split as human-curation control) have a larger audience than hangman process metrics and most evidence already collected. Choose deliberately, not by default toward the project with more code.

### Order of operations

01. Ranking-invariance re-scoring (free, decides the thesis).
02. Prompt ablation.
03. Dominated-miss dialect audit + "provable error" reframing.
04. Fix the nudge-artifact containment claim.
05. Scale-up prerequisites: pinned configs, seed replication, family ladder, effort sweep, paired power calc, and the stratified runtime generator (dev / held-out / fresh-seed tiers) in place of a committed word list.
06. Ablation matrix (replaces single candidate-list ablation).
07. Verbalised-posterior probe + commit calibration.
08. Human endgame study.
09. Contamination hygiene.
10. Deliberate A-vs-B-vs-other-paper decision.

### Disposition (2026-08-09)

Working through the list above; status and divergences:

- **Item 3 — done, result: empty.** Zero of 328 pilot dominated misses are dialect-orthographic (details in section 8 caveats). Wording reframed. Pushback on the union-dictionary suggestion: widening support redefines the metric ("dead in every dialect") rather than fixing it; keep strict as primary and add union as a robustness pair when a scaled dataset makes the caveat live.
- **Item 4 — done.** Containment claim corrected in section 3; the artifact also contests one nano loss (`happy`).
- **Item 1 — done, verdict: thesis A.** Rankings and endgame concentration are invariant across the dictionary grid; magnitudes move (up to 3× for the best model). Details in section 4. The prior axis waits on the `prior=` parameter, and by the reviewer's own support argument it cannot move `dominated_rate`, only the graded metrics — with the caveat that this invariance requires the frequency prior's smoothing floor to preserve support.
- **Item 2 — done, verdict: not compliance.** The belief-eliciting prompt does not collapse `dominated_rate` (no significant difference in any paired comparison; details in section 4). The metric measures capability. The re-run also cleared the nudge artifact: the example letter is gone, nano's residual repeating is organic (one g-spiral on `flopping`), and `repeat_rate` is citable from the ablation arms onward.
- Pushback on dropping `suboptimal_rate` entirely: agreed off headlines, kept in tables (it is the only per-decision rate; `hit_prob_regret` headlines).
- Noted for scale-up: of the confound list, the reasoning-effort sweep is the most urgent (nano reasons, gpt-4o does not; the gradient may be partly test-time compute).
- The verbalised-posterior probe has direct supporting evidence already: nano spontaneously verbalises candidate lists mid-game ("Possible words remain: gawky, …" on `happy`), so existing transcripts may be minable before any probe is built.

### Refocus (2026-09-03)

A review of the whole project against the question "is this interesting, or a paper for its own sake". Outcome, recorded in sections 1, 6 and 7:

- Thesis A retired as the organising thesis: expected, undercut by the budget-curve result, and resting on a confounded three-model gradient. Its evidence stays as robustness material.
- The pinned-endgame miss rate (`analysis/pinned_endgame.py`) promoted from an unremarked feature of the per-guess table to result 1. Most dominated misses are made after the word is already determined; the character-level framing the earlier notes avoided is where the phenomenon lives, and the decomposition matrix from this review is how the project adds to that literature rather than joining it.
- Budget-conditional play promoted to result 2, contingent on replication.
- The benchmark-construction findings demoted from candidate paper to blog post.
- Scale-up made conditional on a gate that two cheap experiments decide. The review's scale-up prerequisites are carried unchanged behind the gate.
- Two harness defects from this review (system prompt describes fields the tool does not return; `Sample.input` is dead code) confirmed still open and listed under hygiene.
