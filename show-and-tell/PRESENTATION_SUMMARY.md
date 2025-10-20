# Hangman Bench Presentation - Summary

## What I've Prepared for You

I've created a complete presentation package for your Hangman Bench show-and-tell:

### 📄 Core Documents

1. **SHOW_AND_TELL.md** - Complete narrative guide
   - Part 1: Building games with tools (architecture, patterns)
   - Part 2: The "apple problem" (research journey)
   - Part 3: Why it's a good eval
   - Includes code examples, key takeaways, Q&A

2. **SLIDES_OUTLINE.md** - 22 slides + backups
   - Ready to convert to presentation format
   - Visual flow with code snippets
   - Backup slides for deep dives
   - Timing estimates included

3. **CODE_WALKTHROUGH.md** - Detailed code guide
   - Line-by-line navigation
   - Key patterns explained
   - Demo scripts included
   - 20-minute walkthrough flow

4. **DEMO_SCRIPT.md** - Step-by-step demos
   - 6 different demo options
   - Terminal commands ready to run
   - Expected outputs documented
   - Timing for each demo

5. **PRESENTATION_SUMMARY.md** - This file!

---

## The Story Arc

### Act 1: The Tool Demo (10-15 min)

**"We wanted to show how to build games with Inspect's new tool features"**

- Hangman is perfect: simple rules, clear win/loss, strategic play
- Walk through the architecture: GameState → Store → Tool → Solver → Scorer
- Key patterns: typed stores, tool feedback, on_continue, message limits
- This is the "how to build game evals" part

### Act 2: The Apple Problem (15-20 min) ⭐

**"Then we tried to measure word difficulty objectively..."**

- Initial dataset classified by intuition: "apple" = v_easy
- Found Wolfram simulation: "apple" = hard (6.67 wrong guesses)
- Built 3 deterministic solvers: frequency, coverage, info gain
- Results: apple = 1, 5, 1 wrong guesses (depending on solver!)
- **The insight**: No single "correct" difficulty metric
- Other surprises: happy/puppy are v_hard, rhythm is medium
- Decision: Keep original labels, include all analysis code

### Act 3: It's Actually Useful (5-10 min)

**"And it turns out to be a good eval in its own right"**

- Tests strategic tool use, language understanding, probabilistic reasoning
- 100 words, grouped scoring, rich metadata
- Configurable parameters, cost estimation utilities
- Published on PyPI as `hangman-bench`

---

## Key Technical Insights

### For Building Game Evals

1. **Typed Store Pattern**

   ```python
   class HangmanStore(StoreModel):
       game_state: GameState | None = None
   
   hstore = store_as(HangmanStore)
   hstore.game_state.guess(letter)
   ```

   - Type-safe per-sample state
   - Persists across tool calls
   - Clean separation of concerns

2. **Tool Feedback Design**
   - Return structured information, not just "correct/wrong"
   - Include enough context for strategic decisions
   - Show current state, remaining resources, history

3. **Graceful Degradation**

   ```python
   async def on_continue(state: AgentState) -> bool | str:
       if game_over:
           return False  # Stop
       if not used_tools:
           return "Try calling hangman_guess('a')..."  # Nudge
       return True  # Continue
   ```

4. **Message Limits**
   - Calculate from game complexity: `(max_turns) * 4 + buffer`
   - Account for commentary, prompts, tool calls, responses

### For Difficulty Analysis

1. **Intuition ≠ Objective Measurement**
   - "Apple" seems easy but is hard for some solvers
   - Repeated letters (happy, puppy) increase difficulty
   - Scary-looking words (awkward) may have common letters

2. **Solver Strategy Matters**
   - Frequency: Maximize raw letter counts
   - Coverage: Maximize probability of hitting any word
   - Info Gain: Minimize expected remaining candidates
   - Each optimizes different objectives → different rankings

3. **Multiple Metrics Are Valuable**
   - No single "correct" measure
   - Different metrics reveal different aspects
   - Variance itself is interesting data

4. **Reproducibility Is Key**
   - All analysis scripts included
   - Deterministic solvers (no randomness)
   - Clear documentation of methodology

---

## The "Apple" Story (Your Hook)

This is your most compelling narrative:

> "We built Hangman Bench to demonstrate Inspect's tool features. Simple game, clear rules, perfect for showing stateful interactions. We classified 100 words by intuition - 'apple' is obviously very easy, right?
>
> Then we tried to measure difficulty objectively. Found a Wolfram simulation from 2010 that rated 'apple' as HARD - 6.67 wrong guesses on average. That didn't match intuition, so we built our own solvers.
>
> Three different strategies, three different results: 1, 5, and 1 wrong guesses for 'apple'. The frequency solver finds 'a' and 'e' immediately. The coverage solver struggles because 'p' appears in many words but often in wrong positions.
>
> The insight: there's no single 'correct' difficulty metric. It depends on your strategy. We kept the original intuitive labels because the variance itself is valuable. And we included all the analysis code so anyone can reproduce it.
>
> What started as a tool demonstration became a research project about the nature of difficulty in games."

---

## Recommended Presentation Flow

### For 30-Minute Slot

1. **Intro** (2 min)
   - What is Hangman Bench
   - Original goal: tool demonstration

2. **The Apple Problem** (10 min) ⭐
   - Show the paradox
   - Live demo: run solvers on "apple"
   - Explain why results vary
   - Show other surprising results

3. **Code Walkthrough** (15 min)
   - GameState and store pattern
   - Tool implementation
   - on_continue logic
   - Quick eval demo

4. **Wrap-up** (3 min)
   - Key takeaways
   - It's a good eval too
   - Questions

### For 45-Minute Slot

1. **Intro** (3 min)
2. **Architecture Deep Dive** (12 min)
   - Show all components
   - Walk through patterns
   - Explain design decisions
3. **The Apple Problem** (15 min) ⭐
   - Full research journey
   - Multiple word comparisons
   - Analysis pipeline
4. **Live Demo** (10 min)
   - Run eval
   - Show analysis scripts
   - Explore results
5. **Wrap-up** (5 min)
   - Takeaways
   - Q&A

### For 60-Minute Slot

- All of the above
- Plus: Cost estimation demo
- Plus: More code deep dives
- Plus: Extended Q&A

---

## Demo Recommendations

### Must-Have Demos

1. **The Apple Comparison** (Demo 2)
   - Most compelling story
   - Shows the core insight
   - Easy to understand
   - Memorable hook

2. **Code Walkthrough** (Demo 4)
   - Shows technical depth
   - Teaches useful patterns
   - Relevant to audience

### Nice-to-Have Demos

3. **Quick Eval** (Demo 1)
   - Shows it working
   - Builds credibility
   - Can run in background

4. **Word Comparison** (Demo 3)
   - Reinforces the insight
   - Shows patterns
   - Quick and visual

### Skip If Short on Time

5. **Analysis Pipeline** (Demo 5)
   - Interesting but not essential
   - Can describe instead of showing

6. **Cost Estimation** (Demo 6)
   - Useful but not core story
   - Mention it exists

---

## Materials Checklist

### Before Presentation

- [ ] Test all demo commands work
- [ ] Have API keys configured (if running evals)
- [ ] Open key files in IDE
- [ ] Have terminals ready
- [ ] Test screen sharing setup
- [ ] Print/have slides ready (if using)

### Files to Have Open

- [ ] `src/hangman_bench/hangman.py`
- [ ] `src/hangman_bench/datasets.py`
- [ ] `analysis/difficulty_report.tsv`
- [ ] `analysis/README.md`
- [ ] `SHOW_AND_TELL.md` (for reference)

### Terminal Windows

- [ ] Terminal 1: Repo root (for evals)
- [ ] Terminal 2: Repo root (for analysis)
- [ ] Both with `uv` available

---

## Key Messages

### For Engineers Building Evals

1. **Store pattern is powerful** - Clean state management for multi-turn interactions
2. **Tool feedback matters** - Design output to guide model strategy
3. **Graceful degradation** - Handle models that don't follow instructions
4. **Message limits are critical** - Calculate based on game complexity
5. **Games make great evals** - Clear objectives, natural tool use

### For Researchers

1. **Intuition ≠ Objective Measurement** - Always validate assumptions
2. **Solver strategy matters** - Different objectives yield different results
3. **Multiple metrics are valuable** - No single "correct" measure
4. **Document the journey** - Failed experiments are interesting
5. **Make it reproducible** - Include all analysis code

### For Everyone

1. **Started as a demo** → became research
2. **"Apple" is hard** (depending on your solver!)
3. **All code included** - try it yourself
4. **It's on PyPI** - `pip install hangman-bench`
5. **Questions welcome** - this is a conversation

---

## Anticipated Questions & Answers

### Technical Questions

**Q: Why not use a simpler state management approach?**
A: Store pattern scales to complex multi-turn interactions. Clean separation between game logic and Inspect integration. Type safety catches bugs early.

**Q: What if models guess the same letter twice?**
A: GameState checks for duplicates and returns early. Tool feedback shows already-guessed letters. We accept some inefficiency.

**Q: How do you handle models that don't use tools?**
A: `on_continue` provides gentle nudges. Message limit eventually stops them. We accept some failures - it's part of what we're measuring.

**Q: Why `parallel=False` on the tool?**
A: Guessing multiple letters simultaneously doesn't make sense for Hangman. Sequential guessing only.

### Research Questions

**Q: Why not use the reclassified difficulties?**
A: Results vary greatly by solver strategy. No single "correct" ranking. We kept intuitive labels for stability and because the variance itself is interesting data.

**Q: Which solver is "correct"?**
A: None! They optimize different objectives. Frequency maximizes raw counts, coverage maximizes hit probability, info gain minimizes remaining candidates. All valid strategies.

**Q: What's the hardest word?**
A: Depends on solver! "pyjamas" (19 wrong) for coverage, "pizza" (22 wrong) for info gain, "puppy" (9 wrong) for frequency. Repeated letters are generally harder.

**Q: Why is "rhythm" not that hard?**
A: No vowels looks scary, but 'r', 't', 'h' are common consonants. Smart solvers find them quickly. It's medium difficulty for deterministic strategies.

### Practical Questions

**Q: How long does a full eval take?**
A: ~100 samples, depends on model. GPT-4o-mini: ~10 min. GPT-4o: ~20 min. Use cost estimation on small runs to project.

**Q: What models work well?**
A: GPT-4o, Claude 3.5 Sonnet work great. GPT-4o-mini is decent. Smaller models struggle with tool use and strategy.

**Q: Can you add more languages?**
A: Yes! Just extend `datasets.py` with new word lists. Need difficulty ratings for each language.

**Q: Can you add more games?**
A: Absolutely! This pattern works for any turn-based game. Just change GameState and tool logic. Tic-tac-toe, Wordle, 20 Questions all possible.
I'm thinking about Minesweeper and Chess (using Stockfish in a docker container), and maybe Blackjack.

---

## Success Metrics

### You'll Know It Went Well If

- [ ] Audience asks about the apple problem
- [ ] Someone wants to try building their own game eval
- [ ] Questions about extending to other languages/games
- [ ] Discussion about difficulty metrics in their own work
- [ ] Requests for the repo link
- [ ] Comments about the store pattern being useful

### You'll Know It Went Poorly If

- [ ] Silence during Q&A
- [ ] Confused looks during code walkthrough
- [ ] No questions about the research
- [ ] People checking phones/laptops
- [ ] Early departures

---

## Post-Presentation

### Follow-Up Materials

Share these files:

- `SHOW_AND_TELL.md` - Full narrative
- `CODE_WALKTHROUGH.md` - Code guide
- Link to repo
- Link to PyPI package

### Potential Next Steps

- Blog post about the apple problem
- Tutorial on building game evals
- Extend to other languages
- Add more games using same patterns
- Paper about difficulty metrics in games

---

## Final Tips

### Do

- ✅ Start with the apple story - it's your hook
- ✅ Show live demos - more engaging than slides
- ✅ Emphasize patterns over implementation details
- ✅ Connect to audience's work (building evals)
- ✅ Leave time for questions and discussion
- ✅ Have fun with it - this is interesting research!

### Don't

- ❌ Get bogged down in code details
- ❌ Apologize for "failed" experiments (they're insights!)
- ❌ Rush through the apple problem (it's the best part)
- ❌ Assume everyone knows Inspect deeply
- ❌ Skip the demos (they make it real)
- ❌ Forget to mention it's on PyPI

---

## One-Sentence Summary

**"We built Hangman Bench to demonstrate Inspect's tool features, then discovered that 'apple' is hard (depending on your solver), and ended up with an interesting eval and a research project about the nature of difficulty in games."**

---

## Good Luck! 🍎

You have a great story to tell. The apple problem is genuinely interesting, the code demonstrates useful patterns, and the eval is actually useful. Your audience will appreciate both the technical depth and the research journey.

Remember: This started as a demonstration and became a discovery. That's the best kind of project.
