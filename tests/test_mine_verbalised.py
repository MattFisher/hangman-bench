"""Tests for the verbalised-candidate miner's scoring core."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ANALYSIS = pathlib.Path(__file__).resolve().parents[1] / "analysis"


def _load():
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    spec = importlib.util.spec_from_file_location(
        "mine_verbalised", ANALYSIS / "mine_verbalised.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mine_verbalised"] = module
    spec.loader.exec_module(module)
    return module


def test_mine_text_keeps_dictionary_words_of_the_right_length_once():
    m = _load()
    dictionary = {"gawky", "happy", "sappy", "coffee"}
    text = "Possible words remain: gawky, happy, HAPPY, coffee, zzzzz. I'll guess p."
    assert m.mine_text(text, 5, dictionary) == ["gawky", "happy"]


def test_recall_only_when_the_true_set_is_small():
    m = _load()
    assert m.recall_of(["gawky", "happy"], ["gawky", "happy", "sappy"]) == 2 / 3
    assert (
        m.recall_of(["gawky"], ["w" + str(i) for i in range(m.RECALL_MAX_CANDIDATES + 1)]) is None
    )
    assert m.recall_of(["gawky"], []) is None


def test_summary_counts_naming_the_target_then_guessing_dead():
    m = _load()
    steps = [
        m.StepMentions(
            model="x",
            word="happy",
            step=3,
            board="ha..y",
            n_candidates=1,
            mentions=["happy", "hardy"],
            consistent=["happy"],
            target_mentioned=True,
            next_guess="r",
            next_dominated=True,
            recall=1.0,
        ),
        m.StepMentions(
            model="x",
            word="happy",
            step=1,
            board=".a...",
            n_candidates=40,
            mentions=["gawky"],
            consistent=["gawky"],
            target_mentioned=False,
            next_guess="p",
            next_dominated=False,
            recall=None,
        ),
    ]
    (row,) = m.summarise(steps)
    assert row["steps_with_mentions"] == 2
    assert row["pinned_steps_with_mentions"] == 1
    assert row["pinned_target_named"] == 1
    assert row["named_target_then_dead_guess"] == 1
    assert row["mean_precision"] == 0.75
