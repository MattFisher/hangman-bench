"""Tests for the static pinned-board probe."""

import pathlib

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from hangman_bench.oracle import DEFAULT_WORDLIST, consistent_candidates, load_dictionary_index
from hangman_bench.probe import (
    PinnedBoard,
    build_prompt,
    is_consistent,
    parse_answer,
    pinned_board_for,
    pinned_probe,
    read_boards,
    sample_pinned_boards,
    write_boards,
)

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def index() -> dict[int, list[str]]:
    return load_dictionary_index(str(DEFAULT_WORDLIST))


class TestGeneration:
    def test_board_is_pinned_and_reproduces_the_pilot_state(
        self, index: dict[int, list[str]]
    ) -> None:
        board = pinned_board_for("coffee", index[6])
        assert board is not None
        # The state gpt-4o faced in the pilot before six consecutive dead guesses.
        assert board.board == ".o..ee"
        assert consistent_candidates(board.board, board.guessed, index[6]) == ["coffee"]

    def test_excluded_letters_are_the_wrong_guesses(self, index: dict[int, list[str]]) -> None:
        board = pinned_board_for("gazebo", index[6])
        assert board is not None
        assert set(board.excluded).isdisjoint("gazebo")
        assert board.excluded == "".join(sorted(board.excluded))

    def test_extra_wrong_letters_keep_the_board_pinned(self, index: dict[int, list[str]]) -> None:
        plain = pinned_board_for("gazebo", index[6])
        padded = pinned_board_for("gazebo", index[6], extra_wrong=4)
        assert plain is not None
        assert padded is not None
        assert padded.board == plain.board
        assert len(padded.excluded) == len(plain.excluded) + 4
        assert set(padded.excluded).isdisjoint("gazebo")
        assert consistent_candidates(padded.board, padded.guessed, index[6]) == ["gazebo"]

    def test_unknown_word_yields_nothing(self, index: dict[int, list[str]]) -> None:
        assert pinned_board_for("zzzzzz", index[6]) is None

    def test_sample_is_seeded_stratified_and_pinned(self, index: dict[int, list[str]]) -> None:
        boards = sample_pinned_boards(index, n=24, seed=0, lengths=(5, 6, 7))
        assert len(boards) == 24
        assert {len(b.word) for b in boards} == {5, 6, 7}
        for b in boards:
            assert consistent_candidates(b.board, b.guessed, index[len(b.word)]) == [b.word]
            assert "." in b.board
        assert boards == sample_pinned_boards(index, n=24, seed=0, lengths=(5, 6, 7))
        assert boards != sample_pinned_boards(index, n=24, seed=1, lengths=(5, 6, 7))

    def test_sample_does_not_depend_on_dictionary_order(self, index: dict[int, list[str]]) -> None:
        shuffled = {k: list(reversed(v)) for k, v in index.items()}
        a = sample_pinned_boards(index, n=12, seed=3, lengths=(5, 6))
        b = sample_pinned_boards(shuffled, n=12, seed=3, lengths=(5, 6))
        assert a == b

    def test_band_stratification(self, index: dict[int, list[str]]) -> None:
        def band(word: str) -> str:
            return "x" if word[0] < "n" else "y"

        boards = sample_pinned_boards(
            index, n=20, seed=0, lengths=(6, 7), band_of=band, bands=("x", "y")
        )
        counts = {(len(b.word), b.zipf_band) for b in boards}
        assert counts == {(6, "x"), (6, "y"), (7, "x"), (7, "y")}


class TestRoundTrip:
    def test_tsv_round_trip(self, tmp_path: pathlib.Path, index: dict[int, list[str]]) -> None:
        boards = sample_pinned_boards(index, n=6, seed=0, lengths=(5, 6))
        path = tmp_path / "boards.tsv"
        write_boards(boards, path)
        assert read_boards(path) == boards

    def test_committed_board_set_is_pinned(self, index: dict[int, list[str]]) -> None:
        boards = read_boards(REPO / "analysis" / "pinned_boards.tsv")
        sources = {b.source for b in boards}
        assert sources == {"pilot", "dictionary"}
        for b in boards:
            assert consistent_candidates(b.board, b.guessed, index[len(b.word)]) == [b.word], b


class TestScoringHelpers:
    def test_parse_answer_prefers_last_token_of_target_length(self) -> None:
        assert parse_answer("The word is probably Coffee.", 6, ALPHABET) == "coffee"
        assert parse_answer("toffee, no wait: coffee", 6, ALPHABET) == "coffee"
        assert parse_answer("banana", 6, ALPHABET) == "banana"
        assert parse_answer("I think it's cat", 6, ALPHABET) == "cat"
        assert parse_answer("", 6, ALPHABET) == ""

    def test_consistency_is_positional_and_respects_exclusions(self) -> None:
        board = PinnedBoard(word="coffee", board=".o..ee", excluded="at")
        assert is_consistent("coffee", board)
        assert is_consistent("bouffee", board) is False  # wrong length
        assert is_consistent("toffee", board) is False  # uses an excluded letter
        assert is_consistent("cosine", board) is False  # e must be exactly at 4,5
        assert is_consistent("coxxee", board)  # consistent but not a word

    def test_prompt_mentions_board_exclusions_and_length(self) -> None:
        board = PinnedBoard(word="coffee", board=".o..ee", excluded="at")
        prompt = build_prompt(board)
        assert "_ o _ _ e e" in prompt
        assert "a, t" in prompt
        assert "6 letters" in prompt
        assert "Exactly one" not in prompt
        assert "Exactly one" in build_prompt(board, told_unique=True)
        assert "_o__ee" in build_prompt(board, style="contiguous")


class TestTask:
    def test_task_from_tsv_filters_by_source(self) -> None:
        task = pinned_probe(
            boards=str(REPO / "analysis" / "pinned_boards.tsv"), source="dictionary"
        )
        assert all((s.metadata or {})["source"] == "dictionary" for s in task.dataset)
        assert len(task.dataset) == 300

    def test_task_generates_when_no_tsv(self) -> None:
        task = pinned_probe(n=6, seed=1)
        assert len(task.dataset) == 6
        assert all((s.metadata or {})["seed"] == 1 for s in task.dataset)

    def test_fresh_seed_is_recorded(self) -> None:
        task = pinned_probe(n=2, seed=None)
        seeds = {(s.metadata or {})["seed"] for s in task.dataset}
        assert len(seeds) == 1
        assert None not in seeds

    def test_bad_style_rejected(self) -> None:
        with pytest.raises(ValueError, match="board_style"):
            pinned_probe(n=1, board_style="vertical")

    def test_end_to_end_scoring(self, tmp_path: pathlib.Path) -> None:
        boards = [
            PinnedBoard(word="coffee", board=".o..ee", excluded="at"),
            PinnedBoard(word="gazebo", board=".a.eb.", excluded="nsy"),
            PinnedBoard(word="igloo", board="...oo", excluded="aes"),
        ]
        path = tmp_path / "boards.tsv"
        write_boards(boards, path)
        outputs = [
            ModelOutput.from_content("mockllm/model", "The word is coffee."),  # exact
            ModelOutput.from_content("mockllm/model", "gazebq"),  # consistent, not a word
            ModelOutput.from_content("mockllm/model", "tattoo"),  # a word, inconsistent
        ]
        log = eval(
            tasks=pinned_probe(boards=str(path)),
            model=get_model("mockllm/model", custom_outputs=outputs),
        )[0]
        assert log.status == "success"
        assert log.results is not None
        metrics = {
            f"{score.name}.{name}": metric.value
            for score in log.results.scores
            for name, metric in score.metrics.items()
        }
        assert metrics["exact.mean"] == pytest.approx(1 / 3)
        assert metrics["consistent.mean"] == pytest.approx(2 / 3)
        assert metrics["in_dictionary.mean"] == pytest.approx(2 / 3)
