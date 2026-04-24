from __future__ import annotations

import json

import numpy as np

from wordle.lists import word_index
from wordle.mcp_logic import (
    parse_history,
    serialize_history,
    wordle_compare_guess,
    wordle_suggest_next_guess,
)
from wordle.patterns import build_pattern_table, compute_pattern, decode_pattern
from wordle.solver import GameData


def _make_game(guesses: list[str], answers: list[str]) -> GameData:
    g_idx = word_index(guesses)
    a_idx = word_index(answers)
    return GameData(
        guesses=guesses,
        answers=answers,
        table=build_pattern_table(guesses, answers),
        priors=np.ones(len(answers), dtype=np.float64),
        g_idx=g_idx,
        a_idx=a_idx,
        ans_in_guess=np.array([g_idx[word] for word in answers], dtype=np.int64),
        fingerprint="test",
    )


def test_parse_history_accepts_compact_and_emoji_feedback():
    turns = parse_history("SLATE:⬛🟨⬛🟩⬛")

    assert serialize_history(turns) == "slate:bybgb"


def test_invalid_history_returns_actionable_error():
    response = wordle_suggest_next_guess("slate:bbg")

    assert response["status"] == "invalid_input"
    assert response["examples"]["compact"] == "slate:bbgyb,crown:bgbbb"


def test_suggest_opening_guess_shape():
    response = wordle_suggest_next_guess("", top_n=3)

    assert response["status"] == "ok"
    assert response["pool"] == "curated"
    assert response["candidates"]["count"] == 2310
    assert len(response["suggestions"]) == 3
    assert len(response["suggestions"][0]["word"]) == 5


def test_compare_guess_shape():
    response = wordle_compare_guess("", "slate", top_n=3)

    assert response["status"] == "ok"
    assert response["guess"]["word"] == "slate"
    assert response["guess"]["rank"] >= 1
    assert len(response["suggestions"]) == 3
    json.dumps(response)


def test_compare_flags_hard_mode_violation():
    response = wordle_compare_guess("crane:bbbbg", "cigar", hard_mode=True)

    assert response["status"] == "violates_hard_mode"
    assert response["violates_hard_mode"] is True
    assert response["guess"]["word"] == "cigar"


def test_broad_fallback_when_curated_pool_is_empty(monkeypatch):
    import wordle.mcp_logic as logic

    guesses = ["crane", "abbey", "cigar", "slate"]
    curated = _make_game(guesses, ["abbey"])
    broad = _make_game(guesses, ["abbey", "cigar"])
    pattern = compute_pattern("crane", "cigar")

    monkeypatch.setattr(logic, "get_curated_game", lambda: curated)
    monkeypatch.setattr(logic, "get_broad_game", lambda: broad)

    response = logic.wordle_suggest_next_guess(
        f"crane:{decode_pattern(pattern)}",
        include_all_candidates=True,
    )

    assert response["status"] == "ok"
    assert response["pool"] == "broad"
    assert response["used_broad_fallback"] is True
    assert response["candidates"]["words"] == ["cigar"]


def test_include_all_candidates_can_return_large_lists():
    response = wordle_suggest_next_guess("", include_all_candidates=True, top_n=1)

    assert response["status"] == "ok"
    assert response["candidates"]["count"] == 2310
    assert len(response["candidates"]["words"]) == 2310
    assert response["candidates"]["truncated"] is False
    json.dumps(response)
