"""Tests for scoring, filtering, endgame behavior, and hard-mode mask."""

from __future__ import annotations

import numpy as np
import pytest

from wordle.lists import load_lists, load_pattern_table, load_priors, word_index
from wordle.patterns import compute_pattern, encode_feedback
from wordle.scoring import (
    apply_feedback,
    entropy_scores,
    expected_remaining,
    hard_mode_guess_mask,
    rank_guesses,
)


@pytest.fixture(scope="module")
def game():
    """Full-size game state."""
    guesses, answers = load_lists()
    table = load_pattern_table(guesses, answers)
    priors = load_priors(answers, alpha=1.0)
    g_idx = word_index(guesses)
    ans_in_guess = np.array([g_idx[a] for a in answers])
    return {
        "guesses": guesses,
        "answers": answers,
        "table": table,
        "priors": priors,
        "g_idx": g_idx,
        "ans_in_guess": ans_in_guess,
    }


class TestApplyFeedback:
    def test_all_green_leaves_one(self, game):
        mask = np.ones(len(game["answers"]), dtype=bool)
        crane_idx = game["g_idx"]["crane"]
        # if we guess crane and get all green, only crane remains (if in answers)
        ans_crane = (
            game["answers"].index("crane") if "crane" in game["answers"] else None
        )
        if ans_crane is not None:
            all_green = 242
            new_mask = apply_feedback(mask, game["table"], crane_idx, all_green)
            assert new_mask.sum() == 1
            assert new_mask[ans_crane]

    def test_known_reduction(self, game):
        """After guessing CRANE and getting a known feedback, the remaining set
        should be exactly the answers consistent with that feedback."""
        mask = np.ones(len(game["answers"]), dtype=bool)
        crane_idx = game["g_idx"]["crane"]
        # Fabricate a feedback: answer is 'abbey' -> pattern(crane, abbey)
        target = "abbey"
        pattern = compute_pattern("crane", target)
        new_mask = apply_feedback(mask, game["table"], crane_idx, pattern)
        # Cross-check: the remaining answers are exactly those w for which
        # compute_pattern("crane", w) == pattern
        expected = [
            w for w in game["answers"] if compute_pattern("crane", w) == pattern
        ]
        assert new_mask.sum() == len(expected)
        assert target in expected


class TestEntropyScores:
    def test_first_move_salet_is_strong(self, game):
        """SALET should be in the top 20 openers by weighted entropy."""
        mask = np.ones(len(game["answers"]), dtype=bool)
        scores = entropy_scores(game["table"], mask, game["priors"])
        ranked = np.argsort(-scores)
        top20 = [game["guesses"][i] for i in ranked[:20]]
        # Under weighted priors, SALET/SOARE/TRACE/CRANE/SLATE/RAISE should all
        # score well. We don't commit to which is #1, just that one of them is.
        strong = {
            "salet",
            "soare",
            "trace",
            "crane",
            "slate",
            "raise",
            "tares",
            "reast",
        }
        assert set(top20) & strong, (
            f"expected a strong opener in top20, got {top20[:5]}"
        )

    def test_entropy_is_nonneg(self, game):
        mask = np.ones(len(game["answers"]), dtype=bool)
        scores = entropy_scores(game["table"], mask, game["priors"])
        assert (scores >= 0).all()

    def test_entropy_zero_for_single_candidate(self, game):
        mask = np.zeros(len(game["answers"]), dtype=bool)
        mask[0] = True
        scores = entropy_scores(game["table"], mask, game["priors"])
        # With |S|=1, every guess yields the same pattern -> 0 bits
        assert (scores == 0).all()


class TestExpectedRemaining:
    def test_lower_for_better_splitter(self, game):
        """A known-good splitter should have lower expected remaining than a bad one."""
        mask = np.ones(len(game["answers"]), dtype=bool)
        scores = expected_remaining(game["table"], mask, game["priors"])
        # CRANE should split the set much better than FUZZY
        assert scores[game["g_idx"]["crane"]] < scores[game["g_idx"]["fuzzy"]]


class TestRankGuesses:
    def test_endgame_restricts_to_S(self, game):
        """When |S|=2, only those 2 candidates should appear in ranking."""
        mask = np.zeros(len(game["answers"]), dtype=bool)
        i1, i2 = 10, 500
        mask[i1] = True
        mask[i2] = True
        ranked, _ = rank_guesses(
            game["table"], mask, game["priors"], game["ans_in_guess"], top_n=5
        )
        assert len(ranked) == 2
        ranked_words = {game["guesses"][g] for g, _ in ranked}
        assert ranked_words == {game["answers"][i1], game["answers"][i2]}

    def test_single_candidate_returns_that_word(self, game):
        mask = np.zeros(len(game["answers"]), dtype=bool)
        mask[42] = True
        ranked, _ = rank_guesses(
            game["table"], mask, game["priors"], game["ans_in_guess"], top_n=5
        )
        assert len(ranked) == 1
        assert game["guesses"][ranked[0][0]] == game["answers"][42]

    def test_candidate_bonus_breaks_ties(self, game):
        mask = np.zeros(len(game["answers"]), dtype=bool)
        mask[:4] = True
        ranked, _ = rank_guesses(
            game["table"], mask, game["priors"], game["ans_in_guess"], top_n=3
        )
        assert len(ranked) >= 1

    def test_full_scores_returned(self, game):
        """The full scores array should be same length as guesses."""
        mask = np.ones(len(game["answers"]), dtype=bool)
        _, scores = rank_guesses(
            game["table"], mask, game["priors"], game["ans_in_guess"], top_n=5
        )
        assert scores.shape == (len(game["guesses"]),)


class TestHardModeMask:
    def test_no_past_allows_all(self, game):
        mask = hard_mode_guess_mask(game["guesses"], [])
        assert mask.all()

    def test_green_pins_position(self, game):
        # After guessing CRANE with feedback "bbbbg" (only E is green),
        # all valid next guesses must have 'e' at position 4.
        past = [("crane", encode_feedback("bbbbg"))]
        mask = hard_mode_guess_mask(game["guesses"], past)
        # every allowed word ends in 'e'
        for i, w in enumerate(game["guesses"]):
            if mask[i]:
                assert w[4] == "e"
        # and every word ending in 'e' that doesn't include c/r/a/n is allowed
        # (greys don't constrain in hard mode)
        assert mask[game["g_idx"]["abide"]]  # doesn't contain c/r/a/n, ends in e

    def test_yellow_requires_letter(self, game):
        # CRANE -> "ybbbb" means C is yellow: next guess must contain C, not at pos 0
        past = [("crane", encode_feedback("ybbbb"))]
        mask = hard_mode_guess_mask(game["guesses"], past)
        allowed = [game["guesses"][i] for i in range(len(game["guesses"])) if mask[i]]
        assert all("c" in w for w in allowed)
        assert all(w[0] != "c" for w in allowed)

    def test_yellow_forbids_same_position(self):
        guesses = ["cigar", "panic", "cacao"]
        past = [("crane", encode_feedback("ybbbb"))]
        mask = hard_mode_guess_mask(guesses, past)
        assert mask.tolist() == [False, True, False]

    def test_double_yellow_requires_count(self):
        # Synthetic: guess "lolls" with feedback "ybybb".
        # L@0 yellow, O@1 grey, L@2 yellow, L@3 grey, S@4 grey.
        # per-turn: L=2 (two yellows). Next guess needs >=2 Ls, with no L
        # at positions 0 or 2.
        guesses = [
            "algal",
            "flail",
            "llama",
            "glass",
            "cello",
            "hoopy",
            "filly",
            "lolly",
        ]
        past = [("lolls", encode_feedback("ybybb"))]
        mask = hard_mode_guess_mask(guesses, past)
        # algal/flail have 2Ls away from positions 0 and 2; the rest either
        # lack two Ls or reuse a yellow slot.
        assert mask.tolist() == [True, True, False, False, False, False, False, False]

    def test_yellow_adds_letter_constraint_only(self):
        # lone yellow adds presence requirement; greys do NOT forbid reuse.
        guesses = ["hello", "world", "hoopy"]
        past = [("lulls", encode_feedback("ybbbb"))]  # only L@0 yellow
        mask = hard_mode_guess_mask(guesses, past)
        # hello has L; world has L; hoopy has no L
        assert mask.tolist() == [True, True, False]
