"""Integration tests: solver actually solves games end-to-end."""

from __future__ import annotations

import numpy as np
import pytest

from wordle.explain import analyze_guess, one_liner, detailed_explanation
from wordle.patterns import compute_pattern, encode_feedback
from wordle.play import selfplay
from wordle.solver import GameData, SolverState


@pytest.fixture(scope="module")
def game():
    return GameData.load(alpha=1.0)


class TestSelfplay:
    def test_solves_known_words(self, game):
        # Three ordinary answers; solver should reach them within 6 turns
        for secret in ["abbey", "crane", "house", "zebra"]:
            result = selfplay(game, secret)
            assert result.solved, f"failed on {secret!r} after {result.words}"
            assert result.words[-1] == secret
            assert 1 <= result.n_guesses <= 6

    def test_forced_opener(self, game):
        result = selfplay(game, "abbey", opener="salet")
        assert result.words[0] == "salet"
        assert result.solved

    def test_cache_gives_same_results(self, game):
        cache: dict = {}
        a = selfplay(game, "abbey", cache=cache)
        b = selfplay(game, "abbey", cache=cache)
        assert a.words == b.words
        assert len(cache) > 0

    def test_hard_mode_solves(self, game):
        for secret in ["crane", "abbey"]:
            result = selfplay(game, secret, hard_mode=True)
            assert result.solved, f"hard mode failed on {secret!r}"


class TestExplainStats:
    def test_analyze_guess_returns_sane_stats(self, game):
        state = SolverState(game)
        ranked, _ = state.rank(top_n=1)
        gi, score = ranked[0]
        stats = analyze_guess(state, gi, score)
        assert stats.word == game.guesses[gi]
        assert stats.bits > 0
        assert stats.n_candidates == len(game.answers)
        assert stats.n_buckets > 0
        assert stats.worst_next > 0
        assert 0 < stats.expected_next <= stats.n_candidates
        # The one-liner renders without error
        assert str(one_liner(stats))
        # detailed renders without error
        _ = detailed_explanation(state, stats)

class TestFeedbackValidation:
    def test_detects_inconsistent_feedback(self, game):
        """Reproduces the user's BLANK-vs-BADAM confusion: feedback bggby for
        BADAM is impossible after TIARE bbgbb + CHALS bbgyb (pins A at pos 2,
        which contradicts A at pos 1 from bggby)."""
        state = SolverState(game)
        state.apply("tiare", encode_feedback("bbgbb"))
        state.apply("chals", encode_feedback("bbgyb"))
        # At this point 9 candidates remain. bggby for BADAM is impossible.
        assert not state.feedback_leaves_candidates("badam", encode_feedback("bggby"))
        # But bybbb (correct for FLAKY) does leave candidates.
        assert state.feedback_leaves_candidates("badam", encode_feedback("bybbb"))
        # And the user's ACTUAL guess (BLANK) with bggby is consistent.
        assert state.feedback_leaves_candidates("blank", encode_feedback("bggby"))


class TestBroadGame:
    def test_broad_loads_and_has_guesses_as_answers(self, game):
        broad = GameData.load_broad(alpha=1.0)
        assert len(broad.answers) == len(broad.guesses)
        assert broad.is_broad
        assert broad.table.shape == (len(broad.guesses), len(broad.answers))

    def test_switch_game_reapplies_history(self, game):
        """After switching to broad, the mask should be consistent with all
        prior turns."""
        state = SolverState(game)
        state.apply("tiare", encode_feedback("bbgbb"))
        state.apply("chals", encode_feedback("bbgyb"))
        assert not state.game.is_broad
        narrow_count = state.candidates_count

        broad = GameData.load_broad(alpha=1.0)
        state.switch_game(broad)
        assert state.game.is_broad
        # Broad pool should be a superset of narrow at this point
        assert state.candidates_count >= narrow_count
        # All remaining words must be consistent with history
        for j in np.flatnonzero(state.mask):
            w = broad.answers[j]
            assert compute_pattern("tiare", w) == encode_feedback("bbgbb")
            assert compute_pattern("chals", w) == encode_feedback("bbgyb")


class TestExplainAfterNarrowing:
    def test_in_s_stats_after_narrowing(self, game):
        state = SolverState(game)
        # Apply CRANE -> pattern for "abbey"
        from wordle.patterns import compute_pattern
        crane_idx = game.g_idx["crane"]
        p = compute_pattern("crane", "abbey")
        state.apply("crane", p)
        ranked, _ = state.rank(top_n=3)
        for gi, score in ranked:
            stats = analyze_guess(state, gi, score)
            if stats.is_in_S:
                assert stats.win_prob > 0
            else:
                assert stats.win_prob == 0
