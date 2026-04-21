"""Integration tests: solver actually solves games end-to-end."""

from __future__ import annotations

import pytest

from wordle.explain import analyze_guess, one_liner, detailed_explanation
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


class TestExplain:
    def test_analyze_guess_returns_sane_stats(self, game):
        state = SolverState(game)
        ranked = state.rank(top_n=1)
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

    def test_in_s_stats_after_narrowing(self, game):
        state = SolverState(game)
        # Apply CRANE -> pattern for "abbey"
        from wordle.patterns import compute_pattern
        crane_idx = game.g_idx["crane"]
        p = compute_pattern("crane", "abbey")
        state.apply("crane", p)
        ranked = state.rank(top_n=3)
        for gi, score in ranked:
            stats = analyze_guess(state, gi, score)
            if stats.is_in_S:
                assert stats.win_prob > 0
            else:
                assert stats.win_prob == 0
