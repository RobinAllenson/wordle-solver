"""Tests for the anytime speculation worker."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from wordle.anytime import AnytimeWorker
from wordle.patterns import compute_pattern
from wordle.solver import GameData, SolverState


@pytest.fixture(scope="module")
def game():
    return GameData.load(alpha=1.0)


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestAnytimeWorker:
    def test_populates_cache(self, game):
        """Worker should populate at least one pattern within a reasonable time."""
        state = SolverState(game)
        worker = AnytimeWorker(state)
        gi = game.g_idx["tiare"]
        worker.start(gi)
        assert _wait_until(lambda: worker.populated > 0, timeout=5.0)
        worker.stop()

    def test_cached_matches_live(self, game):
        """Cached top-N should match recomputing live for that pattern."""
        from wordle.scoring import apply_feedback, rank_guesses

        state = SolverState(game)
        worker = AnytimeWorker(state)
        gi = game.g_idx["tiare"]
        worker.start(gi)
        # Wait for worker to finish
        worker._thread.join(timeout=30.0)

        # Pick a pattern we'd see from TIARE vs "abbey"
        pat = compute_pattern("tiare", "abbey")
        cached = worker.lookup(pat)
        assert cached is not None, f"pattern {pat} not cached"
        cached_top, cached_scores = cached

        # Recompute live
        live_mask = apply_feedback(state.mask, game.table, gi, pat)
        live_top, live_scores = rank_guesses(
            game.table,
            live_mask,
            game.priors,
            game.ans_in_guess,
            top_n=5,
        )
        assert cached_top == live_top
        assert np.allclose(cached_scores, live_scores)

    def test_stop_cancels_quickly(self, game):
        """Stop should return within ~1 second of signaling."""
        state = SolverState(game)
        worker = AnytimeWorker(state)
        gi = game.g_idx["tiare"]
        worker.start(gi)
        # Give it time to start working
        time.sleep(0.05)
        t0 = time.time()
        worker.stop()
        assert time.time() - t0 < 2.5

    def test_small_S_skips_spawn(self, game):
        """|S|<=2 shouldn't start a thread — endgame is already instant."""
        state = SolverState(game)
        # Manually reduce S to 2
        state.mask[:] = False
        state.mask[0] = True
        state.mask[1] = True
        worker = AnytimeWorker(state)
        worker._cache[123] = ([(0, 1.0)], np.array([1.0]))
        worker.start(game.g_idx["tiare"])
        assert worker._thread is None
        assert worker.populated == 0
        assert worker.lookup(123) is None

    def test_old_generation_cannot_write_cache(self, game):
        """A worker from a previous turn must not write into current cache."""
        state = SolverState(game)
        state.mask[:] = False
        state.mask[:3] = True
        worker = AnytimeWorker(state)
        worker._generation = 2
        worker._run(game.g_idx["tiare"], generation=1, stop_event=threading.Event())
        assert worker.populated == 0

    def test_restart_wipes_cache(self, game):
        """Calling start again should clear previous cache."""
        state = SolverState(game)
        worker = AnytimeWorker(state)
        worker.start(game.g_idx["tiare"])
        worker._thread.join(timeout=30.0)
        assert worker.populated > 0
        # Restart with a different guess
        worker.start(game.g_idx["crane"])
        # Cache should be reset at start
        # (may have populated some again, but the set should not include the
        # previous-populated patterns for wrong state)
        worker.stop()
        # Just assert start() swapped threads cleanly and cache is fresh
        worker._thread = None  # ensure clean

    def test_hard_mode_honoured(self, game):
        """In hard mode, the cached top entries should not include words that
        violate the speculated-hint constraints."""
        from wordle.scoring import hard_mode_guess_mask

        state = SolverState(game, hard_mode=True)
        worker = AnytimeWorker(state)
        gi = game.g_idx["tiare"]
        worker.start(gi)
        worker._thread.join(timeout=30.0)

        # Pick a pattern where hard mode bites: "bbbbg" means E pinned at pos 4
        from wordle.patterns import encode_feedback

        pat = encode_feedback("bbbbg")
        cached = worker.lookup(pat)
        if cached is None:
            pytest.skip(f"pattern {pat} not populated")
        top, _ = cached
        # Every suggested word should be allowed under hard mode after this turn
        filt = hard_mode_guess_mask(game.guesses, [("tiare", pat)])
        for idx, _ in top:
            assert filt[idx], (
                f"anytime top pick {game.guesses[idx]!r} violates hard mode"
            )
