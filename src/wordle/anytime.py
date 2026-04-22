"""Background speculation during user latency.

After we suggest a guess, the human spends a few seconds typing the result.
During that window we compute the next-turn top-N for each feedback pattern
we might receive, keyed by the pattern int. When the real feedback arrives
we look up the pre-ranked list instead of running the scorer again.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from wordle.patterns import ALL_GREEN, N_PATTERNS
from wordle.scoring import hard_mode_guess_mask, rank_guesses
from wordle.solver import SolverState

CachedEntry = tuple[list[tuple[int, float]], np.ndarray]


class AnytimeWorker:
    """One worker per turn. Start after the user picks a guess; stop when
    feedback arrives. Thread-safe lookup via an internal lock."""

    def __init__(self, state: SolverState):
        self.state = state
        self._cache: dict[int, CachedEntry] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._guess_idx: Optional[int] = None

    def start(self, guess_idx: int) -> None:
        """Begin speculating next-turn rankings for every probable feedback.

        No-op when |S| <= 2 (endgame is already instant)."""
        self.stop()
        if int(self.state.mask.sum()) <= 2:
            return
        self._stop.clear()
        with self._lock:
            self._cache = {}
            self._guess_idx = guess_idx
        self._thread = threading.Thread(
            target=self._run, args=(guess_idx,), daemon=True
        )
        self._thread.start()

    def _run(self, guess_idx: int) -> None:
        game = self.state.game
        mask = self.state.mask
        row = game.table[guess_idx, mask]
        answer_idxs = np.flatnonzero(mask)
        priors_s = game.priors[mask]

        weights = np.bincount(row, weights=priors_s, minlength=N_PATTERNS)
        nonempty = np.flatnonzero(weights > 0)
        order = nonempty[np.argsort(-weights[nonempty])]

        guess_word = game.guesses[guess_idx]
        history_base = list(self.state.history)
        hard = self.state.hard_mode

        for pattern in order:
            if self._stop.is_set():
                return
            p_int = int(pattern)
            if p_int == ALL_GREEN:
                continue  # game ends on this pattern
            s_next = np.zeros_like(mask)
            s_next[answer_idxs[row == pattern]] = True
            if not s_next.any():
                continue

            guess_filter = None
            if hard:
                guess_filter = hard_mode_guess_mask(
                    game.guesses, history_base + [(guess_word, p_int)]
                )

            top, scores = rank_guesses(
                game.table,
                s_next,
                game.priors,
                game.ans_in_guess,
                guess_filter=guess_filter,
                top_n=5,
            )
            with self._lock:
                self._cache[p_int] = (top, scores)

    def lookup(self, pattern: int) -> Optional[CachedEntry]:
        with self._lock:
            return self._cache.get(int(pattern))

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def populated(self) -> int:
        with self._lock:
            return len(self._cache)
