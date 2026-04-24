"""Game-state container and SolverState that binds evolving mask + history."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from wordle.lists import (
    load_broad_table,
    load_lists,
    load_pattern_table,
    load_priors,
    word_list_fingerprint,
    word_index,
)
from wordle.patterns import ALL_GREEN
from wordle.scoring import (
    apply_feedback,
    hard_mode_guess_mask,
    rank_guesses,
)


def _game_fingerprint(
    guesses: list[str],
    answers: list[str],
    priors: np.ndarray,
) -> str:
    h = hashlib.sha256()
    h.update(word_list_fingerprint(guesses).encode("ascii"))
    h.update(word_list_fingerprint(answers).encode("ascii"))
    h.update(priors.dtype.str.encode("ascii"))
    h.update(priors.shape[0].to_bytes(8, "little"))
    h.update(np.ascontiguousarray(priors).tobytes())
    return h.hexdigest()


@dataclass
class GameData:
    """Immutable per-session data: word lists, pattern table, priors."""

    guesses: list[str]
    answers: list[str]
    table: np.ndarray
    priors: np.ndarray
    g_idx: dict[str, int]
    a_idx: dict[str, int]  # answer word -> index in answers
    ans_in_guess: np.ndarray  # shape (|A|,), index into guesses
    fingerprint: str

    @classmethod
    def load(cls, alpha: float = 1.0, verbose: bool = False) -> "GameData":
        guesses, answers = load_lists()
        table = load_pattern_table(guesses, answers, verbose=verbose)
        priors = load_priors(answers, alpha=alpha)
        g_idx = word_index(guesses)
        a_idx = word_index(answers)
        ans_in_guess = np.array([g_idx[a] for a in answers], dtype=np.int64)
        return cls(
            guesses,
            answers,
            table,
            priors,
            g_idx,
            a_idx,
            ans_in_guess,
            _game_fingerprint(guesses, answers, priors),
        )

    @classmethod
    def load_broad(cls, alpha: float = 1.0, verbose: bool = False) -> "GameData":
        """Broader answer pool: every valid guess is a candidate answer.

        Use when the real answer may lie outside the curated 2,310 list
        (e.g. a post-NYT-curation word). Slower but always correct."""
        guesses, _ = load_lists()
        table = load_broad_table(guesses, verbose=verbose)
        priors = load_priors(guesses, alpha=alpha)
        g_idx = word_index(guesses)
        a_idx = g_idx
        ans_in_guess = np.arange(len(guesses), dtype=np.int64)
        return cls(
            guesses,
            guesses,
            table,
            priors,
            g_idx,
            a_idx,
            ans_in_guess,
            _game_fingerprint(guesses, guesses, priors),
        )

    @property
    def is_broad(self) -> bool:
        return len(self.answers) == len(self.guesses)


@dataclass
class SolverState:
    """Evolving per-game state: the candidate mask + guess history."""

    game: GameData
    hard_mode: bool = False
    mask: np.ndarray = field(init=False)
    history: list[tuple[str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mask = np.ones(len(self.game.answers), dtype=bool)

    @property
    def candidates_count(self) -> int:
        return int(self.mask.sum())

    def candidates(self, limit: int | None = None) -> list[str]:
        """Remaining candidate answers, sorted by prior (descending)."""
        idxs = np.flatnonzero(self.mask)
        order = np.argsort(-self.game.priors[idxs], kind="stable")
        sorted_idxs = idxs[order]
        if limit is not None:
            sorted_idxs = sorted_idxs[:limit]
        return [self.game.answers[j] for j in sorted_idxs]

    def candidate_set(self) -> set[str]:
        return {self.game.answers[j] for j in np.flatnonzero(self.mask)}

    def guess_filter(self) -> np.ndarray | None:
        if self.hard_mode and self.history:
            return hard_mode_guess_mask(self.game.guesses, self.history)
        return None

    def rank(self, top_n: int = 5) -> tuple[list[tuple[int, float]], np.ndarray]:
        """Return (top-N (idx, score), full scores array)."""
        return rank_guesses(
            self.game.table,
            self.mask,
            self.game.priors,
            self.game.ans_in_guess,
            guess_filter=self.guess_filter(),
            top_n=top_n,
        )

    def apply(self, guess_word: str, pattern: int) -> None:
        gi = self.game.g_idx[guess_word]
        self.mask = apply_feedback(self.mask, self.game.table, gi, pattern)
        self.history.append((guess_word, pattern))

    def is_solved(self) -> bool:
        return bool(self.history) and self.history[-1][1] == ALL_GREEN

    def is_broken(self) -> bool:
        """Candidate set empty — user must have entered wrong feedback."""
        return self.candidates_count == 0

    def feedback_leaves_candidates(self, guess_word: str, pattern: int) -> bool:
        """Would applying this (guess, pattern) leave at least one candidate?"""
        if guess_word not in self.game.g_idx:
            return False
        gi = self.game.g_idx[guess_word]
        return bool((self.mask & (self.game.table[gi] == pattern)).any())

    def switch_game(self, new_game: GameData) -> None:
        """Replace game data (e.g. narrow → broad) and rebuild mask from history."""
        self.game = new_game
        self.mask = np.ones(len(new_game.answers), dtype=bool)
        for word, pattern in self.history:
            if word in new_game.g_idx:
                gi = new_game.g_idx[word]
                self.mask = self.mask & (new_game.table[gi] == pattern)
