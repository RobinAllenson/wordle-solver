"""Self-play driver: solver tries to guess a given secret word."""

from __future__ import annotations

from dataclasses import dataclass

from wordle.patterns import ALL_GREEN
from wordle.solver import GameData, SolverState


@dataclass
class SelfplayResult:
    secret: str
    words: list[str]
    patterns: list[int]
    solved: bool

    @property
    def n_guesses(self) -> int:
        return len(self.words)


HistoryKey = tuple[str, tuple[str, ...], tuple[int, ...], bool]


def selfplay(
    game: GameData,
    secret: str,
    hard_mode: bool = False,
    max_turns: int = 6,
    opener: str | None = None,
    cache: dict[HistoryKey, str] | None = None,
) -> SelfplayResult:
    """Solver attempts to guess `secret`. Uses O(1) pattern lookup + optional
    memoization of top-pick by history (big speedup for bench)."""
    state = SolverState(game, hard_mode=hard_mode)
    secret_idx = game.a_idx[secret]
    words: list[str] = []
    patterns: list[int] = []

    for turn in range(max_turns):
        if turn == 0 and opener is not None:
            word = opener
        else:
            key: HistoryKey | None = None
            if cache is not None:
                key = (game.fingerprint, tuple(words), tuple(patterns), hard_mode)
                if key in cache:
                    word = cache[key]
                else:
                    word = None  # type: ignore[assignment]
            else:
                word = None  # type: ignore[assignment]
            if word is None:
                ranked, _ = state.rank(top_n=1)
                if not ranked:
                    return SelfplayResult(secret, words, patterns, solved=False)
                word = game.guesses[ranked[0][0]]
                if cache is not None and key is not None:
                    cache[key] = word

        pattern = int(game.table[game.g_idx[word], secret_idx])
        words.append(word)
        patterns.append(pattern)
        state.apply(word, pattern)

        if pattern == ALL_GREEN:
            return SelfplayResult(secret, words, patterns, solved=True)

    return SelfplayResult(secret, words, patterns, solved=False)
