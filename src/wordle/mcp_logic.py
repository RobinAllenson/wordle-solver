"""Stateless Wordle helper logic for MCP tools.

The MCP layer should stay thin: parse a compact game history, build a
SolverState, then return JSON-serializable dicts that agents can use directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from wordle.explain import GuessStats, analyze_guess, analyze_word
from wordle.patterns import ALL_GREEN, decode_pattern, encode_feedback
from wordle.solver import GameData, SolverState

MAX_TURNS = 6
MAX_HISTORY_CHARS = 4096
MAX_TOP_N = 25
DEFAULT_CANDIDATE_LIMIT = 25
AUTO_ALL_CANDIDATE_THRESHOLD = 50

EXAMPLE_HISTORY = "slate:bbgyb,crown:bgbbb"

_GREEN_SQUARE = "\U0001f7e9"
_YELLOW_SQUARE = "\U0001f7e8"
_BLACK_SQUARE = "\u2b1b"
_WHITE_SQUARE = "\u2b1c"
_VARIATION_SELECTOR = "\ufe0f"

_FEEDBACK_CHAR_MAP = {
    "g": "g",
    _GREEN_SQUARE: "g",
    "y": "y",
    _YELLOW_SQUARE: "y",
    "b": "b",
    ".": "b",
    "-": "b",
    "x": "b",
    _BLACK_SQUARE: "b",
    _WHITE_SQUARE: "b",
    "\u25a0": "b",  # black square
    "\u25a1": "b",  # white square
    "\u25fb": "b",  # white medium square
    "\u25fc": "b",  # black medium square
    "\u2b1a": "b",  # black small square
}


class WordleInputError(ValueError):
    """Input problem that should be returned as invalid_input, not raised to MCP."""


def _invalid_response(message: str) -> dict[str, Any]:
    return {
        "status": "invalid_input",
        "summary": message,
        "message": message,
        "examples": {
            "compact": EXAMPLE_HISTORY,
            "emoji": f"slate:{_BLACK_SQUARE}{_YELLOW_SQUARE}{_BLACK_SQUARE}{_GREEN_SQUARE}{_BLACK_SQUARE}",
            "legend": "b/black/gray = absent, y/yellow = present elsewhere, g/green = correct position",
        },
    }


def _normalize_feedback(raw: str) -> str:
    text = (raw or "").strip().lower().replace(_VARIATION_SELECTOR, "")
    text = "".join(ch for ch in text if not ch.isspace())
    out: list[str] = []
    for i, ch in enumerate(text):
        mapped = _FEEDBACK_CHAR_MAP.get(ch)
        if mapped is None:
            raise WordleInputError(
                f"invalid feedback character {ch!r} at position {i + 1}; "
                "use five characters from b/y/g, '.', '-', 'x', or Wordle square emoji"
            )
        out.append(mapped)
    if len(out) != 5:
        raise WordleInputError(
            f"feedback must describe exactly 5 tiles; got {len(out)} from {raw!r}"
        )
    return "".join(out)


def parse_history(history: str) -> list[tuple[str, int]]:
    """Parse compact history like 'slate:bbgyb,crown:bgbbb'."""
    history = (history or "").strip()
    if not history:
        return []
    if len(history) > MAX_HISTORY_CHARS:
        raise WordleInputError(
            f"history is too long; maximum is {MAX_HISTORY_CHARS} characters"
        )

    chunks = [
        chunk.strip()
        for chunk in history.replace("\n", ",").replace(";", ",").split(",")
        if chunk.strip()
    ]
    if len(chunks) > MAX_TURNS:
        raise WordleInputError(f"Wordle has at most {MAX_TURNS} turns")

    parsed: list[tuple[str, int]] = []
    for turn_number, chunk in enumerate(chunks, start=1):
        if ":" in chunk:
            word, feedback = chunk.split(":", 1)
        else:
            parts = chunk.split()
            if len(parts) != 2:
                raise WordleInputError(
                    f"turn {turn_number} must be 'guess:feedback', for example {EXAMPLE_HISTORY!r}"
                )
            word, feedback = parts

        word = word.strip().lower()
        if len(word) != 5 or not word.isascii() or not word.isalpha():
            raise WordleInputError(
                f"turn {turn_number} guess {word!r} must be exactly 5 ASCII letters"
            )
        try:
            pattern = encode_feedback(_normalize_feedback(feedback))
        except ValueError as e:
            raise WordleInputError(str(e)) from e

        if parsed and parsed[-1][1] == ALL_GREEN:
            raise WordleInputError("all-green feedback must be the final turn")
        parsed.append((word, pattern))

    return parsed


def serialize_history(turns: list[tuple[str, int]]) -> str:
    return ",".join(f"{word}:{decode_pattern(pattern)}" for word, pattern in turns)


@lru_cache(maxsize=1)
def get_curated_game() -> GameData:
    return GameData.load(alpha=1.0)


@lru_cache(maxsize=1)
def get_broad_game() -> GameData:
    return GameData.load_broad(alpha=1.0)


def warm_caches(include_broad: bool = False) -> dict[str, Any]:
    """Load pattern tables so deploys can pay the cold-start cost upfront."""
    curated = get_curated_game()
    result: dict[str, Any] = {
        "curated_answers": len(curated.answers),
        "valid_guesses": len(curated.guesses),
        "curated_fingerprint": curated.fingerprint,
    }
    if include_broad:
        broad = get_broad_game()
        result.update(
            {
                "broad_answers": len(broad.answers),
                "broad_fingerprint": broad.fingerprint,
            }
        )
    return result


def _normalize_top_n(top_n: int) -> tuple[int, bool]:
    try:
        n = int(top_n)
    except (TypeError, ValueError) as e:
        raise WordleInputError("top_n must be an integer from 1 to 25") from e
    if n < 1:
        raise WordleInputError("top_n must be at least 1")
    capped = n > MAX_TOP_N
    return min(n, MAX_TOP_N), capped


def _build_state(game: GameData, turns: list[tuple[str, int]], hard_mode: bool) -> SolverState:
    state = SolverState(game, hard_mode=hard_mode)
    for turn_number, (word, pattern) in enumerate(turns, start=1):
        if word not in game.g_idx:
            raise WordleInputError(
                f"turn {turn_number} guess {word.upper()} is not in the Wordle guess list"
            )
        state.apply(word, pattern)
    return state


def _resolve_state(
    turns: list[tuple[str, int]], hard_mode: bool
) -> tuple[SolverState, str, bool]:
    """Build state against curated answers, falling back to broad answers if needed."""
    curated = _build_state(get_curated_game(), turns, hard_mode)
    if not turns or turns[-1][1] == ALL_GREEN or curated.candidates_count > 0:
        return curated, "curated", False

    broad = _build_state(get_broad_game(), turns, hard_mode)
    return broad, "broad", broad.candidates_count > 0


def _candidate_payload(
    state: SolverState,
    include_all_candidates: bool,
) -> dict[str, Any]:
    count = state.candidates_count
    if include_all_candidates or count <= AUTO_ALL_CANDIDATE_THRESHOLD:
        limit = None
    else:
        limit = DEFAULT_CANDIDATE_LIMIT

    words = state.candidates(limit=limit)
    return {
        "count": count,
        "words": words,
        "truncated": limit is not None and count > len(words),
        "limit": limit,
        "include_all_requested": include_all_candidates,
    }


def _stats_payload(stats: GuessStats, rank: int) -> dict[str, Any]:
    return {
        "word": stats.word,
        "rank": rank,
        "bits": round(float(stats.bits), 6),
        "candidate_count": int(stats.n_candidates),
        "expected_remaining": round(float(stats.expected_next), 6),
        "worst_case_remaining": int(stats.worst_next),
        "feedback_buckets": int(stats.n_buckets),
        "is_candidate": bool(stats.is_in_S),
        "win_probability": round(float(stats.win_prob), 6),
        "letter_coverage": [
            {"letter": letter, "coverage": round(float(frac), 6)}
            for letter, frac in stats.letter_coverage
        ],
        "likely_feedback": [
            {
                "pattern": decode_pattern(pattern),
                "candidate_count": int(count),
                "probability": round(float(prob), 6),
            }
            for pattern, count, prob in stats.top_patterns
        ],
    }


def _suggestions_payload(state: SolverState, top_n: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    ranked, all_scores = state.rank(top_n=top_n)
    suggestions = [
        _stats_payload(analyze_guess(state, guess_idx, score), rank)
        for rank, (guess_idx, score) in enumerate(ranked, start=1)
    ]
    return suggestions, all_scores


def _base_response(
    *,
    status: str,
    summary: str,
    turns: list[tuple[str, int]],
    state: SolverState,
    pool: str,
    used_broad_fallback: bool,
    top_n_capped: bool,
    include_all_candidates: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "history": serialize_history(turns),
        "turns_used": len(turns),
        "hard_mode": state.hard_mode,
        "pool": pool,
        "used_broad_fallback": used_broad_fallback,
        "top_n_capped": top_n_capped,
        "candidates": _candidate_payload(state, include_all_candidates),
    }


def wordle_suggest_next_guess(
    history: str,
    hard_mode: bool = False,
    top_n: int = 5,
    include_all_candidates: bool = False,
) -> dict[str, Any]:
    """Return ranked next guesses and current candidate information."""
    try:
        normalized_top_n, top_n_capped = _normalize_top_n(top_n)
        turns = parse_history(history)
        state, pool, used_broad_fallback = _resolve_state(turns, hard_mode)
    except WordleInputError as e:
        return _invalid_response(str(e))

    if turns and turns[-1][1] == ALL_GREEN:
        word = turns[-1][0]
        return {
            "status": "solved",
            "summary": f"{word.upper()} is marked all green; the puzzle is solved.",
            "history": serialize_history(turns),
            "turns_used": len(turns),
            "hard_mode": hard_mode,
            "pool": pool,
            "used_broad_fallback": used_broad_fallback,
            "solution": word,
            "suggestions": [],
            "candidates": _candidate_payload(state, include_all_candidates),
        }

    if state.candidates_count == 0:
        return {
            "status": "inconsistent_feedback",
            "summary": (
                "No valid Wordle answer matches this history, even after broad fallback. "
                "At least one guess or color is probably wrong."
            ),
            "history": serialize_history(turns),
            "turns_used": len(turns),
            "hard_mode": hard_mode,
            "pool": pool,
            "used_broad_fallback": used_broad_fallback,
            "suggestions": [],
            "candidates": _candidate_payload(state, include_all_candidates),
        }

    suggestions, _ = _suggestions_payload(state, normalized_top_n)
    top = suggestions[0] if suggestions else None
    if top is None:
        summary = "No legal next guesses are available."
    else:
        pool_note = " using the broad answer pool" if used_broad_fallback else ""
        summary = (
            f"Try {top['word'].upper()}{pool_note}. It scores {top['bits']:.2f} bits "
            f"and leaves about {top['expected_remaining']:.1f} candidates on average."
        )

    response = _base_response(
        status="ok",
        summary=summary,
        turns=turns,
        state=state,
        pool=pool,
        used_broad_fallback=used_broad_fallback,
        top_n_capped=top_n_capped,
        include_all_candidates=include_all_candidates,
    )
    response["suggestions"] = suggestions
    return response


def _state_without_hard_mode(state: SolverState) -> SolverState:
    relaxed = SolverState(state.game, hard_mode=False)
    for word, pattern in state.history:
        relaxed.apply(word, pattern)
    return relaxed


def _compare_verdict(rank: int, delta: float) -> str:
    if rank == 1:
        return "Your guess matches the top pick."
    if rank <= 5:
        return f"Your guess is rank #{rank} in the top 5."
    if delta <= 0.3:
        return "Within a hair of the top pick; it is a reasonable choice."
    if delta <= 1.0:
        return "Somewhat weaker than the top picks."
    return "Noticeably weaker; the listed suggestions gather more information."


def wordle_compare_guess(
    history: str,
    guess: str,
    hard_mode: bool = False,
    top_n: int = 5,
    include_all_candidates: bool = False,
) -> dict[str, Any]:
    """Compare a proposed guess with the solver's top-ranked suggestions."""
    try:
        normalized_top_n, top_n_capped = _normalize_top_n(top_n)
        turns = parse_history(history)
        state, pool, used_broad_fallback = _resolve_state(turns, hard_mode)
    except WordleInputError as e:
        return _invalid_response(str(e))

    guess_word = (guess or "").strip().lower()
    if len(guess_word) != 5 or not guess_word.isascii() or not guess_word.isalpha():
        return _invalid_response("guess must be exactly 5 ASCII letters")
    if guess_word not in state.game.g_idx:
        return _invalid_response(f"{guess_word.upper()} is not in the Wordle guess list")

    if turns and turns[-1][1] == ALL_GREEN:
        word = turns[-1][0]
        return {
            "status": "solved",
            "summary": f"{word.upper()} is marked all green; the puzzle is solved.",
            "history": serialize_history(turns),
            "turns_used": len(turns),
            "hard_mode": hard_mode,
            "pool": pool,
            "used_broad_fallback": used_broad_fallback,
            "solution": word,
            "guess": guess_word,
            "suggestions": [],
            "candidates": _candidate_payload(state, include_all_candidates),
        }

    if state.candidates_count == 0:
        return {
            "status": "inconsistent_feedback",
            "summary": (
                "No valid Wordle answer matches this history, even after broad fallback. "
                "At least one guess or color is probably wrong."
            ),
            "history": serialize_history(turns),
            "turns_used": len(turns),
            "hard_mode": hard_mode,
            "pool": pool,
            "used_broad_fallback": used_broad_fallback,
            "guess": guess_word,
            "suggestions": [],
            "candidates": _candidate_payload(state, include_all_candidates),
        }

    hard_filter = state.guess_filter()
    violates_hard_mode = bool(
        hard_filter is not None and not bool(hard_filter[state.game.g_idx[guess_word]])
    )

    suggestions, hard_scores = _suggestions_payload(state, normalized_top_n)
    analysis_state = _state_without_hard_mode(state) if violates_hard_mode else state
    _, analysis_scores = (
        analysis_state.rank(top_n=1) if violates_hard_mode else (suggestions, hard_scores)
    )
    picked_stats, picked_rank = analyze_word(analysis_state, guess_word, analysis_scores)

    top = suggestions[0] if suggestions else None
    top_bits = float(top["bits"]) if top else 0.0
    delta = top_bits - picked_stats.bits if top else 0.0
    status = "violates_hard_mode" if violates_hard_mode else "ok"

    if violates_hard_mode:
        summary = (
            f"{guess_word.upper()} violates hard mode constraints, but would otherwise "
            f"rank #{picked_rank} with {picked_stats.bits:.2f} bits."
        )
    else:
        summary = (
            f"{guess_word.upper()} ranks #{picked_rank} with {picked_stats.bits:.2f} bits. "
            f"{_compare_verdict(picked_rank, delta)}"
        )

    response = _base_response(
        status=status,
        summary=summary,
        turns=turns,
        state=state,
        pool=pool,
        used_broad_fallback=used_broad_fallback,
        top_n_capped=top_n_capped,
        include_all_candidates=include_all_candidates,
    )
    response.update(
        {
            "guess": _stats_payload(picked_stats, picked_rank),
            "violates_hard_mode": violates_hard_mode,
            "top_word": top["word"] if top else None,
            "top_bits": top_bits,
            "delta_bits_vs_top": round(float(delta), 6),
            "verdict": _compare_verdict(picked_rank, delta),
            "suggestions": suggestions,
        }
    )
    return response
