"""Stateless Wordle helper logic for MCP tools.

The MCP layer should stay thin: parse a compact game history, build a
SolverState, then return JSON-serializable dicts that agents can use directly.
"""

from __future__ import annotations

from collections import Counter
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
DEFAULT_WORDLE_POSSIBLE_LIMIT = 100
MAX_WORDLE_POSSIBLE_LIMIT = 5000
DEFAULT_ENGLISH_CANDIDATE_LIMIT = 500
MAX_ENGLISH_CANDIDATE_LIMIT = 1000
MAX_ENGLISH_WORD_LENGTH = 32

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
_WORD_PATTERN_WILDCARDS = {"_", "?", "."}


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


def _normalize_feedback_text(raw: str, expected_len: int) -> str:
    text = (raw or "").strip().lower().replace(_VARIATION_SELECTOR, "")
    text = "".join(ch for ch in text if not ch.isspace())
    out: list[str] = []
    for i, ch in enumerate(text):
        mapped = _FEEDBACK_CHAR_MAP.get(ch)
        if mapped is None:
            raise WordleInputError(
                f"invalid feedback character {ch!r} at position {i + 1}; "
                "use b/y/g, '.', '-', 'x', or Wordle square emoji"
            )
        out.append(mapped)
    if len(out) != expected_len:
        raise WordleInputError(
            f"feedback must describe exactly {expected_len} tiles; got {len(out)} from {raw!r}"
        )
    return "".join(out)


def _normalize_feedback(raw: str) -> str:
    return _normalize_feedback_text(raw, expected_len=5)


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


@lru_cache(maxsize=MAX_ENGLISH_WORD_LENGTH)
def get_english_words_by_length(length: int) -> tuple[str, ...]:
    """Return broad English words of a given length in wordfreq order."""
    from wordfreq import iter_wordlist

    seen: set[str] = set()
    words: list[str] = []
    for raw in iter_wordlist("en", "best"):
        word = raw.lower()
        if (
            len(word) == length
            and word.isascii()
            and word.isalpha()
            and word not in seen
        ):
            seen.add(word)
            words.append(word)
    return tuple(words)


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


def _normalize_limit(
    limit: int,
    *,
    default: int,
    maximum: int,
    field_name: str = "limit",
) -> tuple[int, bool]:
    try:
        n = default if limit is None else int(limit)
    except (TypeError, ValueError) as e:
        raise WordleInputError(f"{field_name} must be an integer") from e
    if n < 1:
        raise WordleInputError(f"{field_name} must be at least 1")
    capped = n > maximum
    return min(n, maximum), capped


def _normalize_ascii_word(raw: str, field_name: str) -> str:
    word = (raw or "").strip().lower()
    if not word:
        return ""
    if not word.isascii() or not word.isalpha():
        raise WordleInputError(f"{field_name} must contain only ASCII letters")
    if len(word) > MAX_ENGLISH_WORD_LENGTH:
        raise WordleInputError(
            f"{field_name} must be at most {MAX_ENGLISH_WORD_LENGTH} letters"
        )
    return word


def _normalize_word_pattern(raw: str) -> str:
    pattern = "".join(ch for ch in (raw or "").strip().lower() if not ch.isspace())
    if not pattern:
        return ""
    if len(pattern) > MAX_ENGLISH_WORD_LENGTH:
        raise WordleInputError(
            f"pattern must be at most {MAX_ENGLISH_WORD_LENGTH} characters"
        )
    for i, ch in enumerate(pattern):
        if ch in _WORD_PATTERN_WILDCARDS:
            continue
        if not ch.isascii() or not ch.isalpha():
            raise WordleInputError(
                f"invalid pattern character {ch!r} at position {i + 1}; "
                "use ASCII letters for known positions and '_' for unknown positions"
            )
    return pattern


def _matches_word_pattern(word: str, pattern: str) -> bool:
    return all(
        pattern_ch in _WORD_PATTERN_WILDCARDS or word_ch == pattern_ch
        for word_ch, pattern_ch in zip(word, pattern)
    )


def _compute_feedback_text(guess: str, answer: str) -> str:
    """Compute b/y/g feedback for any equal-length guess and answer."""
    result = ["b"] * len(guess)
    remaining: Counter[str] = Counter()
    for i, guess_ch in enumerate(guess):
        if guess_ch == answer[i]:
            result[i] = "g"
        else:
            remaining[answer[i]] += 1
    for i, guess_ch in enumerate(guess):
        if result[i] == "g":
            continue
        if remaining[guess_ch] > 0:
            result[i] = "y"
            remaining[guess_ch] -= 1
    return "".join(result)


def _word_candidates_payload(words: list[str], limit: int) -> dict[str, Any]:
    returned = words[:limit]
    return {
        "count": len(words),
        "words": returned,
        "truncated": len(words) > len(returned),
        "limit": limit,
    }


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
    limit: int | None = None,
) -> dict[str, Any]:
    count = state.candidates_count
    if include_all_candidates or (
        limit is None and count <= AUTO_ALL_CANDIDATE_THRESHOLD
    ):
        effective_limit = None
    else:
        effective_limit = limit or DEFAULT_CANDIDATE_LIMIT

    words = state.candidates(limit=effective_limit)
    return {
        "count": count,
        "words": words,
        "truncated": effective_limit is not None and count > len(words),
        "limit": effective_limit,
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


def wordle_list_possible_answers(
    history: str,
    include_all_candidates: bool = False,
    limit: int = DEFAULT_WORDLE_POSSIBLE_LIMIT,
) -> dict[str, Any]:
    """Return possible curated Wordle answers without ranking next guesses."""
    try:
        normalized_limit, limit_capped = _normalize_limit(
            limit,
            default=DEFAULT_WORDLE_POSSIBLE_LIMIT,
            maximum=MAX_WORDLE_POSSIBLE_LIMIT,
        )
        turns = parse_history(history)
        state = _build_state(get_curated_game(), turns, hard_mode=False)
    except WordleInputError as e:
        return _invalid_response(str(e))

    candidates = _candidate_payload(
        state,
        include_all_candidates=include_all_candidates,
        limit=normalized_limit,
    )

    if turns and turns[-1][1] == ALL_GREEN:
        word = turns[-1][0]
        return {
            "status": "solved",
            "summary": f"{word.upper()} is marked all green; the puzzle is solved.",
            "history": serialize_history(turns),
            "turns_used": len(turns),
            "pool": "curated",
            "limit_capped": limit_capped,
            "solution": word,
            "candidates": candidates,
        }

    if state.candidates_count == 0:
        return {
            "status": "inconsistent_feedback",
            "summary": (
                "No curated Wordle answer matches this history. "
                "At least one guess or color is probably wrong, or the answer "
                "is outside the curated Wordle answer list."
            ),
            "history": serialize_history(turns),
            "turns_used": len(turns),
            "pool": "curated",
            "limit_capped": limit_capped,
            "candidates": candidates,
        }

    return {
        "status": "ok",
        "summary": (
            f"Found {state.candidates_count} possible curated Wordle "
            f"answer{'s' if state.candidates_count != 1 else ''}."
        ),
        "history": serialize_history(turns),
        "turns_used": len(turns),
        "pool": "curated",
        "limit_capped": limit_capped,
        "candidates": candidates,
    }


def english_word_candidates(
    pattern: str = "",
    guess: str = "",
    feedback: str = "",
    limit: int = DEFAULT_ENGLISH_CANDIDATE_LIMIT,
) -> dict[str, Any]:
    """Return broad English words matching a pattern and/or one feedback pair."""
    try:
        word_pattern = _normalize_word_pattern(pattern)
        guess_word = _normalize_ascii_word(guess, "guess")
        has_feedback = bool((feedback or "").strip())
        if bool(guess_word) != has_feedback:
            raise WordleInputError("provide both guess and feedback, or neither")
        feedback_text = (
            _normalize_feedback_text(feedback, expected_len=len(guess_word))
            if guess_word
            else ""
        )
        if not word_pattern and not guess_word:
            raise WordleInputError(
                "provide a word pattern like '_e_o___' or a guess with feedback"
            )
        if word_pattern and guess_word and len(word_pattern) != len(guess_word):
            raise WordleInputError("pattern, guess, and feedback must have the same length")
        word_length = len(word_pattern or guess_word)
        normalized_limit, limit_capped = _normalize_limit(
            limit,
            default=DEFAULT_ENGLISH_CANDIDATE_LIMIT,
            maximum=MAX_ENGLISH_CANDIDATE_LIMIT,
        )
    except WordleInputError as e:
        return _invalid_response(str(e))

    words = get_english_words_by_length(word_length)
    matches = [
        word
        for word in words
        if (not word_pattern or _matches_word_pattern(word, word_pattern))
        and (not guess_word or _compute_feedback_text(guess_word, word) == feedback_text)
    ]
    candidates = _word_candidates_payload(list(matches), normalized_limit)
    return {
        "status": "ok",
        "summary": (
            f"Found {len(matches)} English word{'s' if len(matches) != 1 else ''} "
            f"matching the query."
        ),
        "corpus": "wordfreq:en:best",
        "word_length": word_length,
        "pattern": word_pattern,
        "guess": guess_word,
        "feedback": feedback_text,
        "limit_capped": limit_capped,
        "candidates": candidates,
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
