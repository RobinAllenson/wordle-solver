"""Guess scoring, filtering, and ranking."""

from __future__ import annotations

import numpy as np

from wordle.patterns import GREEN, N_PATTERNS, YELLOW


def apply_feedback(
    mask: np.ndarray,
    table: np.ndarray,
    guess_idx: int,
    pattern: int,
) -> np.ndarray:
    """Return new mask after observing `pattern` for `guess_idx`."""
    return mask & (table[guess_idx] == pattern)


def entropy_scores(
    table: np.ndarray,
    mask: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Weighted entropy (bits) per guess. Higher = better info gain."""
    p_mass = priors[mask]
    total = float(p_mass.sum())
    n_g = table.shape[0]
    if total == 0.0:
        return np.zeros(n_g, dtype=np.float64)
    sub = table[:, mask]
    scores = np.empty(n_g, dtype=np.float64)
    for g in range(n_g):
        counts = np.bincount(sub[g], weights=p_mass, minlength=N_PATTERNS)
        nz = counts[counts > 0]
        p = nz / total
        scores[g] = -float((p * np.log2(p)).sum())
    return scores


def expected_remaining(
    table: np.ndarray,
    mask: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Prior-weighted expected remaining mass after this guess. Lower = better."""
    p_mass = priors[mask]
    total = float(p_mass.sum())
    n_g = table.shape[0]
    if total == 0.0:
        return np.zeros(n_g, dtype=np.float64)
    sub = table[:, mask]
    scores = np.empty(n_g, dtype=np.float64)
    for g in range(n_g):
        counts = np.bincount(sub[g], weights=p_mass, minlength=N_PATTERNS)
        scores[g] = float((counts * counts).sum()) / total
    return scores


def bucket_counts(
    table: np.ndarray,
    mask: np.ndarray,
    priors: np.ndarray,
    guess_idx: int,
) -> np.ndarray:
    """Prior-weighted bucket mass per pattern for one guess. Shape (243,)."""
    p_mass = priors[mask]
    return np.bincount(table[guess_idx, mask], weights=p_mass, minlength=N_PATTERNS)


def rank_guesses(
    table: np.ndarray,
    mask: np.ndarray,
    priors: np.ndarray,
    ans_in_guess: np.ndarray,
    guess_filter: np.ndarray | None = None,
    top_n: int = 10,
    candidate_bonus: float = 1e-3,
) -> list[tuple[int, float]]:
    """Return list of (guess_idx, score) in descending score order.

    Endgame (|S|<=2): restrict to in-S candidates so we can win outright.
    Else: entropy with a small prior-weighted bonus for in-S guesses (breaks
    ties toward winning when both options are otherwise equal).
    """
    s_size = int(mask.sum())
    if s_size == 0:
        return []

    scores = entropy_scores(table, mask, priors)
    in_s_guess_idx = ans_in_guess[mask]

    if s_size <= 2:
        restricted = np.full_like(scores, -np.inf)
        restricted[in_s_guess_idx] = scores[in_s_guess_idx]
        # With 2 candidates both score the same (both split 2->{1,1} or resolve
        # fully). Tiebreak by prior so we guess the likelier one first.
        restricted[in_s_guess_idx] += candidate_bonus * priors[mask]
        scores = restricted
    else:
        scores = scores.copy()
        scores[in_s_guess_idx] += candidate_bonus * priors[mask]

    if guess_filter is not None:
        scores = np.where(guess_filter, scores, -np.inf)

    order = np.argsort(-scores, kind="stable")[:top_n]
    return [(int(i), float(scores[i])) for i in order if scores[i] > -np.inf]


def hard_mode_guess_mask(
    guesses: list[str],
    past: list[tuple[str, int]],
) -> np.ndarray:
    """Hard-mode validity mask over guesses given past (word, pattern) turns.

    Rules enforced: greens must stay in place; yellow letters must appear
    somewhere in the next guess. Grey letters impose no constraint. Multiplicity
    of yellow letters is tracked per-turn (if two Os turned yellow in one guess,
    the next guess must contain at least two Os).
    """
    n = len(guesses)
    if not past:
        return np.ones(n, dtype=bool)

    green_pos: dict[int, str] = {}
    # Aggregate minimum required count of each letter, taken as the max across
    # turns (each turn's yellow-count is a lower bound on true count).
    required: dict[str, int] = {}
    for word, pattern in past:
        per_turn: dict[str, int] = {}
        for k in range(5):
            d = (pattern // (3 ** k)) % 3
            if d == GREEN:
                green_pos[k] = word[k]
                per_turn[word[k]] = per_turn.get(word[k], 0) + 1
            elif d == YELLOW:
                per_turn[word[k]] = per_turn.get(word[k], 0) + 1
        for ch, c in per_turn.items():
            if c > required.get(ch, 0):
                required[ch] = c

    mask = np.empty(n, dtype=bool)
    for i, w in enumerate(guesses):
        ok = all(w[k] == c for k, c in green_pos.items())
        if ok and required:
            for ch, need in required.items():
                if w.count(ch) < need:
                    ok = False
                    break
        mask[i] = ok
    return mask
