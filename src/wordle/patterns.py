"""Feedback pattern computation and precomputed table.

A Wordle feedback is a 5-digit base-3 number:
    0 = grey, 1 = yellow, 2 = green
Position 0 is least significant, so pattern = sum(3**k * digit_k for k in 0..4).

Double-letter rule: greens consume answer letters first, then yellows are
assigned left-to-right; a yellow fires only if the letter is still
"available" in the answer's remaining (non-green) positions.
"""

from __future__ import annotations

import numpy as np

GREY = 0
YELLOW = 1
GREEN = 2

N_PATTERNS = 243  # 3 ** 5
ALL_GREEN = 242  # 2 + 2*3 + 2*9 + 2*27 + 2*81

POWERS = np.array([1, 3, 9, 27, 81], dtype=np.uint8)


def compute_pattern(guess: str, answer: str) -> int:
    """Scalar reference implementation. Returns pattern in 0..242."""
    digits = [0, 0, 0, 0, 0]
    remaining: dict[str, int] = {}
    for i in range(5):
        if guess[i] == answer[i]:
            digits[i] = GREEN
        else:
            remaining[answer[i]] = remaining.get(answer[i], 0) + 1
    for i in range(5):
        if digits[i] == GREEN:
            continue
        c = guess[i]
        if remaining.get(c, 0) > 0:
            digits[i] = YELLOW
            remaining[c] -= 1
    return digits[0] + 3 * digits[1] + 9 * digits[2] + 27 * digits[3] + 81 * digits[4]


def decode_pattern(pattern: int) -> str:
    """Turn 0..242 into a 5-char string: b=grey, y=yellow, g=green."""
    chars = []
    for _ in range(5):
        chars.append("byg"[pattern % 3])
        pattern //= 3
    return "".join(chars)


def encode_feedback(s: str) -> int:
    """Parse a user-entered 5-char feedback string into 0..242.

    Accepts b/g/y (grey, green, yellow), case-insensitive, with '.'/'-'/'x'
    as aliases for grey.
    """
    s = s.strip().lower()
    if len(s) != 5:
        raise ValueError(f"feedback must be 5 characters, got {s!r}")
    digit_map = {"g": GREEN, "y": YELLOW, "b": GREY, ".": GREY, "-": GREY, "x": GREY}
    total = 0
    for i, c in enumerate(s):
        if c not in digit_map:
            raise ValueError(f"invalid feedback char {c!r} at position {i}")
        total += digit_map[c] * (3**i)
    return total


def words_to_array(words: list[str]) -> np.ndarray:
    """Convert a list of 5-letter lowercase words to a uint8 (N, 5) array of 0..25."""
    n = len(words)
    arr = np.empty((n, 5), dtype=np.uint8)
    for i, w in enumerate(words):
        for k, c in enumerate(w):
            arr[i, k] = ord(c) - 97
    return arr


def build_pattern_table(guesses: list[str], answers: list[str]) -> np.ndarray:
    """Build the full |guesses| x |answers| feedback-pattern table.

    Returns a uint8 array of shape (|guesses|, |answers|) with values in 0..242.
    Memory: |G| * |A| bytes (e.g. 14854 * 2310 ~= 33 MB).
    """
    G = words_to_array(guesses)
    A = words_to_array(answers)
    n_g = G.shape[0]
    n_a = A.shape[0]
    table = np.empty((n_g, n_a), dtype=np.uint8)

    row_ix = np.arange(n_g)
    for j in range(n_a):
        ans = A[j]
        greens = G == ans  # (n_g, 5)

        available = np.zeros((n_g, 26), dtype=np.int8)
        for k in range(5):
            available[:, ans[k]] += (~greens[:, k]).astype(np.int8)

        digits = np.zeros((n_g, 5), dtype=np.uint8)
        digits[greens] = GREEN

        for k in range(5):
            not_green = ~greens[:, k]
            letters_k = G[:, k]
            avail_here = available[row_ix, letters_k]
            is_yellow = not_green & (avail_here > 0)
            digits[is_yellow, k] = YELLOW
            rows = np.flatnonzero(is_yellow)
            if rows.size:
                available[rows, letters_k[rows]] -= 1

        table[:, j] = digits @ POWERS

    return table
