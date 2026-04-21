"""Tests for feedback-pattern computation, especially double-letter cases."""

from __future__ import annotations

import numpy as np
import pytest

from wordle.patterns import (
    GREEN,
    GREY,
    YELLOW,
    build_pattern_table,
    compute_pattern,
    decode_pattern,
    encode_feedback,
)


def pat(digits: list[int]) -> int:
    """Helper: build pattern from explicit digits at positions 0..4."""
    return sum(d * (3 ** i) for i, d in enumerate(digits))


class TestComputePattern:
    def test_all_green(self):
        assert compute_pattern("crane", "crane") == pat([GREEN] * 5)

    def test_all_grey(self):
        # no shared letters
        assert compute_pattern("fuzzy", "crank") == pat(
            [GREY, GREY, GREY, GREY, GREY]
        )

    def test_single_yellow(self):
        # guess "apple" vs answer "grape":
        # greens: only pos 4 'e'. Remaining after greens = {g,r,a,p}.
        # L->R yellows: pos 0 'a' YELLOW, pos 1 'p' YELLOW, pos 2 'p' grey
        # (p already consumed), pos 3 'l' grey.
        assert compute_pattern("apple", "grape") == pat(
            [YELLOW, YELLOW, GREY, GREY, GREEN]
        )

    def test_double_letter_in_guess_single_in_answer(self):
        # guess 'aabcd' vs answer 'xaxyx': first 'a' in guess -> yellow (matches the
        # only 'a' in answer), second 'a' -> grey (no more 'a' available).
        # Here: guess "allow", answer "calls": guess has a(0), l(1), l(2)
        # answer has c(0), a(1), l(2), l(3), s(4)
        # Pos 0 'a' vs 'c': not green. Available after greens: {c:1, a:1, l:2, l:... wait
        # greens first:
        #  pos 0 a vs c: no. pos 1 l vs a: no. pos 2 l vs l: YES green.
        #  pos 3 o vs l: no. pos 4 w vs s: no.
        # remaining after greens: {c:1, a:1, l:1, s:1} (one l consumed)
        # yellows L->R:
        #  pos 0 'a': available -> yellow, a->0
        #  pos 1 'l': available -> yellow, l->0
        #  pos 3 'o': not available -> grey
        #  pos 4 'w': not available -> grey
        assert compute_pattern("allow", "calls") == pat(
            [YELLOW, YELLOW, GREEN, GREY, GREY]
        )

    def test_double_letter_green_plus_yellow(self):
        # guess 'sheep' vs answer 'steep':
        #  pos 0 s vs s: GREEN
        #  pos 1 h vs t: grey (not in answer)
        #  pos 2 e vs e: GREEN
        #  pos 3 e vs e: GREEN
        #  pos 4 p vs p: GREEN
        assert compute_pattern("sheep", "steep") == pat(
            [GREEN, GREY, GREEN, GREEN, GREEN]
        )

    def test_double_letter_both_yellow_impossible_second_grey(self):
        # guess 'class' vs answer 'sassy':
        # answer letters: s(0), a(1), s(2), s(3), y(4)
        # pos 0 c vs s: no
        # pos 1 l vs a: no
        # pos 2 a vs s: no (a not at 2, but a is in answer at pos 1)
        # pos 3 s vs s: GREEN
        # pos 4 s vs y: no (s at pos 4, but s is in answer elsewhere)
        # remaining after greens: answer minus pos-3 s = {s:2, a:1, y:1}
        # yellows L->R:
        #  pos 0 c: not in remaining -> grey
        #  pos 1 l: not in remaining -> grey
        #  pos 2 a: in remaining -> yellow, a->0
        #  pos 4 s: in remaining -> yellow, s->1
        assert compute_pattern("class", "sassy") == pat(
            [GREY, GREY, YELLOW, GREEN, YELLOW]
        )

    def test_double_in_guess_vs_double_in_answer(self):
        # guess 'geese' vs answer 'sheep':
        # pos 0 g vs s: no
        # pos 1 e vs h: no
        # pos 2 e vs e: GREEN
        # pos 3 s vs e: no
        # pos 4 e vs p: no
        # remaining after greens: answer minus pos-2 e = {s:1, h:1, e:1, p:1}
        # yellows L->R:
        #  pos 0 g: not in remaining -> grey
        #  pos 1 e: in remaining -> yellow, e->0
        #  pos 3 s: in remaining -> yellow, s->0
        #  pos 4 e: e is now 0 -> grey
        assert compute_pattern("geese", "sheep") == pat(
            [GREY, YELLOW, GREEN, YELLOW, GREY]
        )

    def test_guess_no_dupes_answer_has_dupes(self):
        # guess 'abide' vs answer 'daddy':
        # pos 0 a vs d: no
        # pos 1 b vs a: no
        # pos 2 i vs d: no
        # pos 3 d vs d: GREEN
        # pos 4 e vs y: no
        # remaining = {d:2, a:1, y:1}
        # L->R: a -> yellow (a->0), b -> grey, i -> grey, e -> grey
        assert compute_pattern("abide", "daddy") == pat(
            [YELLOW, GREY, GREY, GREEN, GREY]
        )

    def test_range(self):
        """Pattern must fit in 0..242."""
        assert 0 <= compute_pattern("zzzzz", "aaaaa") <= 242
        assert compute_pattern("aaaaa", "aaaaa") == 242


class TestDecodeEncode:
    def test_decode_all_grey(self):
        assert decode_pattern(0) == "bbbbb"

    def test_decode_all_green(self):
        assert decode_pattern(242) == "ggggg"

    def test_decode_mixed(self):
        # pattern with greens at 0, 2, 4 and yellows at 1, 3
        p = pat([GREEN, YELLOW, GREEN, YELLOW, GREEN])
        assert decode_pattern(p) == "gygyg"

    def test_encode_all_grey(self):
        assert encode_feedback("bbbbb") == 0
        assert encode_feedback(".....") == 0
        assert encode_feedback("-----") == 0

    def test_encode_all_green(self):
        assert encode_feedback("ggggg") == 242

    def test_encode_case_insensitive(self):
        assert encode_feedback("GyByG") == encode_feedback("gybyg")

    def test_encode_roundtrip(self):
        for p in [0, 1, 42, 100, 242]:
            assert encode_feedback(decode_pattern(p)) == p

    def test_encode_invalid_length(self):
        with pytest.raises(ValueError):
            encode_feedback("bbbb")

    def test_encode_invalid_char(self):
        with pytest.raises(ValueError):
            encode_feedback("bbbbz")


class TestBuildTable:
    def test_small_table_matches_scalar(self):
        guesses = ["crane", "slate", "allow", "geese", "abide"]
        answers = ["steep", "calls", "sassy", "daddy", "crane", "grape"]
        table = build_pattern_table(guesses, answers)
        assert table.shape == (5, 6)
        assert table.dtype == np.uint8
        for gi, g in enumerate(guesses):
            for ai, a in enumerate(answers):
                assert table[gi, ai] == compute_pattern(g, a), (
                    f"mismatch for guess={g} answer={a}: "
                    f"table={table[gi, ai]} scalar={compute_pattern(g, a)}"
                )

    def test_self_match_is_all_green(self):
        words = ["crane", "allow", "geese"]
        table = build_pattern_table(words, words)
        for i in range(3):
            assert table[i, i] == 242
