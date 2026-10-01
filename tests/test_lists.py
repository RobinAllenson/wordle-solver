"""Tests for word-list and pattern-table cache behavior."""

from __future__ import annotations

import numpy as np

from wordle.lists import load_pattern_table
from wordle.patterns import ALL_GREEN


def test_pattern_cache_rebuilds_without_matching_metadata(tmp_path):
    cache_path = tmp_path / "patterns.npy"
    np.save(cache_path, np.array([[0]], dtype=np.uint8))

    table = load_pattern_table(["ccccc"], ["ccccc"], cache_path=cache_path)

    assert int(table[0, 0]) == ALL_GREEN
    assert (tmp_path / "patterns.npy.json").exists()


def test_expanded_snapshot_preserves_waken_and_observed_partitions():
    from wordle.lists import load_lists
    from wordle.patterns import compute_pattern, encode_feedback

    guesses, answers = load_lists()
    assert len(answers) == 3158
    assert set(answers) <= set(guesses)
    assert "waken" in answers
    for guess, feedback, count in [("slate", "bbyby", 141),
                                    ("brond", "bbbyb", 10),
                                    ("aheap", "ybybb", 3)]:
        answers = [w for w in answers
                   if compute_pattern(guess, w) == encode_feedback(feedback)]
        assert len(answers) == count
    assert set(answers) == {"maven", "waken", "waxen"}
