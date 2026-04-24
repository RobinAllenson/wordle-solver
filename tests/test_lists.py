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
