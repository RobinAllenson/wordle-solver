"""Word-list loading, frequency priors, and pattern-table caching."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wordle.patterns import build_pattern_table  # noqa: F401 (re-exported use)

# Word lists ship inside the package so `pip install` from a git URL works.
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"

# Pattern tables are build artifacts — cache them outside the package so it
# stays read-only when installed. Honor WORDLE_CACHE_DIR for explicit control.
CACHE_DIR = Path(os.environ.get("WORDLE_CACHE_DIR") or (Path.home() / ".cache" / "wordle"))


def _load_word_file(path: Path) -> list[str]:
    with path.open() as f:
        words = [line.strip().lower() for line in f]
    return [w for w in words if len(w) == 5 and w.isalpha() and w.isascii()]


def load_lists(
    guesses_path: Path | None = None,
    answers_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return (guesses, answers). Both sorted; answers ⊆ guesses."""
    guesses_path = guesses_path or (PACKAGE_DATA_DIR / "guesses.txt")
    answers_path = answers_path or (PACKAGE_DATA_DIR / "answers.txt")
    answers = sorted(set(_load_word_file(answers_path)))
    guesses = sorted(set(_load_word_file(guesses_path)) | set(answers))
    return guesses, answers


def load_priors(answers: list[str], alpha: float = 1.0) -> np.ndarray:
    """Probability distribution over answers, proportional to zipf_frequency^alpha.

    Zipf frequencies are log10-scaled (typical English words: 3-7). We convert
    to linear frequency before weighting. alpha=1 means "weight by true frequency";
    alpha=0 gives uniform; alpha>1 sharpens toward common words.
    """
    from wordfreq import zipf_frequency

    zipf = np.array([zipf_frequency(w, "en") for w in answers], dtype=np.float64)
    # Floor so unknown words (zipf==0) still get a tiny nonzero weight
    linear = 10.0 ** np.maximum(zipf, 1.0)
    weights = linear ** alpha
    weights /= weights.sum()
    return weights


def load_pattern_table(
    guesses: list[str],
    answers: list[str],
    cache_path: Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Load the precomputed |G|x|A| pattern table, building and caching if needed.

    Cache validity is checked by shape only; delete the file if you change lists.
    """
    cache_path = cache_path or (CACHE_DIR / "patterns.npy")
    expected_shape = (len(guesses), len(answers))
    if cache_path.exists():
        table = np.load(cache_path)
        if table.shape == expected_shape and table.dtype == np.uint8:
            if verbose:
                print(f"Loaded cached pattern table from {cache_path}")
            return table
        if verbose:
            print(f"Cache shape mismatch {table.shape} != {expected_shape}, rebuilding")
    if verbose:
        print(f"Building pattern table {expected_shape}...")
    table = build_pattern_table(guesses, answers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, table)
    if verbose:
        print(f"Cached to {cache_path}")
    return table


def word_index(words: list[str]) -> dict[str, int]:
    """Map word -> index in list."""
    return {w: i for i, w in enumerate(words)}


def load_broad_table(
    guesses: list[str],
    cache_path: Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Build/load a |G|x|G| pattern table — every guess vs every guess.

    ~220 MB uint8, ~20s to build. Used when the real NYT answer might be
    outside the curated 2,310-word pool.
    """
    cache_path = cache_path or (CACHE_DIR / "patterns_broad.npy")
    expected_shape = (len(guesses), len(guesses))
    if cache_path.exists():
        table = np.load(cache_path)
        if table.shape == expected_shape and table.dtype == np.uint8:
            if verbose:
                print(f"Loaded broad pattern table from {cache_path}")
            return table
    if verbose:
        print(f"Building broad pattern table {expected_shape} (one-time)…")
    table = build_pattern_table(guesses, guesses)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, table)
    return table
