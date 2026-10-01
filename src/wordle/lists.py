"""Word-list loading, frequency priors, and pattern-table caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from wordle.patterns import build_pattern_table  # noqa: F401 (re-exported use)

# Word lists ship inside the package so `pip install` from a git URL works.
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"

# Pattern tables are build artifacts — cache them outside the package so it
# stays read-only when installed. Honor WORDLE_CACHE_DIR for explicit control.
CACHE_DIR = Path(
    os.environ.get("WORDLE_CACHE_DIR") or (Path.home() / ".cache" / "wordle")
)
CACHE_SCHEMA_VERSION = 1


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


def load_priors(answers: list[str], alpha: float = 0.0) -> np.ndarray:
    """Uniform answer probabilities by default; optional corpus-frequency heuristic.

    English usage frequency is not a calibrated probability of NYT selection.
    alpha=0 makes no preference within the chosen pool; alpha=1 explicitly
    restores the old raw-frequency heuristic (10**max(zipf_frequency, 1)).
    Intermediate values temper that heuristic. None is an official NYT prior.
    """
    if not answers:
        raise ValueError("answer pool must not be empty")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if alpha == 0:
        return np.full(len(answers), 1.0 / len(answers), dtype=np.float64)

    from wordfreq import zipf_frequency

    zipf = np.array([zipf_frequency(w, "en") for w in answers], dtype=np.float64)
    # Floor so unknown words (zipf==0) still get a tiny nonzero weight
    log_weights = np.maximum(zipf, 1.0)
    # Subtract the maximum before exponentiating to avoid overflow.
    weights = 10.0 ** ((log_weights - log_weights.max()) * alpha)
    weights /= weights.sum()
    return weights


def word_list_fingerprint(words: list[str]) -> str:
    """Stable fingerprint for an ordered word list."""
    h = hashlib.sha256()
    for word in words:
        h.update(word.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _cache_metadata(guesses: list[str], answers: list[str]) -> dict[str, object]:
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "guesses": word_list_fingerprint(guesses),
        "answers": word_list_fingerprint(answers),
        "n_guesses": len(guesses),
        "n_answers": len(answers),
    }


def _metadata_matches(cache_path: Path, expected: dict[str, object]) -> bool:
    try:
        with _metadata_path(cache_path).open() as f:
            actual = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _write_metadata(cache_path: Path, metadata: dict[str, object]) -> None:
    with _metadata_path(cache_path).open("w") as f:
        json.dump(metadata, f, sort_keys=True)


def load_pattern_table(
    guesses: list[str],
    answers: list[str],
    cache_path: Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Load the precomputed |G|x|A| pattern table, building and caching if needed.

    Cache validity is checked with metadata that fingerprints the ordered word
    lists; stale same-shaped caches are rebuilt automatically.
    """
    cache_path = cache_path or (CACHE_DIR / "patterns.npy")
    expected_shape = (len(guesses), len(answers))
    expected_metadata = _cache_metadata(guesses, answers)
    if cache_path.exists():
        if _metadata_matches(cache_path, expected_metadata):
            table = np.load(cache_path)
            if table.shape == expected_shape and table.dtype == np.uint8:
                if verbose:
                    print(f"Loaded cached pattern table from {cache_path}")
                return table
        if verbose:
            print(f"Cache metadata mismatch for {cache_path}, rebuilding")
    if verbose:
        print(f"Building pattern table {expected_shape}...")
    table = build_pattern_table(guesses, answers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, table)
    _write_metadata(cache_path, expected_metadata)
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
    outside the curated 3,158-word pool.
    """
    cache_path = cache_path or (CACHE_DIR / "patterns_broad.npy")
    expected_shape = (len(guesses), len(guesses))
    expected_metadata = _cache_metadata(guesses, guesses)
    if cache_path.exists():
        if _metadata_matches(cache_path, expected_metadata):
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
    _write_metadata(cache_path, expected_metadata)
    return table
