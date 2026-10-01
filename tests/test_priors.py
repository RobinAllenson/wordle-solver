"""Answer priors are modelling choices, not NYT editorial probabilities."""

import inspect

import numpy as np
import pytest
import wordfreq

from wordle import cli, mcp_logic
from wordle.lists import load_priors
from wordle.patterns import build_pattern_table
from wordle.scoring import entropy_scores
from wordle.solver import GameData


def test_default_does_not_bias_towards_corpus_frequency(monkeypatch):
    def no_corpus_lookup(*args):
        raise AssertionError("uniform prior must not depend on corpus frequency")

    monkeypatch.setattr(wordfreq, "zipf_frequency", no_corpus_lookup)
    words = ["haven", "vegan", "waxen"]
    priors = load_priors(words)
    np.testing.assert_allclose(priors, [1 / 3] * 3)
    # These three answers give distinct feedback for HAVEN; all three should
    # contribute equally to information gain, regardless of their usage.
    scores = entropy_scores(build_pattern_table(["haven"], words),
                            np.ones(3, dtype=bool), priors)
    assert scores[0] == pytest.approx(np.log2(3))


def test_frequency_weighting_remains_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(wordfreq, "zipf_frequency",
                        lambda word, lang: {"haven": 6.0, "waxen": 2.0}[word])
    raw = load_priors(["haven", "waxen"], alpha=1)
    tempered = load_priors(["haven", "waxen"], alpha=0.25)
    assert raw[0] / raw[1] == pytest.approx(10000)
    assert tempered[0] / tempered[1] == pytest.approx(10)
    assert raw.sum() == pytest.approx(1)


@pytest.mark.parametrize("alpha", [-1, float("nan"), float("inf")])
def test_invalid_prior_exponents_are_rejected(alpha):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        load_priors(["haven"], alpha=alpha)


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        load_priors([])


def test_mcp_loaders_use_uniform_defaults(monkeypatch):
    calls = []

    def capture(cls, alpha=0.0, verbose=False):
        calls.append(alpha)
        return object()

    monkeypatch.setattr(GameData, "load", classmethod(capture))
    monkeypatch.setattr(GameData, "load_broad", classmethod(capture))
    # Bypass the cache so this test does not depend on other tests' state.
    mcp_logic.get_curated_game.__wrapped__()
    mcp_logic.get_broad_game.__wrapped__()
    assert calls == [0.0, 0.0]


def test_python_and_cli_defaults_agree():
    for fn in [GameData.load, GameData.load_broad]:
        assert inspect.signature(fn).parameters["alpha"].default == 0.0
    for fn in [cli.play, cli.selfplay_cmd, cli.bench_cmd]:
        assert inspect.signature(fn).parameters["alpha"].default.default == 0.0
