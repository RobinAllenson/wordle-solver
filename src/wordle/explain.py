"""Per-guess rationale and teaching notes — the layer that tries to teach
strategy, not just justify picks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wordle.patterns import N_PATTERNS, decode_pattern
from wordle.solver import SolverState


@dataclass
class GuessStats:
    word: str
    bits: float
    n_candidates: int
    n_buckets: int
    expected_next: float  # expected |S_next| under current priors
    worst_next: int  # max bucket size (unweighted)
    is_in_S: bool
    win_prob: float  # mass of this word in S; 0 if not in S
    letter_coverage: list[tuple[str, float]]  # (letter, fraction of S with it)
    top_patterns: list[tuple[int, int, float]]  # (pattern, bucket size, mass fraction)


def analyze_guess(
    state: SolverState, guess_idx: int, score: float
) -> GuessStats:
    """Compute explanation-ready stats for one guess."""
    game = state.game
    word = game.guesses[guess_idx]
    mask = state.mask
    p_mass = game.priors[mask]
    total = float(p_mass.sum())

    row = game.table[guess_idx, mask]
    sizes = np.bincount(row, minlength=N_PATTERNS)
    weights = np.bincount(row, weights=p_mass, minlength=N_PATTERNS)
    nonempty_idx = np.flatnonzero(sizes > 0)
    n_buckets = int(nonempty_idx.size)
    worst_next = int(sizes.max())
    expected_next = (
        float(((weights / total) * sizes).sum()) if total > 0 else 0.0
    )

    is_in_S = word in game.a_idx and mask[game.a_idx[word]]
    win_prob = (
        float(game.priors[game.a_idx[word]] / total)
        if is_in_S and total > 0
        else 0.0
    )

    # Letter coverage (unique letters only)
    s_words = [game.answers[j] for j in np.flatnonzero(mask)]
    coverage: list[tuple[str, float]] = []
    seen: set[str] = set()
    for c in word:
        if c in seen:
            continue
        seen.add(c)
        frac = sum(1 for w in s_words if c in w) / max(1, len(s_words))
        coverage.append((c, frac))

    # Top 3 most-probable feedback patterns
    order = np.argsort(-weights[nonempty_idx])[:3]
    top_patterns = [
        (
            int(nonempty_idx[i]),
            int(sizes[nonempty_idx[i]]),
            float(weights[nonempty_idx[i]] / total) if total > 0 else 0.0,
        )
        for i in order
    ]

    return GuessStats(
        word=word,
        bits=score,
        n_candidates=int(mask.sum()),
        n_buckets=n_buckets,
        expected_next=expected_next,
        worst_next=worst_next,
        is_in_S=is_in_S,
        win_prob=win_prob,
        letter_coverage=coverage,
        top_patterns=top_patterns,
    )


def one_liner(stats: GuessStats) -> Text:
    """Short always-visible rationale for a guess."""
    t = Text()
    t.append(f"{stats.word.upper()}", style="bold")
    t.append(f" · {stats.bits:.2f} bits", style="cyan")
    t.append(
        f" · narrows {stats.n_candidates}→~{stats.expected_next:.0f}"
        f" (worst {stats.worst_next})",
        style="",
    )
    if stats.letter_coverage:
        letters = "·".join(c.upper() for c, _ in stats.letter_coverage)
        t.append(f" · tests {letters}", style="dim")
    if stats.is_in_S:
        t.append(f" · ∈S, win {stats.win_prob:.1%}", style="green")
    return t


def _coverage_table(stats: GuessStats) -> Table:
    tbl = Table(
        title="Letter coverage in S",
        title_style="dim",
        show_header=True,
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("letter")
    tbl.add_column("% of S with letter", justify="left")
    for c, frac in stats.letter_coverage:
        bar = "█" * int(round(frac * 20))
        tbl.add_row(c.upper(), f"{frac:>4.0%}  [dim]{bar}[/dim]")
    return tbl


def _feedback_preview_table(stats: GuessStats) -> Table:
    tbl = Table(
        title="Most likely feedback",
        title_style="dim",
        show_header=True,
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("pattern")
    tbl.add_column("P", justify="right")
    tbl.add_column("|S_next|", justify="right")
    for p, size, prob in stats.top_patterns:
        tbl.add_row(decode_pattern(p), f"{prob:.0%}", str(size))
    return tbl


def _teaching_note(state: SolverState, stats: GuessStats) -> str:
    n = stats.n_candidates
    turns_left = 6 - len(state.history)

    if n == 1:
        return "One candidate left — guess it."
    if n == 2:
        return (
            "Two candidates. Picking one gives 50% solve now, and guarantees "
            "a win next turn whichever way it lands."
        )
    if turns_left == 1:
        return (
            f"Last turn — it's worth picking an actual candidate even if a "
            f"non-candidate scored higher on information."
        )
    if n > 150:
        return (
            "Early game: spread across the most common letters. The goal "
            "here is to cut the space, not to guess the answer. Non-candidate "
            "guesses often top the ranking."
        )
    if n > 20:
        return (
            "Splitting phase: the strongest guesses divide the remaining "
            "candidates into many small, evenly-sized groups. Extra bits now "
            "pay off as faster convergence."
        )
    return (
        "Narrow middle: starting to matter whether your guess is itself a "
        "possible answer — the ∈S column shows who could win outright."
    )


def detailed_explanation(
    state: SolverState, stats: GuessStats
) -> Group:
    """Full rationale panel for --why / /why command."""
    head = Text()
    head.append(f"{stats.word.upper()}", style="bold cyan")
    head.append(f"  {stats.bits:.3f} bits\n", style="cyan")
    head.append(
        f"Splits {stats.n_candidates} candidates into {stats.n_buckets} "
        f"buckets (avg {stats.expected_next:.1f}, worst {stats.worst_next}).\n"
    )
    if stats.is_in_S:
        head.append(
            f"Is itself a possible answer — solve chance {stats.win_prob:.1%}.\n",
            style="green",
        )
    else:
        head.append(
            "Not a possible answer — pure information gathering.\n",
            style="yellow",
        )

    note = _teaching_note(state, stats)
    note_panel = Panel(
        Text(note, style="italic"),
        title="[yellow]strategy note[/yellow]",
        border_style="dim",
        padding=(0, 1),
    )

    return Group(
        head,
        _coverage_table(stats),
        Text(),  # spacer
        _feedback_preview_table(stats),
        Text(),
        note_panel,
    )
