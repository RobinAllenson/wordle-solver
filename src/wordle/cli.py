"""CLI entry points: `wordle play`, `wordle selfplay`, `wordle bench`."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from collections import Counter

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from wordle.anytime import AnytimeWorker
from wordle.explain import (
    analyze_guess,
    analyze_word,
    detailed_explanation,
    one_liner,
    render_comparison,
)
from wordle.patterns import ALL_GREEN, decode_pattern, encode_feedback
from wordle.play import HistoryKey, selfplay
from wordle.solver import GameData, SolverState

app = typer.Typer(
    help="Wordle solver with frequency-weighted entropy.",
    no_args_is_help=True,
)
console = Console()

MAX_TURNS = 6


def _render_tiles(word: str, pattern: int) -> Text:
    """Render 5 coloured tiles for a (word, pattern) pair."""
    decoded = decode_pattern(pattern)
    t = Text()
    style_for = {"g": "black on green", "y": "black on yellow", "b": "white on grey30"}
    for c, d in zip(word.upper(), decoded):
        t.append(f" {c} ", style=style_for[d])
        t.append(" ")
    return t


def _render_suggestions(
    ranked: list[tuple[int, float]],
    state: SolverState,
) -> Table:
    cand_set = state.candidate_set()
    table = Table(show_header=True, header_style="bold", title="Top guesses")
    table.add_column("#", style="dim", width=3)
    table.add_column("word")
    table.add_column("bits", justify="right")
    table.add_column("∈S", justify="center")
    for i, (gi, s) in enumerate(ranked, 1):
        w = state.game.guesses[gi]
        in_s = w in cand_set
        table.add_row(
            str(i),
            w.upper(),
            f"{s:.4f}",
            "[green]✓[/green]" if in_s else "",
        )
    return table


def _print_candidate_context(state: SolverState) -> None:
    n = state.candidates_count
    if n == 0:
        return
    console.print(f"[cyan]{n}[/cyan] candidate{'s' if n != 1 else ''} remain")
    if n <= 20:
        cands = state.candidates()
        console.print(f"  [dim]{', '.join(w.upper() for w in cands)}[/dim]")
    else:
        cands = state.candidates(limit=5)
        console.print(
            f"  [dim]most likely: {', '.join(w.upper() for w in cands)}…[/dim]"
        )


def _ask_guess(
    state: SolverState,
    ranked: list[tuple[int, float]],
    all_scores,
    broaden_callback=None,
) -> str:
    default_word = state.game.guesses[ranked[0][0]]
    while True:
        picked = typer.prompt(
            f"Guess [blank = {default_word.upper()}, "
            f"/why [N|WORD], /cands, /broad, /quit]",
            default="",
            show_default=False,
        ).strip().lower()
        if not picked:
            return default_word
        if picked in ("/quit", "/q", "quit"):
            raise typer.Exit()
        if picked in ("/cands", "/c"):
            cands = state.candidates()
            console.print(", ".join(w.upper() for w in cands))
            continue
        if picked in ("/broad", "/b"):
            if broaden_callback:
                broaden_callback()
            continue
        if picked.startswith("/why"):
            parts = picked.split()
            arg = parts[1] if len(parts) > 1 else None
            stats = None
            if arg is None:
                gi, sc = ranked[0]
                stats = analyze_guess(state, gi, sc)
            elif arg.isdigit():
                n = int(arg)
                if not (1 <= n <= len(ranked)):
                    console.print(f"[red]rank must be 1..{len(ranked)}[/red]")
                    continue
                gi, sc = ranked[n - 1]
                stats = analyze_guess(state, gi, sc)
            else:
                word = arg.lower()
                if word not in state.game.g_idx:
                    console.print(
                        f"[yellow]{word!r} isn't in the guess list — can't "
                        f"analyze it.[/yellow]"
                    )
                    continue
                stats, rank = analyze_word(state, word, all_scores)
                console.print(detailed_explanation(state, stats))
                # Also compare to top if it isn't itself the top
                top_gi, top_sc = ranked[0]
                if top_gi != state.game.g_idx[word]:
                    top_stats = analyze_guess(state, top_gi, top_sc)
                    console.print(render_comparison(stats, rank, top_stats))
                continue
            console.print(detailed_explanation(state, stats))
            continue
        if picked not in state.game.g_idx:
            console.print(
                f"[yellow]{picked!r} isn't in the guess list — "
                f"Wordle would reject it. Try another.[/yellow]"
            )
            continue
        return picked


def _ask_feedback() -> int:
    while True:
        raw = typer.prompt("Feedback (g/y/b; e.g. bybbg)").strip().lower()
        try:
            return encode_feedback(raw)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")


def _recover_inconsistent_feedback(
    state: SolverState,
    word: str,
    pattern: int,
    alpha: float,
) -> tuple[str, int] | None:
    """Prompt the user to resolve feedback that eliminates every candidate.

    Returns (word, pattern) after recovery, or None if user quits.
    The caller should invalidate any anytime cache after this runs since the
    state (and potentially the whole game data) may have changed.
    """
    while not state.feedback_leaves_candidates(word, pattern):
        console.print(
            f"\n[yellow]⚠ feedback '{decode_pattern(pattern)}' for "
            f"{word.upper()} leaves 0 of {state.candidates_count} "
            f"candidates.[/yellow]"
        )
        console.print(
            "  [bold]r[/bold] re-enter feedback"
            "   [bold]g[/bold] actually guessed a different word"
            "   [bold]b[/bold] widen candidate pool"
            "   [bold]q[/bold] quit"
        )
        choice = typer.prompt("choice", default="r").strip().lower()[:1]
        if choice == "r":
            pattern = _ask_feedback()
        elif choice == "g":
            new_word = typer.prompt("actual guess word").strip().lower()
            if new_word not in state.game.g_idx:
                console.print(
                    f"[red]{new_word!r} not in guess list — try again[/red]"
                )
                continue
            word = new_word
        elif choice == "b":
            if state.game.is_broad:
                console.print(
                    "[red]Already using broad pool — must be a typo.[/red]"
                )
                continue
            with console.status(
                "Building broad pattern table (one-time ~20s)…"
            ):
                broad = GameData.load_broad(alpha=alpha)
            state.switch_game(broad)
            console.print(
                f"[dim]Widened to {len(broad.answers)} candidate answers; "
                f"{state.candidates_count} still consistent with history.[/dim]"
            )
        elif choice == "q":
            return None
        else:
            console.print("[red]pick r, g, b, or q[/red]")
    return word, pattern


@app.command()
def play(
    hard: bool = typer.Option(False, "--hard", help="Play in hard mode."),
    alpha: float = typer.Option(
        1.0,
        "--alpha",
        help="Frequency-prior exponent. 0 = uniform, 1 = raw frequency weighting.",
    ),
    broad: bool = typer.Option(
        False,
        "--broad",
        help="Start with full guess list as candidate pool (slower, always correct).",
    ),
) -> None:
    """Solver assists you in a real Wordle game."""
    with console.status("Loading word lists + pattern table…"):
        game = (
            GameData.load_broad(alpha=alpha)
            if broad
            else GameData.load(alpha=alpha)
        )
    state = SolverState(game, hard_mode=hard)
    console.print(
        f"[dim]{len(game.guesses)} guesses, {len(game.answers)} candidate answers"
        f"{' · hard mode' if hard else ''}"
        f"{' · broad' if game.is_broad else ''} · α={alpha}[/dim]\n"
    )

    def broaden() -> None:
        nonlocal state
        if state.game.is_broad:
            console.print("[dim]already in broad mode[/dim]")
            return
        with console.status("Building broad pattern table (one-time ~20s)…"):
            b = GameData.load_broad(alpha=alpha)
        state.switch_game(b)
        console.print(
            f"[dim]widened to {state.candidates_count} "
            f"candidates[/dim]"
        )

    worker = AnytimeWorker(state)
    pre_ranked: tuple[list[tuple[int, float]], object] | None = None
    used_cache = False

    for turn in range(1, MAX_TURNS + 1):
        console.rule(f"Turn {turn}")
        _print_candidate_context(state)

        if pre_ranked is not None:
            ranked, all_scores = pre_ranked
            pre_ranked = None
            used_cache = True
        else:
            ranked, all_scores = state.rank(top_n=5)
            used_cache = False

        if not ranked:
            console.print(
                "[red]No candidates left — a previous feedback was likely wrong.[/red]"
            )
            raise typer.Exit(code=1)

        console.print(_render_suggestions(ranked, state))
        if used_cache:
            console.print(
                "[dim]⚡ top-5 served from anytime cache[/dim]"
            )

        top_gi, top_score = ranked[0]
        top_stats = analyze_guess(state, top_gi, top_score)
        console.print(one_liner(top_stats))

        word = _ask_guess(state, ranked, all_scores, broaden_callback=broaden)

        if state.game.g_idx[word] != top_gi:
            picked_stats, picked_rank = analyze_word(state, word, all_scores)
            console.print(render_comparison(picked_stats, picked_rank, top_stats))

        # Start background speculation for the chosen word
        worker.start(state.game.g_idx[word])
        try:
            pattern = _ask_feedback()
        finally:
            worker.stop()

        # Validate feedback against current candidates; prompt for recovery
        # if it would leave no matches.
        cached = worker.lookup(pattern)
        if not state.feedback_leaves_candidates(word, pattern):
            recovered = _recover_inconsistent_feedback(state, word, pattern, alpha)
            if recovered is None:
                raise typer.Exit()
            word, pattern = recovered
            cached = None  # worker cache invalid after recovery

        console.print("  ", _render_tiles(word, pattern))

        state.apply(word, pattern)

        if state.is_solved():
            console.print(f"\n[bold green]Solved in {turn}![/bold green]")
            return
        if state.is_broken():
            console.print(
                "\n[red]No answers match all feedback — "
                "check for typos in earlier turns.[/red]"
            )
            raise typer.Exit(code=1)

        if cached is not None:
            pre_ranked = cached

    console.print("\n[yellow]Out of turns.[/yellow]")


@app.command("selfplay")
def selfplay_cmd(
    secret: str = typer.Argument(None, help="Secret word; random answer if omitted."),
    hard: bool = typer.Option(False, "--hard"),
    opener: str = typer.Option(None, "--opener", help="Force a specific opening word."),
    alpha: float = typer.Option(1.0, "--alpha"),
    seed: int = typer.Option(None, "--seed"),
) -> None:
    """Solver plays itself against a secret word."""
    import random

    with console.status("Loading…"):
        game = GameData.load(alpha=alpha)
    if secret is None:
        if seed is not None:
            random.seed(seed)
        secret = random.choice(game.answers)
    secret = secret.lower()
    if secret not in game.a_idx:
        console.print(f"[red]{secret!r} not in answer list[/red]")
        raise typer.Exit(code=1)

    opener_word = opener.lower() if opener else None
    if opener_word and opener_word not in game.g_idx:
        console.print(f"[red]opener {opener_word!r} not in guess list[/red]")
        raise typer.Exit(code=1)

    result = selfplay(
        game, secret, hard_mode=hard, opener=opener_word
    )

    console.print(f"Secret: [bold]{secret.upper()}[/bold]\n")
    for w, p in zip(result.words, result.patterns):
        console.print("  ", _render_tiles(w, p))
    if result.solved:
        console.print(
            f"\n[green]Solved in {result.n_guesses}.[/green]"
        )
    else:
        console.print(
            f"\n[red]Failed after {result.n_guesses} turns.[/red]"
        )


@app.command("bench")
def bench_cmd(
    sample: int = typer.Option(None, "--sample", help="Random sample size; all if omitted."),
    hard: bool = typer.Option(False, "--hard"),
    opener: str = typer.Option(None, "--opener"),
    alpha: float = typer.Option(1.0, "--alpha"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Run self-play over many answers and print the guess-count distribution."""
    import random

    with console.status("Loading…"):
        game = GameData.load(alpha=alpha)

    words = list(game.answers)
    if sample is not None and sample < len(words):
        random.seed(seed)
        words = random.sample(words, sample)
    else:
        words = sorted(words)

    opener_word = opener.lower() if opener else None
    if opener_word and opener_word not in game.g_idx:
        console.print(f"[red]opener {opener_word!r} not in guess list[/red]")
        raise typer.Exit(code=1)

    cache: dict[HistoryKey, str] = {}
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("solving", total=len(words))
        for w in words:
            r = selfplay(
                game, w, hard_mode=hard, opener=opener_word, cache=cache
            )
            results.append(r)
            prog.advance(task)

    # Summary
    counts = Counter(r.n_guesses if r.solved else 7 for r in results)
    n = len(results)
    fails = sum(1 for r in results if not r.solved)
    mean = sum(r.n_guesses for r in results if r.solved) / max(1, n - fails)
    solved_counts = sorted(k for k in counts if k <= 6)
    median = sorted(r.n_guesses for r in results if r.solved)[
        (n - fails) // 2 if n - fails else 0
    ] if (n - fails) else 0
    max_g = max((r.n_guesses for r in results if r.solved), default=0)

    tbl = Table(title="Guess-count distribution", show_header=True, header_style="bold")
    tbl.add_column("guesses", justify="right")
    tbl.add_column("count", justify="right")
    tbl.add_column("histogram")
    biggest = max(counts.values())
    for k in range(1, 7):
        c = counts.get(k, 0)
        bar = "█" * int(round(c / biggest * 40)) if biggest else ""
        tbl.add_row(str(k), str(c), bar)
    if fails:
        c = counts[7]
        bar = "█" * int(round(c / biggest * 40))
        tbl.add_row("fail", str(c), f"[red]{bar}[/red]")
    console.print(tbl)
    console.print(
        f"mean {mean:.3f}  median {median}  max {max_g}  "
        f"fails {fails}/{n} ({fails/n:.1%})"
        f"  · cache hits reused {len(cache)} unique states"
    )


if __name__ == "__main__":
    app()
