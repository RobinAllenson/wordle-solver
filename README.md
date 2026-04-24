# wordle

A Wordle solver that explains *why* it picks each guess. Built around
frequency-weighted information-gain — but the point is as much to teach
the strategy as to run it.

## What it does

- **Play mode** — assists you in a live Wordle game. Suggests five ranked
  candidates, lets you accept or override, then parses your feedback.
- **Strategy explanations** — `/why` shows letter coverage in the remaining
  candidate set, the most-likely feedback outcomes with resulting
  `|S_next|`, and a context-aware strategy note (e.g. "Early game: spread
  across common letters, not guess the answer").
- **Self-play and bench** — solver guesses a known or random answer; or
  iterates the full 2,310-word answer list and prints a guess-count
  histogram.
- **Hard mode** — toggle with `--hard`; enforces green-stays-put and
  yellow-must-appear (with correct multiplicity) on the guess set.

## Install

```
uv sync
```

## Usage

```
uv run wordle play                    # assist in a real Wordle game
uv run wordle play --hard             # hard mode
uv run wordle selfplay abbey          # watch the solver guess a word
uv run wordle selfplay --seed 7       # random secret
uv run wordle bench                   # distribution over all answers
uv run wordle bench --sample 200      # quick sample
uv run wordle bench --opener salet    # force a starting word
```

During play, at the guess prompt:

- blank → pick the top-ranked suggestion
- any 5-letter word → override
- `/why [N]` → full rationale for suggestion #N (default #1)
- `/cands` → list all remaining candidates
- `/quit`

Feedback format: 5 characters of `g` (green) / `y` (yellow) / `b` (grey),
case-insensitive; `.` or `-` also work for grey. Example: `bybbg`.

## MCP server

The package can also run as a stateless MCP server for ChatGPT, Claude, or
other MCP clients:

```
uv sync --extra mcp
uv run --extra mcp wordle-mcp
```

By default this starts a Streamable HTTP MCP endpoint at
`http://127.0.0.1:5010/mcp` plus `GET /health`.

Tools:

- `wordle_suggest_next_guess(history, hard_mode=false, top_n=5, include_all_candidates=false)`
- `wordle_compare_guess(history, guess, hard_mode=false, top_n=5, include_all_candidates=false)`

The canonical `history` format is compact and stateless:
`slate:bbgyb,crown:bgbbb`. Feedback is five left-to-right tiles:
`b` = grey/black/absent, `y` = yellow, `g` = green. Uppercase,
`.`, `-`, `x`, and Wordle square emoji are accepted too.

The server starts with the curated 2,310-answer pool. If a valid history
leaves no curated candidates, it automatically falls back to broad mode where
every valid guess can be an answer. Broad mode uses a separate on-disk pattern
cache of roughly 220 MB, so warm it during deployment if you want to avoid the
first fallback request paying the build cost:

```
uv run --extra mcp python -c "from wordle.mcp_logic import warm_caches; print(warm_caches(include_broad=True))"
```

The MCP server also exposes a `help_with_wordle_screenshot` prompt. The model
should read the screenshot visually, convert rows to compact `b/y/g` history,
then call `wordle_suggest_next_guess`; the server does not perform OCR.

## How it works

For each possible guess `g` and the current set of still-possible answers
`S`:

1. Partition `S` into up to 243 buckets by the feedback pattern each answer
   would produce.
2. Weight each answer by its prior — `zipf_frequency(w, 'en')` raised to
   `α` (default 1.0) — so common words count more. This matches NYT's
   observed bias toward everyday words.
3. Score each guess by the weighted entropy of its bucket distribution.
4. Tiebreak in favour of guesses that are themselves possible answers
   (could win outright).
5. When `|S| ≤ 2`, restrict the ranking to `S` — we should be trying to
   win this turn.

A precomputed `|guesses| × |answers|` `uint8` pattern table (≈33 MB)
makes each scoring pass a handful of NumPy `bincount` calls.

## Benchmark

On the 2,310-word answer list, default α=1.0:

```
guesses   count   histogram
      2      61   ██
      3    1068   ███████████████████████████████████████
      4    1086   ████████████████████████████████████████
      5      92   ███
      6       3
mean 3.527  median 4  max 6  fails 0/2310
```

For reference: the proven optimum under uniform priors is 3.42 (Alex
Selby, SALET-rooted decision tree). Frequency-weighted greedy solvers
generally land in the 3.4–3.6 range.

## Layout

```
src/wordle/
  patterns.py   feedback encoding + vectorised |G|×|A| table builder
  lists.py      word-list loading, frequency priors, cache
  scoring.py    entropy, endgame, hard-mode filter
  solver.py     SolverState (mask + history) and GameData
  explain.py    rationale + teaching notes
  play.py       self-play driver with memoised top-picks
  cli.py        typer entry point
data/
  answers.txt   2,310 candidate answers
  guesses.txt   14,854 NYT valid-guess list
  patterns.npy  cached pattern table (built on first run)
tests/          pytest suite
```

## Development

```
uv run pytest
```

Word lists are vendored from:
- [tabatkins/wordle-list](https://github.com/tabatkins/wordle-list) — NYT valid guesses
- Original Wordle answer list (pre-NYT, 2,310 words)
