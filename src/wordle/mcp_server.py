"""FastMCP server for Wordle assistance."""

from __future__ import annotations

import contextlib
import os
from typing import Annotated, Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from wordle.mcp_logic import (
    EXAMPLE_HISTORY,
    english_word_candidates,
    warm_caches,
    wordle_compare_guess,
    wordle_list_possible_answers,
    wordle_suggest_next_guess,
)

MCP_INSTRUCTIONS = f"""
Use this Wordle Solver when a user wants help choosing their next Wordle guess
or comparing a proposed guess against entropy-ranked alternatives. Use
wordle_list_possible_answers when the user asks what Wordle answers remain
without asking for a recommended next guess.

Canonical history format is compact and stateless: {EXAMPLE_HISTORY}.
Each turn is guess:feedback. Feedback is five left-to-right tiles:
b = gray/black/absent, y = yellow/present elsewhere, g = green/correct.
Wordle square emoji are also accepted.

Use english_word_candidates for non-Wordle word-pattern lookups like '_e_o___'
or arbitrary-length feedback pairs like 'fenotps:bgbgyyy'.

If the user provides a screenshot, read the board visually first, convert each
row to compact feedback, then call wordle_suggest_next_guess.
""".strip()

mcp = FastMCP(
    "Wordle Solver",
    instructions=MCP_INSTRUCTIONS,
    website_url="https://robinallenson.com/wordle",
    stateless_http=True,
    json_response=True,
)


@mcp.tool(
    name="wordle_suggest_next_guess",
    title="Suggest the next Wordle guess",
    description=(
        "Use this when the user gives Wordle guesses and tile colors and wants "
        "the best next guess. Pass history as compact guess:feedback turns, "
        f"for example {EXAMPLE_HISTORY!r}. Feedback accepts b/y/g, '.', '-', "
        "'x', uppercase, and Wordle square emoji. The solver uses the curated "
        "answer pool first and automatically falls back to broad mode when "
        "needed."
    ),
)
def suggest_next_guess(
    history: Annotated[
        str,
        Field(
            description=(
                "Compact turn history like 'slate:bbgyb,crown:bgbbb'. Use b for "
                "gray/black, y for yellow, and g for green."
            )
        ),
    ] = "",
    hard_mode: Annotated[
        bool,
        Field(description="If true, future suggestions obey Wordle hard-mode rules."),
    ] = False,
    top_n: Annotated[
        int,
        Field(description="Number of ranked suggestions to return. Values above 25 are capped."),
    ] = 5,
    include_all_candidates: Annotated[
        bool,
        Field(
            description=(
                "If true, return every remaining candidate. By default, all candidates "
                "are returned only when there are 50 or fewer; otherwise the response "
                "includes the top 25 by prior."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    return wordle_suggest_next_guess(
        history=history,
        hard_mode=hard_mode,
        top_n=top_n,
        include_all_candidates=include_all_candidates,
    )


@mcp.tool(
    name="wordle_list_possible_answers",
    title="List possible Wordle answers",
    description=(
        "Use this when the user gives Wordle guesses and tile colors and wants "
        "to know which curated Wordle answer words are still possible, without "
        "ranking or recommending the best next guess. Pass history as compact "
        f"guess:feedback turns, for example {EXAMPLE_HISTORY!r}."
    ),
)
def list_possible_answers(
    history: Annotated[
        str,
        Field(
            description=(
                "Compact turn history like 'slate:bbgyb,crown:bgbbb'. Use b for "
                "gray/black, y for yellow, and g for green."
            )
        ),
    ] = "",
    include_all_candidates: Annotated[
        bool,
        Field(
            description=(
                "If true, return every remaining curated Wordle answer. Otherwise "
                "return up to `limit` candidates plus the total count."
            )
        ),
    ] = False,
    limit: Annotated[
        int,
        Field(
            description=(
                "Maximum number of candidate words to return when not returning all. "
                "Values above 5000 are capped."
            )
        ),
    ] = 100,
) -> dict[str, Any]:
    return wordle_list_possible_answers(
        history=history,
        include_all_candidates=include_all_candidates,
        limit=limit,
    )


@mcp.tool(
    name="wordle_compare_guess",
    title="Compare a Wordle guess",
    description=(
        "Use this when the user has a specific Wordle guess in mind and wants "
        "to know how it compares with the solver's top entropy-ranked guesses. "
        "The response includes the guess rank, information score, candidate "
        "impact, and whether it violates hard mode."
    ),
)
def compare_guess(
    history: Annotated[
        str,
        Field(
            description=(
                "Compact turn history like 'slate:bbgyb,crown:bgbbb'. Leave empty "
                "when comparing an opening guess."
            )
        ),
    ],
    guess: Annotated[
        str,
        Field(description="The 5-letter Wordle guess to compare, for example 'slate'."),
    ],
    hard_mode: Annotated[
        bool,
        Field(description="If true, flag guesses that violate hard-mode constraints."),
    ] = False,
    top_n: Annotated[
        int,
        Field(description="Number of top alternatives to return. Values above 25 are capped."),
    ] = 5,
    include_all_candidates: Annotated[
        bool,
        Field(description="If true, return every remaining candidate answer."),
    ] = False,
) -> dict[str, Any]:
    return wordle_compare_guess(
        history=history,
        guess=guess,
        hard_mode=hard_mode,
        top_n=top_n,
        include_all_candidates=include_all_candidates,
    )


@mcp.tool(
    name="english_word_candidates",
    title="Find English words matching a pattern or feedback",
    description=(
        "Use this for broad English word lookup outside the Wordle answer list. "
        "Pass a pattern such as '_e_o___' where '_' means unknown, or pass an "
        "arbitrary-length guess and b/y/g feedback pair such as guess='fenotps', "
        "feedback='bgbgyyy'. The response is not a next-guess recommendation."
    ),
)
def find_english_word_candidates(
    pattern: Annotated[
        str,
        Field(
            description=(
                "Known-letter pattern such as '_e_o___'. ASCII letters are fixed "
                "positions; '_', '?', and '.' are unknown positions."
            )
        ),
    ] = "",
    guess: Annotated[
        str,
        Field(
            description=(
                "Optional arbitrary-length word guess. If set, feedback must also "
                "be set and must have the same length."
            )
        ),
    ] = "",
    feedback: Annotated[
        str,
        Field(
            description=(
                "Feedback for guess using b/y/g: b=absent, y=present elsewhere, "
                "g=correct position."
            )
        ),
    ] = "",
    limit: Annotated[
        int,
        Field(
            description=(
                "Maximum number of English candidates to return. Values above "
                "1000 are capped."
            )
        ),
    ] = 500,
) -> dict[str, Any]:
    return english_word_candidates(
        pattern=pattern,
        guess=guess,
        feedback=feedback,
        limit=limit,
    )


@mcp.prompt(
    name="help_with_wordle_screenshot",
    title="Help with a Wordle screenshot",
    description=(
        "Use this prompt when the user attaches or describes a Wordle screenshot. "
        "The model should extract the board visually, convert colors to b/y/g, "
        "then call wordle_suggest_next_guess."
    ),
)
def help_with_wordle_screenshot() -> str:
    return f"""
Read the Wordle board from the screenshot. For each completed row, extract:
1. the five-letter guess, left to right
2. the five colors, left to right

Convert colors to compact feedback:
- gray, black, or white tiles -> b
- yellow tiles -> y
- green tiles -> g

Build a compact history string like {EXAMPLE_HISTORY!r}. Then call
wordle_suggest_next_guess with that history. If a tile is ambiguous, ask the
user to confirm before calling the tool.
""".strip()


async def health(_request) -> PlainTextResponse:
    return PlainTextResponse("OK\n")


@contextlib.asynccontextmanager
async def lifespan(_app: Starlette):
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)

app = CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)


def main() -> None:
    if os.getenv("WORDLE_MCP_WARM_BROAD") == "1":
        warm_caches(include_broad=True)
    host = os.getenv("WORDLE_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("WORDLE_MCP_PORT", "5010"))
    uvicorn.run("wordle.mcp_server:app", host=host, port=port)


if __name__ == "__main__":
    main()
