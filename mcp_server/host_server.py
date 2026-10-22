"""PaperBanana host-delegated MCP server.

The MCP host (Claude Code or Codex) does every LLM step with its own
subagents; this server runs PaperBanana's pipeline glue and renders the
final image (image API or Codex). No LLM API key is needed.

Tools:
    pb_start   — Start a run; returns the first step for the host
    pb_submit  — Submit a subagent's answer to the pending LLM call
    pb_render  — Render the pending image with the run's image provider
    pb_status  — Show what a run is waiting for

Every tool returns the run's next step: ``needs_llm`` (hand the call to a
worker subagent), ``needs_render`` (call pb_render) or ``done``.

Usage:
    paperbanana-host-mcp     # stdio transport
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Literal

import structlog
from fastmcp import FastMCP

from paperbanana.core.config import Settings
from paperbanana.host import session as host_session

mcp = FastMCP(
    "PaperBanana Host",
    instructions=(
        "Generate academic diagrams and plots where you (the host) do the LLM work. "
        "Call pb_start, then follow each result's `next` field: give every needs_llm "
        "call to a fresh worker subagent that submits with pb_submit, and call pb_render "
        "for needs_render, until status is done."
    ),
)

# Steps on one run must not interleave; runs are cheap to serialise.
_lock = asyncio.Lock()


def _load_dotenv_best_effort() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _resolve_image_provider(image_backend: str | None) -> str:
    backend = (image_backend or os.environ.get("PAPERBANANA_HOST_IMAGE_BACKEND") or "codex").lower()
    if backend == "codex":
        return "codex_imagen"
    if backend == "api":
        # Whatever IMAGE_PROVIDER / IMAGE_MODEL say (.env or environment).
        return Settings().image_provider
    raise ValueError(f"image_backend must be 'codex' or 'api', got {image_backend!r}")


@mcp.tool
async def pb_start(
    caption: str,
    source_text: str | None = None,
    source_path: str | None = None,
    diagram_type: Literal["methodology", "statistical_plot"] = "methodology",
    data_path: str | None = None,
    image_backend: Literal["codex", "api"] | None = None,
    aspect_ratio: str | None = None,
    venue: str | None = None,
    iterations: int = 3,
    auto_refine: bool = False,
    optimize: bool = False,
    generate_caption: bool = False,
    input_images: list[str] | None = None,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """Start a PaperBanana run whose LLM steps are done by the host.

    Args:
        caption: Figure caption / what the figure should communicate.
        source_text: Methodology text (methodology diagrams).
        source_path: Path to a .txt/.md/.pdf with the methodology text, instead of source_text.
        diagram_type: "methodology" (rendered image) or "statistical_plot" (matplotlib code).
        data_path: .csv or .json data file (statistical plots only).
        image_backend: Final render for methodology diagrams: "codex" (Codex CLI image
            generation, no key) or "api" (IMAGE_PROVIDER from .env, e.g. google_imagen).
            Defaults to PAPERBANANA_HOST_IMAGE_BACKEND, else "codex".
        aspect_ratio: Force a ratio (1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9);
            otherwise the venue or the planner chooses.
        venue: Venue style pack (neurips, icml, acl, ieee, or a user pack).
        iterations: Maximum render → critique rounds (default 3).
        auto_refine: Keep refining until the critic is satisfied (up to 30 rounds).
        optimize: Enrich the context and sharpen the caption first (2 extra LLM calls).
        generate_caption: Write a publication-ready caption at the end.
        input_images: Paths to user sketches / reference images that guide the layout.
        output_dir: Parent folder for the run directory (default ./outputs).

    Returns:
        The run's first step (see the server instructions).
    """
    async with _lock:
        image_provider = (
            "none" if diagram_type == "statistical_plot" else _resolve_image_provider(image_backend)
        )
        return await host_session.start_run(
            caption=caption,
            source_context=source_text,
            source_path=source_path,
            diagram_type=diagram_type,
            data_path=data_path,
            aspect_ratio=aspect_ratio,
            venue=venue,
            iterations=iterations,
            auto_refine=auto_refine,
            optimize=optimize,
            generate_caption=generate_caption,
            image_provider=image_provider,
            input_images=input_images,
            output_dir=output_dir,
        )


@mcp.tool
async def pb_submit(run_dir: str, call_id: str, output: str) -> dict[str, Any]:
    """Submit the answer to a run's pending LLM call and advance the run.

    Args:
        run_dir: The run directory from the step.
        call_id: The pending call id (e.g. "llm-003").
        output: The complete answer, exactly as the prompt file asks.
            For JSON calls, raw JSON only.

    Returns:
        The run's next step.
    """
    async with _lock:
        return await host_session.submit(run_dir, call_id, output)


@mcp.tool
async def pb_render(run_dir: str) -> dict[str, Any]:
    """Render the run's pending image (image API or Codex) and advance the run.

    Can take several minutes with the Codex backend.

    Args:
        run_dir: The run directory from the step.

    Returns:
        The run's next step.
    """
    async with _lock:
        return await host_session.render(run_dir)


@mcp.tool
async def pb_status(run_dir: str) -> dict[str, Any]:
    """Show what a run is waiting for (use after an interruption).

    Args:
        run_dir: The run directory.

    Returns:
        The run's next step, or its final result when done.
    """
    async with _lock:
        return await host_session.get_status(run_dir)


def main() -> None:
    """MCP server entry point."""
    # stdout carries the MCP protocol; structlog prints to stdout by default.
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )
    _load_dotenv_best_effort()
    mcp.run()


if __name__ == "__main__":
    main()
