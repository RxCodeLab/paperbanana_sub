---
name: paperbanana
description: Generate a publication-quality methodology diagram or statistical plot with PaperBanana, with Claude Code subagents doing every LLM step (no LLM API key). The final image is rendered by Codex or by the image API. Use when the user wants a paper figure, method diagram or plot from methodology text or a data file.
argument-hint: <methodology-file | data-file> [caption]
allowed-tools:
  - mcp__paperbanana-host__pb_start
  - mcp__paperbanana-host__pb_render
  - mcp__paperbanana-host__pb_status
  - Read
---

# PaperBanana (host-delegated)

The `paperbanana-host` MCP server runs PaperBanana's own pipeline (reference retrieval, prompt assembly, response parsing, aspect ratio, critic loop) and pauses at every LLM call. You answer each call with a fresh `paperbanana-worker` subagent; the server renders the image.

## 1. Gather inputs

- **Methodology diagram:** a methodology text file (`.txt`, `.md`, `.pdf`) or pasted text, plus a caption.
- **Statistical plot:** a `.csv` or `.json` data file, plus a caption saying what the plot should show.
- If `$ARGUMENTS` names a file, use it. If there is no caption, ask for one.
- **Image backend** (diagrams only): `codex` by default. Use `api` only if the user asks for the image API (it uses `IMAGE_PROVIDER` and its API key from `.env`).
- Optional, only if the user mentions them: `aspect_ratio`, `venue` (neurips, icml, acl, ieee), `iterations` (default 3), `input_images` (sketches), `generate_caption`.

## 2. Start

Call `pb_start` with absolute paths (`source_path` or `data_path`), `caption`, `diagram_type` (`methodology` or `statistical_plot`) and any options. Every tool returns the next step.

## 3. Loop until `status` is `done`

- **`needs_llm`**: launch one `paperbanana-worker` subagent whose prompt is exactly these three lines:
  ```
  run_dir: <run_dir>
  call_id: <call_id>
  prompt_file: <prompt_file>
  ```
  Do not read the prompt file or the images yourself, and never answer a call yourself. The worker submits its own answer. When it finishes, call `pb_status(run_dir)` for the next step. If the worker could not submit, launch a new worker for the same call.
- **`needs_render`**: call `pb_render(run_dir)`. With Codex this takes a few minutes.
- **`failed`**: stop and report `error`.

Tell the user in one short line what just finished and what is next, e.g. "Planner done, styling next" or "Render 2 of up to 3".

## 4. Finish

Read `final_image` to show it, then report:
- the final image path;
- how many rounds ran and the last critic summary;
- `run_dir` (every prompt and answer is under `run_dir/host/calls/`);
- the generated caption, if one was requested.
