# Host-delegated mode (Claude Code / Codex, no LLM API key)

In host-delegated mode, the agent you are already running (Claude Code or Codex) does every LLM step of PaperBanana with its own subagents. PaperBanana still runs its full pipeline: reference retrieval, prompt assembly, reference images, response parsing, aspect-ratio choice and the critic loop. The final image is rendered by **Codex image generation** (no key) or by the **image API** you already use.

```
Claude Code / Codex  ──pb_start──►  paperbanana-host MCP server
        ▲                              └─ PaperBananaPipeline.generate(), paused at each LLM call
        │  needs_llm: call_id, prompt_file, image_files
        ├─► worker subagent: reads the prompt + images → pb_submit(answer)
        │  needs_render
        ├─► pb_render  →  Codex image generation or image API
        └─ ... until status = done → final_image
```

Each step replays the pipeline from the start with the answers submitted so far, and pauses at the first call without one. All state is in `<run_dir>/host/session.json`, so a run can be resumed with `pb_status` after an interruption. Every prompt and answer is saved under `<run_dir>/host/calls/`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[mcp]"
codex login          # only for the Codex image backend
```

## Claude Code

Already wired in this repo:

- `.mcp.json`: the `paperbanana-host` MCP server (`.venv/bin/paperbanana-host-mcp`). Approve it when Claude Code asks.
- `.claude/skills/paperbanana/`: the orchestrating skill.
- `.claude/agents/paperbanana-worker.md`: the worker subagent (one per LLM call).

Then, from the repo root:

```
/paperbanana examples/sample_inputs/transformer_method.txt "Overview of the Transformer architecture"
```

Workers use the session's model (`model: inherit`). Change it in the worker's frontmatter, for example `model: sonnet`.

## Codex

Add the server to `~/.codex/config.toml` (replace the path):

```toml
[mcp_servers.paperbanana-host]
command = "/path/to/paperbanana_sub/.venv/bin/paperbanana-host-mcp"
cwd = "/path/to/paperbanana_sub"
tool_timeout_sec = 1800   # pb_render can take several minutes
default_tools_approval_mode = "approve"   # otherwise every pb_* call asks for approval
```

The skill is in `.agents/skills/paperbanana/`. Run Codex from the repo root and ask for a diagram ("use the paperbanana skill ..."). Codex loads MCP tools on demand, so the first call may start with a tool search.

## Image backend

| `image_backend` | Renders with | Needs |
|---|---|---|
| `codex` (default) | `codex exec` built-in image generation | `codex login` |
| `api` | `IMAGE_PROVIDER` / `IMAGE_MODEL` from `.env` (default `google_imagen`) | that provider's API key |

Set the default with `PAPERBANANA_HOST_IMAGE_BACKEND=api`. `CODEX_IMAGE_MODEL` picks the Codex model (default: Codex's own default).

Statistical plots never use the image backend: the plot is drawn by running the generated matplotlib code locally.

## Limits

- One run at a time per server; the subagent calls within a run are sequential.
- Parameters such as `temperature` and `max_tokens` are not passed on to the host.
- Codex renders follow the requested aspect ratio, but exact pixel sizes are not guaranteed.
- Not supported in this mode: batch runs, `continue_run`, polish mode and the studio. Use the regular CLI with an API provider for those.
