---
name: paperbanana
description: Generate a publication-quality methodology diagram or statistical plot with PaperBanana, with Codex sub-agents doing every LLM step (no LLM API key). The final image is rendered by Codex image generation or by the image API. Use when the user wants a paper figure, method diagram or plot from methodology text or a data file.
---

# PaperBanana (host-delegated)

The `paperbanana-host` MCP server runs PaperBanana's own pipeline (reference retrieval, prompt assembly, response parsing, aspect ratio, critic loop) and pauses at every LLM call. You answer each call with a fresh sub-agent; the server renders the image.

## 1. Gather inputs

- **Methodology diagram:** a methodology text file (`.txt`, `.md`, `.pdf`) or pasted text, plus a caption.
- **Statistical plot:** a `.csv` or `.json` data file, plus a caption saying what the plot should show.
- If there is no caption, ask for one.
- **Image backend** (diagrams only): `codex` by default. Use `api` only if the user asks for the image API (it uses `IMAGE_PROVIDER` and its API key from `.env`).
- Optional, only if the user mentions them: `aspect_ratio`, `venue` (neurips, icml, acl, ieee), `iterations` (default 3), `input_images` (sketches), `generate_caption`.

## 2. Start

Call the `pb_start` tool with absolute paths (`source_path` or `data_path`), `caption`, `diagram_type` (`methodology` or `statistical_plot`) and any options. Every tool returns the next step.

## 3. Loop until `status` is `done`

- **`needs_llm`**: spawn one fresh sub-agent with `spawn_agent`, giving it the worker instructions below with `run_dir`, `call_id` and `prompt_file` filled in. Wait for it, close it, then call `pb_status(run_dir)` for the next step. Do not read the prompt file or the images yourself. If sub-agents are unavailable, follow the worker instructions yourself, judging the call only on what the prompt file contains.
- **`needs_render`**: call `pb_render(run_dir)`. With Codex this takes a few minutes.
- **`failed`**: stop and report `error`.

Tell the user in one short line what just finished and what is next.

### Worker instructions

```
You answer exactly one PaperBanana pipeline call.
run_dir: <run_dir>
call_id: <call_id>
prompt_file: <prompt_file>

1. Read prompt_file in full (it can be long; read it in chunks if needed, skipping nothing).
2. If it lists attached images, open every image with view_image in the listed order.
   "Image 1" or "reference image 1" in the prompt is the first file.
3. Do the task in the "# Prompt" section yourself, following it and the "# Output format"
   section exactly. Code parses your answer: JSON calls get raw JSON only (no fences, no
   commentary); text calls get only the requested content, complete, keeping marker lines
   such as "RECOMMENDED_RATIO: 16:9".
4. Call the pb_submit tool with run_dir, call_id and your complete answer as output.
   If it returns an error (for example invalid JSON), fix the answer and submit again.
5. Reply with one line: "submitted <call_id>", or the error if you could not submit.
Do not read other files, run commands, or do anything beyond this one call.
```

## 4. Finish

Show `final_image` to the user, then report:
- the final image path;
- how many rounds ran and the last critic summary;
- `run_dir` (every prompt and answer is under `run_dir/host/calls/`);
- the generated caption, if one was requested.
