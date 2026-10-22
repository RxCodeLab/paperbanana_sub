"""Host-delegated generation runs.

The MCP host (Claude Code or Codex) answers every LLM call with a subagent,
while PaperBanana keeps all of its own glue: prompt assembly, reference
images, response parsing, aspect-ratio choice and the critic loop.

Each step re-runs the real ``PaperBananaPipeline.generate()`` from the start
with two replay providers. LLM calls that already have an answer get it back;
the first call without one stops the run by raising :class:`PendingCall`, and
that call is written to disk for the host to pick up. Image renders pause the
same way, but the server performs them itself (image API or Codex) when the
host calls :func:`render`. Calls are keyed by their order in the run, which
is stable because the pipeline is deterministic given the same answers.

All state lives in ``<run_dir>/host/session.json``; nothing is kept in memory
between steps.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional

import structlog
from PIL import Image

import paperbanana.core.pipeline as pipeline_module
from paperbanana.agents.base import BaseAgent
from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.plot_data import load_statistical_plot_payload
from paperbanana.core.source_loader import load_methodology_source
from paperbanana.core.types import DiagramType, GenerationInput, GenerationOutput
from paperbanana.core.utils import extract_json, generate_run_id, save_image
from paperbanana.providers.base import ImageGenProvider, VLMProvider
from paperbanana.providers.registry import ProviderRegistry

logger = structlog.get_logger()

SESSION_VERSION = 1

# Longest edge of the image copies handed to host subagents.
_VIEW_MAX_EDGE = 2000


class PendingCall(BaseException):
    """Raised inside the pipeline when a call has to be answered by the host.

    Derives from ``BaseException`` so the pipeline's ``except Exception``
    fallbacks and retry wrappers let it through instead of swallowing it.
    """

    def __init__(
        self,
        kind: str,
        call_id: str,
        payload: dict[str, Any],
        images: list[Image.Image],
    ):
        super().__init__(f"{kind} call {call_id} needs the host")
        self.kind = kind
        self.call_id = call_id
        self.payload = payload
        self.images = images


# ── Replay providers ─────────────────────────────────────────────────


def _image_digest(img: Image.Image) -> str:
    h = hashlib.sha256(f"{img.mode}{img.size}".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def _fingerprint(payload: Any, images: list[Image.Image]) -> str:
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode())
    for img in images:
        h.update(_image_digest(img).encode())
    return h.hexdigest()[:16]


def _calling_agent() -> str:
    """Name of the agent whose ``run()`` is on the await chain, if any."""
    frame = inspect.currentframe()
    try:
        while frame is not None:
            candidate = frame.f_locals.get("self")
            if isinstance(candidate, BaseAgent):
                return candidate.agent_name
            frame = frame.f_back
    finally:
        del frame
    return "unknown"


class ReplayVLM(VLMProvider):
    """Returns host-submitted answers in call order; pauses at the first gap."""

    def __init__(self, session: dict[str, Any]):
        self._answers: dict[str, Any] = session["answers"]
        self._seq = 0

    @property
    def name(self) -> str:
        return "host"

    @property
    def model_name(self) -> str:
        return "host-delegated"

    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        self._seq += 1
        call_id = f"llm-{self._seq:03d}"
        images = list(images or [])
        fingerprint = _fingerprint([system_prompt, prompt, response_format], images)

        answer = self._answers.get(call_id)
        if answer is not None:
            if answer.get("fingerprint") != fingerprint:
                logger.warning("Replaying an answer whose prompt has changed", call_id=call_id)
            return answer["text"]

        raise PendingCall(
            "llm",
            call_id,
            {
                "agent": _calling_agent(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response_format": response_format,
                "fingerprint": fingerprint,
            },
            images,
        )


class ReplayImageGen(ImageGenProvider):
    """Returns finished renders in call order; pauses at the first gap."""

    def __init__(self, session: dict[str, Any]):
        self._image = session["image"]
        self._renders: dict[str, Any] = session["renders"]
        self._seq = 0

    @property
    def name(self) -> str:
        return self._image["provider"]

    @property
    def model_name(self) -> str:
        return self._image["model"]

    @property
    def supported_ratios(self) -> list[str]:
        return list(self._image["supported_ratios"])

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        quality: Optional[str] = None,
        images: Optional[list[Image.Image]] = None,
    ) -> Image.Image:
        self._seq += 1
        call_id = f"render-{self._seq:03d}"
        images = list(images or [])
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
        }
        fingerprint = _fingerprint(params, images)

        done = self._renders.get(call_id)
        if done is not None:
            if done.get("fingerprint") != fingerprint:
                logger.warning("Replaying a render whose prompt has changed", call_id=call_id)
            with Image.open(done["path"]) as img:
                return img.copy()

        raise PendingCall("render", call_id, {"params": params, "fingerprint": fingerprint}, images)


# ── Session storage ──────────────────────────────────────────────────


def _session_path(run_dir: Path) -> Path:
    return run_dir / "host" / "session.json"


def _load(run_dir: Path) -> dict[str, Any]:
    path = _session_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(f"No host session found in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(run_dir: Path, session: dict[str, Any]) -> None:
    path = _session_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _save_image_copy(folder: Path, img: Image.Image, max_edge: Optional[int]) -> Path:
    """Write *img* once per content hash and return its path."""
    path = folder / f"{_image_digest(img)[:16]}.png"
    if not path.exists():
        copy = img.copy()
        if max_edge:
            copy.thumbnail((max_edge, max_edge))
        if copy.mode not in ("RGB", "RGBA", "L"):
            copy = copy.convert("RGB")
        folder.mkdir(parents=True, exist_ok=True)
        copy.save(path, format="PNG")
    return path


def _prompt_file_text(call: PendingCall, image_files: list[str]) -> str:
    payload = call.payload
    lines = [
        f"<!-- PaperBanana host call {call.call_id} ({payload['agent']}). "
        "Your reply is parsed by code: answer exactly as the prompt asks. -->",
        "",
        "# Output format",
        "",
    ]
    if payload.get("response_format") == "json":
        lines.append("Respond with valid JSON only: no markdown fences, no commentary.")
    else:
        lines.append("Respond with the requested content only: no preamble, no sign-off.")
    if image_files:
        lines += [
            "",
            "# Attached images",
            "",
            "Look at every image, in this order, before answering (Read tool in Claude "
            'Code, view_image in Codex). "Image 1" or "reference image 1" in the prompt '
            "is the first file.",
            "",
        ]
        lines += [f"{i}. {path}" for i, path in enumerate(image_files, start=1)]
    if payload.get("system_prompt"):
        lines += ["", "# System instructions", "", payload["system_prompt"]]
    lines += ["", "# Prompt", "", payload["prompt"]]
    return "\n".join(lines) + "\n"


def _materialize(run_dir: Path, call: PendingCall) -> dict[str, Any]:
    """Write a pending call's prompt and images to disk; return its session record."""
    host_dir = run_dir / "host"
    pending: dict[str, Any] = {
        "call_id": call.call_id,
        "kind": call.kind,
        "fingerprint": call.payload["fingerprint"],
    }
    if call.kind == "llm":
        image_files = [
            str(_save_image_copy(host_dir / "images", img, _VIEW_MAX_EDGE)) for img in call.images
        ]
        agent = call.payload["agent"]
        prompt_file = host_dir / "calls" / f"{call.call_id}_{agent}.prompt.md"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(_prompt_file_text(call, image_files), encoding="utf-8")
        pending.update(
            agent=agent,
            response_format=call.payload["response_format"],
            prompt_file=str(prompt_file),
            image_files=image_files,
        )
    else:
        # Renders get full-resolution inputs: they are edited, not just viewed.
        pending.update(
            params=call.payload["params"],
            image_files=[
                str(_save_image_copy(host_dir / "images", img, None)) for img in call.images
            ],
        )
    return pending


def _result(run_dir: Path, output: GenerationOutput) -> dict[str, Any]:
    return {
        "final_image": output.image_path,
        "iterations": [
            {
                "iteration": rec.iteration,
                "image": rec.image_path,
                "critic_summary": rec.critique.summary if rec.critique else None,
            }
            for rec in output.iterations
        ],
        "generated_caption": output.generated_caption,
        "metadata_file": str(run_dir / "metadata.json"),
    }


def _descriptor(session: dict[str, Any]) -> dict[str, Any]:
    """What the host should do next, as returned by every tool."""
    status = session["status"]
    out: dict[str, Any] = {
        "run_dir": session["run_dir"],
        "status": status,
        "llm_calls_answered": len(session["answers"]),
        "renders_done": len(session["renders"]),
    }
    pending = session.get("pending")
    if status == "needs_llm" and pending:
        out.update(
            call_id=pending["call_id"],
            agent=pending["agent"],
            prompt_file=pending["prompt_file"],
            image_files=pending["image_files"],
            response_format=pending["response_format"] or "text",
            next=(
                "Hand this call to one fresh worker subagent: it reads prompt_file and every "
                "image file, then submits its answer with pb_submit(run_dir, call_id, output)."
            ),
        )
    elif status == "needs_render" and pending:
        params = pending["params"]
        out.update(
            call_id=pending["call_id"],
            image_provider=session["image"]["provider"],
            aspect_ratio=params.get("aspect_ratio"),
            next="Call pb_render(run_dir). Rendering can take a few minutes.",
        )
    elif status == "done":
        out.update(session["result"])
    elif status == "failed":
        out["error"] = session.get("error")
    return out


# ── Driving the pipeline ─────────────────────────────────────────────


class _PlotCodeReplay:
    """Runs each generated plot script once and replays its files afterwards.

    Re-executing on every replay could redraw a plot (unseeded randomness,
    sampling) after the critic has already reviewed it.
    """

    def __init__(self, session: dict[str, Any], run_dir: Path, visualizer: Any):
        self._plots: dict[str, Any] = session.setdefault("plots", {})
        self._store = run_dir / "host" / "plots"
        self._visualizer = visualizer
        self._execute = visualizer._execute_plot_code
        self._seq = 0

    def __call__(
        self,
        code: str,
        output_path: str,
        aspect_ratio: Optional[str] = None,
        vector_formats: Optional[list[str]] = None,
    ) -> bool:
        self._seq += 1
        key = f"plot-{self._seq:03d}"
        fingerprint = _fingerprint([code, aspect_ratio, vector_formats], [])

        cached = self._plots.get(key)
        if cached is None:
            success = self._execute(code, output_path, aspect_ratio, vector_formats)
            files: dict[str, str] = {}
            if success:
                produced = {"raster": output_path, **self._visualizer._last_vector_paths}
                self._store.mkdir(parents=True, exist_ok=True)
                for kind, path in produced.items():
                    stored = self._store / f"{key}{Path(path).suffix}"
                    shutil.copyfile(path, stored)
                    files[kind] = str(stored)
            self._plots[key] = {"success": success, "files": files, "fingerprint": fingerprint}
            return success

        if cached["fingerprint"] != fingerprint:
            logger.warning("Replaying a plot whose code has changed", key=key)
        vector_paths: dict[str, str] = {}
        for kind, stored in cached["files"].items():
            if kind == "raster":
                target = output_path
            else:
                target = str(Path(output_path).with_suffix(f".{kind}"))
                vector_paths[kind] = target
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stored, target)
        self._visualizer._last_vector_paths = vector_paths
        return bool(cached["success"])


@contextlib.contextmanager
def _fixed_run_id(run_id: str) -> Iterator[None]:
    """Make the pipeline reuse *run_id* instead of minting (and creating) a new run dir."""
    original = pipeline_module.generate_run_id
    pipeline_module.generate_run_id = lambda: run_id
    try:
        yield
    finally:
        pipeline_module.generate_run_id = original


async def advance(run_dir: str | Path) -> dict[str, Any]:
    """Replay the run up to its next unanswered call (or to completion)."""
    run_dir = Path(run_dir).resolve()
    session = _load(run_dir)
    if session["status"] in ("done", "failed") or session.get("pending"):
        return _descriptor(session)

    settings = Settings(**session["settings"])
    with _fixed_run_id(session["run_id"]):
        pipeline = PaperBananaPipeline(
            settings=settings,
            vlm_client=ReplayVLM(session),
            image_gen_fn=ReplayImageGen(session),
        )
    pipeline.visualizer._execute_plot_code = _PlotCodeReplay(session, run_dir, pipeline.visualizer)
    gen_input = GenerationInput.model_validate(session["input"])

    try:
        output = await pipeline.generate(gen_input)
    except PendingCall as call:
        session["pending"] = _materialize(run_dir, call)
        session["status"] = "needs_llm" if call.kind == "llm" else "needs_render"
    except Exception as e:
        session["status"] = "failed"
        session["error"] = f"{type(e).__name__}: {e}"
        _save(run_dir, session)
        raise
    else:
        session["status"] = "done"
        session["result"] = _result(run_dir, output)
    _save(run_dir, session)
    return _descriptor(session)


async def get_status(run_dir: str | Path) -> dict[str, Any]:
    """Return the next action for a run, advancing it if it is mid-step."""
    return await advance(run_dir)


async def submit(run_dir: str | Path, call_id: str, output: str) -> dict[str, Any]:
    """Record the host's answer to the pending LLM call and advance the run."""
    run_dir = Path(run_dir).resolve()
    session = _load(run_dir)
    pending = session.get("pending")
    if not pending or pending["kind"] != "llm":
        raise ValueError(f"No LLM call is waiting in this run (status: {session['status']}).")
    if pending["call_id"] != call_id:
        raise ValueError(f"This run is waiting for {pending['call_id']}, not {call_id}.")

    text = output.strip()
    if not text:
        raise ValueError("The answer is empty.")
    if pending.get("response_format") == "json" and extract_json(text) is None:
        raise ValueError(
            "This call needs a JSON answer and the submitted text does not parse as JSON. "
            "Fix it and submit again."
        )

    answer_file = Path(pending["prompt_file"]).with_name(f"{call_id}_{pending['agent']}.answer.md")
    answer_file.write_text(text + "\n", encoding="utf-8")
    session["answers"][call_id] = {
        "text": text,
        "agent": pending["agent"],
        "fingerprint": pending["fingerprint"],
    }
    session["pending"] = None
    session["status"] = "running"
    _save(run_dir, session)
    return await advance(run_dir)


def _create_image_provider(provider: str) -> ImageGenProvider:
    return ProviderRegistry.create_image_gen(Settings(image_provider=provider))


async def render(
    run_dir: str | Path,
    image_gen: Optional[ImageGenProvider] = None,
) -> dict[str, Any]:
    """Perform the pending render with the run's image provider and advance the run."""
    run_dir = Path(run_dir).resolve()
    session = _load(run_dir)
    pending = session.get("pending")
    if not pending or pending["kind"] != "render":
        raise ValueError(f"No render is waiting in this run (status: {session['status']}).")

    provider = image_gen or _create_image_provider(session["image"]["provider"])
    kwargs = dict(pending["params"])
    if pending["image_files"]:
        inputs = []
        for path in pending["image_files"]:
            with Image.open(path) as img:
                inputs.append(img.copy())
        kwargs["images"] = inputs
    image = await provider.generate(**kwargs)

    call_id = pending["call_id"]
    out_path = run_dir / "host" / "renders" / f"{call_id}.png"
    save_image(image, out_path)
    session["renders"][call_id] = {"path": str(out_path), "fingerprint": pending["fingerprint"]}
    session["pending"] = None
    session["status"] = "running"
    _save(run_dir, session)
    return await advance(run_dir)


async def start_run(
    *,
    caption: str,
    source_context: Optional[str] = None,
    source_path: Optional[str] = None,
    diagram_type: str = "methodology",
    data_path: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    venue: Optional[str] = None,
    iterations: int = 3,
    auto_refine: bool = False,
    optimize: bool = False,
    generate_caption: bool = False,
    image_provider: str = "codex_imagen",
    input_images: Optional[list[str]] = None,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """Create a host-delegated run and advance it to its first host call."""
    dtype = DiagramType(diagram_type)
    raw_data = None
    if dtype is DiagramType.METHODOLOGY:
        if source_path:
            source_context = load_methodology_source(Path(source_path).expanduser())
        if not source_context or not source_context.strip():
            raise ValueError("Provide the methodology text (source_text) or a source_path.")
    else:
        if not data_path:
            raise ValueError("A statistical plot needs data_path (.csv or .json).")
        source_context, payload = load_statistical_plot_payload(Path(data_path).expanduser())
        raw_data = {"data": payload}

    images: list[str] = []
    for path in input_images or []:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        images.append(str(resolved))

    if dtype is DiagramType.METHODOLOGY:
        # Built now so missing API keys or a missing codex CLI fail before any LLM work.
        provider = _create_image_provider(image_provider)
        image = {
            "provider": provider.name,
            "model": provider.model_name,
            "supported_ratios": list(provider.supported_ratios),
        }
    else:
        # Plots are drawn by executing generated matplotlib code; nothing is rendered.
        provider = _create_image_provider("none")
        image = {
            "provider": "none",
            "model": "none",
            "supported_ratios": list(provider.supported_ratios),
        }

    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption,
        diagram_type=dtype,
        raw_data=raw_data,
        aspect_ratio=aspect_ratio,
        input_images=images,
    )

    settings: dict[str, Any] = {
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "refinement_iterations": iterations,
        "auto_refine": auto_refine,
        "optimize_inputs": optimize,
        "generate_caption": generate_caption,
        # The host keeps its own copy of every prompt under host/calls/, and the
        # pipeline's recorder would add a duplicate on every replay.
        "save_prompts": False,
    }
    if venue:
        settings["venue"] = venue

    run_id = generate_run_id()
    run_dir = Path(settings["output_dir"]) / run_id
    session = {
        "version": SESSION_VERSION,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "input": gen_input.model_dump(mode="json"),
        "image": image,
        "answers": {},
        "renders": {},
        "pending": None,
        "status": "running",
    }
    _save(run_dir, session)
    return await advance(run_dir)
