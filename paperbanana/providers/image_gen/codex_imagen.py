"""Codex CLI image generation provider.

Renders images with the built-in image generation tool of the locally
installed ``codex`` CLI, so no image API key is needed — it uses the user's
Codex sign-in. Codex saves every generated image under
``$CODEX_HOME/generated_images/<thread_id>/``; the thread id comes from the
``thread.started`` event that ``codex exec --json`` prints.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class CodexImageGen(ImageGenProvider):
    """Image provider that shells out to ``codex exec``.

    Accepts ``images`` on ``generate()`` (attached with ``--image``), so it
    also works for guided edits such as polish mode.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        codex_home: Optional[str] = None,
        timeout_seconds: float = 900.0,
        ignore_user_config: bool = True,
    ):
        self._model = model
        self._codex_home = Path(
            codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex"
        ).expanduser()
        self._timeout_seconds = timeout_seconds
        # The user's config.toml can register MCP servers and hooks that have
        # no business inside a one-shot render; auth still comes from CODEX_HOME.
        self._ignore_user_config = ignore_user_config

    @property
    def name(self) -> str:
        return "codex_imagen"

    @property
    def model_name(self) -> str:
        return f"codex ({self._model or 'default'})"

    @property
    def supported_ratios(self) -> list[str]:
        # The ratio is requested in the prompt, so any of the standard ones work.
        return ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

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
        with tempfile.TemporaryDirectory(prefix="pb_codex_") as tmp:
            workdir = Path(tmp)
            image_paths: list[Path] = []
            for i, img in enumerate(images or []):
                path = workdir / f"input_{i + 1}.png"
                img.save(path, format="PNG")
                image_paths.append(path)

            full_prompt = self._build_prompt(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                num_input_images=len(image_paths),
            )
            cmd = self._build_command(workdir, image_paths)

            logger.info(
                "Codex image generation",
                prompt_length=len(full_prompt),
                num_input_images=len(image_paths),
                aspect_ratio=aspect_ratio,
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir),
                # Same home for the CLI (auth, image output) and _find_generated_image.
                env={**os.environ, "CODEX_HOME": str(self._codex_home)},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(full_prompt.encode()),
                    timeout=self._timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(
                    f"codex image generation timed out after {self._timeout_seconds:.0f}s"
                )

        thread_id, last_message, errors = self._parse_events(stdout.decode(errors="replace"))
        if proc.returncode != 0 and not thread_id:
            detail = "; ".join(errors) or stderr.decode(errors="replace").strip()
            raise RuntimeError(f"codex exited with code {proc.returncode}: {detail[:500]}")
        if not thread_id:
            raise RuntimeError("codex did not report a thread id; cannot locate the image")

        image_file = self._find_generated_image(thread_id)
        if image_file is None:
            detail = "; ".join(errors) or last_message or "no message"
            raise RuntimeError(
                f"codex finished without generating an image (thread {thread_id}): {detail[:500]}"
            )

        logger.info("Codex image generated", path=str(image_file), thread_id=thread_id)
        with Image.open(image_file) as img:
            return img.copy()

    def _build_command(self, workdir: Path, image_paths: list[Path]) -> list[str]:
        cmd = [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            str(workdir),
        ]
        if self._ignore_user_config:
            cmd.append("--ignore-user-config")
        if self._model:
            cmd += ["-m", self._model]
        # Prompt comes from stdin. --image takes several values, so it goes last
        # to keep it from swallowing the "-" positional.
        cmd.append("-")
        for path in image_paths:
            cmd += ["--image", str(path)]
        return cmd

    @staticmethod
    def _build_prompt(
        prompt: str,
        *,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        aspect_ratio: Optional[str],
        num_input_images: int,
    ) -> str:
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        ratio = aspect_ratio or f"{width}:{height}"
        lines = [
            "Generate exactly one image with your built-in image generation tool.",
            "Do not run shell commands, write code, or create files yourself. "
            "Call the image generation tool once, then reply with just: done",
            f"Target aspect ratio: {ratio} ({orientation}, about {width}x{height} pixels).",
        ]
        if num_input_images:
            lines.append(
                f"The {num_input_images} attached image(s) are the existing figure. Edit it "
                "according to the instructions below and keep everything else unchanged."
            )
        if negative_prompt:
            lines.append(f"Avoid: {negative_prompt}")
        lines += ["", "Image specification:", "", prompt]
        return "\n".join(lines)

    @staticmethod
    def _parse_events(stdout: str) -> tuple[Optional[str], Optional[str], list[str]]:
        """Pull the thread id, last agent message and error messages from JSONL events."""
        thread_id: Optional[str] = None
        last_message: Optional[str] = None
        errors: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "thread.started":
                thread_id = event.get("thread_id") or thread_id
            elif etype == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    last_message = item["text"]
            elif etype in ("error", "turn.failed"):
                err = event.get("error") or {}
                msg = event.get("message") or (err.get("message") if isinstance(err, dict) else err)
                if msg:
                    errors.append(str(msg))
        return thread_id, last_message, errors

    def _find_generated_image(self, thread_id: str) -> Optional[Path]:
        folder = self._codex_home / "generated_images" / thread_id
        if not folder.is_dir():
            return None
        files = [p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES]
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
