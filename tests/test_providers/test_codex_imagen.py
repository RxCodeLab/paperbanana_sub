"""Tests for the Codex CLI image generation provider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.providers.image_gen.codex_imagen import CodexImageGen
from paperbanana.providers.registry import ProviderRegistry

THREAD_ID = "01a1094b-b5f7-7d91-a242-41c9c04aa8da"


def _events(*events: dict) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


def _mock_proc(stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


def _write_generated(codex_home, thread_id=THREAD_ID, color=(10, 20, 30)):
    folder = codex_home / "generated_images" / thread_id
    folder.mkdir(parents=True)
    Image.new("RGB", (32, 18), color).save(folder / "exec-abc.png")


def test_registry_creates_codex_imagen_when_cli_available():
    with patch("shutil.which", return_value="/usr/local/bin/codex"):
        gen = ProviderRegistry.create_image_gen(Settings(image_provider="codex_imagen"))
    assert isinstance(gen, CodexImageGen)
    assert gen.name == "codex_imagen"


def test_registry_errors_when_codex_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(ValueError, match="codex CLI not found"):
            ProviderRegistry.create_image_gen(Settings(image_provider="codex_imagen"))


async def test_generate_returns_image_from_thread_folder(tmp_path):
    _write_generated(tmp_path)
    proc = _mock_proc(
        _events(
            {"type": "thread.started", "thread_id": THREAD_ID},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        )
    )
    gen = CodexImageGen(codex_home=str(tmp_path), model="gpt-5.5")
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        img = await gen.generate("A diagram", width=2048, height=1152, aspect_ratio="16:9")

    assert img.getpixel((0, 0)) == (10, 20, 30)
    assert spawn.call_args.kwargs["env"]["CODEX_HOME"] == str(tmp_path)
    cmd = spawn.call_args.args
    assert cmd[:2] == ("codex", "exec")
    assert "--json" in cmd and "--ignore-user-config" in cmd
    assert cmd[cmd.index("-m") + 1] == "gpt-5.5"
    assert cmd[-1] == "-"  # prompt via stdin, no images
    prompt = proc.communicate.call_args.args[0].decode()
    assert "16:9 (landscape" in prompt and prompt.rstrip().endswith("A diagram")


async def test_input_images_are_attached_after_stdin_marker(tmp_path):
    _write_generated(tmp_path)
    proc = _mock_proc(_events({"type": "thread.started", "thread_id": THREAD_ID}))
    gen = CodexImageGen(codex_home=str(tmp_path))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        await gen.generate("Make arrows thicker", images=[Image.new("RGB", (8, 8))])

    cmd = list(spawn.call_args.args)
    dash = cmd.index("-")
    assert cmd[dash + 1] == "--image" and cmd[dash + 2].endswith("input_1.png")
    assert "existing figure" in proc.communicate.call_args.args[0].decode()


async def test_missing_image_reports_codex_message(tmp_path):
    proc = _mock_proc(
        _events(
            {"type": "thread.started", "thread_id": THREAD_ID},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Image generation is unavailable."},
            },
        )
    )
    gen = CodexImageGen(codex_home=str(tmp_path))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="unavailable"):
            await gen.generate("A diagram")


async def test_nonzero_exit_without_thread_raises(tmp_path):
    proc = _mock_proc(
        _events({"type": "error", "message": "Not logged in"}),
        returncode=1,
    )
    gen = CodexImageGen(codex_home=str(tmp_path))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="Not logged in"):
            await gen.generate("A diagram")
