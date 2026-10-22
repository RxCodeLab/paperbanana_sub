"""Tests for the host-delegated MCP server tools."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from PIL import Image

from mcp_server import host_server
from paperbanana.host import session as hs


class FakeImageGen:
    name = "codex_imagen"
    model_name = "fake"
    supported_ratios = ["16:9"]

    async def generate(self, **kwargs):
        return Image.new("RGB", (64, 36), (255, 255, 255))


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    monkeypatch.setattr(hs, "_create_image_provider", lambda name: FakeImageGen())


async def test_tools_listed():
    async with Client(host_server.mcp) as client:
        names = {tool.name for tool in await client.list_tools()}
    assert names == {"pb_start", "pb_submit", "pb_render", "pb_status"}


async def test_start_submit_render_flow(tmp_path):
    async with Client(host_server.mcp) as client:
        step = (
            await client.call_tool(
                "pb_start",
                {
                    "caption": "Overview",
                    "source_text": "We encode and decode.",
                    "output_dir": str(tmp_path),
                    "image_backend": "codex",
                },
            )
        ).data
        assert step["status"] == "needs_llm"
        run_dir = step["run_dir"]

        # Answer until the first render, then render once.
        while step["status"] == "needs_llm":
            answer = (
                json.dumps({"selected_ids": []}) if step["response_format"] == "json" else "Plan"
            )
            step = (
                await client.call_tool(
                    "pb_submit",
                    {"run_dir": run_dir, "call_id": step["call_id"], "output": answer},
                )
            ).data
        assert step["status"] == "needs_render"
        step = (await client.call_tool("pb_render", {"run_dir": run_dir})).data
        assert step["status"] == "needs_llm" and step["agent"] == "critic"

        status = (await client.call_tool("pb_status", {"run_dir": run_dir})).data
        assert status["call_id"] == step["call_id"]


async def test_bad_submission_is_a_tool_error(tmp_path):
    async with Client(host_server.mcp) as client:
        step = (
            await client.call_tool(
                "pb_start",
                {"caption": "c", "source_text": "t", "output_dir": str(tmp_path)},
            )
        ).data
        with pytest.raises(ToolError, match="JSON"):
            await client.call_tool(
                "pb_submit",
                {"run_dir": step["run_dir"], "call_id": step["call_id"], "output": "not json"},
            )
