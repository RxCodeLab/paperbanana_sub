"""Tests for host-delegated runs (paperbanana.host.session)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.host import session as hs


class FakeImageGen:
    name = "fake_imagen"
    model_name = "fake"
    supported_ratios = ["1:1", "16:9"]

    def __init__(self):
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return Image.new("RGB", (kwargs["width"] // 16, kwargs["height"] // 16), (200, 220, 255))


@pytest.fixture
def fake_gen(monkeypatch):
    gen = FakeImageGen()
    monkeypatch.setattr(hs, "_create_image_provider", lambda name: gen)
    return gen


@pytest.fixture(autouse=True)
def ref_set(monkeypatch, tmp_path):
    """A tiny reference set with real image files (keeps the user's cache out of tests)."""
    ref_dir = tmp_path / "refs"
    (ref_dir / "images").mkdir(parents=True)
    examples = []
    for i in range(1, 13):
        Image.new("RGB", (64, 32), (i * 20, 100, 150)).save(ref_dir / "images" / f"r{i}.png")
        examples.append(
            {
                "id": f"r{i}",
                "source_context": f"Method {i} text.",
                "caption": f"Caption {i}",
                "image_path": f"images/r{i}.png",
                "category": "test",
                "aspect_ratio": 2.0,
            }
        )
    (ref_dir / "index.json").write_text(json.dumps({"metadata": {}, "examples": examples}))
    monkeypatch.setenv("REFERENCE_SET_PATH", str(ref_dir))
    return ref_dir


async def _drive(step: dict, answer) -> tuple[dict, list[str]]:
    trace = []
    while step["status"] != "done":
        if step["status"] == "needs_llm":
            trace.append(step["agent"])
            step = await hs.submit(step["run_dir"], step["call_id"], answer(step))
        elif step["status"] == "needs_render":
            trace.append("render")
            step = await hs.render(step["run_dir"])
        else:
            raise AssertionError(step)
    return step, trace


def _diagram_answers(step: dict) -> str:
    agent = step["agent"]
    if agent == "retriever":
        return json.dumps({"selected_ids": ["r2", "r1"]})
    if agent == "planner":
        return "Encoder -> Latent -> Decoder, left to right.\nRECOMMENDED_RATIO: 21:9"
    if agent == "stylist":
        return "Styled description v1"
    if agent == "critic":
        if step["llm_calls_answered"] == 3:
            return json.dumps(
                {"critic_suggestions": ["Label the arrows"], "revised_description": "v2"}
            )
        return json.dumps({"critic_suggestions": [], "revised_description": None})
    raise AssertionError(f"unexpected agent {agent}")


async def test_methodology_run_end_to_end(tmp_path, fake_gen):
    step = await hs.start_run(
        caption="Overview",
        source_context="We encode x into z and decode it back.",
        output_dir=str(tmp_path / "out"),
        image_provider="fake_imagen",
    )

    assert step["status"] == "needs_llm" and step["agent"] == "retriever"
    prompt_text = Path(step["prompt_file"]).read_text()
    assert "Respond with valid JSON only" in prompt_text
    assert "r3" in prompt_text

    final, trace = await _drive(step, _diagram_answers)

    assert trace == ["retriever", "planner", "stylist", "render", "critic", "render", "critic"]
    assert Path(final["final_image"]).is_file()
    assert [it["iteration"] for it in final["iterations"]] == [1, 2]
    # The planner's ratio reached the renderer, and the critic's revision was rendered.
    assert fake_gen.calls[0]["aspect_ratio"] == "21:9"
    assert "v2" in fake_gen.calls[1]["prompt"]
    metadata = json.loads(Path(final["metadata_file"]).read_text())
    assert metadata["vlm_provider"] == "host"


async def test_planner_gets_selected_reference_images_in_order(tmp_path, fake_gen):
    step = await hs.start_run(
        caption="Overview",
        source_context="Method text.",
        output_dir=str(tmp_path / "out"),
        image_provider="fake_imagen",
    )

    step = await hs.submit(step["run_dir"], step["call_id"], _diagram_answers(step))
    assert step["agent"] == "planner"
    assert len(step["image_files"]) == 2
    with Image.open(step["image_files"][0]) as first:
        assert first.getpixel((0, 0)) == (40, 100, 150)  # r2 was selected first
    prompt_text = Path(step["prompt_file"]).read_text()
    assert prompt_text.index(step["image_files"][0]) < prompt_text.index(step["image_files"][1])


async def test_submit_rejects_wrong_call_and_bad_json(tmp_path, fake_gen):
    step = await hs.start_run(
        caption="Overview",
        source_context="Method text.",
        output_dir=str(tmp_path / "out"),
        image_provider="fake_imagen",
    )
    assert step["agent"] == "retriever"
    with pytest.raises(ValueError, match="waiting for llm-001"):
        await hs.submit(step["run_dir"], "llm-009", "{}")
    with pytest.raises(ValueError, match="JSON"):
        await hs.submit(step["run_dir"], step["call_id"], "I pick the first three.")
    with pytest.raises(ValueError, match="No render"):
        await hs.render(step["run_dir"])
    # A rejected answer leaves the call pending.
    status = await hs.get_status(step["run_dir"])
    assert status["status"] == "needs_llm" and status["call_id"] == "llm-001"


async def test_statistical_plot_runs_generated_code(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n2,4\n3,9\n")
    code = (
        "```python\n"
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3], [2, 4, 9])\nplt.savefig(OUTPUT_PATH)\n"
        "```"
    )

    def answer(step):
        if step["agent"] == "retriever":
            return json.dumps({"selected_ids": []})
        if step["agent"] in ("planner", "stylist"):
            return "Line plot of y against x."
        if step["agent"] == "visualizer":
            return code
        if step["agent"] == "critic":
            return json.dumps({"critic_suggestions": [], "revised_description": None})
        raise AssertionError(step["agent"])

    step = await hs.start_run(
        caption="y grows with x",
        diagram_type="statistical_plot",
        data_path=str(data),
        output_dir=str(tmp_path / "out"),
        image_provider="none",
    )
    final, trace = await _drive(step, answer)
    assert "render" not in trace
    assert trace[-2:] == ["visualizer", "critic"]
    with Image.open(final["final_image"]) as img:
        assert img.size[0] > 100


async def test_plot_code_runs_once_and_final_matches_reviewed_image(tmp_path):
    """Replays must reuse the reviewed plot, not redraw it with fresh randomness."""
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n2,4\n")
    counter = tmp_path / "executions.txt"
    code = (
        "```python\n"
        "import random\nimport matplotlib\nmatplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        f"open({str(counter)!r}, 'a').write('x')\n"
        "plt.scatter([random.random() for _ in range(50)], [random.random() for _ in range(50)])\n"
        "plt.savefig(OUTPUT_PATH)\n"
        "```"
    )
    reviewed: list[bytes] = []

    def answer(step):
        if step["agent"] == "retriever":
            return json.dumps({"selected_ids": []})
        if step["agent"] in ("planner", "stylist"):
            return "Scatter plot."
        if step["agent"] == "visualizer":
            return code
        if step["agent"] == "critic":
            with Image.open(step["image_files"][0]) as img:
                reviewed.append(img.convert("RGB").tobytes())
            if len(reviewed) == 1:
                return json.dumps(
                    {"critic_suggestions": ["Add labels"], "revised_description": "v2"}
                )
            return json.dumps({"critic_suggestions": [], "revised_description": None})
        raise AssertionError(step["agent"])

    step = await hs.start_run(
        caption="noise",
        diagram_type="statistical_plot",
        data_path=str(data),
        output_dir=str(tmp_path / "out"),
        image_provider="none",
    )
    final, trace = await _drive(step, answer)

    assert trace.count("visualizer") == 2
    assert counter.read_text() == "xx"  # one execution per round, despite many replays
    with Image.open(final["final_image"]) as img:
        assert img.convert("RGB").tobytes() == reviewed[-1]


async def test_start_validates_inputs(tmp_path, fake_gen):
    with pytest.raises(ValueError, match="methodology text"):
        await hs.start_run(caption="c", output_dir=str(tmp_path), image_provider="fake")
    with pytest.raises(ValueError, match="data_path"):
        await hs.start_run(caption="c", diagram_type="statistical_plot", output_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        await hs.start_run(
            caption="c",
            source_context="text",
            input_images=[str(tmp_path / "missing.png")],
            output_dir=str(tmp_path),
            image_provider="fake",
        )


async def test_pending_call_escapes_pipeline_fallbacks(tmp_path, fake_gen):
    """The stylist and critic fall back on Exception; PendingCall must not be swallowed."""
    step = await hs.start_run(
        caption="Overview",
        source_context="Method text.",
        output_dir=str(tmp_path / "out"),
        image_provider="fake_imagen",
    )
    step = await hs.submit(step["run_dir"], step["call_id"], json.dumps({"selected_ids": []}))
    step = await hs.submit(step["run_dir"], step["call_id"], "Plan")
    assert step["agent"] == "stylist"
    step = await hs.submit(step["run_dir"], step["call_id"], "Styled")
    assert step["status"] == "needs_render"
    step = await hs.render(step["run_dir"])
    assert step["agent"] == "critic"
    assert len(step["image_files"]) == 1
