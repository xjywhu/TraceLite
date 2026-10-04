import copy
from fractions import Fraction

import pytest
from PIL import Image

from swe_memory_policy.offline import render_history, select_observations
from swe_memory_policy.utils import sha256_file


def history(count=5):
    items = []
    for index in range(count):
        items.extend(
            [
                {
                    "type": "function_call",
                    "call_id": f"call_{index}",
                    "name": "bash",
                    "arguments": '{"command":"echo hello"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call_{index}",
                    "output": "<returncode>0</returncode>\n<output>hello</output>",
                },
            ]
        )
    return {"input": items}


def test_recent_partition_preserves_input():
    value = history()
    before = copy.deepcopy(value)
    observations, metadata = select_observations(value)
    assert [o.observation_index for o in observations] == [1, 2]
    assert metadata["text_oa_indexes"] == [3, 4, 5]
    assert value == before
    assert select_observations(history(3))[0] == []


def test_parallel_tools_share_window():
    value = {
        "input": [
            {"type": "function_call", "call_id": "a"},
            {"type": "function_call", "call_id": "b"},
            {"type": "function_call_output", "call_id": "b", "output": "B"},
            {"type": "function_call_output", "call_id": "a", "output": "A"},
        ]
    }
    assert len(select_observations(value, recent=0)[0]) == 2
    assert select_observations(value, recent=1)[0] == []


def test_invalid_linkage_and_output_directory(tmp_path):
    value = history()
    value["input"].append({"type": "function_call", "call_id": "unfinished"})
    with pytest.raises(ValueError):
        select_observations(value)
    with pytest.raises(FileExistsError):
        render_history("hello", tmp_path)
    with pytest.raises(ValueError):
        render_history("hello", tmp_path / "unused", factors=(Fraction(0),))
    assert not (tmp_path / "unused").exists()


def test_trajectory_terminal_batch_is_reported():
    items = history(2)["input"]
    value = {
        "messages": [
            {"object": "response", "output": [items[0]]},
            items[1],
            {"object": "response", "output": [items[2]]},
            items[3],
            {
                "object": "response",
                "output": [{"type": "function_call", "call_id": "submit"}],
            },
        ]
    }
    observations, metadata = select_observations(value, recent=0)
    assert len(observations) == 2
    assert metadata["ignored_terminal_call_ids"] == ["submit"]


def test_all_factors_start_from_unchanged_base(tmp_path):
    output = tmp_path / "render"
    result = render_history(
        "hello world " * 100,
        output,
        factors=tuple(Fraction(k) for k in range(1, 11)),
    )
    images = result["observations"][0]["images"]
    assert len(images) == 10
    assert images[0]["sha256"] == images[0]["base_sha256"]
    assert len({image["base_sha256"] for image in images}) == 1
    assert result["status"] == "complete"
    for index, image in enumerate(images, 1):
        assert image["status"] == "rendered"
        assert (
            image["estimated_visual_tokens"] <= image["baseline_visual_tokens"] // index
        )
        assert sha256_file(output / image["path"]) == image["sha256"]
    assert result["visual_token_estimator"] == "luna_patch32_high_20260906_v2"
    assert "scale_mode" not in result


def test_unattainable_factor_keeps_base_and_reports_partial(tmp_path, monkeypatch):
    def tiny_base(_observation, output_dir):
        output_dir.mkdir()
        page = output_dir / "tiny.png"
        Image.new("RGB", (32, 32), "white").save(page)
        return [page], {"classification": "test"}

    monkeypatch.setattr("swe_memory_policy.offline.render_base", tiny_base)
    output = tmp_path / "render"
    result = render_history("hello", output, factors=(Fraction(1), Fraction(2)))
    base, failed = result["observations"][0]["images"]
    assert result["status"] == "partial"
    assert base["status"] == "rendered"
    assert failed["status"] == "unattainable_budget"
    assert failed["token_budget"] == 1
    assert failed["minimum_visual_tokens"] == 2
    assert failed["budget_satisfied"] is False
    assert failed["path"] is None
    assert (output / failed["base_path"]).is_file()
    assert not list((output / "token_2p1").iterdir())


@pytest.mark.parametrize("condition", [{"model": "unknown"}, {"detail": "auto"}])
def test_unsupported_condition_rejected_before_output(tmp_path, condition):
    output = tmp_path / "unused"
    with pytest.raises(ValueError):
        render_history("hello", output, **condition)
    assert not output.exists()


def test_removed_linear_api_rejected(tmp_path):
    with pytest.raises(TypeError):
        render_history("hello", tmp_path / "unused", scale_mode="linear")
    assert not (tmp_path / "unused").exists()


def test_duplicate_factors_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        render_history("hello", tmp_path / "unused", factors=(1, Fraction(1)))
    assert not (tmp_path / "unused").exists()


def test_cli_reports_partial_failure(tmp_path, monkeypatch, capsys):
    from swe_memory_policy.offline import main

    source = tmp_path / "observation.txt"
    source.write_text("hello", encoding="utf-8")
    output = tmp_path / "render"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tracelite-render",
            str(source),
            "--output-dir",
            str(output),
            "--factors",
            "1",
            "100000",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert (output / "manifest.json").exists()
    assert "unattainable" in capsys.readouterr().err


def test_cli_default_factors(tmp_path, monkeypatch):
    import json

    from swe_memory_policy.offline import main

    source = tmp_path / "observation.txt"
    source.write_text("hello", encoding="utf-8")
    output = tmp_path / "render"
    monkeypatch.setattr(
        "sys.argv",
        ["tracelite-render", str(source), "--output-dir", str(output)],
    )
    main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [row["factor"] for row in manifest["observations"][0]["images"]] == [
        "4",
    ]
    assert manifest["status"] == "complete"
