from __future__ import annotations

import pytest

from swe_memory_policy.image_response_only_observation import (
    build_responses_observation_image_block,
    render_responses_image_response_only_observation,
)


def test_responses_output_has_an_independent_lossless_source() -> None:
    content = "<returncode>0</returncode>\n<output>result</output>"
    block = build_responses_observation_image_block(
        sequence_index=1,
        oa_index=6,
        tool_index=2,
        tool_call_id="call-6-b",
        content=content,
    )
    assert block.stem == "OA0006_Tool002"
    assert block.page_label == "OA 6 - TOOL MESSAGE 2"
    assert block.tool_call_id == "call-6-b"
    assert content in block.source_text
    assert "role: tool" in block.source_text
    assert "tool_call_id: call-6-b" in block.source_text


def test_empty_output_is_still_rendered() -> None:
    assert render_responses_image_response_only_observation(
        oa_index=1,
        tool_index=1,
        tool_call_id="call-empty",
        content="",
    ).endswith("content:\n\n")


def test_invalid_call_mapping_fields_fail_strictly() -> None:
    with pytest.raises(ValueError, match="non-empty tool_call_id"):
        render_responses_image_response_only_observation(
            oa_index=1, tool_index=1, tool_call_id="", content="result"
        )
    with pytest.raises(ValueError, match="string tool content"):
        render_responses_image_response_only_observation(
            oa_index=1,
            tool_index=1,
            tool_call_id="call-1",
            content=[],  # type: ignore[arg-type]
        )
