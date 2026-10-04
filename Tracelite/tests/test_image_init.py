from __future__ import annotations

from swe_memory_policy.history import (
    FrameworkFeedbackUnit,
    OAUnit,
    ParsedHistory,
)
from swe_memory_policy.image_init import (
    build_image_init_blocks,
    build_image_init_history_prompt,
    render_image_init_history,
    render_image_init_oa,
)


def _oa(index: int) -> OAUnit:
    return OAUnit(
        oa_index=index,
        assistant={
            "role": "assistant",
            "content": [
                {"type": "text", "text": "inspect the repository"},
                {"type": "meta", "value": None},
            ],
            "tool_calls": [
                {
                    "id": f"call-{index}-a",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"rg header_rows"}',
                        "provider_extension": {"x": 1},
                    },
                },
                {
                    "id": f"call-{index}-b",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pytest -q"}',
                    },
                },
            ],
            "reasoning_marker": "preserve me",
        },
        observations=(
            {
                "role": "tool",
                "tool_call_id": f"call-{index}-a",
                "name": "bash",
                "content": "first complete observation\n",
                "provider_field": False,
            },
            {
                "role": "tool",
                "tool_call_id": f"call-{index}-b",
                "name": "bash",
                "content": "",
            },
        ),
    )


def test_oa_visualization_preserves_native_message_fields() -> None:
    rendered = render_image_init_oa(_oa(6))
    assert "OA 6 — ASSISTANT MESSAGE" in rendered
    assert "role: assistant" in rendered
    assert '"type": "text"' in rendered
    assert 'id: "call-6-a"' in rendered
    assert 'function.name: "bash"' in rendered
    assert '"command":"rg header_rows"' in rendered
    assert '"provider_extension": {' in rendered
    assert '"reasoning_marker": "preserve me"' in rendered
    assert "OA 6 — TOOL MESSAGE 1" in rendered
    assert 'tool_call_id: "call-6-a"' in rendered
    assert "first complete observation" in rendered
    assert '"provider_field": false' in rendered
    assert "OA 6 — TOOL MESSAGE 2" in rendered
    assert 'tool_call_id: "call-6-b"' in rendered
    assert 'content:\n""' in rendered


def test_blocks_follow_wire_order_and_keep_feedback_separate() -> None:
    feedback = FrameworkFeedbackUnit(
        sequence_index=2,
        feedback_index=1,
        message={
            "role": "user",
            "content": "Every response needs a bash tool call",
            "extra": {"retry": 1},
        },
    )
    parsed = ParsedHistory(
        static_messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ),
        history_units=(_oa(1), feedback, _oa(2)),
    )
    blocks = build_image_init_blocks(parsed)
    assert [(block.kind, block.sequence_index) for block in blocks] == [
        ("oa", 1),
        ("framework_feedback", 2),
        ("oa", 3),
    ]
    assert [block.stem for block in blocks] == [
        "OA0001",
        "Feedback0001",
        "OA0002",
    ]
    assert "FRAMEWORK FEEDBACK 1 — USER MESSAGE" in blocks[1].source_text
    assert '"retry": 1' in blocks[1].source_text
    aggregate = render_image_init_history(blocks)
    assert (
        aggregate.index("OA 1 — ASSISTANT")
        < aggregate.index("FRAMEWORK FEEDBACK 1")
        < aggregate.index("OA 2 — ASSISTANT")
    )


def test_global_prompt_describes_visual_native_history() -> None:
    prompt = build_image_init_history_prompt()
    assert "lossless visual serialization" in prompt
    assert "assistant role" in prompt
    assert "tool-role tool_call_id" in prompt
    assert "historical native messages" in prompt
