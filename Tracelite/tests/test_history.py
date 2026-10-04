from __future__ import annotations

import pytest

from swe_memory_policy.history import (
    HistoryProtocolError,
    parse_chat_history,
    parse_mini_swe_agent_history,
    render_init_history,
    render_init_history_units,
)


def _messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "inspect",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"sed -n \'1,20p\' a.py"}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call-1",
            "content": "def f():\n    return '中文 ✓'\n",
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call-2",
            "content": "",
        },
    ]


def test_parse_multi_tool_oa_and_render_exact_content() -> None:
    parsed = parse_chat_history(_messages())
    assert len(parsed.static_messages) == 2
    assert len(parsed.oa_units) == 1
    assert len(parsed.oa_units[0].observations) == 2
    display = render_init_history(parsed.oa_units)
    assert "[Action 1]" in display
    assert "def f():\n    return '中文 ✓'" in display


def test_framework_feedback_is_part_of_oa() -> None:
    messages = _messages()[:2] + [
        {"role": "assistant", "content": "bad call", "tool_calls": []},
        {"role": "user", "content": "Function call validation failed"},
    ]
    parsed = parse_chat_history(messages)
    assert parsed.oa_units[0].observations[0]["role"] == "user"
    assert parsed.oa_units[0].observations[0]["content"] == (
        "Function call validation failed"
    )


def test_mini_standalone_format_feedback_is_not_faked_as_oa() -> None:
    messages = _messages()[:2] + [
        {"role": "user", "content": "Every response needs a bash tool call"},
        *_messages()[2:],
    ]
    parsed = parse_mini_swe_agent_history(messages)
    assert len(parsed.feedback_units) == 1
    assert len(parsed.oa_units) == 1
    display = render_init_history_units(parsed.history_units)
    assert display.startswith(
        "[Framework Feedback 1]\nEvery response needs a bash tool call\n\n"
    )
    assert "[Action 1]" in display


def test_mini_multi_bash_pairing_is_strict() -> None:
    parsed = parse_mini_swe_agent_history(_messages())
    assert len(parsed.oa_units[0].observations) == 2
    broken = _messages()
    broken[-1]["tool_call_id"] = "unknown"
    with pytest.raises(HistoryProtocolError, match="undeclared"):
        parse_mini_swe_agent_history(broken)


def test_init_display_is_readable_and_omits_transport_ids() -> None:
    parsed = parse_chat_history(_messages())
    display = render_init_history(parsed.oa_units)
    assert display.startswith("[Action 1]\ninspect\n[Tool 1]\nterminal\n")
    assert "[Tool 2]\nterminal\n{}" in display
    assert "[Observation 1]" in display
    assert "[Tool Result 1]" in display
    assert "def f():\n    return" in display
    assert "call-1" not in display
    assert "call_id" not in display


def test_init_display_preserves_unicode_and_multiline_observation() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "检查代码 ✓",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"python -m pytest"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "terminal",
            "content": "Traceback:\n  文件.py:1\n    return '中文'\n",
        },
    ]
    display = render_init_history(parse_chat_history(messages).oa_units)
    assert "检查代码 ✓" in display
    assert "Traceback:\n  文件.py:1\n    return '中文'\n" in display


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data[-1].update(tool_call_id="unknown"), "undeclared"),
        (lambda data: data.pop(), "no observations"),
        (
            lambda data: data.append({"role": "developer", "content": "x"}),
            "unexpected",
        ),
    ],
)
def test_invalid_pairing_fails_closed(mutation, match: str) -> None:  # noqa: ANN001
    messages = _messages()
    mutation(messages)
    with pytest.raises(HistoryProtocolError, match=match):
        parse_chat_history(messages)
