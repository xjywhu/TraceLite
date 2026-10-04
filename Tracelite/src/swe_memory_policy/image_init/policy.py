from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from swe_memory_policy.history import (
    FrameworkFeedbackUnit,
    HistoryUnit,
    OAUnit,
    ParsedHistory,
)
from swe_memory_policy.utils import canonical_json_bytes

SEPARATOR = "━" * 36


@dataclass(frozen=True)
class ImageInitBlock:
    """One independently rendered native-history unit."""

    sequence_index: int
    kind: Literal["oa", "framework_feedback"]
    source_text: str
    oa_index: int | None = None
    feedback_index: int | None = None

    @property
    def stem(self) -> str:
        if self.kind == "oa":
            if self.oa_index is None:  # pragma: no cover
                raise ValueError("OA block lacks oa_index")
            return f"OA{self.oa_index:04d}"
        if self.feedback_index is None:  # pragma: no cover
            raise ValueError("feedback block lacks feedback_index")
        return f"Feedback{self.feedback_index:04d}"

    @property
    def page_label(self) -> str:
        if self.kind == "oa":
            return f"OA {self.oa_index}"
        return f"FRAMEWORK FEEDBACK {self.feedback_index}"


def _json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8").rstrip("\n")


def _content(value: Any) -> list[str]:
    type_name = (
        "null"
        if value is None
        else "array"
        if isinstance(value, list)
        else "object"
        if isinstance(value, dict)
        else "boolean"
        if isinstance(value, bool)
        else "number"
        if isinstance(value, int | float)
        else "string"
    )
    rendered = value if isinstance(value, str) and value else _json(value)
    return [f"content_type: {type_name}", "", "content:", rendered]


def _additional_fields(message: dict[str, Any], known: set[str]) -> list[str]:
    additional = {key: value for key, value in message.items() if key not in known}
    if not additional:
        return []
    return ["", "additional_fields:", _json(additional)]


def _assistant_block(unit: OAUnit) -> str:
    assistant = unit.assistant
    lines = [
        SEPARATOR,
        f"OA {unit.oa_index} — ASSISTANT MESSAGE",
        SEPARATOR,
        "",
        f"role: {assistant.get('role', 'assistant')}",
        "",
        *_content(assistant.get("content")),
        "",
        "tool_calls:",
    ]
    calls = assistant.get("tool_calls")
    if isinstance(calls, list) and calls:
        for number, call in enumerate(calls, start=1):
            if not isinstance(call, dict):
                lines.extend([f"  call {number}:", _json(call)])
                continue
            function = call.get("function")
            function_dict = function if isinstance(function, dict) else {}
            arguments = function_dict.get("arguments")
            rendered_arguments = (
                arguments if isinstance(arguments, str) else _json(arguments)
            )
            lines.extend(
                [
                    f"  call {number}:",
                    f"    id: {_json(call.get('id'))}",
                    f"    type: {_json(call.get('type'))}",
                    f"    function.name: {_json(function_dict.get('name'))}",
                    "    function.arguments:",
                    rendered_arguments,
                ]
            )
            function_extra = {
                key: value
                for key, value in function_dict.items()
                if key not in {"name", "arguments"}
            }
            call_extra = {
                key: value
                for key, value in call.items()
                if key not in {"id", "type", "function"}
            }
            if function_extra:
                lines.extend(["    function.additional_fields:", _json(function_extra)])
            if call_extra:
                lines.extend(["    additional_fields:", _json(call_extra)])
    else:
        lines.append("  []")
    lines.extend(_additional_fields(assistant, {"role", "content", "tool_calls"}))
    return "\n".join(lines)


def _observation_block(unit: OAUnit, observation: dict[str, Any], number: int) -> str:
    role = observation.get("role")
    title = "TOOL MESSAGE" if role == "tool" else f"{str(role).upper()} MESSAGE"
    lines = [
        SEPARATOR,
        f"OA {unit.oa_index} — {title} {number}",
        SEPARATOR,
        "",
        f"role: {role}",
    ]
    known = {"role", "content"}
    if "tool_call_id" in observation:
        lines.extend(["", f"tool_call_id: {_json(observation.get('tool_call_id'))}"])
        known.add("tool_call_id")
    if "name" in observation:
        lines.extend([f"name: {_json(observation.get('name'))}"])
        known.add("name")
    lines.extend(["", *_content(observation.get("content"))])
    lines.extend(_additional_fields(observation, known))
    return "\n".join(lines)


def render_image_init_oa(unit: OAUnit) -> str:
    blocks = [_assistant_block(unit)]
    blocks.extend(
        _observation_block(unit, observation, number)
        for number, observation in enumerate(unit.observations, start=1)
    )
    return "\n\n".join(blocks) + "\n"


def render_image_init_feedback(unit: FrameworkFeedbackUnit) -> str:
    message = unit.message
    lines = [
        SEPARATOR,
        f"FRAMEWORK FEEDBACK {unit.feedback_index} — USER MESSAGE",
        SEPARATOR,
        "",
        f"role: {message.get('role')}",
        "",
        *_content(message.get("content")),
    ]
    lines.extend(_additional_fields(message, {"role", "content"}))
    return "\n".join(lines) + "\n"


def render_image_init_unit(unit: HistoryUnit) -> str:
    if isinstance(unit, OAUnit):
        return render_image_init_oa(unit)
    if isinstance(unit, FrameworkFeedbackUnit):
        return render_image_init_feedback(unit)
    raise TypeError(f"unsupported history unit: {type(unit)!r}")


def build_image_init_blocks(
    parsed_history: ParsedHistory,
) -> tuple[ImageInitBlock, ...]:
    blocks: list[ImageInitBlock] = []
    for sequence_index, unit in enumerate(parsed_history.history_units, start=1):
        if isinstance(unit, OAUnit):
            blocks.append(
                ImageInitBlock(
                    sequence_index=sequence_index,
                    kind="oa",
                    oa_index=unit.oa_index,
                    source_text=render_image_init_oa(unit),
                )
            )
        elif isinstance(unit, FrameworkFeedbackUnit):
            blocks.append(
                ImageInitBlock(
                    sequence_index=sequence_index,
                    kind="framework_feedback",
                    feedback_index=unit.feedback_index,
                    source_text=render_image_init_feedback(unit),
                )
            )
        else:  # pragma: no cover
            raise TypeError(f"unsupported history unit: {type(unit)!r}")
    return tuple(blocks)


def render_image_init_history(blocks: tuple[ImageInitBlock, ...]) -> str:
    return "\n".join(block.source_text.rstrip("\n") for block in blocks) + (
        "\n" if blocks else ""
    )
