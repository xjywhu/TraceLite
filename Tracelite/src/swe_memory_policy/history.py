from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swe_memory_policy.utils import compact_json

ALLOWED_ROLES = {"system", "developer", "user", "assistant", "tool"}


class HistoryProtocolError(ValueError):
    """Raised when an OpenAI history cannot be grouped without guessing."""


@dataclass(frozen=True)
class OAUnit:
    oa_index: int
    assistant: dict[str, Any]
    observations: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "oa_index": self.oa_index,
            "assistant": self.assistant,
            "observations": list(self.observations),
        }


@dataclass(frozen=True)
class FrameworkFeedbackUnit:
    sequence_index: int
    feedback_index: int
    message: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "feedback_index": self.feedback_index,
            "message": self.message,
        }


HistoryUnit = OAUnit | FrameworkFeedbackUnit


@dataclass(frozen=True)
class ParsedHistory:
    static_messages: tuple[dict[str, Any], ...]
    history_units: tuple[HistoryUnit, ...]

    @property
    def oa_units(self) -> tuple[OAUnit, ...]:
        return tuple(unit for unit in self.history_units if isinstance(unit, OAUnit))

    @property
    def feedback_units(self) -> tuple[FrameworkFeedbackUnit, ...]:
        return tuple(
            unit
            for unit in self.history_units
            if isinstance(unit, FrameworkFeedbackUnit)
        )


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise HistoryProtocolError("assistant.tool_calls must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            raise HistoryProtocolError("tool call must be an object")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise HistoryProtocolError("tool call id is missing")
        if call_id in seen:
            raise HistoryProtocolError(f"duplicate tool call id: {call_id}")
        seen.add(call_id)
        result.append(call)
    return result


def parse_chat_history(messages: list[dict[str, Any]]) -> ParsedHistory:
    if not isinstance(messages, list) or not messages:
        raise HistoryProtocolError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict):
            raise HistoryProtocolError("each message must be an object")
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise HistoryProtocolError(f"unsupported message role: {role!r}")

    first_assistant = next(
        (
            index
            for index, message in enumerate(messages)
            if message["role"] == "assistant"
        ),
        len(messages),
    )
    static = messages[:first_assistant]
    if any(message["role"] in {"assistant", "tool"} for message in static):
        raise HistoryProtocolError("static prefix contains historical messages")

    units: list[OAUnit] = []
    cursor = first_assistant
    while cursor < len(messages):
        assistant = messages[cursor]
        if assistant["role"] != "assistant":
            raise HistoryProtocolError(
                f"expected assistant at history position {cursor}, "
                f"got {assistant['role']}"
            )
        declared_calls = _tool_calls(assistant)
        expected_ids = {call["id"] for call in declared_calls}
        cursor += 1
        observations: list[dict[str, Any]] = []
        matched_ids: set[str] = set()
        while cursor < len(messages) and messages[cursor]["role"] != "assistant":
            observation = messages[cursor]
            role = observation["role"]
            if role == "tool":
                call_id = observation.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise HistoryProtocolError(
                        "tool observation is missing tool_call_id"
                    )
                if call_id not in expected_ids:
                    raise HistoryProtocolError(
                        f"observation references undeclared tool call: {call_id}"
                    )
                if call_id in matched_ids:
                    raise HistoryProtocolError(
                        f"duplicate observation for tool call: {call_id}"
                    )
                matched_ids.add(call_id)
            elif role != "user":
                raise HistoryProtocolError(
                    f"unexpected {role} message inside OA {len(units) + 1}"
                )
            observations.append(observation)
            cursor += 1
        missing = expected_ids - matched_ids
        if missing:
            raise HistoryProtocolError(
                "assistant tool calls have no observations: "
                + ", ".join(sorted(missing))
            )
        if not declared_calls and not observations:
            raise HistoryProtocolError(
                "terminal assistant message cannot be historical input "
                "to a later request"
            )
        units.append(
            OAUnit(
                oa_index=len(units) + 1,
                assistant=assistant,
                observations=tuple(observations),
            )
        )
    return ParsedHistory(tuple(static), tuple(units))


def parse_mini_swe_agent_history(
    messages: list[dict[str, Any]],
) -> ParsedHistory:
    """Parse mini-SWE-agent v2's linear OpenAI tool-call history.

    The first two messages are the official system and instance prompts.  A
    format error is recorded by mini as a standalone user message because the
    invalid assistant response is not appended to the agent history.  Preserve
    that message as framework feedback instead of pretending it is an action.
    """

    if not isinstance(messages, list) or len(messages) < 2:
        raise HistoryProtocolError("mini history must contain system and task")
    for message in messages:
        if not isinstance(message, dict):
            raise HistoryProtocolError("each message must be an object")
        if message.get("role") not in ALLOWED_ROLES:
            raise HistoryProtocolError(
                f"unsupported message role: {message.get('role')!r}"
            )
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise HistoryProtocolError("mini static prefix must be system then user")

    static = tuple(messages[:2])
    units: list[HistoryUnit] = []
    oa_index = 0
    feedback_index = 0
    cursor = 2
    while cursor < len(messages):
        message = messages[cursor]
        if message["role"] == "user":
            feedback_index += 1
            units.append(
                FrameworkFeedbackUnit(
                    sequence_index=len(units) + 1,
                    feedback_index=feedback_index,
                    message=message,
                )
            )
            cursor += 1
            continue
        if message["role"] != "assistant":
            raise HistoryProtocolError(
                f"expected assistant or framework feedback at position {cursor}, "
                f"got {message['role']}"
            )

        assistant = message
        declared_calls = _tool_calls(assistant)
        if not declared_calls:
            raise HistoryProtocolError(
                "mini historical assistant message has no bash tool call"
            )
        expected_ids = {call["id"] for call in declared_calls}
        observations: list[dict[str, Any]] = []
        matched_ids: set[str] = set()
        cursor += 1
        while cursor < len(messages) and messages[cursor]["role"] == "tool":
            observation = messages[cursor]
            call_id = observation.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise HistoryProtocolError("tool observation is missing tool_call_id")
            if call_id not in expected_ids:
                raise HistoryProtocolError(
                    f"observation references undeclared tool call: {call_id}"
                )
            if call_id in matched_ids:
                raise HistoryProtocolError(
                    f"duplicate observation for tool call: {call_id}"
                )
            matched_ids.add(call_id)
            observations.append(observation)
            cursor += 1
        missing = expected_ids - matched_ids
        if missing:
            raise HistoryProtocolError(
                "assistant tool calls have no observations: "
                + ", ".join(sorted(missing))
            )
        oa_index += 1
        units.append(
            OAUnit(
                oa_index=oa_index,
                assistant=assistant,
                observations=tuple(observations),
            )
        )
    return ParsedHistory(static, tuple(units))


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            else:
                chunks.append(compact_json(item))
        return "\n".join(chunks)
    return compact_json(content)


def render_init_oa_display(unit: OAUnit) -> str:
    """Render one native action/result step for image history.

    This deliberately omits transport identifiers and audit metadata. The
    canonical :class:`OAUnit` remains the lossless audit representation; this
    function defines only the compact bytes visible to the model in
    ``image_init``.
    """

    index = unit.oa_index
    lines = [f"[Action {index}]"]
    assistant_text = _content_text(unit.assistant.get("content"))
    if assistant_text:
        lines.append(assistant_text)
    for number, call in enumerate(_tool_calls(unit.assistant), start=1):
        function = call.get("function") or {}
        name = function.get("name") or call.get("name") or "unknown"
        arguments = function.get("arguments", call.get("arguments", {}))
        argument_text = (
            arguments if isinstance(arguments, str) else compact_json(arguments)
        )
        lines.extend([f"[Tool {number}]", str(name)])
        if argument_text:
            lines.append(argument_text)

    lines.append(f"[Observation {index}]")
    for number, observation in enumerate(unit.observations, start=1):
        if len(unit.observations) > 1:
            kind = "Tool Result" if observation["role"] == "tool" else "Framework"
            lines.append(f"[{kind} {number}]")
        text = _content_text(observation.get("content"))
        if text:
            lines.append(text)
    return "\n".join(lines) + "\n"


def render_init_history(units: tuple[OAUnit, ...]) -> str:
    """Render the complete request history from scratch in chronological order."""

    return "\n".join(render_init_oa_display(unit).rstrip("\n") for unit in units) + (
        "\n" if units else ""
    )


def render_init_history_units(units: tuple[HistoryUnit, ...]) -> str:
    """Render OA units and standalone framework feedback in wire order."""

    rendered: list[str] = []
    for unit in units:
        if isinstance(unit, OAUnit):
            rendered.append(render_init_oa_display(unit).rstrip("\n"))
        else:
            text = _content_text(unit.message.get("content"))
            block = [f"[Framework Feedback {unit.feedback_index}]"]
            if text:
                block.append(text)
            rendered.append("\n".join(block))
    return "\n\n".join(rendered) + ("\n" if rendered else "")
