import hashlib
import json
import re
from collections import Counter

RECENT_TOOL_CALLS = 3
MIN_MATCHED_LINES = 3

_COMPLETE_WRAPPER = re.compile(
    r"\A(?:<exception>(?P<exception>[\s\S]*?)</exception>\n)?"
    r"<returncode>-?\d+</returncode>\n?"
    r"<output>(?P<body>[\s\S]*)</output>\Z"
)
_TRUNCATED_WRAPPER = re.compile(
    r"\A(?:<exception>(?P<exception>[\s\S]*?)</exception>\n)?"
    r"<returncode>-?\d+</returncode>\n?"
    r"<warning>[\s\S]*?</warning>\n?"
    r"<output_head>(?P<head>[\s\S]*?)</output_head>\n?"
    r"<elided_chars>[^<]*</elided_chars>\n?"
    r"<output_tail>(?P<tail>[\s\S]*?)</output_tail>\Z"
)
_LINE_END = re.compile(r"(?:\r\n|[\n\v\f\r\x1c-\x1e\x85\u2028\u2029])\Z")


def _body_spans(content):
    """Return only visible Observation body spans, excluding protocol wrappers."""
    match = _COMPLETE_WRAPPER.fullmatch(content)
    if match is not None:
        names = ("exception", "body")
    else:
        match = _TRUNCATED_WRAPPER.fullmatch(content)
        names = ("exception", "head", "tail") if match is not None else ()
    if match is None:
        return ((0, len(content)),)
    return tuple(match.span(name) for name in names if match.start(name) >= 0)


def _body_line_counter(content):
    lines = Counter()
    for start, end in _body_spans(content):
        lines.update(line for line in content[start:end].splitlines() if line.strip())
    return lines


def _body_character_counter(content):
    characters = Counter()
    for start, end in _body_spans(content):
        characters.update(
            character for character in content[start:end] if not character.isspace()
        )
    return characters


def _line_records(lines):
    return [
        {
            "sha256": hashlib.sha256(line.encode()).hexdigest(),
            "byte_length": len(line.encode()),
            "count": count,
        }
        for line, count in sorted(
            lines.items(),
            key=lambda item: hashlib.sha256(item[0].encode()).hexdigest(),
        )
    ]


def redact_exact_dependency_lines(content, matched_lines):
    """Delete only exact non-empty body-line occurrences from a render copy."""
    remaining = Counter(matched_lines)
    replacements = []
    removed = 0
    for start, end in _body_spans(content):
        kept = []
        for piece in content[start:end].splitlines(keepends=True):
            ending = _LINE_END.search(piece)
            line = piece[: ending.start()] if ending is not None else piece
            if line.strip() and remaining[line] > 0:
                remaining[line] -= 1
                removed += 1
            else:
                kept.append(piece)
        replacements.append((start, end, "".join(kept)))
    rewritten = content
    for start, end, replacement in reversed(replacements):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    if any(remaining.values()):
        raise ValueError(
            "exact overlap line could not be removed from observation body"
        )
    return rewritten, removed


def _signature(item):
    return hashlib.sha256(
        json.dumps(
            [item.tool_call_id, item.tool_name, item.tool_arguments, item.content],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _replay_overlap(ordered):
    dependency_lines = {}
    events = []
    for index, current in enumerate(ordered):
        current_lines = _body_line_counter(current.content)
        if not current_lines:
            continue
        current_characters = _body_character_counter(current.content)
        for old_index in range(max(0, index - RECENT_TOOL_CALLS), index):
            old = ordered[old_index]
            old_lines = _body_line_counter(old.content)
            matched_lines = old_lines & current_lines
            if sum(matched_lines.values()) < MIN_MATCHED_LINES:
                continue
            old_characters = _body_character_counter(old.content)
            character_intersection = old_characters & current_characters
            character_union = old_characters | current_characters
            numerator = sum(character_intersection.values())
            denominator = sum(character_union.values())
            if numerator < 1 or denominator < numerator:
                raise ValueError("exact line overlap produced invalid character IoU")

            aggregate = dependency_lines.setdefault(old.tool_call_id, Counter())
            for line, count in matched_lines.items():
                aggregate[line] = max(aggregate[line], count)
            events.append(
                {
                    "trigger_call_id": current.tool_call_id,
                    "source_call_id": old.tool_call_id,
                    "distance_in_tool_calls": index - old_index,
                    "matched_line_occurrences": sum(matched_lines.values()),
                    "matched_lines": _line_records(matched_lines),
                    "character_iou_numerator": numerator,
                    "character_iou_denominator": denominator,
                    "character_iou": numerator / denominator,
                }
            )

    dependency_records = {
        call_id: _line_records(lines) for call_id, lines in dependency_lines.items()
    }
    replacements = {}
    for item in ordered:
        lines = dependency_lines.get(item.tool_call_id)
        if not lines:
            continue
        rewritten, removed = redact_exact_dependency_lines(item.content, lines)
        if rewritten == item.content or removed != sum(lines.values()):
            raise ValueError(
                "exact overlap did not produce the expected render rewrite"
            )
        replacements[item.input_index] = rewritten
    return dependency_records, events, replacements
