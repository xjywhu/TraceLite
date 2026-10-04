from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

POLICY_VERSION = "terminal-sanitize-v1"

_ESC = "\x1b"
_BEL = "\x07"
_ST = "\x9c"
_C1_CSI = "\x9b"
_C1_OSC = "\x9d"
_C1_STRINGS = {
    "\x90": "dcs",
    "\x98": "sos",
    "\x9e": "pm",
    "\x9f": "apc",
}
_ESC_STRINGS = {
    "P": "dcs",
    "X": "sos",
    "^": "pm",
    "_": "apc",
}


@dataclass(frozen=True)
class TerminalSanitizeResult:
    """A visual-only terminal normalization and its non-secret audit record."""

    text: str
    manifest: dict[str, Any]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_csi_end(text: str, start: int) -> int | None:
    """Return the exclusive end of a complete CSI sequence.

    ``start`` is the first byte after CSI's introducer.  A newline terminates our
    search: consuming a later diagnostic line because a malformed CSI never
    supplied its final byte would be information loss.
    """

    for index in range(start, len(text)):
        character = text[index]
        if character in "\r\n":
            return None
        codepoint = ord(character)
        if 0x40 <= codepoint <= 0x7E:
            return index + 1
        if not (0x20 <= codepoint <= 0x3F):
            return None
    return None


def _find_st_terminated_end(text: str, start: int, *, allow_bel: bool) -> int | None:
    index = start
    while index < len(text):
        if allow_bel and text[index] == _BEL:
            return index + 1
        if text[index] == _ST:
            return index + 1
        if text.startswith(_ESC + "\\", index):
            return index + 2
        index += 1
    return None


def _find_escape_end(text: str, start: int) -> int | None:
    """Return the end of an ISO-2022 ESC sequence with intermediates."""

    index = start
    while index < len(text) and 0x20 <= ord(text[index]) <= 0x2F:
        index += 1
    if index < len(text) and 0x30 <= ord(text[index]) <= 0x7E:
        return index + 1
    return None


def _strip_terminal_controls(
    text: str,
) -> tuple[str, Counter[str], Counter[str], list[dict[str, Any]], int]:
    output: list[str] = []
    removed: Counter[str] = Counter()
    incomplete: Counter[str] = Counter()
    spans: list[dict[str, Any]] = []
    inserted_markers = 0
    index = 0

    def record(start: int, end: int, kind: str, *, complete: bool = True) -> None:
        removed[kind] += 1
        spans.append(
            {
                "start": start,
                "end": end,
                "kind": kind,
                "complete": complete,
            }
        )

    def preserve_incomplete(start: int, end: int, kind: str) -> None:
        nonlocal inserted_markers
        incomplete[kind] += 1
        record(start, end, kind, complete=False)
        output.append(f"[[INCOMPLETE_{kind.upper()}]]")
        inserted_markers += 1

    while index < len(text):
        character = text[index]

        if character == _ESC:
            if index + 1 >= len(text):
                record(index, index + 1, "single_escape")
                index += 1
                continue
            introducer = text[index + 1]
            if introducer == "[":
                end = _find_csi_end(text, index + 2)
                if end is None:
                    preserve_incomplete(index, index + 2, "csi")
                    index += 2
                else:
                    record(index, end, "csi")
                    index = end
                continue
            if introducer == "]":
                end = _find_st_terminated_end(text, index + 2, allow_bel=True)
                if end is None:
                    preserve_incomplete(index, index + 2, "osc")
                    index += 2
                else:
                    record(index, end, "osc")
                    index = end
                continue
            string_kind = _ESC_STRINGS.get(introducer)
            if string_kind is not None:
                end = _find_st_terminated_end(text, index + 2, allow_bel=False)
                if end is None:
                    preserve_incomplete(index, index + 2, string_kind)
                    index += 2
                else:
                    record(index, end, string_kind)
                    index = end
                continue
            if introducer == "\\":
                record(index, index + 2, "st")
                index += 2
                continue
            end = _find_escape_end(text, index + 1)
            if end is not None:
                record(index, end, "escape")
                index = end
                continue
            # A truly stray ESC has no printable meaning.  Remove only ESC and
            # preserve the following byte verbatim instead of guessing a range.
            record(index, index + 1, "single_escape")
            index += 1
            continue

        if character == _C1_CSI:
            end = _find_csi_end(text, index + 1)
            if end is None:
                preserve_incomplete(index, index + 1, "csi")
                index += 1
            else:
                record(index, end, "csi")
                index = end
            continue
        if character == _C1_OSC:
            end = _find_st_terminated_end(text, index + 1, allow_bel=True)
            if end is None:
                preserve_incomplete(index, index + 1, "osc")
                index += 1
            else:
                record(index, end, "osc")
                index = end
            continue
        string_kind = _C1_STRINGS.get(character)
        if string_kind is not None:
            end = _find_st_terminated_end(text, index + 1, allow_bel=False)
            if end is None:
                preserve_incomplete(index, index + 1, string_kind)
                index += 1
            else:
                record(index, end, string_kind)
                index = end
            continue
        if character == _ST:
            record(index, index + 1, "st")
            index += 1
            continue
        if character == _BEL:
            record(index, index + 1, "bel")
            index += 1
            continue
        if 0x80 <= ord(character) <= 0x9F:
            record(index, index + 1, "c1_control")
            index += 1
            continue

        output.append(character)
        index += 1

    return "".join(output), removed, incomplete, spans, inserted_markers


def _apply_terminal_line_editing(text: str) -> tuple[str, dict[str, int]]:
    """Apply CR/backspace cursor semantics without touching ordinary newlines."""

    output: list[str] = []
    cells: list[str] = []
    cursor = 0
    index = 0
    carriage_returns = 0
    backspaces = 0
    overwritten = 0
    edited_lines = 0
    line_edited = False
    crlf_normalized = 0

    def flush(*, newline: bool) -> None:
        nonlocal cells, cursor, line_edited, edited_lines
        output.extend(cells)
        if newline:
            output.append("\n")
        if line_edited:
            edited_lines += 1
        cells = []
        cursor = 0
        line_edited = False

    while index < len(text):
        character = text[index]
        if character == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
            crlf_normalized += 1
            flush(newline=True)
            index += 2
            continue
        if character == "\n":
            flush(newline=True)
            index += 1
            continue
        if character == "\r":
            carriage_returns += 1
            line_edited = True
            cursor = 0
            index += 1
            continue
        if character == "\b":
            backspaces += 1
            line_edited = True
            cursor = max(0, cursor - 1)
            index += 1
            continue
        if cursor < len(cells):
            if cells[cursor] != character:
                overwritten += 1
            cells[cursor] = character
        else:
            cells.append(character)
        cursor += 1
        index += 1
    flush(newline=False)
    return "".join(output), {
        "carriage_returns_applied": carriage_returns,
        "backspaces_applied": backspaces,
        "overwritten_characters": overwritten,
        "edited_lines": edited_lines,
        "crlf_normalized": crlf_normalized,
    }


def sanitize_terminal_text(text: str) -> TerminalSanitizeResult:
    """Return a safe visual copy while leaving the caller's raw text untouched.

    Complete terminal-control strings are removed.  An incomplete string removes
    only its introducer, inserts an explicit marker, and preserves the remaining
    payload.  This is intentionally fail-closed for diagnostic text.
    """

    text.encode("utf-8", errors="strict")
    stripped, removed, incomplete, spans, inserted_markers = _strip_terminal_controls(
        text
    )
    normalized, line_editing = _apply_terminal_line_editing(stripped)
    removed_characters = sum(span["end"] - span["start"] for span in spans)
    manifest: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "visual_copy_only": True,
        "input_sha256": _digest(text),
        "output_sha256": _digest(normalized),
        "input_characters": len(text),
        "output_characters": len(normalized),
        "changed": normalized != text,
        "complete_sequences_removed": sum(removed.values()) - sum(incomplete.values()),
        "removed_sequence_counts": dict(sorted(removed.items())),
        "removed_source_characters": removed_characters,
        "removed_spans": spans,
        "incomplete_sequence_counts": dict(sorted(incomplete.items())),
        "incomplete_markers_inserted": inserted_markers,
        **line_editing,
        "literal_unicode_labels_preserved": True,
    }
    return TerminalSanitizeResult(normalized, manifest)
