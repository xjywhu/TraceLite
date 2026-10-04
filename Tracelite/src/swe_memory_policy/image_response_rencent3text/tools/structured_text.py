from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

from swe_memory_policy.image_response_rencent3text.tools.model import (
    HARD_NEWLINE_GLYPH,
    HARD_NEWLINE_MARKER,
    CompactDocument,
    CompactObservationInput,
    CompactRegion,
    StyledAtom,
)
from swe_memory_policy.image_response_rencent3text.tools.parsing import (
    ParsedObservationWrapper,
)
from swe_memory_policy.image_response_rencent3text.tools.terminal_sanitize import (
    TerminalSanitizeResult,
    sanitize_terminal_text,
)


@dataclass(frozen=True)
class StructuredTextKind:
    classification: str
    title: str
    preserve_original: bool = False


_TABLE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"(?m)^\s*\|?[-+: ]{5,}\|")
_GRID_TABLE = re.compile(r"(?m)^\s*\+(?:[-=]{3,}\+){2,}\s*$")
_PANDAS_FOOTER = re.compile(r"(?m)^\s*\[\d+ rows? x \d+ columns?\]\s*$")
_NUMPY_MATRIX = re.compile(r"(?s)(?:\barray|\bmatrix)\s*\(\s*\[\[")
_MATRIX_VALUE = re.compile(
    r"(?<![\w.])(?:[-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?|True|False|nan|inf)"
    r"(?![\w.])",
    re.IGNORECASE,
)
_PATH_TOKEN = re.compile(r"(?P<path>(?:~/|/testbed/)[^\s\"'<>|]+?)(?=$|[\s\"'<>|,;)])")
_NUMBER_TOKEN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")
_RETURNCODE = re.compile(r"<returncode>(?P<value>[^<]*)</returncode>")
_VISUAL_PADDING = re.compile(r"[ \t\x00]{32,}")
_MODERATE_VISUAL_PADDING = re.compile(r"[ \t\x00]{8,31}")
_VISUAL_WHITESPACE_POLICY = "compact_visual_whitespace_v2"
_GIT_BLAME_LINE = re.compile(
    r"(?m)^(?P<prefix>\^?[0-9a-fA-F]{7,40}"
    r"(?:\s+[^\s()]+){0,2}\s+\()"
    r"(?P<author>.*?)\s+"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} [0-2]\d:[0-5]\d:[0-5]\d [+-]\d{4})"
    r"\s+(?P<line_number>\d+)\)(?P<code>.*)$"
)


def compact_git_blame_timestamps(text: str) -> tuple[str, dict[str, object]]:
    """Omit timestamps only from structurally valid ``git blame`` lines.

    This operates on the visual copy after terminal sanitization.  Ordinary
    dates in logs, test output, source code, and prose remain untouched.
    """

    input_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    timestamps_omitted = 0
    timestamp_characters_omitted = 0

    def omit_timestamp(match: re.Match[str]) -> str:
        nonlocal timestamps_omitted, timestamp_characters_omitted
        timestamps_omitted += 1
        timestamp_characters_omitted += len(match.group("timestamp"))
        return "{}{} :{}){}".format(
            match.group("prefix"),
            match.group("author").rstrip(),
            match.group("line_number"),
            match.group("code"),
        )

    compacted = _GIT_BLAME_LINE.sub(omit_timestamp, text)
    return compacted, {
        "policy_version": "git-blame-timestamp-visual-omission-v1",
        "visual_copy_only": True,
        "changed": compacted != text,
        "timestamps_omitted": timestamps_omitted,
        "timestamp_characters_omitted": timestamp_characters_omitted,
        "input_sha256": input_sha256,
        "output_sha256": hashlib.sha256(compacted.encode("utf-8")).hexdigest(),
    }


def compact_visual_whitespace(text: str, policy: str) -> tuple[str, dict[str, object]]:
    """Compact horizontal padding only in the visual copy.

    The retained observation and its hashes remain byte-for-byte unchanged.  A
    visible marker records exact counts for long or binary-like runs.  Pure
    space runs of 8--31 characters become four spaces; they are layout, not
    payload, and do not need an omission marker.
    """

    input_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if policy == "preserve":
        return text, {
            "policy_version": "preserve",
            "visual_copy_only": True,
            "changed": False,
            "run_count": 0,
            "source_characters_compacted": 0,
            "markers_inserted": 0,
            "moderate_space_runs_compacted": 0,
            "moderate_space_characters_compacted": 0,
            "binary_like_runs_marked": 0,
            "input_sha256": input_sha256,
            "output_sha256": input_sha256,
        }
    if policy != _VISUAL_WHITESPACE_POLICY:
        raise ValueError("unsupported sparse-padding policy")

    run_count = 0
    source_characters = 0
    markers_inserted = 0
    moderate_space_runs = 0
    moderate_space_characters = 0
    binary_like_runs = 0

    def omission_marker(match: re.Match[str]) -> str:
        nonlocal run_count, source_characters, markers_inserted
        value = match.group(0)
        run_count += 1
        source_characters += len(value)
        markers_inserted += 1
        return " [PAD omitted={} spaces={} tabs={} nul={}] ".format(
            len(value), value.count(" "), value.count("\t"), value.count("\x00")
        )

    compacted = _VISUAL_PADDING.sub(omission_marker, text)
    minimum_marker_run = 32

    def compact_moderate(match: re.Match[str]) -> str:
        nonlocal run_count, source_characters, markers_inserted
        nonlocal moderate_space_runs, moderate_space_characters
        nonlocal binary_like_runs
        value = match.group(0)
        run_count += 1
        source_characters += len(value)
        if "\t" in value or "\x00" in value:
            binary_like_runs += 1
            markers_inserted += 1
            return " [PAD omitted={} spaces={} tabs={} nul={}] ".format(
                len(value),
                value.count(" "),
                value.count("\t"),
                value.count("\x00"),
            )
        moderate_space_runs += 1
        moderate_space_characters += len(value)
        return " " * 4

    compacted = _MODERATE_VISUAL_PADDING.sub(compact_moderate, compacted)
    return compacted, {
        "policy_version": policy,
        "visual_copy_only": True,
        "minimum_marker_run_characters": minimum_marker_run,
        "moderate_space_run_minimum": (
            8
        ),
        "moderate_space_run_maximum": (
            31
        ),
        "moderate_space_replacement_characters": (
            4
        ),
        "changed": compacted != text,
        "run_count": run_count,
        "source_characters_compacted": source_characters,
        "markers_inserted": markers_inserted,
        "moderate_space_runs_compacted": moderate_space_runs,
        "moderate_space_characters_compacted": moderate_space_characters,
        "binary_like_runs_marked": binary_like_runs,
        "input_sha256": input_sha256,
        "output_sha256": hashlib.sha256(compacted.encode("utf-8")).hexdigest(),
    }


def _command_payload(arguments: str | None) -> str:
    try:
        value = json.loads(arguments or "")
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return ""
    return value["command"]


def is_table_or_matrix(body: str) -> str | None:
    """Return a strict allowlisted legacy-layout reason, if any.

    Only strong visual table/matrix signatures qualify.  Source code containing a
    list literal is deliberately not enough; Python-file observations are handled
    before this function is called.
    """

    stripped = body.lstrip("\r\n")
    if stripped.startswith("Traceback (most recent call last):"):
        return None
    prefix = stripped[:800]
    table = _TABLE.search(prefix)
    rule = _TABLE_RULE.search(prefix)
    grid = _GRID_TABLE.search(prefix)
    if (table is not None and rule is not None) or grid is not None:
        return "preserve_table_layout"
    if _PANDAS_FOOTER.search(body):
        return "preserve_table_layout"
    numpy_matrix = _NUMPY_MATRIX.search(prefix)
    unsafe_matrix_context = bool(
        ">>>" in prefix
        or "Warning:" in prefix
        or re.search(r"(?m)^(?:/testbed/|\d+:)", prefix)
    )
    special_array_display = stripped.startswith("<xarray.") or (
        stripped.startswith("<class ") and "Parameters:" in prefix
    )
    bracket_rows: list[tuple[int, str]] = []
    offset = 0
    for line in prefix.splitlines(keepends=True):
        candidate = line.strip().rstrip(",")
        if (
            candidate.startswith("[")
            and candidate.endswith("]")
            and len(_MATRIX_VALUE.findall(candidate)) >= 2
            and "{" not in candidate
            and "}" not in candidate
            and ("," not in candidate or candidate.startswith("[["))
        ):
            bracket_rows.append((offset, candidate))
        offset += len(line)
    if (
        numpy_matrix is not None
        and (special_array_display or numpy_matrix.start() < 200)
        and not unsafe_matrix_context
    ) or (
        len(bracket_rows) >= 2
        and bracket_rows[0][0] < 200
        and not unsafe_matrix_context
    ):
        return "preserve_matrix_layout"
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if (
        isinstance(parsed, list)
        and len(parsed) >= 2
        and all(isinstance(row, list) and len(row) >= 2 for row in parsed)
    ):
        return "preserve_matrix_layout"
    return None


def classify_structured_text(
    observation: CompactObservationInput,
    wrapper: ParsedObservationWrapper,
) -> StructuredTextKind:
    if wrapper.truncated:
        return StructuredTextKind("compact_truncated_v2", "Truncated output")
    body = wrapper.body
    preserved = is_table_or_matrix(body)
    if preserved is not None:
        return StructuredTextKind(preserved, "Original table/matrix", True)
    command = _command_payload(observation.tool_arguments)
    stripped = body.lstrip("\r\n")
    if not stripped.strip():
        return StructuredTextKind("compact_empty_v1", "Empty output")
    if (
        "<exception>" in wrapper.prefix
        or "<returncode>0</returncode>" not in wrapper.prefix
    ):
        return StructuredTextKind("compact_error_v1", "Command / environment error")
    if re.search(r"(?m)^(diff --git |@@ |--- a/|\+\+\+ b/)", body):
        return StructuredTextKind("compact_diff_v1", "Diff")
    if "Traceback (most recent call last):" in body:
        return StructuredTextKind("compact_traceback_v1", "Traceback")
    if re.search(
        r"(?im)^=+ (test session starts|failures|errors) =+", body
    ) or re.search(r"(?m)^(FAILED|ERROR)\s+[^\n]+::", body):
        return StructuredTextKind("compact_pytest_v1", "Pytest")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict | list):
        return StructuredTextKind("compact_json_v1", "JSON")
    if re.search(r"(^|[;&|]\s*)(find|ls|tree)(\s|$)", command) or re.search(
        r"(?m)^total \d+\s*$", body
    ):
        return StructuredTextKind("compact_listing_v1", "Files / directories")
    if re.search(
        r"(?m)^(\d{4}-\d\d-\d\d|\[?(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]?)",
        stripped,
    ):
        return StructuredTextKind("compact_log_v1", "Log")
    if len(body) < 4000 and body.count("\n") < 80:
        return StructuredTextKind("compact_shell_v1", "Shell output")
    return StructuredTextKind("compact_text_v1", "Structured text")


def _line_style(line: str, classification: str) -> str:
    if classification == "compact_diff_v1":
        if line.startswith("+") and not line.startswith("+++"):
            return "definition"
        if line.startswith("-") and not line.startswith("---"):
            return "string"
        if line.startswith(("@@", "diff --git", "---", "+++")):
            return "metadata"
    if classification == "compact_log_v1":
        return "comment"
    return "plain"


def _styled_line(line: str, classification: str) -> list[StyledAtom]:
    base_style = _line_style(line, classification)
    matches = [
        (match.start(), match.end(), "path") for match in _PATH_TOKEN.finditer(line)
    ]
    matches.extend(
        (match.start(), match.end(), "number") for match in _NUMBER_TOKEN.finditer(line)
    )
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    atoms: list[StyledAtom] = []
    cursor = 0
    for start, end, style in matches:
        if start < cursor:
            continue
        if start > cursor:
            atoms.append(StyledAtom(line[cursor:start], base_style))
        atoms.append(StyledAtom(line[start:end], style))
        cursor = end
    if cursor < len(line):
        atoms.append(StyledAtom(line[cursor:], base_style))
    return atoms


def _flow_atoms(
    text: str, classification: str, *, preserve_indentation: bool = False
) -> list[StyledAtom]:
    # A leading space in a unified diff is a semantic context-line marker, not
    # indentation.  Other flow layouts encode hard line boundaries with arrows,
    # so physical indentation can be removed without leaving ambiguous joins.
    strip_indentation = classification != "compact_diff_v1" and not preserve_indentation
    atoms: list[StyledAtom] = []
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        display = raw.lstrip(" \t") if strip_indentation else raw
        atoms.extend(_styled_line(display, classification))
        if len(raw) != len(line):
            atoms.append(
                StyledAtom(
                    HARD_NEWLINE_MARKER,
                    "structure",
                    generated=True,
                    generated_kind="newline",
                )
            )
    return atoms


def _tag_value(text: str, tag: str) -> str:
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    start = text.find(opening)
    if start < 0:
        return ""
    start += len(opening)
    end = text.find(closing, start)
    return "" if end < 0 else text[start:end]


def _terminal_fields_manifest(
    fields: dict[str, TerminalSanitizeResult],
) -> dict[str, object]:
    return {
        "policy_version": "terminal-sanitize-v1",
        "visual_copy_only": True,
        "changed": any(item.manifest["changed"] for item in fields.values()),
        "fields": {name: item.manifest for name, item in sorted(fields.items())},
    }


def _build_truncated_document(
    observation: CompactObservationInput,
    wrapper: ParsedObservationWrapper,
) -> CompactDocument:
    raw_fields = {
        name: _tag_value(wrapper.prefix, name)
        for name in ("warning", "output_head", "elided_chars", "output_tail")
    }
    visual_fields = {
        name: sanitize_terminal_text(value) for name, value in raw_fields.items()
    }
    blame_compacted_fields: dict[str, str] = {}
    git_blame_timestamp_fields: dict[str, dict[str, object]] = {}
    for name, value in visual_fields.items():
        compacted, manifest = compact_git_blame_timestamps(value.text)
        blame_compacted_fields[name] = compacted
        git_blame_timestamp_fields[name] = manifest
    compacted_fields: dict[str, str] = {}
    sparse_padding_fields: dict[str, dict[str, object]] = {}
    for name, value in blame_compacted_fields.items():
        policy = (
            observation.visual_whitespace_policy
            if name in {"output_head", "output_tail"}
            else "preserve"
        )
        compacted, manifest = compact_visual_whitespace(value, policy)
        compacted_fields[name] = compacted
        sparse_padding_fields[name] = manifest
    raw_hash = hashlib.sha256(observation.content.encode("utf-8")).hexdigest()
    summary: list[StyledAtom] = [
        StyledAtom("warning=", "label", generated=True),
        StyledAtom(compacted_fields["warning"] or "unspecified", "plain"),
        StyledAtom(" · omitted=", "label", generated=True),
        StyledAtom(compacted_fields["elided_chars"] or "unspecified", "number"),
        StyledAtom(" · raw_sha256=", "label", generated=True),
        StyledAtom(raw_hash[:12], "metadata", generated=True),
    ]
    timestamps_omitted = sum(
        int(item["timestamps_omitted"]) for item in git_blame_timestamp_fields.values()
    )
    if timestamps_omitted:
        summary.extend(
            [
                StyledAtom(" · ", "label", generated=True),
                StyledAtom(
                    f"[git-blame timestamps omitted={timestamps_omitted}]",
                    "structure",
                    generated=True,
                    generated_kind="git_blame_timestamp_omission",
                ),
            ]
        )
    regions = [
        CompactRegion(
            kind="truncated_metadata",
            title="Truncated output",
            atoms=summary,
        ),
        CompactRegion(
            kind="truncated_head",
            title="Output head",
            atoms=_flow_atoms(
                compacted_fields["output_head"],
                "compact_truncated_v2",
                preserve_indentation=True,
            ),
        ),
        CompactRegion(
            kind="truncated_tail",
            title="Output tail",
            atoms=_flow_atoms(
                compacted_fields["output_tail"],
                "compact_truncated_v2",
                preserve_indentation=True,
            ),
        ),
    ]
    return CompactDocument(
        observation=observation,
        content_prefix=wrapper.prefix,
        content_suffix=wrapper.suffix,
        body=wrapper.body,
        wrapper_kind=wrapper.kind,
        regions=regions,
        source_file=None,
        path_mappings=[],
        source_spans=list(wrapper.spans),
        complete_python=False,
        generated_open_braces=0,
        generated_close_braces=0,
        classification="compact_truncated_v2",
        semantic_encoding="anchored_head_tail_omitted_hash_v2",
        information_preservation={
            "raw_observation_retained": True,
            "raw_observation_sha256": raw_hash,
            "source_text_deleted": True,
            "truncated_visual_policy": "anchored_head_tail_omitted_hash_v2",
            "missing_content_reconstructed": False,
            "warning_sha256": hashlib.sha256(
                raw_fields["warning"].encode("utf-8")
            ).hexdigest(),
            "head_sha256": hashlib.sha256(
                raw_fields["output_head"].encode("utf-8")
            ).hexdigest(),
            "tail_sha256": hashlib.sha256(
                raw_fields["output_tail"].encode("utf-8")
            ).hexdigest(),
            "omitted_sha256": hashlib.sha256(
                raw_fields["elided_chars"].encode("utf-8")
            ).hexdigest(),
            "terminal_sanitizer": _terminal_fields_manifest(visual_fields),
            "git_blame_timestamp_compaction": {
                "policy_version": "git-blame-timestamp-visual-omission-v1",
                "visual_copy_only": True,
                "changed": any(
                    item["changed"] for item in git_blame_timestamp_fields.values()
                ),
                "timestamps_omitted": timestamps_omitted,
                "timestamp_characters_omitted": sum(
                    int(item["timestamp_characters_omitted"])
                    for item in git_blame_timestamp_fields.values()
                ),
                "fields": git_blame_timestamp_fields,
            },
            "sparse_padding_compaction": {
                "policy_version": observation.visual_whitespace_policy,
                "visual_copy_only": True,
                "minimum_marker_run_characters": (
                    32
                    if observation.visual_whitespace_policy == _VISUAL_WHITESPACE_POLICY
                    else 64
                ),
                "moderate_space_run_minimum": (
                    8
                    if observation.visual_whitespace_policy == _VISUAL_WHITESPACE_POLICY
                    else None
                ),
                "moderate_space_run_maximum": (
                    31
                    if observation.visual_whitespace_policy == _VISUAL_WHITESPACE_POLICY
                    else None
                ),
                "fields": sparse_padding_fields,
                "changed": any(
                    item["changed"] for item in sparse_padding_fields.values()
                ),
                "run_count": sum(
                    int(item["run_count"]) for item in sparse_padding_fields.values()
                ),
                "source_characters_compacted": sum(
                    int(item["source_characters_compacted"])
                    for item in sparse_padding_fields.values()
                ),
                "markers_inserted": sum(
                    int(item["markers_inserted"])
                    for item in sparse_padding_fields.values()
                ),
                "moderate_space_runs_compacted": sum(
                    int(item["moderate_space_runs_compacted"])
                    for item in sparse_padding_fields.values()
                ),
                "moderate_space_characters_compacted": sum(
                    int(item["moderate_space_characters_compacted"])
                    for item in sparse_padding_fields.values()
                ),
                "binary_like_runs_marked": sum(
                    int(item["binary_like_runs_marked"])
                    for item in sparse_padding_fields.values()
                ),
            },
        },
    )


def build_structured_text_document(
    observation: CompactObservationInput,
    wrapper: ParsedObservationWrapper,
) -> CompactDocument | None:
    if wrapper.truncated:
        return _build_truncated_document(observation, wrapper)

    terminal = sanitize_terminal_text(wrapper.body)
    blame_compacted_body, git_blame_timestamps = compact_git_blame_timestamps(
        terminal.text
    )
    compacted_body, sparse_padding = compact_visual_whitespace(
        blame_compacted_body, observation.visual_whitespace_policy
    )
    visual_wrapper = replace(wrapper, body=compacted_body)
    kind = classify_structured_text(observation, visual_wrapper)
    if (
        kind.preserve_original
        and not sparse_padding["changed"]
        and not git_blame_timestamps["changed"]
    ):
        return None
    strip_indentation = (
        not wrapper.truncated
        and kind.classification != "compact_diff_v1"
        and not kind.preserve_original
    )
    leading_whitespace_removed = 0
    if strip_indentation:
        for line in visual_wrapper.body.splitlines(keepends=True):
            raw = line.rstrip("\r\n")
            leading_whitespace_removed += len(raw) - len(raw.lstrip(" \t"))
    if not visual_wrapper.body.strip():
        returncode = _RETURNCODE.search(wrapper.prefix)
        returncode_value = returncode.group("value") if returncode else "unknown"
        sentinel = "empty-success" if returncode_value == "0" else "empty-nonzero"
        regions = [
            CompactRegion(
                kind="empty_output",
                title="Empty output",
                atoms=[StyledAtom(sentinel, "structure", generated=True)],
                source_start_line=1,
                source_end_line=1,
            )
        ]
    else:
        region_atoms: list[StyledAtom] = []
        if git_blame_timestamps["changed"]:
            region_atoms.extend(
                [
                    StyledAtom(
                        "[git-blame timestamps omitted={}]".format(
                            git_blame_timestamps["timestamps_omitted"]
                        ),
                        "structure",
                        generated=True,
                        generated_kind="git_blame_timestamp_omission",
                    ),
                    StyledAtom(
                        HARD_NEWLINE_MARKER,
                        "structure",
                        generated=True,
                        generated_kind="newline",
                    ),
                ]
            )
        region_atoms.extend(_flow_atoms(visual_wrapper.body, kind.classification))
        regions = [
            CompactRegion(
                kind="structured_text",
                title=kind.title,
                atoms=region_atoms,
                source_start_line=1,
                source_end_line=wrapper.body.count("\n") + 1,
            )
        ]
    body_hash = hashlib.sha256(wrapper.body.encode("utf-8")).hexdigest()
    visual_body_hash = hashlib.sha256(visual_wrapper.body.encode("utf-8")).hexdigest()
    return CompactDocument(
        observation=observation,
        content_prefix=wrapper.prefix,
        content_suffix=wrapper.suffix,
        body=wrapper.body,
        wrapper_kind=wrapper.kind,
        regions=regions,
        source_file=None,
        path_mappings=[],
        source_spans=list(wrapper.spans),
        complete_python=False,
        generated_open_braces=0,
        generated_close_braces=0,
        classification=kind.classification,
        semantic_encoding=(
            "line_leading_whitespace_stripped_hard_newline_flow_v2"
            if strip_indentation
            else "lossless_hard_newline_flow_v1"
        ),
        information_preservation={
            "source_body_sha256": body_hash,
            "visual_body_sha256": visual_body_hash,
            "source_characters": len(observation.content),
            "raw_observation_retained": True,
            "source_text_deleted": (
                leading_whitespace_removed > 0
                or terminal.manifest["changed"]
                or git_blame_timestamps["changed"]
                or sparse_padding["changed"]
            ),
            "hard_newlines_encoded_as": HARD_NEWLINE_GLYPH,
            "soft_wrap_marker": None,
            "line_leading_whitespace_policy": (
                "strip_spaces_and_tabs" if strip_indentation else "preserve"
            ),
            "line_leading_whitespace_removed": leading_whitespace_removed,
            "terminal_sanitizer": terminal.manifest,
            "git_blame_timestamp_compaction": git_blame_timestamps,
            "sparse_padding_compaction": sparse_padding,
        },
    )
