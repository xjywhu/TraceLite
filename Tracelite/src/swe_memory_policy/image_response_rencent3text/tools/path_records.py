from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from swe_memory_policy.image_response_rencent3text.tools.model import (
    HARD_NEWLINE_MARKER,
    CompactDocument,
    CompactObservationInput,
    CompactRegion,
    PathMapping,
    StyledAtom,
)
from swe_memory_policy.image_response_rencent3text.tools.parsing import (
    ParsedObservationWrapper,
    display_project_path,
)
from swe_memory_policy.image_response_rencent3text.tools.python_compact import (
    PythonCommentRemoval,
    strip_python_comments,
)


@dataclass(frozen=True)
class PathRecordCommand:
    command_kind: str
    cwd: PurePosixPath
    includes_column: bool


@dataclass(frozen=True)
class ParsedPathRecord:
    observed_path: str
    resolved_path: str
    display_path: str
    line: str
    column: str | None
    payload: str
    line_ending: str
    raw_text: str
    record_index: int
    record_line: int
    source_start: int
    source_end: int
    path_source_start: int
    path_source_end: int

    def manifest(self) -> dict[str, object]:
        return {
            "record_index": self.record_index,
            "record_line": self.record_line,
            "observed_path": self.observed_path,
            "resolved_path": self.resolved_path,
            "display_path": self.display_path,
            "line": self.line,
            "column": self.column,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "path_source_start": self.path_source_start,
            "path_source_end": self.path_source_end,
            "payload_sha256": hashlib.sha256(self.payload.encode("utf-8")).hexdigest(),
            "payload_characters": len(self.payload),
            "line_ending": self.line_ending,
        }


def _inside_root(path: PurePosixPath, root: PurePosixPath) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalize_path(path: PurePosixPath) -> PurePosixPath:
    return PurePosixPath(posixpath.normpath(path.as_posix()))


def _resolve_inside_root(
    observed: str, *, cwd: PurePosixPath, root: PurePosixPath
) -> PurePosixPath | None:
    candidate = PurePosixPath(observed)
    resolved = candidate if candidate.is_absolute() else cwd / candidate
    resolved = _normalize_path(resolved)
    if not resolved.is_absolute() or not _inside_root(resolved, root):
        return None
    return resolved


def _command_payload(arguments: str | None) -> str:
    if not arguments:
        return ""
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return ""
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return ""
    return value["command"]


def _short_flags(tokens: list[str]) -> set[str]:
    flags: set[str] = set()
    for item in tokens:
        if not item.startswith("-") or item.startswith("--") or item == "-":
            continue
        flags.update(character for character in item[1:] if character.isalpha())
    return flags


def parse_path_record_command(
    arguments: str | None, project_root: str
) -> PathRecordCommand | None:
    """Recognize only grep/rg commands that explicitly request line records."""

    command = _command_payload(arguments)
    if not command:
        return None
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None
    root = PurePosixPath(project_root)
    cwd = root
    cursor = 0
    if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] in {"&&", ";"}:
        resolved_cwd = _resolve_inside_root(tokens[1], cwd=root, root=root)
        if resolved_cwd is None:
            return None
        cwd = resolved_cwd
        cursor = 3

    separators = {"|", "||", "&&", ";"}
    while cursor < len(tokens):
        end = cursor
        while end < len(tokens) and tokens[end] not in separators:
            end += 1
        segment = tokens[cursor:end]
        if segment:
            command_name = PurePosixPath(segment[0]).name
            flags = _short_flags(segment[1:])
            long_flags = {item for item in segment[1:] if item.startswith("--")}
            if command_name in {"grep", "egrep", "fgrep"}:
                has_lines = "n" in flags or "--line-number" in long_flags
                has_filenames = (
                    "H" in flags
                    or "--with-filename" in long_flags
                    or "r" in flags
                    or "R" in flags
                    or "--recursive" in long_flags
                )
                if has_lines and has_filenames:
                    return PathRecordCommand("grep", cwd, False)
            elif command_name in {"rg", "ripgrep"}:
                has_lines = "n" in flags or "--line-number" in long_flags
                if has_lines:
                    return PathRecordCommand("rg", cwd, "--column" in long_flags)
        cursor = end + 1
    return None


def _split_line_ending(text: str) -> tuple[str, str]:
    if text.endswith("\r\n"):
        return text[:-2], "\r\n"
    if text.endswith("\n") or text.endswith("\r"):
        return text[:-1], text[-1]
    return text, ""


def _record_pattern(*, includes_column: bool) -> re.Pattern[str]:
    if includes_column:
        return re.compile(
            r"^(?P<path>[^:\r\n]+):(?P<line>[1-9]\d*):"
            r"(?P<column>[1-9]\d*):(?P<payload>.*)$"
        )
    return re.compile(r"^(?P<path>[^:\r\n]+):(?P<line>[1-9]\d*):(?P<payload>.*)$")


def _python_record_comment_removal(record: ParsedPathRecord) -> PythonCommentRemoval:
    if PurePosixPath(record.resolved_path).suffix.lower() not in {".py", ".pyi"}:
        return PythonCommentRemoval(record.payload, 0, 0, 0)
    return strip_python_comments(record.payload)


def _record_atoms(record: ParsedPathRecord) -> list[StyledAtom]:
    visible_payload = _python_record_comment_removal(record).text
    atoms = [StyledAtom(record.line, "number")]
    if record.column is not None:
        atoms.extend([StyledAtom(":", "operator"), StyledAtom(record.column, "number")])
    atoms.extend([StyledAtom(":", "operator"), StyledAtom(visible_payload, "plain")])
    if record.line_ending:
        atoms.append(
            StyledAtom(
                HARD_NEWLINE_MARKER,
                "structure",
                generated=True,
                generated_kind="newline",
            )
        )
    return atoms


def _reconstruct(prefix: str, records: list[ParsedPathRecord], suffix: str) -> str:
    return prefix + "".join(record.raw_text for record in records) + suffix


def build_path_record_document(
    observation: CompactObservationInput,
    wrapper: ParsedObservationWrapper,
) -> CompactDocument | None:
    """Build a reversible grouped layout for strict grep/rg location output."""

    if wrapper.truncated or wrapper.kind != "output":
        return None
    if "<returncode>0</returncode>" not in wrapper.prefix:
        return None
    command = parse_path_record_command(
        observation.tool_arguments, observation.project_root
    )
    if command is None:
        return None
    lines = wrapper.body.splitlines(keepends=True)
    if not lines:
        return None
    first = 0
    while first < len(lines) and not _split_line_ending(lines[first])[0].strip():
        first += 1
    last = len(lines)
    while last > first and not _split_line_ending(lines[last - 1])[0].strip():
        last -= 1
    if first == last:
        return None
    body_prefix = "".join(lines[:first])
    body_suffix = "".join(lines[last:])
    pattern = _record_pattern(includes_column=command.includes_column)
    root = PurePosixPath(observation.project_root)
    records: list[ParsedPathRecord] = []
    body_cursor = len(body_prefix)
    for source_line, raw_line in enumerate(lines[first:last], start=first + 1):
        line, ending = _split_line_ending(raw_line)
        if "\x00" in line or "\x1b" in line:
            return None
        match = pattern.fullmatch(line)
        if match is None:
            return None
        observed = match.group("path")
        if observed.startswith("-"):
            return None
        resolved = _resolve_inside_root(observed, cwd=command.cwd, root=root)
        if resolved is None:
            return None
        display = display_project_path(resolved.as_posix(), observation.project_root)
        if display is None:
            return None
        absolute_start = len(wrapper.prefix) + body_cursor
        path_start = absolute_start + match.start("path")
        records.append(
            ParsedPathRecord(
                observed_path=observed,
                resolved_path=resolved.as_posix(),
                display_path=display,
                line=match.group("line"),
                column=(match.group("column") if command.includes_column else None),
                payload=match.group("payload"),
                line_ending=ending,
                raw_text=raw_line,
                record_index=len(records) + 1,
                record_line=source_line,
                source_start=absolute_start,
                source_end=absolute_start + len(raw_line),
                path_source_start=path_start,
                path_source_end=path_start + len(observed),
            )
        )
        body_cursor += len(raw_line)

    # Even a short result benefits from the reversible path/line layout and should
    # not fall back merely because every match happens to be in a different file.
    reconstructed = _reconstruct(body_prefix, records, body_suffix)
    if reconstructed != wrapper.body:
        return None

    regions: list[CompactRegion] = []
    group_records: list[ParsedPathRecord] = []

    def flush_group() -> None:
        if not group_records:
            return
        atoms: list[StyledAtom] = []
        for item in group_records:
            atoms.extend(_record_atoms(item))
        regions.append(
            CompactRegion(
                kind="path_records",
                title=f"FILE {group_records[0].display_path}",
                atoms=atoms,
                source_start_line=group_records[0].record_line,
                source_end_line=group_records[-1].record_line,
            )
        )

    for record in records:
        if group_records and group_records[-1].resolved_path != record.resolved_path:
            flush_group()
            group_records = []
        group_records.append(record)
    flush_group()

    path_mappings = [
        PathMapping(
            original=record.resolved_path,
            display=record.display_path,
            source_kind=(
                f"{command.command_kind}_absolute_location_record"
                if PurePosixPath(record.observed_path).is_absolute()
                else f"{command.command_kind}_resolved_relative_location_record"
            ),
            source_start=record.path_source_start,
            source_end=record.path_source_end,
            observed=record.observed_path,
            resolution_base=command.cwd.as_posix(),
        )
        for record in records
    ]
    body_hash = hashlib.sha256(wrapper.body.encode("utf-8")).hexdigest()
    python_comment_removals = [
        _python_record_comment_removal(record) for record in records
    ]
    return CompactDocument(
        observation=observation,
        content_prefix=wrapper.prefix,
        content_suffix=wrapper.suffix,
        body=wrapper.body,
        wrapper_kind=wrapper.kind,
        regions=regions,
        source_file=None,
        path_mappings=path_mappings,
        source_spans=list(wrapper.spans),
        complete_python=False,
        generated_open_braces=0,
        generated_close_braces=0,
        classification="compact_path_records_v1",
        semantic_encoding="grouped_location_records_v1",
        body_prefix=body_prefix,
        body_suffix=body_suffix,
        path_records=[record.manifest() for record in records],
        information_preservation={
            "exact_body_reconstruction": True,
            "body_sha256": body_hash,
            "reconstructed_body_sha256": hashlib.sha256(
                reconstructed.encode("utf-8")
            ).hexdigest(),
            "body_characters": len(wrapper.body),
            "reconstructed_characters": len(reconstructed),
            "record_count": len(records),
            "group_count": len(regions),
            "command_kind": command.command_kind,
            "command_cwd": command.cwd.as_posix(),
            "relative_paths_resolved_from_cwd": any(
                not PurePosixPath(record.observed_path).is_absolute()
                for record in records
            ),
            "raw_observation_retained": True,
            "visual_python_comment_policy": "strip_token_comments_v1",
            "visual_python_comments_removed": sum(
                removal.comment_count for removal in python_comment_removals
            ),
            "visual_python_comment_characters_removed": sum(
                removal.comment_characters for removal in python_comment_removals
            ),
            "visual_python_comment_lines_removed": sum(
                removal.comment_lines for removal in python_comment_removals
            ),
            "visual_python_docstring_policy": (
                "preserve_without_complete_ast_context_fail_closed"
            ),
            "visual_python_docstrings_removed": 0,
            "ordinary_triple_quoted_strings_retained": True,
        },
    )
