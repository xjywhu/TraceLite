from __future__ import annotations

import ast
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from swe_memory_policy.image_response_rencent3text.tools.model import (
    CompactObservationInput,
    PathMapping,
    SourceSpan,
)


@dataclass(frozen=True)
class ParsedObservationWrapper:
    prefix: str
    body: str
    suffix: str
    kind: str
    spans: tuple[SourceSpan, ...]
    truncated: bool


def _consume_tag(text: str, position: int, tag: str) -> tuple[int, str] | None:
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    if not text.startswith(opening, position):
        return None
    end = text.find(closing, position + len(opening))
    if end < 0:
        return None
    end += len(closing)
    return end, text[position:end]


def parse_mini_swe_observation(content: str) -> ParsedObservationWrapper | None:
    """Parse only the known, anchored mini-SWE observation envelope.

    This deliberately avoids treating arbitrary XML-like text inside command output
    as protocol metadata.
    """

    position = 0
    if content.startswith("<exception>"):
        consumed = _consume_tag(content, position, "exception")
        if consumed is None:
            return None
        position = consumed[0]
        if content.startswith("\n", position):
            position += 1
    consumed = _consume_tag(content, position, "returncode")
    if consumed is None:
        return None
    position = consumed[0]
    if content.startswith("\n", position):
        position += 1

    if content.startswith("<output>", position):
        opening_end = position + len("<output>")
        closing_start = content.rfind("</output>")
        if closing_start < opening_end or closing_start + len("</output>") != len(
            content
        ):
            return None
        return ParsedObservationWrapper(
            prefix=content[:opening_end],
            body=content[opening_end:closing_start],
            suffix=content[closing_start:],
            kind="output",
            spans=(
                SourceSpan(0, opening_end, "wrapper_prefix"),
                SourceSpan(opening_end, closing_start, "output_body"),
                SourceSpan(closing_start, len(content), "wrapper_suffix"),
            ),
            truncated=False,
        )

    # A long-output envelope is kept intact and sent to the legacy renderer.  We
    # still recognize it so that the body is never mistaken for contiguous Python.
    long_tags = ("warning", "output_head", "elided_chars", "output_tail")
    cursor = position
    for tag in long_tags:
        consumed = _consume_tag(content, cursor, tag)
        if consumed is None:
            return None
        cursor = consumed[0]
        if cursor < len(content) and content[cursor] == "\n":
            cursor += 1
    if cursor != len(content):
        return None
    return ParsedObservationWrapper(
        prefix=content,
        body="",
        suffix="",
        kind="truncated_output",
        spans=(SourceSpan(0, len(content), "truncated_wrapper"),),
        truncated=True,
    )


def _command_payload(arguments: str | None) -> str:
    if not arguments:
        return ""
    try:
        value: Any = json.loads(arguments)
    except json.JSONDecodeError:
        return ""
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return ""
    return value["command"]


def _inside_root(path: PurePosixPath, root: PurePosixPath) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def display_project_path(path: str, project_root: str) -> str | None:
    candidate = PurePosixPath(path)
    root = PurePosixPath(project_root)
    if not candidate.is_absolute() or not _inside_root(candidate, root):
        return None
    relative = candidate.relative_to(root)
    return "~/" + relative.as_posix() if relative.parts else "~/"


def source_file_from_tool(
    arguments: str | None, project_root: str
) -> tuple[str | None, list[PathMapping]]:
    command = _command_payload(arguments)
    if not command:
        return None, []
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None, []
    root = PurePosixPath(project_root)
    cwd = root
    if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] in {"&&", ";"}:
        raw_cwd = PurePosixPath(tokens[1])
        if raw_cwd.is_absolute() and _inside_root(raw_cwd, root):
            cwd = raw_cwd

    source: PurePosixPath | None = None
    readers = {"cat", "sed", "head", "tail"}
    for index, token in enumerate(tokens):
        command_name = PurePosixPath(token).name
        if command_name not in readers:
            continue
        for candidate in tokens[index + 1 :]:
            if candidate in {"|", "||", "&&", ";"}:
                break
            if candidate.startswith("-") or (
                candidate.endswith("p") and "," in candidate
            ):
                continue
            if candidate.endswith(".py"):
                raw = PurePosixPath(candidate)
                source = raw if raw.is_absolute() else cwd / raw
                break
        if source is not None:
            break
    if source is None or not _inside_root(source, root):
        return None, []
    display = display_project_path(source.as_posix(), project_root)
    if display is None:
        return None, []
    return display, [
        PathMapping(
            original=source.as_posix(),
            display=display,
            source_kind="tool_argument",
        )
    ]


_ABSOLUTE_LOCATION = re.compile(
    r"(?P<path>/[^\s\",]+?)(?P<location>:\d+(?::\d+)?)?(?=$|[\s\",])"
)


def rewrite_location_paths(
    text: str, project_root: str, *, source_offset: int = 0
) -> tuple[str, list[PathMapping]]:
    """Rewrite paths only in syntactically file-bearing record lines."""

    result: list[str] = []
    mappings: list[PathMapping] = []
    absolute = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        file_context = (
            stripped.startswith('File "')
            or stripped.startswith("--- /")
            or stripped.startswith("+++ /")
            or stripped.startswith("/" + PurePosixPath(project_root).name + "/")
        )
        if not file_context:
            result.append(line)
            absolute += len(line)
            continue
        cursor = 0
        rewritten: list[str] = []
        for match in _ABSOLUTE_LOCATION.finditer(line):
            display = display_project_path(match.group("path"), project_root)
            if display is None:
                continue
            location = match.group("location") or ""
            replacement = display + location
            rewritten.append(line[cursor : match.start()])
            rewritten.append(replacement)
            mappings.append(
                PathMapping(
                    original=match.group(0),
                    display=replacement,
                    source_kind="location_record",
                    source_start=source_offset + absolute + match.start(),
                    source_end=source_offset + absolute + match.end(),
                )
            )
            cursor = match.end()
        rewritten.append(line[cursor:])
        result.append("".join(rewritten))
        absolute += len(line)
    return "".join(result), mappings


def _looks_like_non_python(body: str) -> bool:
    stripped = body.lstrip("\r\n")
    if "Traceback (most recent call last):" in body:
        return True
    if re.search(r"(?m)^(diff --git |@@ |--- a/|\+\+\+ b/)", body):
        return True
    if re.search(r"(?im)^=+ (test session starts|failures|errors) =+", body):
        return True
    if re.search(r"(?im)^(FAILED|ERROR)\s+[^\n]+::", body):
        return True
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    return isinstance(parsed, dict | list)


def classify_long_python(
    observation: CompactObservationInput,
    wrapper: ParsedObservationWrapper | None,
) -> tuple[bool, bool]:
    """Return ``(is_long_python, parses_as_complete_module)``."""

    if wrapper is None or wrapper.truncated or wrapper.kind != "output":
        return False, False
    body = wrapper.body
    if _looks_like_non_python(body):
        return False, False
    source_file, _ = source_file_from_tool(
        observation.tool_arguments, observation.project_root
    )
    command_confirms_python = source_file is not None and source_file.endswith(".py")
    complete = False
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError):
        tree = None
    else:
        complete = True
    # Python source excerpts may begin inside a suite, so a full-module AST
    # or length cutoff would miss them.
    if command_confirms_python:
        return bool(body.strip()), complete
    if len(body) < 4000 and body.count("\n") + 1 < 80:
        return False, False
    if tree is None:
        return False, False
    has_source_nodes = any(
        isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        for node in tree.body
    )
    return has_source_nodes, complete
