from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from swe_memory_policy.image_response_rencent3text.tools.model import (
    HARD_NEWLINE_GLYPH,
    HARD_NEWLINE_MARKER,
    CompactDocument,
    CompactObservationInput,
    CompactRenderResult,
    StyledAtom,
)
from swe_memory_policy.image_response_rencent3text.tools.parsing import (
    classify_long_python,
    parse_mini_swe_observation,
    source_file_from_tool,
)
from swe_memory_policy.image_response_rencent3text.tools.path_records import (
    build_path_record_document,
)
from swe_memory_policy.image_response_rencent3text.tools.python_compact import (
    brace_counts,
    build_python_regions,
    strip_python_nonsemantic_text,
)
from swe_memory_policy.image_response_rencent3text.tools.structured_text import (
    build_structured_text_document,
    compact_git_blame_timestamps,
    compact_visual_whitespace,
)
from swe_memory_policy.image_response_rencent3text.tools.terminal_sanitize import (
    sanitize_terminal_text,
)
from swe_memory_policy.rendering import FontFace, FontResolver

WIDTH = 1536
WIDTH_TIERS = (768, 1152, 1536)
FONT_SIZE = 18
LINE_HEIGHT = 25
OUTER_PADDING = 20
SECTION_PADDING_X = 12
SECTION_PADDING_Y = 9
SECTION_GAP = 10
HEADER_HEIGHT = 44
BOTTOM_PADDING = 20

COLORS = {
    "canvas": "#F7F8FA",
    "header_background": "#E6EDF8",
    "header_text": "#172033",
    "content_background": "#FFFFFF",
    "path_background": "#EAF7EF",
    "path_badge_background": "#DDF3E5",
    "path_badge": "#16803C",
    "region_background": "#EEF0F3",
    "region_border": "#D7DADE",
    "plain": "#20242A",
    "metadata": "#172033",
    "markup": "#2563EB",
    "path": "#16803C",
    "keyword": "#2563EB",
    "definition": "#16803C",
    "string": "#C65D18",
    "number": "#7C3AED",
    "decorator": "#A21CAF",
    "operator": "#334155",
    "builtin": "#0F766E",
    "comment": "#59616B",
    "structure": "#6B7280",
    "label": "#59616B",
}

_MARKUP = re.compile(r"(<[^>]+>)")
_RETURNCODE = re.compile(r"<returncode>(?P<value>[^<]*)</returncode>")
_EXCEPTION = re.compile(r"<exception>(?P<value>.*?)</exception>", re.DOTALL)


@dataclass(frozen=True)
class VisualLine:
    atoms: tuple[tuple[str, str, FontFace], ...]


@dataclass(frozen=True)
class LayoutSection:
    kind: str
    title: str | None
    background: str
    lines: tuple[VisualLine, ...]
    y: int
    height: int


def _visible_markup(text: str) -> list[StyledAtom]:
    text = (
        text.replace("\r\n", HARD_NEWLINE_MARKER)
        .replace("\r", HARD_NEWLINE_MARKER)
        .replace("\n", HARD_NEWLINE_MARKER)
    )
    atoms: list[StyledAtom] = []
    cursor = 0
    for match in _MARKUP.finditer(text):
        if match.start() > cursor:
            atoms.append(StyledAtom(text[cursor : match.start()], "plain"))
        atoms.append(StyledAtom(match.group(0), "markup"))
        cursor = match.end()
    if cursor < len(text):
        atoms.append(StyledAtom(text[cursor:], "plain"))
    return atoms


def build_compact_document(
    observation: CompactObservationInput,
) -> CompactDocument | None:
    wrapper = parse_mini_swe_observation(observation.content)
    if wrapper is None:
        return None
    path_records = build_path_record_document(observation, wrapper)
    if path_records is not None:
        terminal = sanitize_terminal_text(wrapper.body)
        path_records.information_preservation["terminal_sanitizer"] = terminal.manifest
        return path_records
    if wrapper.truncated:
        return build_structured_text_document(observation, wrapper)

    terminal = sanitize_terminal_text(wrapper.body)
    blame_compacted_body, git_blame_timestamps = compact_git_blame_timestamps(
        terminal.text
    )
    visual_wrapper = replace(wrapper, body=blame_compacted_body)
    is_python, complete = classify_long_python(observation, visual_wrapper)
    if not is_python:
        return build_structured_text_document(observation, wrapper)
    source_file, path_mappings = source_file_from_tool(
        observation.tool_arguments, observation.project_root
    )
    regions = build_python_regions(blame_compacted_body, complete=complete)
    if git_blame_timestamps["changed"] and regions:
        regions[0].atoms[0:0] = [
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
    opens, closes = brace_counts(regions)
    visual_removal = strip_python_nonsemantic_text(blame_compacted_body)
    return CompactDocument(
        observation=observation,
        content_prefix=wrapper.prefix,
        content_suffix=wrapper.suffix,
        body=wrapper.body,
        wrapper_kind=wrapper.kind,
        regions=regions,
        source_file=source_file,
        path_mappings=path_mappings,
        source_spans=list(wrapper.spans),
        complete_python=complete,
        generated_open_braces=opens,
        generated_close_braces=closes,
        information_preservation={
            "raw_observation_retained": True,
            "visual_python_comment_policy": "strip_token_comments_v1",
            "visual_python_comments_removed": visual_removal.comment_count,
            "visual_python_comment_characters_removed": (
                visual_removal.comment_characters
            ),
            "visual_python_comment_lines_removed": visual_removal.comment_lines,
            "visual_python_docstring_policy": (
                "strip_ast_module_class_function_docstrings_v1"
            ),
            "visual_python_docstrings_removed": visual_removal.docstring_count,
            "visual_python_docstring_characters_removed": (
                visual_removal.docstring_characters
            ),
            "visual_python_docstring_lines_removed": visual_removal.docstring_lines,
            "visual_python_docstring_ast_complete": (
                visual_removal.docstring_ast_complete
            ),
            "ordinary_triple_quoted_strings_retained": True,
            "source_body_sha256": hashlib.sha256(
                wrapper.body.encode("utf-8")
            ).hexdigest(),
            "visual_body_sha256": hashlib.sha256(
                blame_compacted_body.encode("utf-8")
            ).hexdigest(),
            "terminal_sanitizer": terminal.manifest,
            "git_blame_timestamp_compaction": git_blame_timestamps,
            "source_text_deleted": bool(
                terminal.manifest["changed"] or git_blame_timestamps["changed"]
            ),
        },
    )


def _wrap_atoms(
    atoms: list[StyledAtom], resolver: FontResolver, available_width: int
) -> tuple[list[VisualLine], int]:
    lines: list[VisualLine] = []
    current: list[tuple[str, str, FontFace]] = []
    width = 0.0
    soft_wraps = 0
    for atom in atoms:
        for source_character in atom.text:
            for character, face in resolver.display_units(source_character):
                advance = face.font.getlength(character)
                if current and width + advance > available_width:
                    lines.append(VisualLine(tuple(current)))
                    current = []
                    width = 0.0
                    soft_wraps += 1
                current.append((character, atom.style, face))
                width += advance
    lines.append(VisualLine(tuple(current)))
    return lines, soft_wraps


def _draw_line(draw: ImageDraw.ImageDraw, line: VisualLine, x: int, y: int) -> None:
    cursor = float(x)
    run: list[str] = []
    run_style: str | None = None
    run_face: FontFace | None = None

    def flush() -> None:
        nonlocal cursor, run
        if not run or run_face is None or run_style is None:
            return
        text = "".join(run)
        advance = run_face.font.getlength(text)
        if run_style == "path_badge":
            draw.rounded_rectangle(
                (cursor - 3, y - 2, cursor + advance + 3, y + LINE_HEIGHT - 2),
                radius=5,
                fill=COLORS["path_badge_background"],
            )
        draw.text(
            (cursor, y),
            text,
            font=run_face.font,
            fill=COLORS.get(run_style, COLORS["plain"]),
        )
        cursor += advance
        run = []

    for character, style, face in line.atoms:
        if run and (style != run_style or face != run_face):
            flush()
        run_style = style
        run_face = face
        run.append(character)
    flush()


def _status_atoms(document: CompactDocument) -> list[StyledAtom]:
    """Render non-return-code protocol status without the output envelope."""

    # The v2 truncated document has explicit summary/head/tail regions.  Reusing
    # its raw wrapper here would duplicate content and reintroduce terminal bytes.
    if document.wrapper_kind == "truncated_output":
        return []

    atoms: list[StyledAtom] = []
    exception = _EXCEPTION.search(document.content_prefix)
    if exception is not None:
        atoms.append(StyledAtom("exception=", "label"))
        atoms.extend(_visible_markup(exception.group("value")))

    if _RETURNCODE.search(document.content_prefix) is None:
        # The anchored parser normally makes this unreachable.  Failing closed
        # keeps malformed future envelopes visible rather than deleting metadata.
        atoms.extend(_visible_markup(document.content_prefix))
    atoms.extend(_visible_markup(document.body_prefix))
    return atoms


def _header_atoms(document: CompactDocument) -> list[StyledAtom]:
    """Build a stable, compact visual header; retain full metadata in audit."""

    observation = document.observation
    call_digest = hashlib.sha256(observation.tool_call_id.encode("utf-8")).hexdigest()[
        :10
    ]
    atoms = [
        StyledAtom(
            f"OA{observation.observation_index} · "
            f"Tool {observation.tool_message_index} · "
            f"page {observation.page_index}/{observation.page_count} · "
            f"call:{call_digest}",
            "metadata",
            generated=True,
        )
    ]
    returncode = _RETURNCODE.search(document.content_prefix)
    if returncode is not None:
        atoms.append(StyledAtom(" · ", "structure", generated=True))
        atoms.append(StyledAtom("returncode=", "label", generated=True))
        atoms.append(StyledAtom(returncode.group("value"), "number"))
    return atoms


def _region_title_atoms(
    document: CompactDocument, region_index: int
) -> list[StyledAtom]:
    region = document.regions[region_index]
    if region.kind == "path_records" and region.title.startswith("FILE "):
        return [
            StyledAtom("FILE ", "label"),
            StyledAtom(region.title.removeprefix("FILE "), "path"),
        ]
    atoms = [StyledAtom(region.title, "label")]
    if region_index == 0 and document.source_file:
        atoms.extend(
            [
                StyledAtom(" ·  ", "structure", generated=True),
                StyledAtom(
                    f"FILE {document.source_file}",
                    "path_badge",
                    generated=True,
                ),
            ]
        )
    return atoms


def _logical_atom_width(atoms: list[StyledAtom], resolver: FontResolver) -> float:
    width = 0.0
    maximum = 0.0
    for atom in atoms:
        for source_character in atom.text:
            for character, face in resolver.display_units(
                source_character, record_fallback=False
            ):
                width += face.font.getlength(character)
        if atom.generated_kind == "newline":
            maximum = max(maximum, width)
            width = 0.0
    return max(maximum, width)


def _select_canvas_width(
    document: CompactDocument, resolver: FontResolver
) -> tuple[int, dict[str, Any]]:
    critical = {
        "compact_python_v1",
        "compact_path_records_v1",
        "compact_diff_v1",
        "compact_error_v1",
        "compact_traceback_v1",
        "compact_pytest_v1",
    }
    minimum = 1152 if document.classification in critical else 768
    candidates = [
        _logical_atom_width(_header_atoms(document), resolver) + 2 * OUTER_PADDING,
        _logical_atom_width(_content_atoms(document), resolver) + 2 * OUTER_PADDING,
        _logical_atom_width(_suffix_atoms(document), resolver) + 2 * OUTER_PADDING,
    ]
    for index, region in enumerate(document.regions):
        candidates.append(
            _logical_atom_width(_region_title_atoms(document, index), resolver)
            + 2 * (OUTER_PADDING + SECTION_PADDING_X)
        )
        candidates.append(
            _logical_atom_width(region.atoms, resolver)
            + 2 * (OUTER_PADDING + SECTION_PADDING_X)
        )
    natural = max(candidates, default=float(minimum))
    required = max(float(minimum), natural)
    selected = _select_width_tier(required)
    return selected, {
        "policy_version": "content-width-tiers-v1",
        "tiers": list(WIDTH_TIERS),
        "critical_minimum_width": 1152,
        "classification_minimum_width": minimum,
        "measured_natural_width": int(natural + 0.999999),
        "selected_base_width": selected,
        "generated_at_selected_width": True,
        "resampled_up": False,
        "overflow_soft_wrapped": natural > selected,
    }


def _select_width_tier(required_width: float) -> int:
    """Choose a deterministic base canvas without resizing source pixels."""

    if required_width < 0:
        raise ValueError("required_width must be non-negative")
    return next(
        (tier for tier in WIDTH_TIERS if tier >= required_width), WIDTH_TIERS[-1]
    )


def _content_atoms(document: CompactDocument) -> list[StyledAtom]:
    return _status_atoms(document)


def _suffix_atoms(document: CompactDocument) -> list[StyledAtom]:
    # ``</output>`` is protocol framing, not observation content.  A path-record
    # document may still have ungrouped body suffix text that must remain visible.
    return _visible_markup(document.body_suffix)


def _transcript(document: CompactDocument) -> str:
    status = "".join(atom.text for atom in _status_atoms(document))
    header = "".join(atom.text for atom in _header_atoms(document))
    parts = [header]
    if status:
        parts.append(status)
    for region_index, region in enumerate(document.regions):
        title = f"[{region.title}]"
        if region_index == 0 and document.source_file:
            title += f" · FILE {document.source_file}"
        parts.append(title)
        parts.append(region.plain_text)
    if document.body_suffix:
        parts.append("".join(atom.text for atom in _suffix_atoms(document)))
    return "\n".join(parts) + "\n"


def _debug_fragments(document: CompactDocument) -> dict[str, str]:
    paths = ""
    if document.source_file:
        paths += f"FILE {document.source_file}\n"
    for mapping in document.path_mappings:
        paths += (
            f"- `{mapping.display}` ← `{mapping.original}` ({mapping.source_kind})\n"
        )
    code: list[str] = []
    if document.classification == "compact_path_records_v1":
        for region in document.regions:
            paths += f"\n## {region.title}\n\n{region.plain_text}\n"
    else:
        for region in document.regions:
            code.extend([f"## {region.title}", "", region.plain_text, ""])
    other = document.content_prefix + document.content_suffix
    return {
        "lujing.md": paths,
        "defcode.md": "\n".join(code),
        "other.md": other,
    }


def _compact_document_region_whitespace(document: CompactDocument) -> None:
    """Apply the shared visual-only whitespace policy to every compact layout."""

    if "sparse_padding_compaction" in document.information_preservation:
        return
    policy = document.observation.visual_whitespace_policy
    before_parts: list[str] = []
    after_parts: list[str] = []
    manifests: list[dict[str, object]] = []
    for region in document.regions:
        compacted_atoms: list[StyledAtom] = []
        for atom in region.atoms:
            before_parts.append(atom.text)
            compacted, manifest = compact_visual_whitespace(atom.text, policy)
            after_parts.append(compacted)
            manifests.append(manifest)
            compacted_atoms.append(replace(atom, text=compacted))
        region.atoms = compacted_atoms
    if not manifests:
        _empty, manifest = compact_visual_whitespace("", policy)
        manifests.append(manifest)
    before = "".join(before_parts)
    after = "".join(after_parts)
    first = manifests[0]
    audit = {
        "policy_version": policy,
        "visual_copy_only": True,
        "minimum_marker_run_characters": first.get("minimum_marker_run_characters"),
        "moderate_space_run_minimum": first.get("moderate_space_run_minimum"),
        "moderate_space_run_maximum": first.get("moderate_space_run_maximum"),
        "moderate_space_replacement_characters": first.get(
            "moderate_space_replacement_characters"
        ),
        "changed": before != after,
        "run_count": sum(int(item["run_count"]) for item in manifests),
        "source_characters_compacted": sum(
            int(item["source_characters_compacted"]) for item in manifests
        ),
        "markers_inserted": sum(int(item["markers_inserted"]) for item in manifests),
        "moderate_space_runs_compacted": sum(
            int(item["moderate_space_runs_compacted"]) for item in manifests
        ),
        "moderate_space_characters_compacted": sum(
            int(item["moderate_space_characters_compacted"]) for item in manifests
        ),
        "binary_like_runs_marked": sum(
            int(item["binary_like_runs_marked"]) for item in manifests
        ),
        "input_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
    }
    document.information_preservation["sparse_padding_compaction"] = audit
    document.information_preservation["source_text_deleted"] = bool(
        document.information_preservation.get("source_text_deleted") or audit["changed"]
    )


def render_compact_observation(
    observation: CompactObservationInput,
    output_dir: Path,
    *,
    stem: str | None = None,
    write_supporting_files: bool = False,
    emit_debug_markdown: bool = False,
) -> CompactRenderResult | None:
    document = build_compact_document(observation)
    if document is None:
        return None
    _compact_document_region_whitespace(document)
    resolver = FontResolver()
    width, width_selection = _select_canvas_width(document, resolver)
    available = width - 2 * OUTER_PADDING
    header_atoms = _header_atoms(document)

    sections: list[LayoutSection] = []
    y = 0
    header_lines, header_wraps = _wrap_atoms(header_atoms, resolver, available)
    header_height = max(
        HEADER_HEIGHT,
        2 * SECTION_PADDING_Y + len(header_lines) * LINE_HEIGHT,
    )
    sections.append(
        LayoutSection(
            kind="header",
            title=None,
            background=COLORS["header_background"],
            lines=tuple(header_lines),
            y=y,
            height=header_height,
        )
    )
    y += header_height

    soft_wraps = header_wraps
    content_atoms = _content_atoms(document)
    if content_atoms:
        content_lines, count = _wrap_atoms(content_atoms, resolver, available)
        soft_wraps += count
        content_height = 2 * SECTION_PADDING_Y + len(content_lines) * LINE_HEIGHT
        sections.append(
            LayoutSection(
                kind="content",
                title=None,
                background=COLORS["content_background"],
                lines=tuple(content_lines),
                y=y,
                height=content_height,
            )
        )
        y += content_height + SECTION_GAP

    for region_index, region in enumerate(document.regions):
        title_atoms = _region_title_atoms(document, region_index)
        title_lines, title_wraps = _wrap_atoms(
            title_atoms, resolver, available - 2 * SECTION_PADDING_X
        )
        body_lines, body_wraps = _wrap_atoms(
            region.atoms, resolver, available - 2 * SECTION_PADDING_X
        )
        soft_wraps += title_wraps + body_wraps
        lines = [*title_lines, *body_lines]
        height = 2 * SECTION_PADDING_Y + len(lines) * LINE_HEIGHT
        sections.append(
            LayoutSection(
                kind=region.kind,
                title=region.title,
                background=(
                    COLORS["path_background"]
                    if region.kind == "path_records"
                    else COLORS["region_background"]
                ),
                lines=tuple(lines),
                y=y,
                height=height,
            )
        )
        y += height + SECTION_GAP

    suffix_atoms = _suffix_atoms(document)
    if suffix_atoms:
        suffix_lines, count = _wrap_atoms(suffix_atoms, resolver, available)
        soft_wraps += count
        suffix_height = 2 * SECTION_PADDING_Y + len(suffix_lines) * LINE_HEIGHT
        sections.append(
            LayoutSection(
                kind="suffix",
                title=None,
                background=COLORS["content_background"],
                lines=tuple(suffix_lines),
                y=y,
                height=suffix_height,
            )
        )
        y += suffix_height
    height = y + BOTTOM_PADDING

    image = Image.new("RGB", (width, height), COLORS["canvas"])
    draw = ImageDraw.Draw(image)
    for section in sections:
        if section.kind == "header":
            draw.rectangle(
                (0, section.y, width, section.y + section.height),
                fill=section.background,
            )
            for index, line in enumerate(section.lines):
                _draw_line(
                    draw,
                    line,
                    OUTER_PADDING,
                    section.y + SECTION_PADDING_Y + index * LINE_HEIGHT,
                )
            continue
        left = OUTER_PADDING if section.kind not in {"content", "suffix"} else 0
        right = width - OUTER_PADDING if left else width
        draw.rounded_rectangle(
            (left, section.y, right, section.y + section.height),
            radius=8 if left else 0,
            fill=section.background,
            outline=COLORS["region_border"] if left else None,
            width=1,
        )
        text_x = left + SECTION_PADDING_X if left else OUTER_PADDING
        text_y = section.y + SECTION_PADDING_Y
        for index, line in enumerate(section.lines):
            _draw_line(draw, line, text_x, text_y + index * LINE_HEIGHT)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem or observation.stem}_p001.png"
    image.save(png_path, format="PNG", optimize=False, compress_level=9)
    transcript = resolver.renderable_text(_transcript(document))
    document.visual_transcript = transcript
    debug = _debug_fragments(document)
    raw_bytes = observation.content.encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": (
            "compact-path-record-observation-v1"
            if document.classification == "compact_path_records_v1"
            else (
                "compact-python-observation-v1"
                if document.classification == "compact_python_v1"
                else "compact-structured-text-observation-v1"
            )
        ),
        "classification": document.classification,
        "classification_strategy": {
            "policy_version": "compact-observation-classifier-v2",
            "classification": document.classification,
            "critical_high_detail_class": document.classification
            in {
                "compact_python_v1",
                "compact_path_records_v1",
                "compact_diff_v1",
                "compact_error_v1",
                "compact_traceback_v1",
                "compact_pytest_v1",
            },
            "width_selection": width_selection,
        },
        "semantic_encoding": document.semantic_encoding,
        "visual_whitespace_policy": observation.visual_whitespace_policy,
        "content_rewritten": True,
        "header": observation.header,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_bytes": len(raw_bytes),
        "raw_characters": len(observation.content),
        "wrapper_kind": document.wrapper_kind,
        "complete_python": document.complete_python,
        "source_spans": [span.as_dict() for span in document.source_spans],
        "source_span_coverage": {
            "start": 0,
            "end": len(observation.content),
            "complete": bool(document.source_spans)
            and document.source_spans[0].start == 0
            and document.source_spans[-1].end == len(observation.content)
            and all(
                left.end == right.start
                for left, right in zip(
                    document.source_spans, document.source_spans[1:], strict=False
                )
            ),
        },
        "source_file": document.source_file,
        "path_mappings": [mapping.as_dict() for mapping in document.path_mappings],
        "path_records": document.path_records,
        "information_preservation": document.information_preservation,
        "terminal_sanitizer": document.information_preservation.get(
            "terminal_sanitizer"
        ),
        "visual_envelope": {
            "returncode_encoding": "returncode=<value>",
            "returncode_location": "metadata_header",
            "output_tags_visible": False,
        },
        "regions": [
            {
                "kind": region.kind,
                "title": region.title,
                "source_start_line": region.source_start_line,
                "source_end_line": region.source_end_line,
            }
            for region in document.regions
        ],
        "generated_structure": {
            "open_braces": document.generated_open_braces,
            "close_braces": document.generated_close_braces,
            "balanced": document.generated_open_braces
            == document.generated_close_braces,
        },
        "visual_layout": {
            "width": width,
            "height": height,
            "font_size": FONT_SIZE,
            "line_height": LINE_HEIGHT,
            "bottom_padding": BOTTOM_PADDING,
            "hard_newline_marker": HARD_NEWLINE_GLYPH,
            "soft_wrap_marker": None,
            "soft_wrap_count": soft_wraps,
            "header_height": header_height,
            "header_line_count": len(header_lines),
            "header_soft_wrap_count": header_wraps,
            "header_overlaps_body": False,
            "width_selection": width_selection,
            "fixed_height": False,
            "page_count": 1,
        },
        "colors": COLORS,
        "fonts": resolver.manifest(),
        "unicode_glyph_fallbacks": resolver.unicode_fallback_manifest(),
        "unsupported_codepoint_visualization": "[[U+XXXX]] or [[U+XXXXXXXX]]",
        "visual_transcript_sha256": hashlib.sha256(
            transcript.encode("utf-8")
        ).hexdigest(),
        "png": {
            "path": png_path.name,
            "sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
            "byte_length": png_path.stat().st_size,
            "width": width,
            "height": height,
        },
    }
    if write_supporting_files:
        (output_dir / "visual_transcript.txt").write_text(transcript, encoding="utf-8")
        import json

        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if emit_debug_markdown:
        for name, text in debug.items():
            (output_dir / name).write_text(text, encoding="utf-8")
    return CompactRenderResult(
        png_path=str(png_path),
        manifest=manifest,
        visual_transcript=transcript,
        debug_fragments=debug,
    )


def compact_renderer_manifest() -> dict[str, Any]:
    resolver = FontResolver()
    return {
        "schema_version": "compact-python-renderer-v1",
        "supported_classifications": [
            "compact_python_v1",
            "compact_path_records_v1",
            "compact_empty_v1",
            "compact_error_v1",
            "compact_truncated_v2",
            "compact_diff_v1",
            "compact_traceback_v1",
            "compact_pytest_v1",
            "compact_json_v1",
            "compact_listing_v1",
            "compact_log_v1",
            "compact_shell_v1",
            "compact_text_v1",
            "preserve_table_layout",
            "preserve_matrix_layout",
        ],
        "width": WIDTH,
        "width_tiers": list(WIDTH_TIERS),
        "width_policy": "content-width-tiers-v1",
        "font_size": FONT_SIZE,
        "line_height": LINE_HEIGHT,
        "fixed_height": False,
        "bottom_padding": BOTTOM_PADDING,
        "hard_newline_marker": HARD_NEWLINE_GLYPH,
        "python_comment_policy": "strip_token_comments_v1",
        "python_docstring_policy": (
            "strip_ast_module_class_function_docstrings_v1_fail_closed"
        ),
        "ordinary_triple_quoted_strings_retained": True,
        "colors": COLORS,
        "unsupported_codepoint_visualization": "[[U+XXXX]] or [[U+XXXXXXXX]]",
        "fonts": resolver.manifest(),
    }
