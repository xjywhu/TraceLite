from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont

from swe_memory_policy.image_response_rencent3text.tools.terminal_sanitize import (
    POLICY_VERSION as TERMINAL_SANITIZER_POLICY_VERSION,
)
from swe_memory_policy.image_response_rencent3text.tools.terminal_sanitize import (
    sanitize_terminal_text,
)
from swe_memory_policy.utils import sha256_file
from swe_memory_policy.vision_tokens import (
    DETAIL,
    ESTIMATOR,
    MODEL,
    plan_visual_token_resize,
)

BACKGROUND = "#FFFFFF"
FOREGROUND = "#111111"
BASE_WIDTH = 2048
MAX_PAGE_HEIGHT = 4096
FONT_SIZE = 18
LINE_HEIGHT = 25
PADDING = 24


class FontCoverageError(ValueError):
    pass


@dataclass(frozen=True)
class FontFace:
    path: Path
    index: int
    coverage: frozenset[int]
    font: ImageFont.FreeTypeFont


def _font_candidates() -> list[Path]:
    configured = os.getenv("IMAGE_MEMORY_FONT_PATHS")
    if configured:
        return [Path(item) for item in configured.split(os.pathsep) if item]
    return [
        Path("/usr/local/share/fonts/CascadiaMono.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/seguisym.ttf"),
    ]


def _coverage(font: TTFont) -> frozenset[int]:
    result: set[int] = set()
    for table in font["cmap"].tables:
        result.update(table.cmap)
    return frozenset(result)


@lru_cache(maxsize=1)
def load_font_faces() -> tuple[FontFace, ...]:
    """Load immutable font faces once per renderer process."""

    faces: list[FontFace] = []
    for path in _font_candidates():
        if not path.is_file():
            continue
        if path.suffix.lower() == ".ttc":
            collection = TTCollection(path, lazy=True)
            for index, font_data in enumerate(collection.fonts):
                faces.append(
                    FontFace(
                        path=path,
                        index=index,
                        coverage=_coverage(font_data),
                        font=ImageFont.truetype(str(path), FONT_SIZE, index=index),
                    )
                )
            collection.close()
        else:
            font_data = TTFont(path, lazy=True)
            faces.append(
                FontFace(
                    path=path,
                    index=0,
                    coverage=_coverage(font_data),
                    font=ImageFont.truetype(str(path), FONT_SIZE),
                )
            )
            font_data.close()
    if not faces:
        raise FontCoverageError("none of the configured fonts could be loaded")
    return tuple(faces)


class FontResolver:
    def __init__(self) -> None:
        self.faces = load_font_faces()
        self._cache: dict[str, FontFace] = {}
        self._unicode_fallbacks: Counter[str] = Counter()

    def face_for(self, character: str) -> FontFace:
        cached = self._cache.get(character)
        if cached is not None:
            return cached
        codepoint = ord(character)
        for face in self.faces:
            if codepoint in face.coverage:
                self._cache[character] = face
                return face
        raise FontCoverageError(
            f"no configured font covers U+{codepoint:04X} ({character!r})"
        )

    @staticmethod
    def codepoint_label(character: str) -> str:
        """Return an ASCII, reversible visual label for an unavailable glyph."""

        width = 4 if ord(character) <= 0xFFFF else 8
        return f"[[U+{ord(character):0{width}X}]]"

    def display_units(
        self, character: str, *, record_fallback: bool = True
    ) -> tuple[tuple[str, FontFace], ...]:
        """Resolve one source character to visible glyph units without data loss.

        Normal characters remain unchanged.  If no configured font has a glyph,
        the exact Unicode scalar value is shown as an ASCII codepoint label.  The
        label is unambiguous in the rendered image and the original source remains
        authoritative in the request-history audit.
        """

        try:
            return ((character, self.face_for(character)),)
        except FontCoverageError:
            if record_fallback:
                self._unicode_fallbacks[character] += 1
            label = self.codepoint_label(character)
            return tuple((item, self.face_for(item)) for item in label)

    def renderable_text(self, text: str) -> str:
        """Apply the same reversible fallback used by the visual renderers."""

        output: list[str] = []
        for character in text:
            if character in {"\r", "\n"}:
                output.append(character)
                continue
            units = self.display_units(character, record_fallback=False)
            output.extend(item for item, _face in units)
        return "".join(output)

    def unicode_fallback_manifest(self) -> list[dict[str, str | int]]:
        return [
            {
                "codepoint": f"U+{ord(character):04X}",
                "source_character": character,
                "visual_label": self.codepoint_label(character),
                "occurrences": count,
            }
            for character, count in sorted(
                self._unicode_fallbacks.items(), key=lambda item: ord(item[0])
            )
        ]

    def manifest(self) -> list[dict[str, str | int]]:
        records: list[dict[str, str | int]] = []
        seen: set[tuple[Path, int]] = set()
        for face in self.faces:
            key = (face.path, face.index)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "path": str(face.path),
                    "index": face.index,
                    "sha256": sha256_file(face.path),
                }
            )
        return records


def _expand_tabs(text: str) -> str:
    output: list[str] = []
    column = 0
    for character in text:
        if character == "\t":
            count = 4 - (column % 4)
            output.append(" " * count)
            column += count
        else:
            output.append(character)
            column = 0 if character == "\n" else column + 1
    return "".join(output)


def _strip_ansi(text: str) -> str:
    """Return the terminal-sanitized visual copy used by the source-line fallback."""

    return sanitize_terminal_text(text).text


def _visual_lines(
    text: str, resolver: FontResolver, available_width: int
) -> list[list[tuple[str, FontFace]]]:
    text = _expand_tabs(text.replace("\r\n", "\n").replace("\r", "\n"))
    logical_lines = text.split("\n")
    visual: list[list[tuple[str, FontFace]]] = []
    for logical in logical_lines:
        current: list[tuple[str, FontFace]] = []
        width = 0.0
        if not logical:
            visual.append(current)
            continue
        for source_character in logical:
            for character, face in resolver.display_units(source_character):
                advance = face.font.getlength(character)
                if current and width + advance > available_width:
                    visual.append(current)
                    current = []
                    width = 0.0
                current.append((character, face))
                width += advance
        visual.append(current)
    if text.endswith("\n") and visual and not visual[-1]:
        visual.pop()
    return visual or [[]]


def _draw_visual_line(
    draw: ImageDraw.ImageDraw,
    line: list[tuple[str, FontFace]],
    x: int,
    y: int,
) -> None:
    cursor = float(x)
    run: list[str] = []
    run_face: FontFace | None = None

    def flush() -> None:
        nonlocal cursor, run
        if run_face is None or not run:
            return
        text = "".join(run)
        draw.text((cursor, y), text, font=run_face.font, fill=FOREGROUND)
        cursor += run_face.font.getlength(text)
        run = []

    for character, face in line:
        if run_face is not None and face != run_face:
            flush()
        run_face = face
        run.append(character)
    flush()


def render_text_pages(
    text: str,
    output_dir: Path,
    stem: str,
    *,
    page_label: str | None = None,
    always_page_suffix: bool = False,
) -> list[Path]:
    text = _strip_ansi(text)
    text.encode("utf-8", errors="strict")
    resolver = FontResolver()
    lines = _visual_lines(text, resolver, BASE_WIDTH - 2 * PADDING)
    total_lines_per_page = max(1, (MAX_PAGE_HEIGHT - 2 * PADDING) // LINE_HEIGHT)
    header_line_count = 2 if page_label else 0
    lines_per_page = max(1, total_lines_per_page - header_line_count)
    chunks = [
        lines[index : index + lines_per_page]
        for index in range(0, len(lines), lines_per_page)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for page_index, chunk in enumerate(chunks, start=1):
        display_lines = chunk
        if page_label:
            header = _visual_lines(
                f"{page_label} — PAGE {page_index}/{len(chunks)}",
                resolver,
                BASE_WIDTH - 2 * PADDING,
            )
            display_lines = [*header, [], *chunk]
        height = min(
            MAX_PAGE_HEIGHT,
            max(
                2 * PADDING + LINE_HEIGHT,
                2 * PADDING + len(display_lines) * LINE_HEIGHT,
            ),
        )
        image = Image.new("RGB", (BASE_WIDTH, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        for line_index, line in enumerate(display_lines):
            _draw_visual_line(draw, line, PADDING, PADDING + line_index * LINE_HEIGHT)
        filename = (
            f"{stem}_p{page_index:03d}.png"
            if len(chunks) > 1 or always_page_suffix
            else f"{stem}.png"
        )
        path = output_dir / filename
        image.save(path, format="PNG", optimize=False, compress_level=9)
        paths.append(path)
    return paths


def downscale_png_to_visual_token_ratio(
    path: Path,
    *,
    target_numerator: int,
    target_denominator: int,
    model: str = MODEL,
    detail: str = DETAIL,
) -> dict[str, object]:
    """Resize one PNG once to a strict estimated visual Token budget.

    An unchanged size leaves the original file bytes intact. An unattainable
    budget raises before writing; it never silently exceeds the requested cost.
    """

    with Image.open(path) as opened:
        geometry = plan_visual_token_resize(
            *opened.size,
            target_numerator=target_numerator,
            target_denominator=target_denominator,
            model=model,
            detail=detail,
        )
        size = (geometry["width"], geometry["height"])
        if size != opened.size:
            image = opened.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            image.save(path, format="PNG", optimize=False, compress_level=9)
    return geometry


def rendering_manifest() -> dict[str, object]:
    resolver = FontResolver()
    return {
        "background": BACKGROUND,
        "foreground": FOREGROUND,
        "base_width": BASE_WIDTH,
        "max_page_height": MAX_PAGE_HEIGHT,
        "font_size": FONT_SIZE,
        "line_height": LINE_HEIGHT,
        "padding": PADDING,
        "terminal_sanitizer_policy": TERMINAL_SANITIZER_POLICY_VERSION,
        "terminal_sanitizer_visual_copy_only": True,
        "compression_definition": "visual_token_ratio",
        "visual_token_estimator": ESTIMATOR,
        "model": MODEL,
        "input_image_detail": DETAIL,
        "resize_policy": "single_lanczos_from_base_png; identity_preserves_bytes",
        "unsupported_codepoint_visualization": "[[U+XXXX]] or [[U+XXXXXXXX]]",
        "fonts": resolver.manifest(),
    }
