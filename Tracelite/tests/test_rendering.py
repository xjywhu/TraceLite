from __future__ import annotations

from pathlib import Path

from PIL import Image

from swe_memory_policy.rendering import (
    BASE_WIDTH,
    FontResolver,
    load_font_faces,
    render_text_pages,
    rendering_manifest,
)
from swe_memory_policy.utils import sha256_file


def test_white_render_is_deterministic_and_preserves_unicode(tmp_path: Path) -> None:
    text = "OA 0001 | ACTION\nprint('中文 ✓')\n\treturn 1\\2\n"
    first = render_text_pages(text, tmp_path / "a", "OA0001")
    second = render_text_pages(text, tmp_path / "b", "OA0001")
    assert len(first) == len(second) == 1
    assert sha256_file(first[0]) == sha256_file(second[0])
    with Image.open(first[0]) as image:
        assert image.width == BASE_WIDTH
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_render_strips_ansi_terminal_formatting(tmp_path: Path) -> None:
    pages = render_text_pages(
        "plain \x1b[31mred\x1b[0m text and stray \x1b escape\n",
        tmp_path / "ansi",
        "OA0001",
    )
    assert len(pages) == 1
    with Image.open(pages[0]) as image:
        assert image.width == BASE_WIDTH


def test_fallback_render_uses_full_terminal_sanitizer_policy(tmp_path: Path) -> None:
    pages = render_text_pages(
        "\x1b]8;;https://example.invalid\x1b\\label\x1b]8;;\x1b\\ "
        "\x1bPpayload\x1b\\ done\n",
        tmp_path / "terminal-families",
        "OA0001",
    )
    assert len(pages) == 1
    manifest = rendering_manifest()
    assert manifest["terminal_sanitizer_policy"] == "terminal-sanitize-v1"
    assert manifest["terminal_sanitizer_visual_copy_only"] is True


def test_unavailable_unicode_uses_reversible_codepoint_label(tmp_path: Path) -> None:
    resolver = FontResolver()
    unavailable = "\u0a00"
    assert resolver.renderable_text(f"before {unavailable} after") == (
        "before [[U+0A00]] after"
    )
    pages = render_text_pages(
        f"before {unavailable} after\n", tmp_path / "unicode-fallback", "OA0001"
    )
    assert len(pages) == 1
    assert resolver.codepoint_label(unavailable) == "[[U+0A00]]"


def test_font_faces_are_cached_per_process() -> None:
    assert load_font_faces() is load_font_faces()


def test_labeled_pages_are_deterministic_and_always_numbered(
    tmp_path: Path,
) -> None:
    first = render_text_pages(
        "role: assistant\ncontent:\nhello\n",
        tmp_path / "first",
        "request_0001_OA0001",
        page_label="OA 1",
        always_page_suffix=True,
    )
    second = render_text_pages(
        "role: assistant\ncontent:\nhello\n",
        tmp_path / "second",
        "request_0001_OA0001",
        page_label="OA 1",
        always_page_suffix=True,
    )
    assert first[0].name == "request_0001_OA0001_p001.png"
    assert sha256_file(first[0]) == sha256_file(second[0])
