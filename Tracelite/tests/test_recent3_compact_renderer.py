from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from swe_memory_policy.image_response_rencent3text.tools import (
    CompactObservationInput,
    build_compact_document,
    build_path_record_document,
    display_project_path,
    parse_mini_swe_observation,
    parse_path_record_command,
    render_compact_observation,
    rewrite_location_paths,
)
from swe_memory_policy.image_response_rencent3text.tools.python_compact import (
    brace_counts,
    build_python_regions,
    compact_python_text,
    strip_python_comments,
    strip_python_nonsemantic_text,
)
from swe_memory_policy.image_response_rencent3text.tools.renderer import (
    WIDTH_TIERS,
    _select_width_tier,
)
from swe_memory_policy.image_response_rencent3text.tools.terminal_sanitize import (
    sanitize_terminal_text,
)


def _long_python() -> str:
    prelude = """import contextlib
from pathlib import Path

VALUE = {"key": 3}

class Example:
    @classmethod
    def run(cls, values: list[int]):
        # comment /testbed/not-a-location.py remains literal
        label = f"value: {values!r}"
        transform = lambda item: {"item": item}
        if values:
            for value in values:
                if value > 2:
                    yield transform(value)
                else:
                    yield {"small": value}
        elif label:
            while values:
                values.pop()
        else:
            try:
                with contextlib.nullcontext():
                    match label:
                        case "value":
                            return Path("/testbed/string.py")
                        case _:
                            return None
            except ValueError as error:
                raise RuntimeError(str(error))
            finally:
                values.clear()
"""
    helpers = "\n".join(
        f"\ndef helper_{index}(value={index}):\n    return value\n"
        for index in range(30)
    )
    return prelude + helpers


def _observation(content: str | None = None) -> CompactObservationInput:
    body = content if content is not None else _long_python()
    wrapped = f"<returncode>0</returncode>\n<output>\n{body}</output>"
    return CompactObservationInput(
        observation_index=2,
        tool_message_index=1,
        page_index=1,
        page_count=1,
        role="tool",
        tool_call_id="call_complete_identifier_123456",
        project_root="/testbed",
        content=wrapped,
        tool_name="bash",
        tool_arguments=json.dumps(
            {"command": "cd /testbed && sed -n '1,400p' pkg/example.py"}
        ),
    )


def _path_record_observation(
    *, command: str | None = None, body: str | None = None
) -> CompactObservationInput:
    records = body or "\n".join(
        [
            "pkg/a.py:12:value = '/testbed/literal.py'",
            "pkg/a.py:20:# comment: keep every colon",
            "pkg/a.py:21:mapping = {'key': 'value'}",
            "pkg/a.py:35:call(one, two)",
            "pkg/a.py:48:",
            "pkg/b.py:3:class Example:",
            "pkg/b.py:9:    def run(self):",
            "pkg/b.py:10:        return 'unchanged'",
        ]
    )
    content = f"<returncode>0</returncode>\n<output>\n{records}\n</output>"
    return CompactObservationInput(
        observation_index=2,
        tool_message_index=2,
        page_index=1,
        page_count=1,
        role="tool",
        tool_call_id="call_path_records_complete_123",
        project_root="/testbed",
        content=content,
        tool_name="bash",
        tool_arguments=json.dumps(
            {"command": command or 'cd /testbed && grep -RIn "value" pkg | head -200'}
        ),
    )


def test_wrapper_parser_is_anchored_and_preserves_nested_markup() -> None:
    content = (
        "<returncode>0</returncode>\n<output>\n"
        "value = '<returncode>9</returncode>'\n</output>"
    )
    parsed = parse_mini_swe_observation(content)
    assert parsed is not None
    assert parsed.prefix == "<returncode>0</returncode>\n<output>"
    assert "<returncode>9</returncode>" in parsed.body
    assert parsed.suffix == "</output>"
    assert [span.kind for span in parsed.spans] == [
        "wrapper_prefix",
        "output_body",
        "wrapper_suffix",
    ]


def test_truncated_wrapper_uses_lossless_compact_layout() -> None:
    content = (
        "<returncode>0</returncode>\n"
        "<warning>too long</warning>\n"
        "<output_head>class A:\n    pass</output_head>\n"
        "<elided_chars>10 characters elided</elided_chars>\n"
        "<output_tail>pass</output_tail>"
    )
    parsed = parse_mini_swe_observation(content)
    assert parsed is not None and parsed.truncated
    observation = _observation()
    object.__setattr__(observation, "content", content)
    document = build_compact_document(observation)
    assert document is not None
    assert document.classification == "compact_truncated_v2"
    assert document.content_prefix == content
    assert document.semantic_encoding == "anchored_head_tail_omitted_hash_v2"
    assert [region.title for region in document.regions] == [
        "Truncated output",
        "Output head",
        "Output tail",
    ]
    assert document.information_preservation["missing_content_reconstructed"] is False


def test_visual_whitespace_compaction_is_auditable_and_preserves_binary_markers(
    tmp_path: Path,
) -> None:
    padding = " " * 4096 + "\x00" * 512 + "\t" * 128
    content = (
        "<returncode>0</returncode>\n"
        "<warning>output too long</warning>\n"
        f"<output_head>FITS-A{padding}FITS-B</output_head>\n"
        "<elided_chars>9000 characters elided</elided_chars>\n"
        f"<output_tail>TAIL-A{padding}TAIL-B</output_tail>"
    )
    preserved = _observation()
    object.__setattr__(preserved, "content", content)
    preserved_result = render_compact_observation(preserved, tmp_path / "preserved")
    assert preserved_result is not None

    optimized = _observation()
    object.__setattr__(optimized, "content", content)
    object.__setattr__(
        optimized, "visual_whitespace_policy", "compact_visual_whitespace_v2"
    )
    optimized_result = render_compact_observation(optimized, tmp_path / "optimized")
    assert optimized_result is not None
    assert "[PAD omitted=4736 spaces=4096 tabs=128 nul=512]" in (
        optimized_result.visual_transcript
    )
    assert optimized_result.manifest["raw_sha256"] == optimized.raw_sha256
    assert optimized_result.manifest["visual_whitespace_policy"] == (
        "compact_visual_whitespace_v2"
    )
    audit = optimized_result.manifest["information_preservation"][
        "sparse_padding_compaction"
    ]
    assert audit["changed"] is True
    assert audit["run_count"] == 2
    assert audit["source_characters_compacted"] == 9472
    assert audit["markers_inserted"] == 2
    assert audit["moderate_space_runs_compacted"] == 0
    assert audit["binary_like_runs_marked"] == 0
    assert audit["minimum_marker_run_characters"] == 32
    assert audit["moderate_space_run_minimum"] == 8
    assert (
        optimized_result.manifest["visual_layout"]["height"]
        < preserved_result.manifest["visual_layout"]["height"] / 4
    )


def test_visual_whitespace_compacts_moderate_spaces_but_marks_binary_runs(
    tmp_path: Path,
) -> None:
    content = (
        "<returncode>0</returncode>\n<output>"
        "left" + " " * 8 + "middle" + " " * 31 + "right\n"
        "binary" + "\x00" * 8 + "tail\n"
        "wide" + " " * 32 + "tail"
        "</output>"
    )
    observation = _observation()
    object.__setattr__(observation, "content", content)
    object.__setattr__(
        observation, "visual_whitespace_policy", "compact_visual_whitespace_v2"
    )
    result = render_compact_observation(observation, tmp_path)
    assert result is not None
    assert "left    middle    right" in result.visual_transcript
    assert "[PAD omitted=8 spaces=0 tabs=0 nul=8]" in result.visual_transcript
    assert "[PAD omitted=32 spaces=32 tabs=0 nul=0]" in result.visual_transcript
    audit = result.manifest["information_preservation"]["sparse_padding_compaction"]
    assert audit["moderate_space_runs_compacted"] == 2
    assert audit["moderate_space_characters_compacted"] == 39
    assert audit["binary_like_runs_marked"] == 1
    assert audit["markers_inserted"] == 2
    assert result.manifest["raw_sha256"] == observation.raw_sha256


def test_visual_whitespace_policy_rejects_unknown_values() -> None:
    observation = _observation()
    object.__setattr__(observation, "visual_whitespace_policy", "unknown")
    try:
        observation.__post_init__()
    except ValueError as error:
        assert str(error) == "unsupported visual whitespace policy"
    else:
        raise AssertionError("unknown whitespace policy was accepted")


def test_git_blame_timestamps_are_precisely_omitted_from_visual_copy(
    tmp_path: Path,
) -> None:
    blame_lines = (
        "8656cfc4e01 django/forms/models.py "
        "(Tim Graham 2015-08-06 17:27:21 -0400 420) def _save_m2m(self):\n"
        "00000000000 django/forms/models.py "
        "(Not Committed Yet 2026-08-15 09:30:00 +0800 421) pending = True\n"
    )
    ordinary_dates = (
        "benchmark started 2015-08-06 17:27:21 -0400\n"
        "timeout at 2026-08-15 09:30:00 +0800 after 30 seconds\n"
    )
    content = (
        f"<returncode>0</returncode>\n<output>{blame_lines}{ordinary_dates}</output>"
    )
    observation = _observation()
    object.__setattr__(observation, "content", content)
    object.__setattr__(
        observation, "visual_whitespace_policy", "compact_visual_whitespace_v2"
    )

    result = render_compact_observation(observation, tmp_path)

    assert result is not None
    assert "[git-blame timestamps omitted=2]" in result.visual_transcript
    assert (
        "8656cfc4e01 django/forms/models.py (Tim Graham :420) def _save_m2m(self):"
    ) in result.visual_transcript
    assert (
        "00000000000 django/forms/models.py (Not Committed Yet :421) pending = True"
    ) in result.visual_transcript
    assert "benchmark started 2015-08-06 17:27:21 -0400" in (result.visual_transcript)
    assert "timeout at 2026-08-15 09:30:00 +0800" in result.visual_transcript
    assert result.manifest["raw_sha256"] == observation.raw_sha256
    audit = result.manifest["information_preservation"][
        "git_blame_timestamp_compaction"
    ]
    assert audit["policy_version"] == "git-blame-timestamp-visual-omission-v1"
    assert audit["visual_copy_only"] is True
    assert audit["changed"] is True
    assert audit["timestamps_omitted"] == 2
    assert audit["timestamp_characters_omitted"] == 50


def test_git_blame_timestamp_omission_rejects_near_matches(tmp_path: Path) -> None:
    lines = (
        "log 8656cfc4e01 (Tim Graham 2015-08-06 17:27:21 -0400 420) code\n"
        "8656cfc4e01 (Tim Graham 2015/08/06 17:27:21 -0400 420) code\n"
        "8656cfc4e01 (Tim Graham 2015-08-06 17:27:21 UTC 420) code\n"
        "release (Tim Graham 2015-08-06 17:27:21 -0400 420) code\n"
    )
    content = f"<returncode>0</returncode>\n<output>{lines}</output>"
    observation = _observation()
    object.__setattr__(observation, "content", content)
    result = render_compact_observation(observation, tmp_path)

    assert result is not None
    assert "[git-blame timestamps omitted=" not in result.visual_transcript
    assert "2015-08-06 17:27:21 -0400" in result.visual_transcript
    audit = result.manifest["information_preservation"][
        "git_blame_timestamp_compaction"
    ]
    assert audit["changed"] is False
    assert audit["timestamps_omitted"] == 0


def test_truncated_git_blame_timestamps_are_omitted_from_head_and_tail(
    tmp_path: Path,
) -> None:
    head = (
        "8656cfc4e01 django/forms/models.py "
        "(Tim Graham 2015-08-06 17:27:21 -0400 420) head = True"
    )
    tail = (
        "00000000000 django/forms/models.py "
        "(Not Committed Yet 2026-08-15 09:30:00 +0800 421) tail = True"
    )
    content = (
        "<returncode>0</returncode>\n"
        "<warning>output too long</warning>\n"
        f"<output_head>{head}</output_head>\n"
        "<elided_chars>9000 characters elided</elided_chars>\n"
        f"<output_tail>{tail}</output_tail>"
    )
    observation = _observation()
    object.__setattr__(observation, "content", content)
    result = render_compact_observation(observation, tmp_path)

    assert result is not None
    assert "[git-blame timestamps omitted=2]" in result.visual_transcript
    assert "(Tim Graham :420) head = True" in result.visual_transcript
    assert "(Not Committed Yet :421) tail = True" in result.visual_transcript
    audit = result.manifest["information_preservation"][
        "git_blame_timestamp_compaction"
    ]
    assert audit["changed"] is True
    assert audit["timestamps_omitted"] == 2
    assert audit["timestamp_characters_omitted"] == 50


def test_terminal_sanitizer_removes_complete_control_families_and_keeps_label() -> None:
    raw = (
        "plain \x1b[31mred\x1b[0m "
        "\x1b]8;;https://example.invalid\x1b\\link\x1b]8;;\x1b\\ "
        "\x1bPprivate-dcs\x1b\\"
        "\x1b_private-apc\x1b\\"
        "\x1b^private-pm\x1b\\"
        "\x1bXprivate-sos\x1b\\"
        "bell\x07 end \x1b"
    )
    result = sanitize_terminal_text(raw)

    assert result.text == "plain red link bell end "
    assert "\x1b" not in result.text
    assert "\x07" not in result.text
    counts = result.manifest["removed_sequence_counts"]
    assert counts["csi"] == 2
    assert counts["osc"] == 2
    assert counts["dcs"] == 1
    assert counts["apc"] == 1
    assert counts["pm"] == 1
    assert counts["sos"] == 1
    assert counts["bel"] == 1
    assert counts["single_escape"] == 1


def test_terminal_sanitizer_supports_c1_forms_and_st() -> None:
    raw = (
        "a\x9b31mred\x9b0m "
        "\x9dtitle\x9c"
        "\x90dcs\x9c"
        "\x9fapc\x9c"
        "\x9epm\x9c"
        "\x98sos\x9c"
        "b\x9c"
    )
    result = sanitize_terminal_text(raw)

    assert result.text == "ared b"
    counts = result.manifest["removed_sequence_counts"]
    assert counts == {
        "apc": 1,
        "csi": 2,
        "dcs": 1,
        "osc": 1,
        "pm": 1,
        "sos": 1,
        "st": 1,
    }


def test_terminal_sanitizer_incomplete_sequences_fail_closed_and_literal_survives() -> (
    None
):
    raw = "literal [[U+001B]]\nbefore \x1b[31\nerror \x1b]unterminated diagnostic"
    result = sanitize_terminal_text(raw)

    assert "literal [[U+001B]]" in result.text
    assert "[[INCOMPLETE_CSI]]31" in result.text
    assert "[[INCOMPLETE_OSC]]unterminated diagnostic" in result.text
    assert result.manifest["incomplete_sequence_counts"] == {"csi": 1, "osc": 1}
    assert result.manifest["incomplete_markers_inserted"] == 2


def test_terminal_sanitizer_applies_cr_and_backspace_screen_semantics() -> None:
    result = sanitize_terminal_text("progress 10%\rprogress 20%\nabc\bZ\n")

    assert result.text == "progress 20%\nabZ\n"
    assert result.manifest["carriage_returns_applied"] == 1
    assert result.manifest["backspaces_applied"] == 1
    assert result.manifest["edited_lines"] == 2


def test_paths_are_rewritten_only_in_file_contexts_and_are_reversible() -> None:
    text = (
        '  File "/testbed/pkg/a.py", line 12, in run\n'
        "/testbed/pkg/b.py:8:3: diagnostic\n"
        "/testbed2/pkg/c.py:9: not project\n"
        'message mentions "/testbed/pkg/natural.py"\n'
    )
    rewritten, mappings = rewrite_location_paths(text, "/testbed")
    assert 'File "~/pkg/a.py", line 12' in rewritten
    assert "~/pkg/b.py:8:3" in rewritten
    assert "/testbed2/pkg/c.py" in rewritten
    assert 'message mentions "/testbed/pkg/natural.py"' in rewritten
    assert len(mappings) == 2
    assert display_project_path("/testbed/pkg/a.py", "/testbed") == "~/pkg/a.py"
    assert display_project_path("/testbed2/pkg/a.py", "/testbed") is None


def test_tokenizer_compacts_all_suites_without_touching_literals() -> None:
    source = _long_python()
    compacted = compact_python_text(source, complete=True)
    rendered = "".join(atom.text for atom in compacted.atoms)
    assert compacted.tokenize_complete
    assert compacted.open_braces == compacted.close_braces
    assert "⏎" in rendered
    assert '{"key": 3}' in rendered
    assert 'f"value: {values!r}"' in rendered
    assert 'lambda item: {"item": item}' in rendered
    assert "# comment /testbed/not-a-location.py remains literal" not in rendered
    assert "..." not in rendered


def test_python_comments_are_removed_without_touching_hashes_in_literals() -> None:
    source = (
        "# whole line\n"
        "url = 'https://example.test/a#fragment'  # trailing\n"
        "pattern = r'#[a-z]+'\n"
        'text = "# literal"\n'
        "def run():  # header\n"
        "    return '# result'  # result comment\n"
    )
    removal = strip_python_comments(source)
    compacted = compact_python_text(source, complete=True)
    rendered = "".join(atom.text for atom in compacted.atoms)

    assert removal.comment_count == 4
    assert removal.comment_lines == 4
    assert "whole line" not in rendered
    assert "trailing" not in rendered
    assert "header" not in rendered
    assert "result comment" not in rendered
    assert "https://example.test/a#fragment" in rendered
    assert "#[a-z]+" in rendered
    assert '"# literal"' in rendered
    assert "'# result'" in rendered
    assert not any(atom.style == "comment" for atom in compacted.atoms)


def test_ast_docstrings_are_removed_but_ordinary_triple_strings_are_retained() -> None:
    source = (
        '"""module documentation"""\n'
        'MODULE_DATA = """keep module data"""\n'
        "class Example:\n"
        '    """class documentation"""\n'
        '    value = """keep class data"""\n'
        "    def run(self):\n"
        '        """function documentation"""\n'
        '        payload = """keep function data"""\n'
        "        return payload\n"
        "async def fetch():\n"
        '    """async documentation"""\n'
        '    return """keep return data"""\n'
    )
    removal = strip_python_nonsemantic_text(source)
    compacted = compact_python_text(source, complete=True)
    rendered = "".join(atom.text for atom in compacted.atoms)

    assert removal.docstring_ast_complete is True
    assert removal.docstring_count == 4
    assert removal.docstring_lines == 4
    assert "module documentation" not in rendered
    assert "class documentation" not in rendered
    assert "function documentation" not in rendered
    assert "async documentation" not in rendered
    assert "keep module data" in rendered
    assert "keep class data" in rendered
    assert "keep function data" in rendered
    assert "keep return data" in rendered


def test_inline_docstring_is_removed_without_damaging_following_code() -> None:
    source = 'def run(): "doc"; return """payload"""\n'
    removal = strip_python_nonsemantic_text(source)

    assert removal.docstring_count == 1
    assert "doc" not in removal.text
    assert 'return """payload"""' in removal.text
    assert "def run(): return" in removal.text


def test_incomplete_python_preserves_triple_strings_fail_closed() -> None:
    source = 'def run():\n    """possibly a docstring"""\n    value = (\n'
    removal = strip_python_nonsemantic_text(source)

    assert removal.docstring_ast_complete is False
    assert removal.docstring_count == 0
    assert "possibly a docstring" in removal.text


def test_docstring_only_prelude_does_not_leave_an_empty_region() -> None:
    source = '"""module documentation"""\n\ndef run():\n    return 1\n'
    document = build_compact_document(_observation(source))

    assert document is not None
    assert [region.title for region in document.regions] == ["def run"]


def test_partial_python_removes_comments_emitted_before_token_error() -> None:
    fragment = "value = (1  # inline\n# whole line\n"
    removal = strip_python_comments(fragment)
    compacted = compact_python_text(fragment, complete=False)
    rendered = "".join(atom.text for atom in compacted.atoms)

    assert removal.comment_count == 2
    assert "inline" not in rendered
    assert "whole line" not in rendered
    assert rendered == "value = (1 ⏎ "


def test_python_tokenize_fallback_strips_only_line_leading_whitespace() -> None:
    fragment = "        )\n    if ready:\n        value = 'a  b'\n"
    compacted = compact_python_text(fragment, complete=False)
    rendered = "".join(atom.text for atom in compacted.atoms)

    assert compacted.tokenize_complete is False
    assert rendered.startswith(") ⏎ if ready: ⏎ value = 'a  b'")
    assert "    )" not in rendered
    assert "⏎     " not in rendered
    # Interior whitespace remains byte-for-byte visible.
    assert "'a  b'" in rendered
    assert rendered.count("⏎") == fragment.count("\n")


def test_partial_python_does_not_invent_eof_closures() -> None:
    partial = "def run():\n    if ready:\n        return 1\n"
    compacted = compact_python_text(partial, complete=False)
    assert compacted.open_braces == 2
    assert compacted.close_braces == 0


def test_document_has_prelude_class_and_independent_method_regions() -> None:
    document = build_compact_document(_observation())
    assert document is not None
    kinds = [region.kind for region in document.regions]
    assert "prelude" in kinds
    assert "class" in kinds
    assert "method" in kinds
    assert "function" in kinds
    assert document.source_file == "~/pkg/example.py"
    assert document.generated_open_braces == document.generated_close_braces


def test_nested_class_before_first_method_keeps_generated_braces_balanced() -> None:
    source = """class Outer:
    class Inner:
        value = 1

    def method(self):
        return self.Inner.value
"""
    regions = build_python_regions(source, complete=True)
    opens, closes = brace_counts(regions)

    assert opens == closes
    outer = next(region for region in regions if region.title == "class Outer")
    assert any(atom.generated_kind == "close_brace" for atom in outer.atoms)


def test_outer_indent_with_column_zero_multiline_string_keeps_braces_balanced(
    tmp_path: Path,
) -> None:
    sources = [
        'class Example:\n    def method(self):\n        value = """first\n'
        'column_zero\nlast"""\n        return value\n',
        'class Example:\n    def method(self):\n        if True:\n'
        '            value = """first\ncolumn_zero\nlast"""\n'
        '            return value\n        return None\n',
    ]
    for index, source in enumerate(sources):
        regions = build_python_regions(source, complete=True)
        opens, closes = brace_counts(regions)
        assert opens > 0
        assert opens == closes
        method = next(region for region in regions if region.kind == "method")
        assert 'value = """first' in method.plain_text
        assert 'column_zero ⏎ last"""' in method.plain_text
        assert "return value" in method.plain_text

        result = render_compact_observation(_observation(source), tmp_path / str(index))
        assert result is not None
        assert result.manifest["classification"] == "compact_python_v1"
        with Image.open(result.png_path) as rendered:
            assert rendered.width >= 1152
            assert rendered.height > 0


def test_renderer_uses_content_width_tier_and_compact_audited_header(
    tmp_path: Path,
) -> None:
    observation = _observation()
    result = render_compact_observation(
        observation,
        tmp_path,
        write_supporting_files=True,
        emit_debug_markdown=True,
    )
    assert result is not None
    with Image.open(result.png_path) as image:
        assert image.width in WIDTH_TIERS
        assert image.height == result.manifest["visual_layout"]["height"]
    lines = result.visual_transcript.splitlines()
    assert lines[0].startswith("OA2 · Tool 1 · page 1/1 · call:")
    assert lines[0].endswith(" · returncode=0")
    assert lines[1] == "[Prelude] · FILE ~/pkg/example.py"
    assert "<returncode>" not in result.visual_transcript
    assert "<output>" not in result.visual_transcript
    assert "</output>" not in result.visual_transcript
    assert result.manifest["source_span_coverage"]["complete"] is True
    assert result.manifest["generated_structure"]["balanced"] is True
    assert result.manifest["visual_envelope"]["output_tags_visible"] is False
    assert (
        result.manifest["visual_envelope"]["returncode_location"] == "metadata_header"
    )
    assert result.manifest["colors"]["path_badge"] == "#16803C"
    assert result.manifest["colors"]["path_badge_background"] == "#DDF3E5"
    assert result.manifest["visual_layout"]["bottom_padding"] == 20
    width_selection = result.manifest["visual_layout"]["width_selection"]
    assert width_selection["policy_version"] == "content-width-tiers-v1"
    assert width_selection["selected_base_width"] == image.width
    assert width_selection["resampled_up"] is False
    assert result.manifest["classification_strategy"]["classification"] == (
        "compact_python_v1"
    )
    assert result.manifest["terminal_sanitizer"]["visual_copy_only"] is True
    assert (tmp_path / "lujing.md").is_file()
    assert (tmp_path / "defcode.md").is_file()
    assert (tmp_path / "other.md").is_file()


def test_nonzero_returncode_and_exception_remain_visible_without_xml(
    tmp_path: Path,
) -> None:
    observation = _observation("failure details\n")
    object.__setattr__(
        observation,
        "content",
        "<exception>command failed</exception>\n"
        "<returncode>7</returncode>\n"
        "<output>failure details\n</output>",
    )
    result = render_compact_observation(observation, tmp_path)

    assert result is not None
    lines = result.visual_transcript.splitlines()
    assert lines[0].startswith("OA2 · Tool 1 · page 1/1 · call:")
    assert lines[0].endswith(" · returncode=7")
    assert lines[1] == "exception=command failed"
    assert "failure details" in result.visual_transcript
    assert "<exception>" not in result.visual_transcript
    assert "<returncode>" not in result.visual_transcript
    assert "<output>" not in result.visual_transcript
    assert "</output>" not in result.visual_transcript
    # The untouched source envelope remains auditable in the manifest.
    assert result.manifest["raw_sha256"] == observation.raw_sha256


def test_compact_renderer_audits_reversible_unicode_fallback(tmp_path: Path) -> None:
    source = _long_python().replace(
        'label = f"value: {values!r}"', 'label = "unassigned: \u0a00"'
    )
    result = render_compact_observation(_observation(source), tmp_path)
    assert result is not None
    assert "[[U+0A00]]" in result.visual_transcript
    assert result.manifest["unicode_glyph_fallbacks"] == [
        {
            "codepoint": "U+0A00",
            "source_character": "\u0a00",
            "visual_label": "[[U+0A00]]",
            "occurrences": 1,
        }
    ]


def test_long_diff_uses_lossless_compact_layout() -> None:
    diff = "diff --git a/a.py b/a.py\n" + "+value = 1\n" * 400
    document = build_compact_document(_observation(diff))
    assert document is not None
    assert document.classification == "compact_diff_v1"
    assert (
        "".join(region.plain_text for region in document.regions).count("+value = 1")
        == 400
    )


def test_grep_path_records_are_grouped_and_exactly_reconstructable() -> None:
    observation = _path_record_observation()
    command = parse_path_record_command(
        observation.tool_arguments, observation.project_root
    )
    assert command is not None
    assert command.command_kind == "grep"
    assert command.cwd.as_posix() == "/testbed"

    document = build_compact_document(observation)
    assert document is not None
    assert document.classification == "compact_path_records_v1"
    assert [region.title for region in document.regions] == [
        "FILE ~/pkg/a.py",
        "FILE ~/pkg/b.py",
    ]
    assert len(document.path_records) == 8
    assert len(document.path_mappings) == 8
    preservation = document.information_preservation
    assert preservation["exact_body_reconstruction"] is True
    assert preservation["record_count"] == 8
    assert preservation["group_count"] == 2
    assert preservation["command_kind"] == "grep"
    assert preservation["command_cwd"] == "/testbed"
    assert preservation["relative_paths_resolved_from_cwd"] is True
    for mapping in document.path_mappings:
        assert mapping.original.startswith("/testbed/pkg/")
        assert mapping.display.startswith("~/pkg/")
        assert mapping.source_start is not None and mapping.source_end is not None
        assert observation.content[mapping.source_start : mapping.source_end] == (
            mapping.observed
        )
    compact_text = "".join(region.plain_text for region in document.regions)
    assert "value = '/testbed/literal.py'" in compact_text
    assert "# comment: keep every colon" not in compact_text
    assert "mapping = {'key': 'value'}" in compact_text
    assert document.information_preservation["visual_python_comments_removed"] == 1


def test_path_record_parser_fails_closed_on_unproven_or_unsafe_output() -> None:
    def path_document(observation: CompactObservationInput):
        parsed = parse_mini_swe_observation(observation.content)
        assert parsed is not None
        return build_path_record_document(observation, parsed)

    assert (
        path_document(_path_record_observation(command="cd /testbed && cat pkg/a.py"))
        is None
    )
    malformed = "\n".join(
        ["pkg/a.py:1:value"] * 7 + ["ordinary log: not a location record"]
    )
    assert path_document(_path_record_observation(body=malformed)) is None
    escaping = "\n".join(["../outside.py:1:value"] * 8)
    assert path_document(_path_record_observation(body=escaping)) is None
    prefix_collision = "\n".join(["/testbed2/pkg/a.py:1:value"] * 8)
    assert path_document(_path_record_observation(body=prefix_collision)) is None
    colored = "\n".join(["pkg/a.py:1:\x1b[31mvalue\x1b[0m"] * 8)
    assert path_document(_path_record_observation(body=colored)) is None


def test_one_strict_search_record_is_compacted() -> None:
    document = build_compact_document(
        _path_record_observation(body="pkg/a.py:12:only_match = True")
    )
    assert document is not None
    assert document.classification == "compact_path_records_v1"
    assert document.information_preservation["record_count"] == 1


def test_short_python_file_excerpt_is_compacted_as_fragment() -> None:
    fragment = "    if ready:\n        return {'value': 1}\n"
    document = build_compact_document(_observation(fragment))
    assert document is not None
    assert document.classification == "compact_python_v1"
    assert document.complete_python is False
    assert document.regions[0].kind == "python_fragment"
    assert "return {'value': 1}" in document.regions[0].plain_text


def test_tables_and_matrices_keep_original_layout() -> None:
    markdown = "| name | value |\n| ---- | ----- |\n| a | 1 |\n"
    matrix = "array([[ True, False],\n       [False,  True]])\n"
    shell = json.dumps({"command": "python -c 'print(1)'"})
    for body in (markdown, matrix):
        observation = _observation(body)
        object.__setattr__(observation, "tool_arguments", shell)
        assert build_compact_document(observation) is None


def test_matrix_examples_inside_warning_or_doctest_are_not_layout_allowlisted() -> None:
    warning = (
        "/testbed/pkg/a.py:2: UserWarning: values\n  array([[1, 2],\n       [3, 4]])\n"
    )
    doctest = (
        ">>> values = array([[1, 2], [3, 4]])\n"
        ">>> values\n"
        "array([[1, 2],\n       [3, 4]])\n"
    )
    shell = json.dumps({"command": "python -c 'print(1)'"})
    for body in (warning, doctest):
        observation = _observation(body)
        object.__setattr__(observation, "tool_arguments", shell)
        document = build_compact_document(observation)
        assert document is not None
        assert document.classification == "compact_shell_v1"


def test_short_shell_output_preserves_every_character_in_flow() -> None:
    body = "first line\n/testbed/pkg/a.py:12: value\nlast line\n"
    observation = _observation(body)
    object.__setattr__(
        observation,
        "tool_arguments",
        json.dumps({"command": "python -c 'print(1)'"}),
    )
    document = build_compact_document(observation)
    assert document is not None
    assert document.classification == "compact_shell_v1"
    visual = "".join(region.plain_text for region in document.regions)
    assert "first line" in visual
    assert "/testbed/pkg/a.py:12: value" in visual
    assert "last line" in visual
    assert visual.count("⏎") == document.body.count("\n")


def test_shell_flow_strips_line_indent_but_preserves_interior_spaces() -> None:
    body = "    first line\n\tsecond  line\n  third line\n"
    observation = _observation(body)
    object.__setattr__(
        observation,
        "tool_arguments",
        json.dumps({"command": "python -c 'print(1)'"}),
    )
    document = build_compact_document(observation)
    assert document is not None
    visual = "".join(region.plain_text for region in document.regions)

    assert "first line ⏎ second  line ⏎ third line" in visual
    assert "    first line" not in visual
    assert "\tsecond" not in visual
    assert "second  line" in visual
    assert document.semantic_encoding == (
        "line_leading_whitespace_stripped_hard_newline_flow_v2"
    )
    assert document.information_preservation["source_text_deleted"] is True
    assert document.information_preservation["line_leading_whitespace_removed"] == 7


def test_diff_flow_preserves_context_line_marker() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        " context line\n"
        "-old line\n"
        "+new line\n"
    )
    document = build_compact_document(_observation(diff))
    assert document is not None
    visual = "".join(region.plain_text for region in document.regions)

    assert "⏎  context line" in visual
    assert document.semantic_encoding == "lossless_hard_newline_flow_v1"
    assert document.information_preservation["source_text_deleted"] is False
    assert document.information_preservation["line_leading_whitespace_removed"] == 0


def test_rg_columns_and_nonconsecutive_groups_preserve_record_order() -> None:
    body = "\n".join(
        [
            "pkg/a.py:1:7:first",
            "pkg/a.py:2:8:second",
            "pkg/a.py:3:9:third",
            "pkg/b.py:4:10:fourth",
            "pkg/b.py:5:11:fifth",
            "pkg/a.py:6:12:sixth",
            "pkg/a.py:7:13:seventh",
            "pkg/a.py:8:14:eighth",
        ]
    )
    document = build_compact_document(
        _path_record_observation(
            command='cd /testbed && rg -n --column "value" pkg', body=body
        )
    )
    assert document is not None
    assert [region.title for region in document.regions] == [
        "FILE ~/pkg/a.py",
        "FILE ~/pkg/b.py",
        "FILE ~/pkg/a.py",
    ]
    assert [record["record_index"] for record in document.path_records] == list(
        range(1, 9)
    )
    assert [record["column"] for record in document.path_records] == [
        str(value) for value in range(7, 15)
    ]
    assert document.information_preservation["group_count"] == 3


def test_path_record_renderer_preserves_payloads_and_audits_geometry(
    tmp_path: Path,
) -> None:
    result = render_compact_observation(
        _path_record_observation(), tmp_path, write_supporting_files=True
    )
    assert result is not None
    assert result.manifest["classification"] == "compact_path_records_v1"
    assert result.manifest["semantic_encoding"] == "grouped_location_records_v1"
    preservation = result.manifest["information_preservation"]
    assert preservation["exact_body_reconstruction"] is True
    assert preservation["body_sha256"] == preservation["reconstructed_body_sha256"]
    assert result.manifest["generated_structure"] == {
        "open_braces": 0,
        "close_braces": 0,
        "balanced": True,
    }
    assert result.visual_transcript.count("FILE ~/pkg/a.py") == 1
    assert result.visual_transcript.count("FILE ~/pkg/b.py") == 1
    assert "⏎" in result.visual_transcript
    with Image.open(result.png_path) as image:
        assert image.width in WIDTH_TIERS
        assert image.height == result.manifest["visual_layout"]["height"]


def test_width_tier_boundaries_are_deterministic() -> None:
    assert _select_width_tier(767) == 768
    assert _select_width_tier(768) == 768
    assert _select_width_tier(769) == 1152
    assert _select_width_tier(1152) == 1152
    assert _select_width_tier(1153) == 1536
    assert _select_width_tier(1536) == 1536
    assert _select_width_tier(1537) == 1536


def test_short_medium_and_long_shell_outputs_use_content_width_tiers(
    tmp_path: Path,
) -> None:
    widths = []
    for index, body in enumerate(("ok\n", "m" * 85 + "\n", "w" * 180 + "\n")):
        observation = _observation(body)
        object.__setattr__(
            observation,
            "tool_arguments",
            json.dumps({"command": "python -c 'print(1)'"}),
        )
        result = render_compact_observation(
            observation, tmp_path / str(index), stem=f"tier-{index}"
        )
        assert result is not None
        widths.append(result.manifest["visual_layout"]["width"])

    assert widths == [768, 1152, 1536]


def test_empty_and_truncated_outputs_are_compact_and_auditable(
    tmp_path: Path,
) -> None:
    empty = _observation("")
    empty_result = render_compact_observation(empty, tmp_path / "empty")
    assert empty_result is not None
    assert empty_result.manifest["classification"] == "compact_empty_v1"
    assert empty_result.manifest["visual_layout"]["width"] == 768
    assert "empty-success" in empty_result.visual_transcript
    assert empty_result.manifest["raw_sha256"] == empty.raw_sha256

    truncated_content = (
        "<returncode>9</returncode>\n"
        "<warning>limit reached</warning>\n"
        "<output_head>first line\n\x1b[31mred\x1b[0m</output_head>\n"
        "<elided_chars>123 characters elided</elided_chars>\n"
        "<output_tail>Traceback tail\nValueError: final failure</output_tail>"
    )
    truncated = _observation()
    object.__setattr__(truncated, "content", truncated_content)
    result = render_compact_observation(truncated, tmp_path / "truncated")
    assert result is not None
    assert result.manifest["classification"] == "compact_truncated_v2"
    assert result.manifest["visual_layout"]["width"] in {768, 1152}
    assert "warning=limit reached" in result.visual_transcript
    assert "omitted=123 characters elided" in result.visual_transcript
    assert "[Output head]" in result.visual_transcript
    assert "red" in result.visual_transcript
    assert "[Output tail]" in result.visual_transcript
    assert "ValueError: final failure" in result.visual_transcript
    preservation = result.manifest["information_preservation"]
    assert preservation["raw_observation_sha256"] == truncated.raw_sha256
    assert preservation["missing_content_reconstructed"] is False
    assert preservation["terminal_sanitizer"]["changed"] is True


def test_visual_sanitization_never_mutates_recent_raw_observation(
    tmp_path: Path,
) -> None:
    raw_body = "literal [[U+001B]] \x1b[32mgreen\x1b[0m\n"
    observation = _observation(raw_body)
    original = observation.content
    result = render_compact_observation(observation, tmp_path)

    assert result is not None
    assert observation.content == original
    assert result.manifest["raw_sha256"] == observation.raw_sha256
    assert "literal [[U+001B]] green" in result.visual_transcript
    assert "\x1b" not in result.visual_transcript
    sanitizer = result.manifest["terminal_sanitizer"]
    assert sanitizer["changed"] is True
    assert sanitizer["input_sha256"] != sanitizer["output_sha256"]


def test_wrapped_header_has_reserved_height_and_does_not_overlap_body(
    tmp_path: Path,
) -> None:
    observation = _observation("body sentinel\n")
    object.__setattr__(observation, "observation_index", int("9" * 120))
    result = render_compact_observation(observation, tmp_path, stem="long-header")

    assert result is not None
    layout = result.manifest["visual_layout"]
    assert layout["header_line_count"] > 1
    assert layout["header_height"] > 44
    assert layout["header_overlaps_body"] is False
    assert "body sentinel" in result.visual_transcript
