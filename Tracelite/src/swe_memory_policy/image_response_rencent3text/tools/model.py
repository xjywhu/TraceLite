from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

HARD_NEWLINE_GLYPH = "⏎"
HARD_NEWLINE_MARKER = f" {HARD_NEWLINE_GLYPH} "


@dataclass(frozen=True)
class CompactObservationInput:
    """Complete input required to render one Responses tool observation."""

    observation_index: int
    tool_message_index: int
    page_index: int
    page_count: int
    role: str
    tool_call_id: str
    project_root: str
    content: str
    input_index: int | None = None
    sequence_index: int | None = None
    tool_name: str | None = None
    tool_arguments: str | None = None
    visual_whitespace_policy: str = "preserve"

    def __post_init__(self) -> None:
        if self.observation_index < 1 or self.tool_message_index < 1:
            raise ValueError("observation and tool-message indexes must be positive")
        if self.page_index < 1 or self.page_count < self.page_index:
            raise ValueError("page indexes are invalid")
        if self.role != "tool":
            raise ValueError("compact observations require role='tool'")
        if not self.tool_call_id:
            raise ValueError("tool_call_id must be complete and non-empty")
        root = PurePosixPath(self.project_root)
        if not root.is_absolute() or str(root) == "/":
            raise ValueError("project_root must be a non-root absolute POSIX path")
        self.content.encode("utf-8", errors="strict")
        if self.visual_whitespace_policy not in {
            "preserve",
            "compact_visual_whitespace_v2",
        }:
            raise ValueError("unsupported visual whitespace policy")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CompactObservationInput:
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def stem(self) -> str:
        return f"OA{self.observation_index:04d}_Tool{self.tool_message_index:03d}"

    @property
    def header(self) -> str:
        return (
            f"OA{self.observation_index} · "
            f"Tool Message {self.tool_message_index} · "
            f"page {self.page_index}/{self.page_count} · "
            f"role:{self.role} · "
            f"tool_call_id:{self.tool_call_id} · "
            f"root:{self.project_root} ≡ ~/"
        )


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    kind: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class PathMapping:
    original: str
    display: str
    source_kind: str
    source_start: int | None = None
    source_end: int | None = None
    observed: str | None = None
    resolution_base: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Style = Literal[
    "plain",
    "metadata",
    "markup",
    "path",
    "path_badge",
    "keyword",
    "definition",
    "string",
    "number",
    "decorator",
    "operator",
    "builtin",
    "comment",
    "structure",
    "label",
]


@dataclass(frozen=True)
class StyledAtom:
    text: str
    style: Style = "plain"
    generated: bool = False
    generated_kind: str | None = None


@dataclass
class CompactRegion:
    kind: str
    title: str
    atoms: list[StyledAtom] = field(default_factory=list)
    source_start_line: int | None = None
    source_end_line: int | None = None

    @property
    def plain_text(self) -> str:
        return "".join(atom.text for atom in self.atoms)


@dataclass
class CompactDocument:
    observation: CompactObservationInput
    content_prefix: str
    content_suffix: str
    body: str
    wrapper_kind: str
    regions: list[CompactRegion]
    source_file: str | None
    path_mappings: list[PathMapping]
    source_spans: list[SourceSpan]
    complete_python: bool
    generated_open_braces: int
    generated_close_braces: int
    classification: str = "compact_python_v1"
    semantic_encoding: str = "python_token_compact_v1"
    body_prefix: str = ""
    body_suffix: str = ""
    path_records: list[dict[str, Any]] = field(default_factory=list)
    information_preservation: dict[str, Any] = field(default_factory=dict)
    visual_transcript: str = ""


@dataclass(frozen=True)
class CompactRenderResult:
    png_path: str
    manifest: dict[str, Any]
    visual_transcript: str
    debug_fragments: dict[str, str]
