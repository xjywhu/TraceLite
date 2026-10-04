from __future__ import annotations

from dataclasses import dataclass

SEPARATOR = "=" * 36


@dataclass(frozen=True)
class ObservationImageBlock:
    """One Responses function output rendered independently as images."""

    sequence_index: int
    oa_index: int
    tool_index: int
    tool_call_id: str
    content: str
    source_text: str

    @property
    def stem(self) -> str:
        return f"OA{self.oa_index:04d}_Tool{self.tool_index:03d}"

    @property
    def page_label(self) -> str:
        return f"OA {self.oa_index} - TOOL MESSAGE {self.tool_index}"


def render_responses_image_response_only_observation(
    *, oa_index: int, tool_index: int, tool_call_id: str, content: str
) -> str:
    """Render one API-visible Responses function output without modification."""

    if oa_index < 1 or tool_index < 1:
        raise ValueError("observation image indexes must be positive")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("observation image requires a non-empty tool_call_id")
    if not isinstance(content, str):
        raise ValueError("observation image requires string tool content")
    return (
        "\n".join(
            [
                SEPARATOR,
                f"OA {oa_index} - TOOL MESSAGE {tool_index}",
                SEPARATOR,
                "",
                "role: tool",
                "",
                f"tool_call_id: {tool_call_id}",
                "",
                "content:",
                content,
            ]
        )
        + "\n"
    )


def build_responses_observation_image_block(
    *,
    sequence_index: int,
    oa_index: int,
    tool_index: int,
    tool_call_id: str,
    content: str,
) -> ObservationImageBlock:
    """Build one deterministic block from a Responses function output."""

    return ObservationImageBlock(
        sequence_index=sequence_index,
        oa_index=oa_index,
        tool_index=tool_index,
        tool_call_id=tool_call_id,
        content=content,
        source_text=render_responses_image_response_only_observation(
            oa_index=oa_index,
            tool_index=tool_index,
            tool_call_id=tool_call_id,
            content=content,
        ),
    )
