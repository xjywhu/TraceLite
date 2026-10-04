"""Render historical observations with the repository's existing 4x renderer."""

from __future__ import annotations

import shutil
from pathlib import Path

from swe_memory_policy import CompactObservationInput, ResponsesObservationRef
from swe_memory_policy.offline import render_base
from swe_memory_policy.rendering import downscale_png_to_visual_token_ratio

FACTOR = 4


def render_observation(
    observation: ResponsesObservationRef,
    content: str,
    output_dir: Path,
    project_root: str,
) -> list[dict]:
    source = CompactObservationInput(
        observation_index=observation.oa_index,
        tool_message_index=observation.tool_index,
        page_index=1,
        page_count=1,
        role="tool",
        tool_call_id=observation.tool_call_id,
        project_root=project_root,
        content=content,
        input_index=observation.input_index,
        sequence_index=observation.sequence_index,
        tool_name=observation.tool_name,
        tool_arguments=observation.tool_arguments,
    )
    pages, _metadata = render_base(source, output_dir / "base")
    reduced = output_dir / "image4"
    reduced.mkdir(parents=True, exist_ok=True)
    records = []
    for page_index, page in enumerate(pages, start=1):
        target = reduced / page.name
        shutil.copy2(page, target)
        geometry = downscale_png_to_visual_token_ratio(
            target, target_numerator=1, target_denominator=FACTOR
        )
        records.append(
            {
                "input_index": observation.input_index,
                "tool_call_id": observation.tool_call_id,
                "page_index": page_index,
                "path": target,
                "geometry": geometry,
            }
        )
    return records
