"""Render recent3 observation history with strict visual Token budgets, offline."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from swe_memory_policy import (
    CompactObservationInput,
    build_responses_observation_image_block,
    compact_renderer_manifest,
    downscale_png_to_visual_token_ratio,
    enumerate_responses_observations,
    partition_responses_observations,
    render_compact_observation,
    render_text_pages,
)
from swe_memory_policy.utils import sha256_file
from swe_memory_policy.vision_tokens import (
    DETAIL,
    ESTIMATOR,
    MODEL,
    UnattainableVisualTokenBudget,
    estimate_visual_tokens,
)

VISUAL_WHITESPACE_POLICY = "compact_visual_whitespace_v2"


def select_observations(
    value: Any, *, recent: int = 3, project_root: str = "/testbed"
) -> tuple[list[CompactObservationInput], dict[str, Any]]:
    """Accept a selected observation, native request, or Responses trajectory."""
    if recent < 0:
        raise ValueError("recent must be non-negative")
    if isinstance(value, str):
        value = {"content": value}
    if not isinstance(value, dict):
        raise ValueError("input must be a text observation or JSON object")
    if isinstance(value.get("content"), str):
        selected = CompactObservationInput(
            observation_index=value.get("observation_index", 1),
            tool_message_index=value.get("tool_message_index", 1),
            page_index=1,
            page_count=1,
            role="tool",
            tool_call_id=value.get("tool_call_id", "offline_observation"),
            project_root=value.get("project_root", project_root),
            content=value["content"],
            tool_name=value.get("tool_name"),
            tool_arguments=value.get("tool_arguments"),
            visual_whitespace_policy=VISUAL_WHITESPACE_POLICY,
        )
        return [selected], {"selection": "explicit_observation", "recent": None}

    payload = value.get("transformed_request", value)
    items = payload.get("input")
    ignored_terminal_calls = []
    if items is None and isinstance(value.get("messages"), list):
        items = []
        for message in value["messages"]:
            if message.get("object") == "response":
                items.extend(message.get("output", []))
            else:
                items.append(message)
        # A finished mini trajectory can end with submit calls and no tool result.
        # Only that terminal batch is excluded; earlier incomplete linkage errors.
        last_output = max(
            (
                i
                for i, item in enumerate(items)
                if item.get("type") == "function_call_output"
            ),
            default=-1,
        )
        ignored_terminal_calls = [
            item.get("call_id")
            for item in items[last_output + 1 :]
            if item.get("type") == "function_call"
        ]
        items = items[: last_output + 1] + [
            item
            for item in items[last_output + 1 :]
            if item.get("type") != "function_call"
        ]
    if not isinstance(items, list):
        raise ValueError(
            "expected content, input, transformed_request.input, or messages"
        )
    observations = enumerate_responses_observations(items)
    partition = partition_responses_observations(observations, recent_oa_count=recent)
    selected = [
        CompactObservationInput(
            observation_index=obs.oa_index,
            tool_message_index=obs.tool_index,
            page_index=1,
            page_count=1,
            role="tool",
            tool_call_id=obs.tool_call_id,
            project_root=project_root,
            content=obs.content,
            input_index=obs.input_index,
            sequence_index=obs.sequence_index,
            tool_name=obs.tool_name,
            tool_arguments=obs.tool_arguments,
            visual_whitespace_policy=VISUAL_WHITESPACE_POLICY,
        )
        for obs in partition.image_observations
    ]
    return selected, {
        "selection": "history_window",
        "recent": recent,
        "image_oa_indexes": list(partition.image_oa_indexes),
        "text_oa_indexes": list(partition.text_oa_indexes),
        "text_input_indexes": list(partition.text_input_indexes),
        "ignored_terminal_call_ids": ignored_terminal_calls,
        "native_history_modified": False,
    }


def render_base(
    observation: CompactObservationInput, output_dir: Path
) -> tuple[list[Path], dict[str, Any]]:
    observation = replace(
        observation, visual_whitespace_policy=VISUAL_WHITESPACE_POLICY
    )
    compact = render_compact_observation(observation, output_dir)
    if compact is not None:
        pages = [Path(compact.png_path)]
        metadata = {
            "classification": compact.manifest["classification"],
            "compact_rendering": compact.manifest,
        }
    else:
        block = build_responses_observation_image_block(
            sequence_index=observation.sequence_index or 1,
            oa_index=observation.observation_index,
            tool_index=observation.tool_message_index,
            tool_call_id=observation.tool_call_id,
            content=observation.content,
        )
        pages = render_text_pages(
            block.source_text,
            output_dir,
            block.stem,
            page_label=block.page_label,
            always_page_suffix=True,
        )
        metadata = {
            "classification": "preserve_source_lines_fallback",
            "compact_rendering": None,
        }
    return pages, metadata


def render_history(
    value: Any,
    output_dir: Path,
    *,
    factors: tuple[Fraction, ...] = (Fraction(4),),
    model: str = MODEL,
    detail: str = DETAIL,
    recent: int = 3,
    project_root: str = "/testbed",
) -> dict[str, Any]:
    estimate_visual_tokens(1, 1, model=model, detail=detail)
    if any(
        isinstance(factor, bool) or not isinstance(factor, int | Fraction)
        for factor in factors
    ):
        raise TypeError("factors must be integers or Fractions")
    factors = tuple(Fraction(factor) for factor in factors)
    if not factors or any(factor < 1 for factor in factors):
        raise ValueError("provide one or more factors >= 1")
    if len(set(factors)) != len(factors):
        raise ValueError("duplicate factors are not allowed")
    observations, selection = select_observations(
        value, recent=recent, project_root=project_root
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "swe-memory-offline-render-v2",
        "compression_definition": "visual_token_ratio",
        "visual_token_estimator": ESTIMATOR,
        "model": model,
        "input_image_detail": detail,
        "renderer": compact_renderer_manifest(),
        **selection,
        "observations": [],
    }
    for observation in observations:
        pages, metadata = render_base(observation, output_dir / "base")
        record = {
            "oa_index": observation.observation_index,
            "tool_index": observation.tool_message_index,
            "call_id": observation.tool_call_id,
            "source_sha256": observation.raw_sha256,
            **metadata,
            "images": [],
        }
        for factor in factors:
            label = f"{factor.numerator}p{factor.denominator}"
            target_dir = output_dir / f"token_{label}"
            target_dir.mkdir(exist_ok=True)
            for page in pages:
                target = target_dir / page.name
                shutil.copy2(page, target)
                image_record = {
                    "factor": str(factor),
                    "base_path": page.relative_to(output_dir).as_posix(),
                    "base_sha256": sha256_file(page),
                }
                try:
                    geometry = downscale_png_to_visual_token_ratio(
                        target,
                        target_numerator=factor.denominator,
                        target_denominator=factor.numerator,
                        model=model,
                        detail=detail,
                    )
                except UnattainableVisualTokenBudget as error:
                    target.unlink()
                    image_record.update(
                        status="unattainable_budget",
                        path=None,
                        sha256=None,
                        error=str(error),
                        **error.metadata,
                    )
                else:
                    image_record.update(
                        status="rendered",
                        path=target.relative_to(output_dir).as_posix(),
                        sha256=sha256_file(target),
                        **geometry,
                    )
                record["images"].append(image_record)
        result["observations"].append(record)
    result["status"] = (
        "partial"
        if any(
            image["status"] != "rendered"
            for observation in result["observations"]
            for image in observation["images"]
        )
        else "complete"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="UTF-8 .txt observation or JSON history"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--factors", nargs="+", default=[Fraction(4)], type=Fraction
    )
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--detail", default=DETAIL, choices=[DETAIL])
    parser.add_argument("--recent", type=int, default=3)
    parser.add_argument("--project-root", default="/testbed")
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    value = text if args.input.suffix.lower() == ".txt" else json.loads(text)
    result = render_history(
        value,
        args.output_dir,
        factors=tuple(args.factors),
        model=args.model,
        detail=args.detail,
        recent=args.recent,
        project_root=args.project_root,
    )
    print(args.output_dir / "manifest.json")
    if result["status"] == "partial":
        parser.exit(
            2, "some requested Token budgets are unattainable; see manifest.json\n"
        )


if __name__ == "__main__":
    main()
