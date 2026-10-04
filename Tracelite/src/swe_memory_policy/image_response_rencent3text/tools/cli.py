from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swe_memory_policy.image_response_rencent3text.tools.model import (
    CompactObservationInput,
)
from swe_memory_policy.image_response_rencent3text.tools.renderer import (
    render_compact_observation,
)


def _payload(value: dict[str, Any]) -> dict[str, Any]:
    transformed = value.get("transformed_request")
    return transformed if isinstance(transformed, dict) else value


def extract_observation(
    request: dict[str, Any], *, call_id: str, project_root: str
) -> CompactObservationInput:
    items = _payload(request).get("input")
    if not isinstance(items, list):
        raise ValueError("Responses request has no input list")
    pending: dict[str, tuple[int, int, str | None, str | None]] = {}
    oa_index = 0
    sequence_index = 0
    for input_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            if not pending:
                oa_index += 1
            current_id = item.get("call_id")
            if not isinstance(current_id, str):
                continue
            pending[current_id] = (
                oa_index,
                len(pending) + 1,
                item.get("name") if isinstance(item.get("name"), str) else None,
                item.get("arguments")
                if isinstance(item.get("arguments"), str)
                else None,
            )
            continue
        if item.get("type") != "function_call_output":
            continue
        current_id = item.get("call_id")
        metadata = pending.pop(current_id, None)
        sequence_index += 1
        if current_id != call_id:
            continue
        if metadata is None or not isinstance(item.get("output"), str):
            raise ValueError("selected output is not linked text")
        return CompactObservationInput(
            observation_index=metadata[0],
            tool_message_index=metadata[1],
            page_index=1,
            page_count=1,
            role="tool",
            tool_call_id=call_id,
            project_root=project_root,
            content=item["output"],
            input_index=input_index,
            sequence_index=sequence_index,
            tool_name=metadata[2],
            tool_arguments=metadata[3],
        )
    raise ValueError(f"call_id not found: {call_id}")


def _extract(args: argparse.Namespace) -> None:
    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    observation = extract_observation(
        request, call_id=args.call_id, project_root=args.project_root
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(observation.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _render(args: argparse.Namespace) -> None:
    value = json.loads(args.input_json.read_text(encoding="utf-8"))
    observation = CompactObservationInput.from_dict(value)
    result = render_compact_observation(
        observation,
        args.output_dir,
        write_supporting_files=True,
        emit_debug_markdown=args.emit_debug_md,
    )
    if result is None:
        raise SystemExit("input is not a high-confidence long Python observation")
    print(result.png_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="recent3 compact Python renderer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--request-json", type=Path, required=True)
    extract.add_argument("--call-id", required=True)
    extract.add_argument("--project-root", default="/testbed")
    extract.add_argument("--output", type=Path, required=True)
    extract.set_defaults(handler=_extract)
    render = subparsers.add_parser("render")
    render.add_argument("--input-json", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--emit-debug-md", action="store_true")
    render.set_defaults(handler=_render)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
