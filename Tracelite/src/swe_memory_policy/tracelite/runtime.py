"""Stateless Responses adapter for TraceLite's compression policy."""

from __future__ import annotations

import base64
import copy
from pathlib import Path

from swe_memory_policy import (
    enumerate_responses_observations,
    partition_responses_observations,
)
from swe_memory_policy.tracelite.policy import OverlapClearPolicy
from swe_memory_policy.tracelite.rendering import render_observation

PAGE_CLEARED_TEXT = "[observation images cleared; latest 5 images retained]"


class TraceLitePolicy:
    """Transform each complete, append-only raw Responses trajectory."""

    def __init__(self, output_dir: str | Path, *, project_root: str = "/testbed"):
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root
        self.policy = OverlapClearPolicy(self.root / "overlap_clear_state.json")

    def record_file_change(self, call_id: str, before: str, after: str) -> bool:
        """Notify the policy when a completed tool call changed repository files."""
        return self.policy.record_file_change(call_id, before, after)

    def transform(self, payload: dict) -> tuple[dict, dict]:
        """Return the provider-ready request and the policy audit."""
        if payload.get("previous_response_id") is not None:
            raise ValueError("TraceLite requires stateless full-history replay")
        if payload.get("model") != "gpt-5.6-luna":
            raise ValueError("image4 requires the gpt-5.6-luna visual-token estimator")
        items = payload.get("input")
        if not isinstance(items, list):
            raise ValueError("Responses request.input must be a list")
        observations = enumerate_responses_observations(items)
        partition = partition_responses_observations(observations, recent_oa_count=3)
        rewritten, suppressed, audit = self.policy.plan(observations, partition)
        page_cleared = set(self.policy.state.get("page_cleared_call_ids", []))
        page_cleared_indexes = {
            observation.input_index
            for observation in partition.image_observations
            if observation.tool_call_id in page_cleared
            and observation.input_index not in suppressed
        }
        suppressed.update(
            {index: "image_count_clear_keep_latest_5" for index in page_cleared_indexes}
        )

        transformed = copy.deepcopy(payload)
        for index, content in rewritten.items():
            transformed["input"][index]["output"] = content
        for index, reason in suppressed.items():
            transformed["input"][index]["output"] = f"[observation removed: {reason}]"
        for index in page_cleared_indexes:
            transformed["input"][index]["output"] = [
                {
                    "type": "input_text",
                    "text": PAGE_CLEARED_TEXT,
                }
            ]

        image_observations = [
            observation
            for observation in partition.image_observations
            if observation.input_index not in suppressed
        ]
        records = []
        if image_observations:
            request_index = len(list(self.root.glob("request_*"))) + 1
            request_dir = self.root / f"request_{request_index:04d}"
            request_dir.mkdir(exist_ok=False)
            for observation in image_observations:
                records.extend(
                    render_observation(
                        observation,
                        rewritten.get(observation.input_index, observation.content),
                        request_dir,
                        self.project_root,
                    )
                )
            records = self.policy.filter_image_pages(records)

        by_input: dict[int, list[dict]] = {}
        for record in records:
            by_input.setdefault(record["input_index"], []).append(record)
        for observation in image_observations:
            image_blocks = [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(record["path"].read_bytes()).decode("ascii"),
                    "detail": "high",
                }
                for record in by_input.get(observation.input_index, [])
            ]
            transformed["input"][observation.input_index]["output"] = image_blocks or [
                {
                    "type": "input_text",
                    "text": PAGE_CLEARED_TEXT,
                }
            ]
        audit["retained_page_ids"] = [
            f"{record['tool_call_id']}:{record['page_index']}" for record in records
        ]
        return transformed, audit
