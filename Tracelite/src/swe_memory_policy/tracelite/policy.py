import json
from pathlib import Path

from swe_memory_policy.tracelite.clear import (
    IMAGE_CLEAR_THRESHOLD,
    filter_image_pages,
    plan_edit_stage_clear,
    record_file_change,
)
from swe_memory_policy.tracelite.overlap import (
    MIN_MATCHED_LINES,
    _replay_overlap,
    _signature,
)

POLICY_NAME = "image4*_recent3_overlap2_clear"


class OverlapClearPolicy:
    def __init__(self, path):
        self.path = Path(path)
        self.state = {
            "version": 3,
            "seen": [],
            "dependency_lines": {},
            "overlap_events": [],
            "edit_stage_started": None,
            "clear_events": [],
            "cleared": {},
            "last_image_call_ids": [],
        }
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("version") != 3:
                raise ValueError("unsupported overlap2 policy state")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(self.path)

    def filter_image_pages(self, records, *, force=False):
        active = filter_image_pages(self.state, records, force=force)
        self.save()
        return active

    def record_file_change(self, call_id, before, after):
        changed = record_file_change(self.state, call_id, before, after)
        if changed:
            self.save()
        return changed

    def plan(self, observations, partition):
        if partition is None:
            raise ValueError("overlap2 requires a recent3 observation partition")
        ordered = sorted(
            observations, key=lambda item: (item.oa_index, item.tool_index)
        )
        signatures = [_signature(item) for item in ordered]
        seen = self.state["seen"]
        if len(signatures) < len(seen) or signatures[: len(seen)] != seen:
            raise ValueError("policy requires the same append-only raw trajectory")

        dependency_lines, overlap_events, all_replacements = _replay_overlap(ordered)
        image_indexes = set(partition.image_input_indexes)
        replacements = {
            index: content
            for index, content in all_replacements.items()
            if index in image_indexes
        }
        self.state["dependency_lines"] = dependency_lines
        self.state["overlap_events"] = overlap_events
        self.state["seen"] = signatures

        image_ids, cleared_indexes = plan_edit_stage_clear(
            self.state, ordered, partition
        )
        self.save()

        audit = {
            "name": POLICY_NAME,
            "overlap_window": "previous_3_tool_calls",
            "overlap_match_rule": "exact_nonempty_observation_body_line_multiset",
            "overlap_min_matched_lines": MIN_MATCHED_LINES,
            "overlap_iou": "non_whitespace_character_multiset",
            "overlap_action": (
                "record_in_recent3_then_delete_exact_lines_on_image_transition"
            ),
            "image_window": "recent_3_OA_groups",
            "image_clear_rule": {
                "stage": "after_first_actual_edit",
                "operator": ">",
                "threshold": IMAGE_CLEAR_THRESHOLD,
            },
            "dependency_lines": dependency_lines,
            "overlap_events": overlap_events,
            "overlap_event_count": len(overlap_events),
            "rewritten_image_call_ids": [
                item.tool_call_id
                for item in ordered
                if item.input_index in replacements
            ],
            "edit_stage_started": self.state["edit_stage_started"],
            "clear_events": list(self.state["clear_events"]),
            "clear_count": len(self.state["clear_events"]),
            "cleared_call_ids": dict(self.state["cleared"]),
            "image_call_ids": list(image_ids),
        }
        return replacements, cleared_indexes, audit
