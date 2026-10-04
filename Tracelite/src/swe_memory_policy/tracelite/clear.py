"""Edit-stage Observation clearing and rendered-page 20/5 clearing."""

IMAGE_CLEAR_THRESHOLD = 7
PAGE_CLEAR_THRESHOLD = 20
PAGE_KEEP_LATEST = 5


def filter_image_pages(state, records, *, force=False):
    cleared = set(state.get("cleared_page_ids", []))
    ids = [f"{record['tool_call_id']}:{record['page_index']}" for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rendered image page identity")
    active = [
        (key, record)
        for key, record in zip(ids, records, strict=True)
        if key not in cleared
    ]
    events = state.setdefault("page_clear_events", [])
    if force or len(active) > PAGE_CLEAR_THRESHOLD:
        removed = active[:-PAGE_KEEP_LATEST]
        retained = active[-PAGE_KEEP_LATEST:]
        cleared.update(key for key, _ in removed)
        events.append(
            {
                "ordinal": len(events) + 1,
                "trigger": "explicit_recovery_clear"
                if force
                else "rendered_image_count_gt_20",
                "image_count_before_clear": len(active),
                "threshold": PAGE_CLEAR_THRESHOLD,
                "keep_latest_images": PAGE_KEEP_LATEST,
                "cleared_page_ids": [key for key, _ in removed],
                "retained_page_ids": [key for key, _ in retained],
            }
        )
        active = retained
    by_call = {}
    for key, record in zip(ids, records, strict=True):
        by_call.setdefault(record["tool_call_id"], []).append(key)
    fully_cleared = set(state.get("page_cleared_call_ids", []))
    fully_cleared.update(
        call_id
        for call_id, keys in by_call.items()
        if all(key in cleared for key in keys)
    )
    state["cleared_page_ids"] = sorted(cleared)
    state["page_cleared_call_ids"] = sorted(fully_cleared)
    state["page_clear_last_image_count"] = len(active)
    return [record for _, record in active]


def record_file_change(state, call_id, before, after):
    if state["edit_stage_started"] is not None or before == after:
        return False
    if not call_id:
        raise ValueError("file change requires an executed tool call ID")
    state["edit_stage_started"] = {
        "call_id": call_id,
        "before_sha256": before,
        "after_sha256": after,
    }
    return True


def plan_edit_stage_clear(state, ordered, partition):
    cleared = state["cleared"]
    image_ids = [
        item.tool_call_id
        for item in partition.image_observations
        if item.tool_call_id not in cleared
    ]
    if (
        state["edit_stage_started"] is not None
        and len(image_ids) > IMAGE_CLEAR_THRESHOLD
    ):
        event = {
            "ordinal": len(state["clear_events"]) + 1,
            "trigger_call_count": len(ordered),
            "image_count_before_clear": len(image_ids),
            "cleared_call_ids": image_ids,
        }
        state["clear_events"].append(event)
        for call_id in image_ids:
            cleared[call_id] = "edit_stage_image_count_gt_7_clear"
        image_ids = []
    state["last_image_call_ids"] = image_ids
    cleared_indexes = {
        item.input_index: cleared[item.tool_call_id]
        for item in ordered
        if item.tool_call_id in cleared
    }
    return image_ids, cleared_indexes
