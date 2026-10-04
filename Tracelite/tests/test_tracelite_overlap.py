from swe_memory_policy.image_response_rencent3text.policy import ResponsesObservationRef
from swe_memory_policy.tracelite.overlap import _replay_overlap


def _observation(index: int, content: str) -> ResponsesObservationRef:
    return ResponsesObservationRef(
        input_index=index,
        sequence_index=index,
        oa_index=index + 1,
        tool_index=1,
        tool_call_id=f"call-{index}",
        content=content,
    )


def test_overlap_requires_three_matching_nonempty_lines() -> None:
    older = _observation(0, "alpha\nbeta\ngamma\nolder-only\n")

    _, two_line_events, two_line_replacements = _replay_overlap(
        [older, _observation(1, "alpha\nbeta\nnewer-only\n")]
    )
    assert two_line_events == []
    assert two_line_replacements == {}

    _, three_line_events, three_line_replacements = _replay_overlap(
        [older, _observation(1, "alpha\nbeta\ngamma\nnewer-only\n")]
    )
    assert three_line_events[0]["matched_line_occurrences"] == 3
    assert three_line_replacements == {0: "older-only\n"}
