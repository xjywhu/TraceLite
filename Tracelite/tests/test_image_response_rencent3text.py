from __future__ import annotations

import pytest

from swe_memory_policy import (
    enumerate_responses_observations,
    partition_responses_observations,
)


def _items(oa_count: int, *, parallel_first: bool = False) -> list[dict[str, object]]:
    items: list[dict[str, object]] = [
        {"type": "message", "role": "system", "content": "system"},
        {"type": "message", "role": "user", "content": "task"},
    ]
    for oa_index in range(1, oa_count + 1):
        tool_count = 2 if parallel_first and oa_index == 1 else 1
        for tool_index in range(1, tool_count + 1):
            call_id = f"call-{oa_index}-{tool_index}"
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "bash",
                    "arguments": "{}",
                }
            )
        for tool_index in range(1, tool_count + 1):
            call_id = f"call-{oa_index}-{tool_index}"
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": f"output {oa_index}.{tool_index}",
                }
            )
    return items


@pytest.mark.parametrize("oa_count", [0, 1, 3])
def test_short_histories_keep_every_observation_textual(oa_count: int) -> None:
    observations = enumerate_responses_observations(_items(oa_count))
    partition = partition_responses_observations(observations)
    assert partition.image_observations == ()
    assert partition.text_observations == observations
    assert partition.text_oa_indexes == tuple(range(1, oa_count + 1))


@pytest.mark.parametrize(
    ("oa_count", "image_oa", "text_oa"),
    [
        (4, (1,), (2, 3, 4)),
        (5, (1, 2), (3, 4, 5)),
    ],
)
def test_long_histories_image_only_early_observations(
    oa_count: int, image_oa: tuple[int, ...], text_oa: tuple[int, ...]
) -> None:
    partition = partition_responses_observations(
        enumerate_responses_observations(_items(oa_count))
    )
    assert partition.image_oa_indexes == image_oa
    assert partition.text_oa_indexes == text_oa


def test_parallel_outputs_share_the_same_transport() -> None:
    partition = partition_responses_observations(
        enumerate_responses_observations(_items(4, parallel_first=True))
    )
    assert [item.oa_index for item in partition.image_observations[:2]] == [1, 1]
    assert [item.tool_index for item in partition.image_observations[:2]] == [1, 2]


def test_enumeration_preserves_tool_metadata_for_renderer_classification() -> None:
    observations = enumerate_responses_observations(_items(1))
    assert observations[0].tool_name == "bash"
    assert observations[0].tool_arguments == "{}"


def test_unknown_output_call_id_fails_closed() -> None:
    items = _items(1)
    items[-1]["call_id"] = "unknown"
    with pytest.raises(ValueError, match="preceding function_call"):
        enumerate_responses_observations(items)


def test_non_text_output_fails_closed() -> None:
    items = _items(1)
    items[-1]["output"] = [{"type": "input_image"}]
    with pytest.raises(ValueError, match="must contain text"):
        enumerate_responses_observations(items)


def test_recent_zero_images_every_observation() -> None:
    observations = enumerate_responses_observations(_items(6))
    partition = partition_responses_observations(observations, recent_oa_count=0)
    assert partition.image_observations == observations
    assert partition.text_observations == ()


def test_recent_oa_count_must_be_non_negative() -> None:
    observations = enumerate_responses_observations(_items(1))
    with pytest.raises(ValueError, match="non-negative"):
        partition_responses_observations(observations, recent_oa_count=-1)
