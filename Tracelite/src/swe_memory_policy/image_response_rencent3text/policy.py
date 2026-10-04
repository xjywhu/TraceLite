from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_RECENT_OA_COUNT = 3


@dataclass(frozen=True)
class ResponsesObservationRef:
    """One native Responses function output and its causal OA location."""

    input_index: int
    sequence_index: int
    oa_index: int
    tool_index: int
    tool_call_id: str
    content: str
    tool_name: str | None = None
    tool_arguments: str | None = None


@dataclass(frozen=True)
class ResponsesObservationPartition:
    """Early image observations and the latest native-text OA window."""

    image_observations: tuple[ResponsesObservationRef, ...]
    text_observations: tuple[ResponsesObservationRef, ...]
    image_oa_indexes: tuple[int, ...]
    text_oa_indexes: tuple[int, ...]

    @property
    def uses_images(self) -> bool:
        return bool(self.image_observations)

    @property
    def image_input_indexes(self) -> tuple[int, ...]:
        return tuple(item.input_index for item in self.image_observations)

    @property
    def text_input_indexes(self) -> tuple[int, ...]:
        return tuple(item.input_index for item in self.text_observations)


def enumerate_responses_observations(
    items: Sequence[Any],
) -> tuple[ResponsesObservationRef, ...]:
    """Validate native Responses call linkage and enumerate visible outputs.

    Consecutive ``function_call`` items form one OA/tool batch.  Their matching
    ``function_call_output`` items may be separated by opaque reasoning or
    message items, but every call ID must be unique and consumed exactly once.
    """

    pending: dict[str, tuple[int, int, str | None, str | None]] = {}
    known_call_ids: set[str] = set()
    consumed_call_ids: set[str] = set()
    observations: list[ResponsesObservationRef] = []
    oa_index = 0
    for input_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Responses history has an empty function call_id")
            if call_id in known_call_ids:
                raise ValueError("Responses history has a duplicate function call_id")
            if not pending:
                oa_index += 1
            name = item.get("name")
            arguments = item.get("arguments")
            pending[call_id] = (
                oa_index,
                len(pending) + 1,
                name if isinstance(name, str) else None,
                arguments if isinstance(arguments, str) else None,
            )
            known_call_ids.add(call_id)
            continue
        if item_type != "function_call_output":
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("function_call_output has an empty call_id")
        if call_id in consumed_call_ids:
            raise ValueError("Responses history repeats a function_call_output")
        location = pending.pop(call_id, None)
        if location is None:
            raise ValueError(
                "function_call_output does not match a preceding function_call"
            )
        output = item.get("output")
        if not isinstance(output, str):
            raise ValueError("function_call_output must contain text before rendering")
        consumed_call_ids.add(call_id)
        observations.append(
            ResponsesObservationRef(
                input_index=input_index,
                sequence_index=len(observations) + 1,
                oa_index=location[0],
                tool_index=location[1],
                tool_call_id=call_id,
                content=output,
                tool_name=location[2],
                tool_arguments=location[3],
            )
        )
    if pending:
        raise ValueError("Responses history contains function calls without outputs")
    return tuple(observations)


def _unique_oa_indexes(
    observations: Sequence[ResponsesObservationRef],
) -> tuple[int, ...]:
    return tuple(dict.fromkeys(item.oa_index for item in observations))


def partition_responses_observations(
    observations: Sequence[ResponsesObservationRef],
    *,
    recent_oa_count: int = DEFAULT_RECENT_OA_COUNT,
) -> ResponsesObservationPartition:
    """Image observations outside the exact recent-OA text window.

    Assistant reasoning, function calls, opaque reasoning items and all other
    Responses items are outside this partition and remain native.  Every tool
    output in the same OA is assigned to the same transport.
    """

    if recent_oa_count < 0:
        raise ValueError("recent_oa_count must be non-negative")
    observations = tuple(observations)
    oa_indexes = _unique_oa_indexes(observations)
    if oa_indexes and oa_indexes != tuple(range(1, oa_indexes[-1] + 1)):
        raise ValueError("Responses observation OA indexes are not contiguous")
    if not observations:
        return ResponsesObservationPartition(
            image_observations=(),
            text_observations=observations,
            image_oa_indexes=(),
            text_oa_indexes=oa_indexes,
        )

    cutoff_oa = oa_indexes[-1] - recent_oa_count
    image = tuple(item for item in observations if item.oa_index <= cutoff_oa)
    text = tuple(item for item in observations if item.oa_index > cutoff_oa)
    return ResponsesObservationPartition(
        image_observations=image,
        text_observations=text,
        image_oa_indexes=_unique_oa_indexes(image),
        text_oa_indexes=_unique_oa_indexes(text),
    )
