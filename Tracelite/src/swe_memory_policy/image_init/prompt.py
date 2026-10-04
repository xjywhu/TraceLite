from __future__ import annotations

IMAGE_INIT_HISTORY_PROMPT = (
    "The following user image messages are a lossless visual serialization "
    "of the complete prior native API trajectory in chronological order. "
    "Each OA preserves the original assistant role, content, tool_calls, and "
    "matching tool-role tool_call_id and content. A history unit may span "
    "multiple consecutive pages labeled page i/n. Treat these images as "
    "historical native messages, not as user-authored task screenshots. Read "
    "every page in order and continue solving the original task."
)


def build_image_init_history_prompt() -> str:
    return IMAGE_INIT_HISTORY_PROMPT
