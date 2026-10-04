"""Offline Luna/high patch estimator and strict per-image token budget selector."""

from fractions import Fraction
from math import isqrt

ESTIMATOR = "luna_patch32_high_20260906_v2"
MODEL = "gpt-5.6-luna"
DETAIL = "high"
PATCH_SIZE = 32
MAX_DIMENSION = 2048
PATCH_BUDGET = 2500
DOCUMENTATION = "https://developers.openai.com/api/docs/guides/images-vision#patch-based-image-tokenization"


class UnattainableVisualTokenBudget(ValueError):
    """A strict budget below the frozen estimator's minimum image cost."""

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        super().__init__(
            f"budget {metadata['token_budget']} is below this estimator's minimum "
            f"{metadata['minimum_visual_tokens']}; no image was resized"
        )


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def patch_count(width: int, height: int) -> int:
    return ((width + PATCH_SIZE - 1) // PATCH_SIZE) * (
        (height + PATCH_SIZE - 1) // PATCH_SIZE
    )


def estimate_visual_tokens(
    width: int, height: int, *, model: str = MODEL, detail: str = DETAIL
) -> dict:
    _positive_integer(width, "width")
    _positive_integer(height, "height")
    if model != MODEL or detail != DETAIL:
        raise ValueError(
            "this version supports only gpt-5.6-luna with explicit detail=high"
        )
    scale = min(Fraction(1), Fraction(MAX_DIMENSION, max(width, height)))
    limited_width = max(1, width * scale.numerator // scale.denominator)
    limited_height = max(1, height * scale.numerator // scale.denominator)
    processed_width, processed_height = limited_width, limited_height
    limited_patches = patch_count(limited_width, limited_height)
    if limited_patches > PATCH_BUDGET:
        # Algebraic equivalent of the documented sqrt + patch-boundary adjustment,
        # using integer square roots and rationals to avoid boundary drift.
        columns = isqrt(PATCH_BUDGET * limited_width // limited_height)
        rows = isqrt(PATCH_BUDGET * limited_height // limited_width)
        patch_scale = min(
            Fraction(PATCH_SIZE * columns, limited_width),
            Fraction(PATCH_SIZE * rows, limited_height),
        )
        processed_width = max(
            1, limited_width * patch_scale.numerator // patch_scale.denominator
        )
        processed_height = max(
            1, limited_height * patch_scale.numerator // patch_scale.denominator
        )
    patches = patch_count(processed_width, processed_height)
    if patches > PATCH_BUDGET:
        raise ValueError("patch preprocessing exceeded the model budget")
    return {
        "estimator": ESTIMATOR,
        "model": model,
        "detail": detail,
        "original_width": width,
        "original_height": height,
        "dimension_limited_width": limited_width,
        "dimension_limited_height": limited_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "patch_size": PATCH_SIZE,
        "patch_budget": PATCH_BUDGET,
        "dimension_limit": MAX_DIMENSION,
        "patch_count": patches,
        "patch_multiplier_numerator": 6,
        "patch_multiplier_denominator": 5,
        "tokens": (patches * 6 + 4) // 5,
        "dimension_rounding": "floor_with_minimum_one_pixel",
        "provider_usage_measured": False,
    }


def plan_visual_token_resize(
    width: int,
    height: int,
    *,
    target_numerator: int,
    target_denominator: int,
    model: str = MODEL,
    detail: str = DETAIL,
) -> dict:
    _positive_integer(target_numerator, "target_numerator")
    _positive_integer(target_denominator, "target_denominator")
    if target_numerator > target_denominator:
        raise ValueError("target ratio must not exceed one")
    baseline = estimate_visual_tokens(width, height, model=model, detail=detail)
    baseline_tokens = baseline["tokens"]
    target = Fraction(baseline_tokens * target_numerator, target_denominator)
    budget = target.numerator // target.denominator
    minimum = estimate_visual_tokens(1, 1, model=model, detail=detail)["tokens"]
    if budget < minimum:
        raise UnattainableVisualTokenBudget(
            {
                "compression_definition": "visual_token_ratio",
                "visual_token_estimator": ESTIMATOR,
                "model": model,
                "input_image_detail": detail,
                "target_visual_token_ratio_numerator": target_numerator,
                "target_visual_token_ratio_denominator": target_denominator,
                "baseline_visual_tokens": baseline_tokens,
                "target_visual_tokens": float(target),
                "token_budget": budget,
                "minimum_visual_tokens": minimum,
                "budget_satisfied": False,
                "original_width": width,
                "original_height": height,
                "baseline_preprocessing": baseline,
                "provider_usage_measured": False,
            }
        )
    if target_numerator == target_denominator:
        out_width, out_height = width, height
    else:
        # Candidates are API-ready: no subsequent dimension/patch resizing. Raw patch
        # coverage is monotone along this aspect-preserving integer long-side family.
        long_side, short_side = max(width, height), min(width, height)
        patch_limit = min(PATCH_BUDGET, budget * 5 // 6)
        left, right, best = 1, min(long_side, MAX_DIMENSION), 1
        while left <= right:
            candidate = (left + right) // 2
            short = max(1, short_side * candidate // long_side)
            if patch_count(candidate, short) <= patch_limit:
                best = candidate
                left = candidate + 1
            else:
                right = candidate - 1
        out_short = max(1, short_side * best // long_side)
        out_width, out_height = (
            (best, out_short) if width >= height else (out_short, best)
        )
    output = estimate_visual_tokens(out_width, out_height, model=model, detail=detail)
    achieved = output["tokens"]
    if achieved > budget:
        raise ValueError("selected dimensions exceed the strict token budget")
    return {
        "compression_definition": "visual_token_ratio",
        "visual_token_estimator": ESTIMATOR,
        "model": model,
        "input_image_detail": detail,
        "target_visual_token_ratio_numerator": target_numerator,
        "target_visual_token_ratio_denominator": target_denominator,
        "baseline_visual_tokens": baseline_tokens,
        "target_visual_tokens": float(target),
        "token_budget": budget,
        "estimated_visual_tokens": achieved,
        "achieved_visual_token_ratio": achieved / baseline_tokens,
        "achieved_estimated_compression_factor": baseline_tokens / achieved,
        "budget_satisfied": True,
        "exact_target_achieved": Fraction(achieved) == target,
        "budget_slack_tokens": budget - achieved,
        "original_width": width,
        "original_height": height,
        "width": out_width,
        "height": out_height,
        "resampling": "none"
        if (out_width, out_height) == (width, height)
        else "lanczos",
        "selection_policy": (
            "largest_api_ready_integer_long_side_within_hard_token_budget_v2"
        ),
        "aspect_policy": "original_aspect_ratio_short_side_floor_minimum_one_pixel",
        "baseline_preprocessing": baseline,
        "output_preprocessing": output,
        "provider_usage_measured": False,
    }
