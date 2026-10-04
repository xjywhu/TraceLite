from fractions import Fraction

import pytest
from PIL import Image

from swe_memory_policy.rendering import downscale_png_to_visual_token_ratio
from swe_memory_policy.utils import sha256_file
from swe_memory_policy.vision_tokens import (
    UnattainableVisualTokenBudget,
    estimate_visual_tokens,
    plan_visual_token_resize,
)


@pytest.mark.parametrize(
    ("size", "processed", "tokens"),
    [
        ((1, 1), (1, 1), 2),
        ((32, 32), (32, 32), 2),
        ((33, 32), (33, 32), 3),
        ((1024, 1024), (1024, 1024), 1229),
        ((2048, 2048), (1600, 1600), 3000),
        ((4096, 4096), (1600, 1600), 3000),
        ((1, 100000), (1, 2048), 77),
    ],
)
def test_frozen_estimator_boundaries(size, processed, tokens):
    result = estimate_visual_tokens(*size)
    assert (result["processed_width"], result["processed_height"]) == processed
    assert result["tokens"] == tokens
    assert result["provider_usage_measured"] is False


@pytest.mark.parametrize("size", [(2048, 2048), (500, 3300), (1175, 347), (1, 6000)])
@pytest.mark.parametrize(
    "factor", [Fraction(k) for k in range(2, 11)] + [Fraction(5, 2)]
)
def test_budget_selection_is_maximal_in_aspect_family(size, factor):
    baseline = estimate_visual_tokens(*size)["tokens"]
    result = plan_visual_token_resize(
        *size,
        target_numerator=factor.denominator,
        target_denominator=factor.numerator,
    )
    budget = int(Fraction(baseline) / factor)
    assert result["token_budget"] == budget
    assert result["estimated_visual_tokens"] <= budget
    width, height = result["width"], result["height"]
    assert width <= size[0] and height <= size[1]
    assert max(width, height) <= 2048
    assert result["output_preprocessing"]["processed_width"] == width
    assert result["output_preprocessing"]["processed_height"] == height
    next_long = max(width, height) + 1
    if next_long <= min(max(size), 2048):
        next_short = max(1, min(size) * next_long // max(size))
        next_patches = ((next_long + 31) // 32) * ((next_short + 31) // 32)
        next_tokens = (next_patches * 6 + 4) // 5
        assert next_tokens > budget or next_patches > 2500


def test_identity_preserves_bytes_even_above_preprocessing_limit(tmp_path):
    page = tmp_path / "large.png"
    Image.new("P", (2200, 2200), 1).save(page, compress_level=0)
    original_hash = sha256_file(page)
    result = downscale_png_to_visual_token_ratio(
        page,
        target_numerator=1,
        target_denominator=1,
    )
    assert result["width"] == result["height"] == 2200
    assert result["resampling"] == "none"
    assert sha256_file(page) == original_hash


def test_one_resize_from_original_and_deterministic_output(tmp_path, monkeypatch):
    from PIL import ImageDraw

    image = Image.new("RGB", (2048, 1400), "white")
    ImageDraw.Draw(image).line((0, 0, 2047, 1399), fill="black", width=7)
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    image.save(first)
    second.write_bytes(first.read_bytes())
    original_resize = Image.Image.resize
    calls = []

    def record_resize(self, size, *args, **kwargs):
        calls.append((self.size, size, args[0]))
        return original_resize(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", record_resize)
    for page in [first, second]:
        downscale_png_to_visual_token_ratio(
            page,
            target_numerator=1,
            target_denominator=4,
        )
    assert len(calls) == 2
    assert all(original == (2048, 1400) for original, _target, _filter in calls)
    assert all(
        resampling == Image.Resampling.LANCZOS for _source, _target, resampling in calls
    )
    assert sha256_file(first) == sha256_file(second)


def test_unattainable_budget_raises_with_metadata_without_writing(tmp_path):
    page = tmp_path / "tiny.png"
    Image.new("RGB", (32, 32), "white").save(page)
    before = sha256_file(page)
    with pytest.raises(UnattainableVisualTokenBudget) as error:
        downscale_png_to_visual_token_ratio(
            page,
            target_numerator=1,
            target_denominator=2,
        )
    assert error.value.metadata["token_budget"] == 1
    assert error.value.metadata["minimum_visual_tokens"] == 2
    assert error.value.metadata["budget_satisfied"] is False
    assert sha256_file(page) == before


@pytest.mark.parametrize("size", [(0, 1), (1, -1), (True, 1), (1.0, 1)])
def test_invalid_dimensions_rejected(size):
    with pytest.raises(ValueError):
        estimate_visual_tokens(*size)


@pytest.mark.parametrize("condition", [{"model": "gpt-unknown"}, {"detail": "auto"}])
def test_unknown_conditions_do_not_inherit_luna_estimates(condition):
    with pytest.raises(ValueError):
        estimate_visual_tokens(1024, 1024, **condition)


@pytest.mark.parametrize("ratio", [(0, 1), (1, 0), (2, 1), (True, 2), (1.5, 2)])
def test_invalid_target_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        plan_visual_token_resize(
            1024,
            1024,
            target_numerator=ratio[0],
            target_denominator=ratio[1],
        )
