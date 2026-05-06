"""Tests for Image evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from data_gen.image.evaluation import _calculate_ssim_proxy, _get_pixel_stats, evaluate_images


@pytest.fixture
def black_img() -> Image.Image:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def white_img() -> Image.Image:
    arr = np.ones((32, 32, 3), dtype=np.uint8) * 255
    return Image.fromarray(arr)


@pytest.fixture
def random_img() -> Image.Image:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestImageEvaluation:
    def test_get_pixel_stats(self, black_img: Image.Image, white_img: Image.Image) -> None:
        stats_b = _get_pixel_stats([black_img])
        assert stats_b["mean"] == 0.0
        assert stats_b["std"] == 0.0
        
        stats_w = _get_pixel_stats([white_img])
        assert stats_w["mean"] == 1.0
        assert stats_w["std"] == 0.0

    def test_get_pixel_stats_empty(self) -> None:
        stats = _get_pixel_stats([])
        assert stats["mean"] == 0.0

    def test_ssim_proxy_identical(self, random_img: Image.Image) -> None:
        score = _calculate_ssim_proxy(random_img, random_img)
        assert score >= 0.99  # Should be close to 1

    def test_ssim_proxy_different(self, black_img: Image.Image, white_img: Image.Image) -> None:
        score = _calculate_ssim_proxy(black_img, white_img)
        assert score < 0.1  # Should be very low

    def test_evaluate_images_identical(self, random_img: Image.Image) -> None:
        # Two identical lists
        res = evaluate_images([random_img, random_img], [random_img, random_img])
        assert res["pixel_stats_similarity"] >= 0.99
        assert res["ssim_score"] >= 0.99
        assert res["diversity_score"] < 0.05  # No diversity since they are identical

    def test_evaluate_images_empty(self) -> None:
        res = evaluate_images([], [])
        assert res["overall_score"] == 0.0
