"""Integration tests for the image module.

Validates the full lifecycle: data preparation -> synthesis -> evaluation.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_gen.image.augmentor import ImageAugmentor
from data_gen.image.evaluation import evaluate_images


def test_image_lifecycle_augmentor(tmp_path: Path) -> None:
    """Test full image generation and evaluation pipeline with Augmentor."""
    # 1. Provide corpus of dummy images
    real_images = []
    rng = np.random.default_rng(42)
    for _ in range(5):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        real_images.append(Image.fromarray(arr))

    # 2. Prepare Synthesizer
    aug = ImageAugmentor(intensity="light")
    aug.prepare(real_images)
    assert aug.is_prepared is True

    # 3. Generate Synthetic Data
    synthetic_images = aug.generate(num_samples=10, seed=42)
    assert len(synthetic_images) == 10

    # Test saving
    aug.save_images(synthetic_images, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 10

    # 4. Evaluate
    metrics = evaluate_images(real_images, synthetic_images)
    
    assert "overall_score" in metrics
    assert "pixel_stats_similarity" in metrics
    assert "ssim_score" in metrics
    assert "diversity_score" in metrics
    assert 0.0 <= metrics["overall_score"] <= 1.0
