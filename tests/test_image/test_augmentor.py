"""Tests for ImageAugmentor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_gen.image.augmentor import ImageAugmentor


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple 64x64 solid color image."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # Red image
    return Image.fromarray(arr)


class TestImageAugmentorInit:
    def test_default_config(self) -> None:
        aug = ImageAugmentor()
        assert aug.intensity == "medium"
        assert aug._lifecycle == "prepare"
        assert aug.is_prepared is False

    def test_invalid_intensity(self) -> None:
        with pytest.raises(ValueError, match="intensity must be"):
            ImageAugmentor(intensity="invalid")


class TestImageAugmentorPrepare:
    def test_prepare_list_of_images(self, sample_image: Image.Image) -> None:
        aug = ImageAugmentor()
        aug.prepare([sample_image, sample_image])
        assert aug.is_prepared
        assert len(aug._images) == 2

    def test_prepare_empty_list(self) -> None:
        aug = ImageAugmentor()
        with pytest.raises(ValueError, match="No valid images found"):
            aug.prepare([])

    def test_prepare_invalid_type(self) -> None:
        aug = ImageAugmentor()
        with pytest.raises(TypeError, match="Unsupported data type"):
            aug.prepare({"image": "invalid"})

    def test_prepare_directory(self, tmp_path: Path, sample_image: Image.Image) -> None:
        img_path = tmp_path / "test.png"
        sample_image.save(img_path)
        
        aug = ImageAugmentor()
        aug.prepare(tmp_path)
        assert aug.is_prepared
        assert len(aug._images) == 1


class TestImageAugmentorGenerate:
    def test_generate_without_prepare_raises(self) -> None:
        aug = ImageAugmentor()
        with pytest.raises(RuntimeError, match="must be prepared"):
            aug.generate(5)

    def test_generate_success(self, sample_image: Image.Image) -> None:
        aug = ImageAugmentor(intensity="light")
        aug.prepare([sample_image])
        results = aug.generate(3, seed=42)
        
        assert len(results) == 3
        assert isinstance(results[0], Image.Image)
        # Check image sizes remain same
        assert results[0].size == sample_image.size


class TestImageAugmentorOps:
    def test_augment_specific_ops(self, sample_image: Image.Image) -> None:
        aug = ImageAugmentor()
        # Flip, rotate, and color
        results = aug.augment([sample_image], ops=["flip", "rotate", "color"], seed=42)
        assert len(results) == 1
        assert isinstance(results[0], Image.Image)

    def test_augment_unsupported_op(self, sample_image: Image.Image) -> None:
        aug = ImageAugmentor()
        # Should log warning and return image unchanged
        results = aug.augment([sample_image], ops=["invalid_op"])
        assert len(results) == 1
        assert np.array_equal(np.array(results[0]), np.array(sample_image))


class TestImageAugmentorSave:
    def test_save_images(self, tmp_path: Path, sample_image: Image.Image) -> None:
        aug = ImageAugmentor()
        aug.save_images([sample_image, sample_image], tmp_path)
        
        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) == 2
