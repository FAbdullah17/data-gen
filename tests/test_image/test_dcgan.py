"""Tests for DCGANSynthesizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_gen.image.dcgan import DCGANSynthesizer

# Mock torch and torchvision
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
mock_torch.float32 = "float32"

mock_nn = MagicMock()
mock_torch.nn = mock_nn

mock_torchvision = MagicMock()
mock_transforms = MagicMock()
mock_datasets = MagicMock()
mock_torchvision.transforms = mock_transforms
mock_torchvision.datasets = mock_datasets


class TestDCGANSynthesizerInit:
    def test_default_config(self) -> None:
        with patch.dict("sys.modules", {"torch": mock_torch, "torchvision": mock_torchvision}):
            gan = DCGANSynthesizer()
            assert gan.image_size == 64
            assert gan.latent_dim == 100
            assert gan._lifecycle == "fit"
            assert gan.is_fitted is False
            assert gan.device == "cpu"


class TestDCGANSynthesizerFit:
    @patch("data_gen.image.utils.require_torch_vision")
    def test_fit_invalid_path(self, mock_require: MagicMock) -> None:
        with patch.dict("sys.modules", {"torch": mock_torch, "torchvision": mock_torchvision}):
            gan = DCGANSynthesizer()
            with pytest.raises(ValueError, match="must be a directory"):
                gan.fit("non_existent_dir_path_12345")


class TestDCGANSynthesizerGenerate:
    def test_generate_without_fit_raises(self) -> None:
        with patch.dict("sys.modules", {"torch": mock_torch, "torchvision": mock_torchvision}):
            gan = DCGANSynthesizer()
            with pytest.raises(RuntimeError, match="must be fitted"):
                gan.generate(5)
