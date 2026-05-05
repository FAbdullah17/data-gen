"""Tests for TransformerTextGenerator."""

from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch

from data_gen.text.transformer import TransformerTextGenerator


# Create mock modules
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
mock_torch.float32 = "float32"

mock_pipeline_func = MagicMock()
mock_transformers = MagicMock()
mock_transformers.pipeline = mock_pipeline_func


class TestTransformerTextGeneratorInit:
    def test_default_config(self) -> None:
        gen = TransformerTextGenerator()
        assert gen._lifecycle == "prepare"
        assert gen.is_prepared is False
        assert gen.model_name == "HuggingFaceTB/SmolLM2-360M-Instruct"


class TestTransformerTextGeneratorPrepare:
    @patch("data_gen.text.utils.require_transformers")
    def test_prepare_without_data(self, mock_require: MagicMock) -> None:
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            gen = TransformerTextGenerator(device="cpu")
            gen.prepare()
            
            assert gen.is_prepared
            assert gen._context is None
            mock_pipeline_func.assert_called()

    @patch("data_gen.text.utils.require_transformers")
    def test_prepare_with_data(self, mock_require: MagicMock) -> None:
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            gen = TransformerTextGenerator(device="cpu")
            gen.prepare(data="Style context.")
            
            assert gen.is_prepared
            assert gen._context == "Style context."


class TestTransformerTextGeneratorGenerate:
    def test_generate_without_prepare_raises(self) -> None:
        gen = TransformerTextGenerator()
        with pytest.raises(RuntimeError, match="must be prepared"):
            gen.generate(5)

    @patch("data_gen.text.utils.require_transformers")
    def test_generate_success(self, mock_require: MagicMock) -> None:
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            # Setup mock pipeline to return a fixed string
            mock_pipe_instance = MagicMock()
            mock_pipe_instance.return_value = [{"generated_text": "Mock output"}]
            mock_pipeline_func.return_value = mock_pipe_instance
            
            gen = TransformerTextGenerator(device="cpu")
            gen.prepare()
            
            # Reset call count before generate
            mock_pipe_instance.reset_mock()
            
            generated = gen.generate(2, instructions="Test", seed=42)
            assert len(generated) == 2
            assert generated[0] == "Mock output"
            assert mock_pipe_instance.call_count == 2
