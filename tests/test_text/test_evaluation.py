"""Tests for text evaluation metrics."""

from __future__ import annotations

from data_gen.text.evaluation import _tokenize, evaluate_text


class TestTokenization:
    def test_tokenize_string(self) -> None:
        tokens = _tokenize("Hello, World! 123.")
        assert tokens == ["hello", "world", "123"]

    def test_tokenize_list(self) -> None:
        tokens = _tokenize(["Hello", "World"])
        assert tokens == ["hello", "world"]

    def test_tokenize_empty(self) -> None:
        assert _tokenize("") == []


class TestEvaluateText:
    def test_identical_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        result = evaluate_text(text, text)
        assert result["vocab_overlap"] == 1.0
        assert result["length_similarity"] == 1.0
        assert result["overall_score"] == 1.0

    def test_different_text(self) -> None:
        real = "A completely different sentence."
        synth = "Another completely weird thing."
        result = evaluate_text(real, synth)
        assert result["vocab_overlap"] < 1.0
        assert result["vocab_overlap"] > 0.0

    def test_empty_text(self) -> None:
        result = evaluate_text("", "")
        assert result["overall_score"] == 0.0

    def test_list_input(self) -> None:
        real = ["First sentence.", "Second sentence."]
        synth = ["First sentence.", "Third sentence."]
        result = evaluate_text(real, synth)
        assert result["vocab_overlap"] >= 0.5
        assert result["length_similarity"] > 0.8
