"""Tests for MarkovTextGenerator."""

from __future__ import annotations

import pytest

from syntharc.text.markov import MarkovTextGenerator


class TestMarkovTextGeneratorInit:
    def test_default_config(self) -> None:
        gen = MarkovTextGenerator()
        assert gen.order == 2
        assert gen._lifecycle == "fit"
        assert gen.is_fitted is False

    def test_invalid_order(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            MarkovTextGenerator(order=0)


class TestMarkovTextGeneratorFit:
    def test_fit_string(self) -> None:
        gen = MarkovTextGenerator(order=1)
        text = "Hello world. Hello world again."
        gen.fit(text)
        assert gen.is_fitted
        assert len(gen._starts) > 0

    def test_fit_list_of_strings(self) -> None:
        gen = MarkovTextGenerator(order=1)
        text_list = ["Hello world.", "How are you?"]
        gen.fit(text_list)
        assert gen.is_fitted
        assert len(gen._chain) > 0

    def test_fit_empty_text(self) -> None:
        gen = MarkovTextGenerator()
        with pytest.raises(ValueError, match="empty text"):
            gen.fit("   ")

    def test_fit_text_too_short(self) -> None:
        gen = MarkovTextGenerator(order=3)
        with pytest.raises(ValueError, match="too short"):
            gen.fit("One two")


class TestMarkovTextGeneratorGenerate:
    def test_generate_without_fit_raises(self) -> None:
        gen = MarkovTextGenerator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            gen.generate(5)

    def test_generate_text(self) -> None:
        gen = MarkovTextGenerator(order=1)
        text = "The quick brown fox jumps over the lazy dog."
        gen.fit(text)
        generated = gen.generate(3, max_length=10, seed=42)
        assert len(generated) == 3
        assert all(isinstance(t, str) for t in generated)
        assert len(generated[0]) > 0
