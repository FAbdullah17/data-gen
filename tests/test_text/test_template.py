"""Tests for TemplateTextGenerator."""

from __future__ import annotations

import pytest

from syntharc.text.template import TemplateTextGenerator


class TestTemplateTextGeneratorInit:
    def test_default_config(self) -> None:
        gen = TemplateTextGenerator()
        assert gen._lifecycle == "fit"
        assert gen.is_fitted is False
        assert len(gen._vocab["noun"]) == 0


class TestTemplateTextGeneratorFit:
    def test_fit_extracts_vocab(self) -> None:
        gen = TemplateTextGenerator()
        text = "The beautiful application performed wonderfully."
        gen.fit(text)
        assert gen.is_fitted
        assert "beautiful" in gen._vocab["adjective"]
        assert "performed" in gen._vocab["verb"]

    def test_fit_list(self) -> None:
        gen = TemplateTextGenerator()
        gen.fit(["First string.", "Second string jumping."])
        assert gen.is_fitted
        assert "jumping" in gen._vocab["verb"]

    def test_fit_empty_raises(self) -> None:
        gen = TemplateTextGenerator()
        with pytest.raises(ValueError, match="empty text"):
            gen.fit("   ")


class TestTemplateTextGeneratorGenerate:
    def test_generate_without_fit_raises(self) -> None:
        gen = TemplateTextGenerator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            gen.generate(5)

    def test_generate_builtin_template(self) -> None:
        gen = TemplateTextGenerator()
        gen.fit("A test string.")
        generated = gen.generate(3, instructions="reviews", seed=42)
        assert len(generated) == 3
        assert all(isinstance(t, str) for t in generated)

    def test_generate_custom_template(self) -> None:
        gen = TemplateTextGenerator()
        gen.fit("Dog cat mouse.")
        templates = ["The {noun} is {adjective}."]
        generated = gen.generate(2, templates=templates, seed=42)
        assert len(generated) == 2
        assert " is " in generated[0]

    def test_generate_invalid_category_raises(self) -> None:
        gen = TemplateTextGenerator()
        gen.fit("Test.")
        with pytest.raises(ValueError, match="not found"):
            gen.generate(1, instructions="invalid_cat")
