"""syntharc.text — Text synthetic data generation."""

from __future__ import annotations

from syntharc.text.markov import MarkovTextGenerator
from syntharc.text.template import TemplateTextGenerator


def __getattr__(name: str) -> object:
    """Lazy imports for optional transformers dependency."""
    if name == "TransformerTextGenerator":
        from syntharc.text.transformer import TransformerTextGenerator

        return TransformerTextGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MarkovTextGenerator",
    "TemplateTextGenerator",
    "TransformerTextGenerator",
]
