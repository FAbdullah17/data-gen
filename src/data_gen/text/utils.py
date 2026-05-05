"""Shared utility functions for text synthesizers."""

from __future__ import annotations


def require_transformers() -> None:
    """Raise a clear error if transformers/torch are not installed."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for TransformerTextGenerator. "
            "Install them with: pip install data-gen[text]"
        ) from exc
