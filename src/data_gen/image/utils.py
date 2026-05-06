"""Shared utility functions for image synthesizers."""

from __future__ import annotations


def require_torch_vision() -> None:
    """Raise a clear error if torch/torchvision are not installed."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torch and torchvision are required for DCGANSynthesizer. "
            "Install them with: pip install data-gen[image]"
        ) from exc
