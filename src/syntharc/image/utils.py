"""Shared utility functions for image synthesizers."""

from __future__ import annotations


def require_albumentations() -> None:
    """Ensure albumentations and opencv are installed."""
    try:
        import albumentations  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "albumentations and opencv-python are required for ImageAugmentor. "
            "Install them with: pip install syntharc[image]"
        ) from exc
