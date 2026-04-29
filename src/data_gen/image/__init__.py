"""data_gen.image — Image synthetic data generation."""

from __future__ import annotations

from data_gen.image.augmentor import ImageAugmentor


def __getattr__(name: str) -> object:
    """Lazy imports for optional torch dependency."""
    if name == "DCGANSynthesizer":
        from data_gen.image.dcgan import DCGANSynthesizer

        return DCGANSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ImageAugmentor", "DCGANSynthesizer"]
