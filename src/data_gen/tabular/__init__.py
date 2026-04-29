"""data_gen.tabular — Tabular synthetic data generation."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Lazy imports for optional SDV dependency."""
    if name == "CTGANSynthesizer":
        from data_gen.tabular.ctgan import CTGANSynthesizer

        return CTGANSynthesizer
    if name == "GaussianCopulaSynthesizer":
        from data_gen.tabular.gaussian_copula import GaussianCopulaSynthesizer

        return GaussianCopulaSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CTGANSynthesizer", "GaussianCopulaSynthesizer"]
