"""syntharc.timeseries — Time-series synthetic data generation."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Lazy imports for optional SDV dependency."""
    if name == "TimeSeriesSynthesizer":
        from syntharc.timeseries.par import TimeSeriesSynthesizer

        return TimeSeriesSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TimeSeriesSynthesizer"]
