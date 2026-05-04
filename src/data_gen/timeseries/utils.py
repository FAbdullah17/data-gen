"""Shared utility functions for time-series synthesizers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def require_sdv_timeseries() -> None:
    """Raise a clear error if SDV is not installed."""
    try:
        import sdv  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SDV is required for time-series synthesis. "
            "Install it with: pip install data-gen[timeseries]"
        ) from exc


def generate_parametric_sequences(
    num_sequences: int,
    sequence_length: int,
    features: dict[str, dict[str, Any]],
    sequence_key: str = "sequence_id",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate random time-series sequences based on parametric distributions.

    Parameters
    ----------
    num_sequences : int
        Number of sequences to generate.
    sequence_length : int
        Length of each sequence.
    features : dict[str, dict[str, Any]]
        Dictionary specifying the features for each timestep.
        Example: ``{"value": {"type": "float", "min": 0, "max": 100},
        "state": {"type": "category", "values": ["A", "B"]}}``
    sequence_key : str
        Column name for the sequence identifier.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated parametric sequences.
    """
    rng = np.random.default_rng(seed)
    total_rows = num_sequences * sequence_length

    data: dict[str, Any] = {
        sequence_key: np.repeat(np.arange(num_sequences), sequence_length)
    }

    for col_name, spec in features.items():
        col_type = spec.get("type", "float")

        if col_type == "int":
            low = spec.get("min", 0)
            high = spec.get("max", 100)
            data[col_name] = rng.integers(low, high + 1, size=total_rows)

        elif col_type == "float":
            low = spec.get("min", 0.0)
            high = spec.get("max", 1.0)
            data[col_name] = rng.uniform(low, high, size=total_rows)

        elif col_type == "category":
            values = spec.get("values", ["A", "B", "C"])
            data[col_name] = rng.choice(values, size=total_rows)

        elif col_type == "bool":
            data[col_name] = rng.choice([True, False], size=total_rows)

        else:
            data[col_name] = rng.standard_normal(total_rows)

    return pd.DataFrame(data)
