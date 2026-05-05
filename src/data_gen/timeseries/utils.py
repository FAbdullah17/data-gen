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

    data: dict[str, Any] = {sequence_key: np.repeat(np.arange(num_sequences), sequence_length)}

    for col_name, spec in features.items():
        if "type" not in spec:
            raise ValueError(
                f"Feature '{col_name}' missing 'type'. "
                f'Example: {{"type": "float", "min": 0.0, "max": 1.0}}'
            )
        col_type = spec["type"]

        if col_type in ("int", "float"):
            if "min" not in spec or "max" not in spec:
                raise ValueError(
                    f"Feature '{col_name}' of type '{col_type}' requires 'min' and 'max'. "
                    f'Example: {{"type": "{col_type}", "min": 0, "max": 100}}'
                )
            low = spec["min"]
            high = spec["max"]
            if low > high:
                raise ValueError(f"Feature '{col_name}' has min > max.")

            if col_type == "int":
                data[col_name] = rng.integers(int(low), int(high) + 1, size=total_rows)
            else:
                data[col_name] = rng.uniform(float(low), float(high), size=total_rows)

        elif col_type == "category":
            if "values" not in spec or not spec["values"]:
                raise ValueError(
                    f"Feature '{col_name}' of type 'category' requires 'values' list. "
                    f'Example: {{"type": "category", "values": ["A", "B"]}}'
                )
            data[col_name] = rng.choice(spec["values"], size=total_rows)

        elif col_type == "bool":
            data[col_name] = rng.choice([True, False], size=total_rows)

        else:
            raise ValueError(
                f"Unsupported type '{col_type}' for feature '{col_name}'. "
                "Supported types: 'int', 'float', 'category', 'bool'."
            )

    return pd.DataFrame(data)
