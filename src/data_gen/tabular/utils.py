"""Shared utility functions for tabular synthesizers.

Contains helpers used across CTGAN, GaussianCopula, and other tabular
modules: SDV availability check, instruction filtering, and
schema-based generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def require_sdv() -> None:
    """Raise a clear error if SDV is not installed."""
    try:
        import sdv  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SDV is required for tabular synthesis. "
            "Install it with: pip install data-gen[tabular]"
        ) from exc


def apply_instructions(
    df: pd.DataFrame,
    instructions: str,
    num_needed: int,
    synthesizer: Any,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Filter generated rows using pandas query and resample if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Initially generated synthetic data.
    instructions : str
        A pandas-compatible query string (e.g. ``"age > 18 and salary > 30000"``).
    num_needed : int
        Target number of rows after filtering.
    synthesizer : Any
        The SDV synthesizer to sample more rows from if filtering removes too many.
    max_retries : int
        Maximum resample attempts before returning whatever we have.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame, up to *num_needed* rows.
    """
    try:
        filtered = df.query(instructions)
    except Exception:
        return df

    retries = 0
    while len(filtered) < num_needed and retries < max_retries:
        extra = synthesizer.sample(num_rows=num_needed)
        try:
            extra_filtered = extra.query(instructions)
        except Exception:
            break
        filtered = pd.concat([filtered, extra_filtered], ignore_index=True)
        retries += 1

    return filtered.head(num_needed).reset_index(drop=True)


def generate_from_schema(
    num_samples: int,
    schema: dict[str, dict[str, Any]],
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate random data from a schema dict without fitting.

    Parameters
    ----------
    num_samples : int
        Number of rows to generate.
    schema : dict[str, dict[str, Any]]
        Column specifications. Supported types:
        - ``{"type": "int", "min": 0, "max": 100}``
        - ``{"type": "float", "min": 0.0, "max": 1.0}``
        - ``{"type": "category", "values": ["A", "B", "C"]}``
        - ``{"type": "bool"}``
        - ``{"type": "string", "prefix": "item"}``
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated DataFrame matching the schema.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {}

    for col_name, spec in schema.items():
        col_type = spec.get("type", "float")

        if col_type == "int":
            low = spec.get("min", 0)
            high = spec.get("max", 100)
            data[col_name] = rng.integers(low, high + 1, size=num_samples)

        elif col_type == "float":
            low = spec.get("min", 0.0)
            high = spec.get("max", 1.0)
            data[col_name] = rng.uniform(low, high, size=num_samples)

        elif col_type == "category":
            values = spec.get("values", ["A", "B", "C"])
            data[col_name] = rng.choice(values, size=num_samples)

        elif col_type == "bool":
            data[col_name] = rng.choice([True, False], size=num_samples)

        elif col_type == "string":
            prefix = spec.get("prefix", col_name)
            data[col_name] = [f"{prefix}_{i}" for i in range(num_samples)]

        else:
            data[col_name] = rng.standard_normal(num_samples)

    return pd.DataFrame(data)
