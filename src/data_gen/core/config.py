"""Configuration loading and validation utilities.

Provides helpers for reading YAML config files and validating that
required keys are present before a synthesizer runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load configuration from a YAML file path or pass through a dict.

    Parameters
    ----------
    source : str | Path | dict
        Either a path to a ``.yaml`` / ``.yml`` file, or an existing
        configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If *source* is a path that does not exist.
    TypeError
        If *source* is neither a path-like nor a dict.
    ValueError
        If the YAML file does not parse to a dictionary.

    Examples
    --------
    >>> cfg = load_config({"epochs": 50, "batch_size": 64})
    >>> cfg["epochs"]
    50

    >>> cfg = load_config("config.yaml")  # doctest: +SKIP
    """
    # Already a dict — return as-is.
    if isinstance(source, dict):
        return source

    # Treat as file path.
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Config file must be .yaml or .yml, got: {path.suffix!r}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML file to contain a mapping (dict), " f"got {type(data).__name__}"
        )
    return data


def validate_config(
    config: dict[str, Any],
    required_keys: list[str],
    context: str = "",
) -> None:
    """Validate that all *required_keys* are present in *config*.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to validate.
    required_keys : list[str]
        Keys that **must** be present.
    context : str
        Optional label (e.g. class name) for clearer error messages.

    Raises
    ------
    ValueError
        If any required key is missing.

    Examples
    --------
    >>> validate_config({"a": 1, "b": 2}, ["a", "b"])
    >>> validate_config({"a": 1}, ["a", "b"], context="MyModule")
    Traceback (most recent call last):
        ...
    ValueError: MyModule config missing required keys: {'b'}
    """
    missing = set(required_keys) - set(config)
    if missing:
        prefix = f"{context} config" if context else "Config"
        raise ValueError(f"{prefix} missing required keys: {missing}")
