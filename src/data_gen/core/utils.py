"""Shared utility functions for the data-gen package.

Provides device detection, reproducibility helpers, and logging setup.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all relevant libraries.

    Sets seeds for Python's ``random``, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The seed value. Must be a non-negative integer.

    Raises
    ------
    ValueError
        If *seed* is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO") -> None:
    """Configure rich-formatted logging for data-gen.

    Uses the ``rich`` library for coloured, structured log output.
    Falls back to basic ``logging`` config if ``rich`` is unavailable
    (shouldn't happen since it's a core dependency).

    Parameters
    ----------
    level : str
        Logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).

    Raises
    ------
    ValueError
        If *level* is not a valid logging level name.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    try:
        from rich.logging import RichHandler

        logging.basicConfig(
            level=numeric_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)],
            force=True,
        )
    except ImportError:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            force=True,
        )

    logging.getLogger("data_gen").setLevel(numeric_level)
