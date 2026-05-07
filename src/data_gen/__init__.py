"""data-gen — Unified synthetic data generation.

A lightweight Python package for synthetic data generation across
tabular, time-series, image, and text domains using sample-based
learning, augmentation, and lightweight generative techniques.

Quick Start
-----------
>>> from data_gen.core import BaseSynthesizer, set_seed, setup_logging

Tabular (requires ``pip install data-gen[tabular]``):

>>> from data_gen.tabular import CTGANSynthesizer  # doctest: +SKIP
>>> from data_gen.tabular import GaussianCopulaSynthesizer  # doctest: +SKIP

Time-series (requires ``pip install data-gen[timeseries]``):

>>> from data_gen.timeseries import TimeSeriesSynthesizer  # doctest: +SKIP

Image (requires ``pip install data-gen[image]``):

>>> from data_gen.image import ImageAugmentor  # doctest: +SKIP

Text (markov/template work out of the box, transformer needs
``pip install data-gen[text]``):

>>> from data_gen.text import MarkovTextGenerator  # doctest: +SKIP
>>> from data_gen.text import TemplateTextGenerator  # doctest: +SKIP
>>> from data_gen.text import TransformerTextGenerator  # doctest: +SKIP
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Fahad Abdullah"
__email__ = "fahadai.co@gmail.com"

__all__ = ["__version__"]
