"""syntharc — Unified synthetic data generation.

A lightweight Python package for synthetic data generation across
tabular, time-series, image, and text domains using sample-based
learning, augmentation, and lightweight generative techniques.

Quick Start
-----------
>>> from syntharc.core import BaseSynthesizer, set_seed, setup_logging

Tabular (requires ``pip install syntharc[tabular]``):

>>> from syntharc.tabular import CTGANSynthesizer  # doctest: +SKIP
>>> from syntharc.tabular import GaussianCopulaSynthesizer  # doctest: +SKIP

Time-series (requires ``pip install syntharc[timeseries]``):

>>> from syntharc.timeseries import TimeSeriesSynthesizer  # doctest: +SKIP

Image (requires ``pip install syntharc[image]``):

>>> from syntharc.image import ImageAugmentor  # doctest: +SKIP

Text (markov/template work out of the box, transformer needs
``pip install syntharc[text]``):

>>> from syntharc.text import MarkovTextGenerator  # doctest: +SKIP
>>> from syntharc.text import TemplateTextGenerator  # doctest: +SKIP
>>> from syntharc.text import TransformerTextGenerator  # doctest: +SKIP
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Fahad Abdullah"
__email__ = "fahadai.co@gmail.com"

__all__ = ["__version__"]
