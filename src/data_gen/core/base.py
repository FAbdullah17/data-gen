"""Base synthesizer abstraction for all data-gen modules.

Provides the ``BaseSynthesizer`` ABC that every generator inherits from.
The API is split into three lifecycle methods:

* ``prepare()`` — load / preprocess / cache resources (no learning).
* ``fit()``     — learn / train / estimate parameters from sample data.
* ``generate()`` — produce *N* synthetic samples (abstract, always required).

"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseSynthesizer(ABC):
    """Abstract base class for all data-gen synthesizers.

    Parameters
    ----------
    config : dict | None
        Optional configuration dictionary for the synthesizer.
        Keys and values are module-specific.
    """

    _lifecycle: str = ""
    """Subclasses set this to ``"fit"`` or ``"prepare"`` to declare
    which lifecycle method they use. ``process()`` reads this directly."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.is_fitted: bool = False
        self.is_prepared: bool = False
        self._logger: logging.Logger = logging.getLogger(f"data_gen.{self.__class__.__name__}")

    # Lifecycle methods

    def process(self, data: Any, **kwargs: Any) -> BaseSynthesizer:
        """Primary entry point that routes data to the correct lifecycle method.

        Reads the ``_lifecycle`` class attribute to determine whether to
        call ``fit()`` or ``prepare()``. Each subclass declares its
        lifecycle type explicitly.

        Parameters
        ----------
        data : Any
            Sample data, file paths, corpus, or any input the module needs.
        **kwargs : Any
            Passed through to the resolved lifecycle method.

        Returns
        -------
        BaseSynthesizer
            ``self``, for method chaining.

        Raises
        ------
        NotImplementedError
            If ``_lifecycle`` is not set on the subclass.

        Examples
        --------
        >>> synth = CTGANSynthesizer()
        >>> synth.process(sample_df)          # routes to fit()
        >>> aug = ImageAugmentor()
        >>> aug.process("./images/")          # routes to prepare()
        """
        if self._lifecycle == "fit":
            return self.fit(data, **kwargs)
        if self._lifecycle == "prepare":
            return self.prepare(data, **kwargs)

        raise NotImplementedError(
            f"{self.__class__.__name__} must set _lifecycle to 'fit' or "
            f"'prepare' to use process()."
        )

    def prepare(self, data: Any, **kwargs: Any) -> BaseSynthesizer:
        """Load, preprocess, cache, or set up resources.

        Subclasses override this when no learning occurs — only loading,
        caching, or preprocessing (e.g. ImageAugmentor, TransformerTextGenerator).

        Parameters
        ----------
        data : Any
            Resource to prepare (paths, text, images, etc.).
        **kwargs : Any
            Module-specific preparation options.

        Returns
        -------
        BaseSynthesizer
            ``self``, for method chaining.

        Raises
        ------
        NotImplementedError
            If the subclass does not support ``prepare()``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support prepare(). "
            f"This module uses fit() to learn from sample data."
        )

    def fit(self, data: Any, **kwargs: Any) -> BaseSynthesizer:
        """Learn, train, or estimate parameters from sample data.

        Subclasses override this when genuine learning occurs — training
        neural networks, building transition tables, estimating distributions
        (e.g. CTGANSynthesizer, MarkovTextGenerator, DCGANSynthesizer).

        Parameters
        ----------
        data : Any
            Sample data to learn from (DataFrame, text, image paths, etc.).
        **kwargs : Any
            Module-specific training options.

        Returns
        -------
        BaseSynthesizer
            ``self``, for method chaining.

        Raises
        ------
        NotImplementedError
            If the subclass does not support ``fit()``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support fit(). "
            f"This module uses prepare() to load and cache resources."
        )

    @abstractmethod
    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate *num_samples* synthetic samples.

        This is the only **required** method that every subclass must
        implement.

        Parameters
        ----------
        num_samples : int
            Number of synthetic samples to produce.
        instructions : str | None
            Optional natural-language instructions that guide generation
            (e.g. ``"ensure age > 18"``, ``"formal tone"``).
        **kwargs : Any
            Module-specific generation options.

        Returns
        -------
        Any
            Generated data in module-appropriate format
            (DataFrame, list[str], list[Image], etc.).
        """

    def evaluate(self, real_data: Any, synthetic_data: Any) -> dict[str, Any]:
        """Compare real vs. synthetic data quality.

        Override in subclasses to return domain-specific metrics.
        The default implementation returns an empty dict.

        Parameters
        ----------
        real_data : Any
            Original / reference data.
        synthetic_data : Any
            Data produced by ``generate()``.

        Returns
        -------
        dict[str, Any]
            Evaluation metrics.
        """
        return {}

    # Serialization

    def save(self, path: str | Path) -> None:
        """Serialize the synthesizer state to disk.

        The default implementation uses ``pickle``.  Subclasses may
        override this for model-specific serialization (e.g.
        ``torch.save``).

        Parameters
        ----------
        path : str | Path
            Destination file path.

        Raises
        ------
        RuntimeError
            If the synthesizer has not been fitted or prepared.
        """
        self._check_ready()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        self._logger.info("Saved %s to %s", self.__class__.__name__, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseSynthesizer:
        """Load a previously saved synthesizer from disk.

        Parameters
        ----------
        path : str | Path
            Path to the saved file.

        Returns
        -------
        BaseSynthesizer
            The restored synthesizer instance.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved synthesizer found at {path}")
        with open(path, "rb") as fh:
            instance = pickle.load(fh)
        if not isinstance(instance, cls):
            raise TypeError(
                f"Loaded object is {type(instance).__name__}, " f"expected {cls.__name__}"
            )
        return instance

    # State guards

    def _check_is_fitted(self) -> None:
        """Raise if the synthesizer has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted first. "
                f"Call fit(data) before generating."
            )

    def _check_is_prepared(self) -> None:
        """Raise if the synthesizer has not been prepared."""
        if not self.is_prepared:
            raise RuntimeError(
                f"{self.__class__.__name__} must be prepared first. "
                f"Call prepare(data) before generating."
            )

    def _check_ready(self) -> None:
        """Raise if the synthesizer is neither fitted nor prepared."""
        if not self.is_fitted and not self.is_prepared:
            raise RuntimeError(
                f"{self.__class__.__name__} is not ready. "
                f"Call fit(data) or prepare(data) first."
            )

    def __repr__(self) -> str:
        status = (
            "fitted" if self.is_fitted else "prepared" if self.is_prepared else "not initialized"
        )
        config_str = ", ".join(f"{k}={v!r}" for k, v in self.config.items()) if self.config else ""
        return (
            f"{self.__class__.__name__}("
            f"status={status}"
            f"{', ' + config_str if config_str else ''}"
            f")"
        )
