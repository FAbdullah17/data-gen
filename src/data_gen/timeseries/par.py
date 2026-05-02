"""PAR-based time-series synthesizer.

Wraps SDV's ``PARSynthesizer`` to learn deep generative models for
multi-sequence time-series data. Requires a sequence identifier
and can conditionally generate based on context columns.

Requires: ``pip install data-gen[timeseries]``
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_gen.core.base import BaseSynthesizer
from data_gen.timeseries.utils import (
    generate_parametric_sequences,
    require_sdv_timeseries,
)


class TimeSeriesSynthesizer(BaseSynthesizer):
    """Time-series synthesizer using PAR (Probabilistic AutoRegressive) via SDV.

    Learns patterns across multiple sequences of data. Requires a
    DataFrame where each sequence is identified by a ``sequence_key``.

    Parameters
    ----------
    epochs : int
        Number of training epochs. SDV default is 100.
    config : dict | None
        Additional configuration passed to BaseSynthesizer.

    Examples
    --------
    >>> synth = TimeSeriesSynthesizer(epochs=50)
    >>> synth.fit(df, sequence_key="sensor_id", context_columns=["city"])
    >>> synthetic = synth.generate(num_sequences=20)

    Parametric mode (no fitting required):

    >>> features = {"temperature": {"type": "float", "min": -10, "max": 40}}
    >>> synthetic = synth.generate(50, sequence_length=100, features=features)
    """

    _lifecycle = "fit"

    def __init__(
        self,
        epochs: int = 100,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        self.epochs = epochs
        self._synthesizer: Any = None
        self._metadata: Any = None
        self._sample_data: pd.DataFrame | None = None
        self.sequence_key: str = ""

    def fit(self, data: Any, **kwargs: Any) -> TimeSeriesSynthesizer:
        """Train the PAR model on a time-series DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Sample data containing multiple sequences. Works best with 3+ sequences.
        sequence_key : str
            The column name identifying individual sequences (e.g., "user_id").
            Required parameter.
        context_columns : list[str] | None
            Optional list of columns that do not change within a sequence
            (e.g., "country", "device_type").
        **kwargs : Any
            Extra keyword arguments passed to SDV's ``PARSynthesizer``.

        Returns
        -------
        TimeSeriesSynthesizer
            ``self``, for method chaining.

        Raises
        ------
        TypeError
            If *data* is not a pandas DataFrame.
        ValueError
            If *data* is empty or *sequence_key* is missing.
        """
        require_sdv_timeseries()

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")
        if data.empty:
            raise ValueError("Cannot fit on an empty DataFrame.")

        sequence_key = kwargs.pop("sequence_key", None)
        if not sequence_key:
            raise ValueError("sequence_key is required for TimeSeriesSynthesizer.")
        if sequence_key not in data.columns:
            raise ValueError(f"sequence_key '{sequence_key}' not found in data.")

        self.sequence_key = sequence_key
        context_columns = kwargs.pop("context_columns", [])

        from sdv.metadata import SingleTableMetadata
        from sdv.sequential import PARSynthesizer as SdvPar

        self._sample_data = data.copy()

        self._metadata = SingleTableMetadata()
        self._metadata.detect_from_dataframe(data)
        
        self._metadata.update_column(column_name=sequence_key, sdtype="id")
        self._metadata.set_sequence_key(column_name=sequence_key)

        self._synthesizer = SdvPar(
            metadata=self._metadata,
            epochs=self.epochs,
            context_columns=context_columns,
            **kwargs,
        )

        self._logger.info(
            "Fitting PAR on %d rows x %d columns for %d epochs...",
            len(data),
            len(data.columns),
            self.epochs,
        )
        self._synthesizer.fit(data)

        self.is_fitted = True
        self._logger.info("PAR fitting complete.")
        return self

    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        *,
        sequence_length: int | None = None,
        features: dict[str, dict[str, Any]] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate synthetic time-series data.

        Parameters
        ----------
        num_samples : int
            Number of sequences to generate (referred to as num_sequences).
        instructions : str | None
            Currently unused for PAR. Reserved for API consistency.
        sequence_length : int | None
            If provided alongside ``features``, generates parametrically
            without using the fitted model.
        features : dict | None
            If provided alongside ``sequence_length``, generates parametrically.
        seed : int | None
            Random seed for reproducibility.
        **kwargs : Any
            Extra arguments passed to SDV's ``sample()`` method.

        Returns
        -------
        pd.DataFrame
            Generated synthetic time-series sequences.

        Raises
        ------
        RuntimeError
            If neither fitted nor (sequence_length + features) provided.
        """
        if features is not None and sequence_length is not None:
            self._logger.info(
                "Generating %d parametric sequences of length %d.",
                num_samples,
                sequence_length,
            )
            return generate_parametric_sequences(
                num_sequences=num_samples,
                sequence_length=sequence_length,
                features=features,
                sequence_key=self.sequence_key or "sequence_id",
                seed=seed,
            )

        self._check_is_fitted()

        self._logger.info("Generating %d synthetic sequences via PAR...", num_samples)
        
        # In SDV PAR, num_sequences determines how many sequence groups to generate
        synthetic: pd.DataFrame = self._synthesizer.sample(
            num_sequences=num_samples, **kwargs
        )

        if instructions:
            self._logger.warning(
                "instructions are not applied to time-series sequences post-generation."
            )

        return synthetic

    def evaluate(self, real_data: Any, synthetic_data: Any) -> dict[str, Any]:
        """Evaluate synthetic time-series quality against original.

        Parameters
        ----------
        real_data : pd.DataFrame
            Original sample data.
        synthetic_data : pd.DataFrame
            Generated synthetic data.

        Returns
        -------
        dict[str, Any]
            Quality metrics from ``evaluate_timeseries()``.
        """
        from data_gen.timeseries.evaluation import evaluate_timeseries

        return evaluate_timeseries(
            real_data, 
            synthetic_data, 
            sequence_key=self.sequence_key,
            metadata=self._metadata
        )
