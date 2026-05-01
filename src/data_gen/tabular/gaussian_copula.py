"""Gaussian Copula tabular synthesizer.

Wraps SDV's ``GaussianCopulaSynthesizer`` to learn statistical
distributions from sample DataFrames using copula-based modeling.
Faster than CTGAN and works well with small samples (~10 rows).

Requires: ``pip install data-gen[tabular]``
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_gen.core.base import BaseSynthesizer
from data_gen.tabular.utils import (
    apply_instructions,
    generate_from_schema,
    require_sdv,
)


class GaussianCopulaSynthesizer(BaseSynthesizer):
    """Tabular synthesizer using Gaussian Copula via SDV.

    Uses statistical copula modeling to capture correlations between
    columns. Faster than CTGAN (no neural network training) and
    effective with as few as 10 sample rows.

    Parameters
    ----------
    config : dict | None
        Additional configuration passed to BaseSynthesizer.

    Examples
    --------
    >>> synth = GaussianCopulaSynthesizer()
    >>> synth.fit(sample_df)
    >>> synthetic = synth.generate(500)

    Schema-only mode (no fitting required):

    >>> schema = {"age": {"type": "int", "min": 18, "max": 65}}
    >>> synthetic = synth.generate(500, schema=schema)
    """

    _lifecycle = "fit"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._synthesizer: Any = None
        self._metadata: Any = None
        self._sample_data: pd.DataFrame | None = None

    def fit(self, data: Any, **kwargs: Any) -> GaussianCopulaSynthesizer:
        """Fit the Gaussian Copula model on a sample DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Sample data to learn from. Works with as few as ~10 rows.
        **kwargs : Any
            Extra keyword arguments passed to SDV's ``GaussianCopulaSynthesizer``.

        Returns
        -------
        GaussianCopulaSynthesizer
            ``self``, for method chaining.

        Raises
        ------
        TypeError
            If *data* is not a pandas DataFrame.
        ValueError
            If *data* is empty.
        """
        require_sdv()

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")
        if data.empty:
            raise ValueError("Cannot fit on an empty DataFrame.")

        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import (
            GaussianCopulaSynthesizer as SdvGaussianCopula,
        )

        self._sample_data = data.copy()

        self._metadata = SingleTableMetadata()
        self._metadata.detect_from_dataframe(data)

        self._synthesizer = SdvGaussianCopula(
            metadata=self._metadata,
            **kwargs,
        )

        self._logger.info(
            "Fitting GaussianCopula on %d rows x %d columns...",
            len(data),
            len(data.columns),
        )
        self._synthesizer.fit(data)

        self.is_fitted = True
        self._logger.info("GaussianCopula fitting complete.")
        return self

    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        *,
        schema: dict[str, dict[str, Any]] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate synthetic tabular data.

        Parameters
        ----------
        num_samples : int
            Number of rows to generate.
        instructions : str | None
            Optional pandas query string to filter output
            (e.g. ``"age > 18 and salary > 30000"``).
        schema : dict | None
            If provided, generates from schema without fitting.
        seed : int | None
            Random seed for reproducibility.
        **kwargs : Any
            Extra arguments passed to SDV's ``sample()`` method.

        Returns
        -------
        pd.DataFrame
            Generated synthetic data.

        Raises
        ------
        RuntimeError
            If neither fitted nor schema provided.
        """
        if schema is not None:
            self._logger.info(
                "Generating %d rows from schema (%d columns).",
                num_samples,
                len(schema),
            )
            return generate_from_schema(num_samples, schema, seed=seed)

        self._check_is_fitted()

        self._logger.info(
            "Generating %d synthetic rows via GaussianCopula...", num_samples
        )
        synthetic: pd.DataFrame = self._synthesizer.sample(
            num_rows=num_samples, **kwargs
        )

        if instructions:
            self._logger.info("Applying instructions: %s", instructions)
            synthetic = apply_instructions(
                synthetic, instructions, num_samples, self._synthesizer
            )

        return synthetic

    def evaluate(self, real_data: Any, synthetic_data: Any) -> dict[str, Any]:
        """Evaluate synthetic data quality against original.

        Parameters
        ----------
        real_data : pd.DataFrame
            Original sample data.
        synthetic_data : pd.DataFrame
            Generated synthetic data.

        Returns
        -------
        dict[str, Any]
            Quality metrics from ``evaluate_tabular()``.
        """
        from data_gen.tabular.evaluation import evaluate_tabular

        return evaluate_tabular(real_data, synthetic_data, metadata=self._metadata)
