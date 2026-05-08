"""Tests for GaussianCopulaSynthesizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from syntharc.tabular.gaussian_copula import GaussianCopulaSynthesizer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small sample DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=30),
            "salary": rng.uniform(30000, 150000, size=30).round(2),
            "department": rng.choice(["Engineering", "Sales", "HR"], size=30),
        }
    )


class TestGaussianCopulaInit:
    """Tests for GaussianCopulaSynthesizer initialization."""

    def test_default_config(self) -> None:
        synth = GaussianCopulaSynthesizer()
        assert synth._lifecycle == "fit"
        assert synth.is_fitted is False

    def test_repr(self) -> None:
        synth = GaussianCopulaSynthesizer()
        assert "GaussianCopulaSynthesizer" in repr(synth)


class TestGaussianCopulaValidation:
    """Tests for input validation."""

    def test_fit_rejects_non_dataframe(self) -> None:
        synth = GaussianCopulaSynthesizer()
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            synth.fit({"a": [1, 2, 3]})

    def test_fit_rejects_empty_dataframe(self) -> None:
        synth = GaussianCopulaSynthesizer()
        with pytest.raises(ValueError, match="empty DataFrame"):
            synth.fit(pd.DataFrame())

    def test_generate_without_fit_raises(self) -> None:
        synth = GaussianCopulaSynthesizer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            synth.generate(10)

    def test_schema_generation(self) -> None:
        synth = GaussianCopulaSynthesizer()
        schema = {"x": {"type": "float", "min": 0, "max": 1}}
        result = synth.generate(50, schema=schema)
        assert len(result) == 50
        assert synth.is_fitted is False


class TestProcessRouting:
    """Tests for process() routing."""

    def test_lifecycle_is_fit(self) -> None:
        synth = GaussianCopulaSynthesizer()
        assert synth._lifecycle == "fit"
