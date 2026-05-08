"""Tests for TimeSeriesSynthesizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from syntharc.timeseries.par import TimeSeriesSynthesizer
from syntharc.timeseries.utils import generate_parametric_sequences


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Create a small sample time-series DataFrame for testing."""
    rng = np.random.default_rng(42)
    # 3 sequences of length 5
    num_seqs = 3
    seq_len = 5
    return pd.DataFrame(
        {
            "sensor_id": np.repeat(np.arange(num_seqs), seq_len),
            "city": np.repeat(["NY", "SF", "LA"], seq_len),
            "temperature": rng.uniform(20, 80, size=num_seqs * seq_len),
            "pressure": rng.uniform(1000, 1020, size=num_seqs * seq_len),
        }
    )


class TestTimeSeriesSynthesizerInit:
    """Tests for TimeSeriesSynthesizer initialization."""

    def test_default_config(self) -> None:
        synth = TimeSeriesSynthesizer()
        assert synth.epochs == 100
        assert synth._lifecycle == "fit"
        assert synth.is_fitted is False

    def test_custom_config(self) -> None:
        synth = TimeSeriesSynthesizer(epochs=50)
        assert synth.epochs == 50

    def test_repr(self) -> None:
        synth = TimeSeriesSynthesizer()
        assert "TimeSeriesSynthesizer" in repr(synth)


class TestTimeSeriesSynthesizerValidation:
    """Tests for input validation."""

    def test_fit_rejects_non_dataframe(self) -> None:
        synth = TimeSeriesSynthesizer()
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            synth.fit({"a": [1, 2, 3]}, sequence_key="id")

    def test_fit_rejects_empty_dataframe(self) -> None:
        synth = TimeSeriesSynthesizer()
        with pytest.raises(ValueError, match="empty DataFrame"):
            synth.fit(pd.DataFrame(), sequence_key="id")

    def test_fit_requires_sequence_key(self, sample_timeseries_df: pd.DataFrame) -> None:
        synth = TimeSeriesSynthesizer()
        with pytest.raises(ValueError, match="sequence_key is required"):
            synth.fit(sample_timeseries_df)

    def test_fit_rejects_missing_sequence_key(self, sample_timeseries_df: pd.DataFrame) -> None:
        synth = TimeSeriesSynthesizer()
        with pytest.raises(ValueError, match="not found in data"):
            synth.fit(sample_timeseries_df, sequence_key="missing_id")

    def test_generate_without_fit_raises(self) -> None:
        synth = TimeSeriesSynthesizer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            synth.generate(10)


class TestParametricGeneration:
    """Tests for parametric sequence generation (no SDV needed)."""

    def test_parametric_int(self) -> None:
        features = {"value": {"type": "int", "min": 0, "max": 10}}
        result = generate_parametric_sequences(
            num_sequences=2, sequence_length=5, features=features, sequence_key="seq_id"
        )
        assert len(result) == 10
        assert "seq_id" in result.columns
        assert "value" in result.columns
        assert result["seq_id"].nunique() == 2

    def test_parametric_float(self) -> None:
        features = {"temp": {"type": "float", "min": -5.0, "max": 5.0}}
        result = generate_parametric_sequences(
            num_sequences=3, sequence_length=4, features=features
        )
        assert len(result) == 12
        assert result["temp"].min() >= -5.0
        assert result["temp"].max() <= 5.0

    def test_parametric_category(self) -> None:
        features = {"status": {"type": "category", "values": ["A", "B"]}}
        result = generate_parametric_sequences(
            num_sequences=2, sequence_length=2, features=features
        )
        assert set(result["status"].unique()).issubset({"A", "B"})

    def test_parametric_via_generate(self) -> None:
        synth = TimeSeriesSynthesizer()
        features = {"x": {"type": "float", "min": 0.0, "max": 1.0}}
        result = synth.generate(num_samples=2, sequence_length=3, features=features)
        assert len(result) == 6
        assert synth.is_fitted is False

    def test_parametric_validation_missing_type(self) -> None:
        features = {"x": {"min": 0, "max": 10}}
        with pytest.raises(ValueError, match="missing 'type'"):
            generate_parametric_sequences(1, 5, features)

    def test_parametric_validation_missing_min_max(self) -> None:
        features = {"x": {"type": "int", "min": 0}}
        with pytest.raises(ValueError, match="requires 'min' and 'max'"):
            generate_parametric_sequences(1, 5, features)

    def test_parametric_validation_missing_values(self) -> None:
        features = {"x": {"type": "category"}}
        with pytest.raises(ValueError, match="requires 'values' list"):
            generate_parametric_sequences(1, 5, features)

    def test_parametric_validation_min_greater_than_max(self) -> None:
        features = {"x": {"type": "float", "min": 10.0, "max": 5.0}}
        with pytest.raises(ValueError, match="min > max"):
            generate_parametric_sequences(1, 5, features)


class TestProcessRouting:
    """Tests for process() routing."""

    def test_lifecycle_is_fit(self) -> None:
        synth = TimeSeriesSynthesizer()
        assert synth._lifecycle == "fit"
