"""Tests for timeseries evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from syntharc.timeseries.evaluation import (
    _autocorrelation,
    _mean_std_similarity,
    _sequence_autocorrelation,
    evaluate_timeseries,
)


@pytest.fixture
def real_ts_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "seq_id": np.repeat([0, 1], 10),
            "value1": rng.uniform(0, 10, size=20),
            "value2": rng.normal(5, 2, size=20),
        }
    )


@pytest.fixture
def synth_ts_df() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "seq_id": np.repeat([0, 1], 10),
            "value1": rng.uniform(0, 10, size=20),
            "value2": rng.normal(5, 2, size=20),
        }
    )


class TestAutocorrelation:
    def test_autocorrelation_constant(self) -> None:
        series = pd.Series([1.0, 1.0, 1.0, 1.0])
        ac = _autocorrelation(series)
        assert ac == 0.0 or np.isnan(ac) or pd.isna(ac)

    def test_autocorrelation_alternating(self) -> None:
        series = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        ac = _autocorrelation(series)
        assert ac < 0

    def test_sequence_autocorrelation(self, real_ts_df: pd.DataFrame) -> None:
        ac = _sequence_autocorrelation(real_ts_df, "seq_id", "value1")
        assert isinstance(ac, float)


class TestMeanStdSimilarity:
    def test_identical(self) -> None:
        col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _mean_std_similarity(col, col) == 1.0

    def test_different(self) -> None:
        a = pd.Series([1.0, 1.0, 1.0])
        b = pd.Series([10.0, 10.0, 10.0])
        score = _mean_std_similarity(a, b)
        assert score < 1.0


class TestEvaluateTimeseries:
    def test_returns_all_keys(self, real_ts_df: pd.DataFrame, synth_ts_df: pd.DataFrame) -> None:
        result = evaluate_timeseries(real_ts_df, synth_ts_df, sequence_key="seq_id")
        assert "mean_std_similarity" in result
        assert "autocorr_similarity" in result
        assert "overall_score" in result

    def test_scores_in_range(self, real_ts_df: pd.DataFrame, synth_ts_df: pd.DataFrame) -> None:
        result = evaluate_timeseries(real_ts_df, synth_ts_df, sequence_key="seq_id")
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            evaluate_timeseries([1, 2], pd.DataFrame(), sequence_key="id")  # type: ignore[arg-type]

    def test_missing_sequence_key(
        self, real_ts_df: pd.DataFrame, synth_ts_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="missing from data"):
            evaluate_timeseries(real_ts_df, synth_ts_df, sequence_key="wrong_key")

    def test_no_common_numeric_columns(self) -> None:
        a = pd.DataFrame({"id": [1, 1], "x": ["A", "B"]})
        b = pd.DataFrame({"id": [1, 1], "y": ["C", "D"]})
        with pytest.raises(ValueError, match="No common numeric columns"):
            evaluate_timeseries(a, b, sequence_key="id")
