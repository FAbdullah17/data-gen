"""Tests for tabular evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_gen.tabular.evaluation import (
    _column_shape_score,
    _correlation_similarity,
    _cs_test,
    _ks_complement,
    evaluate_tabular,
)


@pytest.fixture
def real_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age": rng.integers(18, 65, size=100),
        "salary": rng.uniform(30000, 150000, size=100),
        "dept": rng.choice(["A", "B", "C"], size=100),
    })


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "age": rng.integers(18, 65, size=100),
        "salary": rng.uniform(30000, 150000, size=100),
        "dept": rng.choice(["A", "B", "C"], size=100),
    })


class TestKSComplement:
    def test_identical_columns(self) -> None:
        col = pd.Series([1, 2, 3, 4, 5])
        assert _ks_complement(col, col) == 1.0

    def test_different_columns(self) -> None:
        a = pd.Series(np.arange(100))
        b = pd.Series(np.arange(100, 200))
        score = _ks_complement(a, b)
        assert 0.0 <= score <= 1.0
        assert score < 0.5


class TestCSTest:
    def test_identical_categories(self) -> None:
        col = pd.Series(["A", "B", "C", "A", "B", "C"])
        assert _cs_test(col, col) == 1.0

    def test_different_categories(self) -> None:
        a = pd.Series(["A"] * 50 + ["B"] * 50)
        b = pd.Series(["A"] * 10 + ["B"] * 90)
        score = _cs_test(a, b)
        assert 0.0 <= score <= 1.0
        assert score < 1.0


class TestColumnShape:
    def test_numeric_identical(self) -> None:
        col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _column_shape_score(col, col) == 1.0

    def test_categorical(self) -> None:
        col = pd.Series(["X", "Y", "Z", "X", "Y"])
        score = _column_shape_score(col, col)
        assert score == 1.0


class TestCorrelationSimilarity:
    def test_identical(self, real_df: pd.DataFrame) -> None:
        score = _correlation_similarity(real_df, real_df)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_single_numeric_col(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert _correlation_similarity(df, df) == 1.0


class TestEvaluateTabular:
    def test_returns_all_keys(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> None:
        result = evaluate_tabular(real_df, synthetic_df)
        assert "ks_complement" in result
        assert "cs_test" in result
        assert "column_shapes" in result
        assert "correlation_similarity" in result
        assert "overall_score" in result

    def test_scores_in_range(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> None:
        result = evaluate_tabular(real_df, synthetic_df)
        assert 0.0 <= result["overall_score"] <= 1.0
        assert 0.0 <= result["correlation_similarity"] <= 1.0

    def test_numeric_cols_in_ks(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> None:
        result = evaluate_tabular(real_df, synthetic_df)
        assert "age" in result["ks_complement"]
        assert "salary" in result["ks_complement"]

    def test_categorical_cols_in_cs(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> None:
        result = evaluate_tabular(real_df, synthetic_df)
        assert "dept" in result["cs_test"]

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            evaluate_tabular([1, 2], pd.DataFrame({"a": [1]}))

    def test_no_common_columns(self) -> None:
        a = pd.DataFrame({"x": [1]})
        b = pd.DataFrame({"y": [1]})
        with pytest.raises(ValueError, match="No common columns"):
            evaluate_tabular(a, b)
