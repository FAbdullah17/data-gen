"""Tests for CTGANSynthesizer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from syntharc.tabular.ctgan import CTGANSynthesizer
from syntharc.tabular.utils import generate_from_schema


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small sample DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=50),
            "salary": rng.uniform(30000, 150000, size=50).round(2),
            "department": rng.choice(["Engineering", "Sales", "HR"], size=50),
            "active": rng.choice([True, False], size=50),
        }
    )


class TestCTGANSynthesizerInit:
    """Tests for CTGANSynthesizer initialization."""

    def test_default_config(self) -> None:
        synth = CTGANSynthesizer()
        assert synth.epochs == 300
        assert synth.batch_size == 500
        assert synth._lifecycle == "fit"
        assert synth.is_fitted is False

    def test_custom_config(self) -> None:
        synth = CTGANSynthesizer(epochs=50, batch_size=64)
        assert synth.epochs == 50
        assert synth.batch_size == 64

    def test_repr(self) -> None:
        synth = CTGANSynthesizer()
        assert "CTGANSynthesizer" in repr(synth)
        assert "not initialized" in repr(synth)


class TestCTGANSynthesizerValidation:
    """Tests for input validation."""

    def test_fit_rejects_non_dataframe(self) -> None:
        synth = CTGANSynthesizer()
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            synth.fit([1, 2, 3])

    def test_fit_rejects_empty_dataframe(self) -> None:
        synth = CTGANSynthesizer()
        with pytest.raises(ValueError, match="empty DataFrame"):
            synth.fit(pd.DataFrame())

    def test_generate_without_fit_raises(self) -> None:
        synth = CTGANSynthesizer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            synth.generate(10)


class TestSchemaGeneration:
    """Tests for schema-based generation (no SDV needed)."""

    def test_int_column(self) -> None:
        schema = {"age": {"type": "int", "min": 18, "max": 65}}
        result = generate_from_schema(100, schema, seed=42)
        assert len(result) == 100
        assert "age" in result.columns
        assert result["age"].min() >= 18
        assert result["age"].max() <= 65

    def test_float_column(self) -> None:
        schema = {"price": {"type": "float", "min": 0.0, "max": 100.0}}
        result = generate_from_schema(50, schema, seed=42)
        assert len(result) == 50
        assert result["price"].min() >= 0.0
        assert result["price"].max() <= 100.0

    def test_category_column(self) -> None:
        values = ["A", "B", "C"]
        schema = {"grade": {"type": "category", "values": values}}
        result = generate_from_schema(200, schema, seed=42)
        assert set(result["grade"].unique()).issubset(set(values))

    def test_bool_column(self) -> None:
        schema = {"active": {"type": "bool"}}
        result = generate_from_schema(100, schema, seed=42)
        assert result["active"].dtype == bool

    def test_string_column(self) -> None:
        schema = {"name": {"type": "string", "prefix": "user"}}
        result = generate_from_schema(10, schema, seed=42)
        assert all(s.startswith("user_") for s in result["name"])

    def test_multi_column_schema(self) -> None:
        schema: dict[str, dict[str, Any]] = {
            "id": {"type": "int", "min": 1, "max": 9999},
            "score": {"type": "float", "min": 0, "max": 100},
            "status": {"type": "category", "values": ["pass", "fail"]},
        }
        result = generate_from_schema(200, schema, seed=42)
        assert list(result.columns) == ["id", "score", "status"]
        assert len(result) == 200

    def test_schema_via_generate(self) -> None:
        synth = CTGANSynthesizer()
        schema = {"x": {"type": "int", "min": 0, "max": 10}}
        result = synth.generate(50, schema=schema)
        assert len(result) == 50
        assert synth.is_fitted is False

    def test_reproducibility(self) -> None:
        schema = {"val": {"type": "float", "min": 0, "max": 1}}
        r1 = generate_from_schema(20, schema, seed=123)
        r2 = generate_from_schema(20, schema, seed=123)
        pd.testing.assert_frame_equal(r1, r2)


class TestProcessRouting:
    """Tests for process() routing to fit()."""

    def test_process_routes_to_fit(self, sample_df: pd.DataFrame) -> None:
        synth = CTGANSynthesizer()
        assert synth._lifecycle == "fit"
