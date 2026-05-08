"""Integration tests for the tabular module.

Validates the full lifecycle: data preparation -> synthesis -> evaluation.
"""

import pandas as pd

from syntharc.tabular.evaluation import evaluate_tabular
from syntharc.tabular.gaussian_copula import GaussianCopulaSynthesizer


def test_tabular_lifecycle_integration() -> None:
    """Test full tabular generation and evaluation pipeline."""
    # 1. Create realistic dummy dataset
    real_data = pd.DataFrame(
        {
            "id": range(1, 101),
            "age": [20 + (i % 40) for i in range(100)],
            "income": [30000 + (i * 500) for i in range(100)],
            "department": ["IT" if i % 2 == 0 else "HR" for i in range(100)],
            "is_active": [1 if i % 3 == 0 else 0 for i in range(100)],
        }
    )

    metadata = {
        "columns": {
            "id": {"type": "id", "subtype": "integer"},
            "age": {"type": "numerical", "subtype": "integer"},
            "income": {"type": "numerical", "subtype": "integer"},
            "department": {"type": "categorical"},
            "is_active": {"type": "numerical", "subtype": "integer"},
        },
        "primary_key": "id",
    }

    # 2. Fit Synthesizer
    synth = GaussianCopulaSynthesizer()
    synth.fit(real_data)
    assert synth.is_fitted is True

    # 3. Generate Synthetic Data
    synthetic_data = synth.generate(num_samples=100)
    assert len(synthetic_data) == 100
    assert set(synthetic_data.columns) == set(real_data.columns)

    # 4. Evaluate
    metrics = evaluate_tabular(real_data, synthetic_data, metadata=metadata)

    assert "overall_score" in metrics
    assert "column_shapes" in metrics
    assert 0.0 <= metrics["overall_score"] <= 1.0
