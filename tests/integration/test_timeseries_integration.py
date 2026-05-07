"""Integration tests for the timeseries module.

Validates the full lifecycle: data preparation -> synthesis -> evaluation.
"""

import numpy as np
import pandas as pd

from data_gen.timeseries.evaluation import evaluate_timeseries
from data_gen.timeseries.par import TimeSeriesSynthesizer


def test_timeseries_lifecycle_integration() -> None:
    """Test full timeseries generation and evaluation pipeline."""
    # 1. Create a dummy timeseries dataset
    np.random.seed(42)
    sequences = []
    for seq_id in range(5):
        base_val = np.random.normal(10, 2)
        for t in range(20):
            sequences.append(
                {
                    "sequence_id": seq_id,
                    "timestep": t,
                    "value": base_val + np.sin(t / 2.0) + np.random.normal(0, 0.1),
                    "category": "A" if seq_id % 2 == 0 else "B",
                }
            )

    real_data = pd.DataFrame(sequences)

    # 2. Fit Synthesizer
    synth = TimeSeriesSynthesizer()
    # Adding a context column for SDV PAR compatibility
    synth.fit(real_data, sequence_key="sequence_id", context_columns=["category"])
    assert synth.is_fitted is True

    # 3. Generate Synthetic Data
    synthetic_data = synth.generate(num_samples=5)
    assert len(synthetic_data) > 0
    assert "sequence_id" in synthetic_data.columns

    # 4. Evaluate
    metrics = evaluate_timeseries(real_data, synthetic_data, sequence_key="sequence_id")

    assert "overall_score" in metrics
    assert "mean_std_similarity" in metrics
    assert "autocorr_similarity" in metrics
    assert 0.0 <= metrics["overall_score"] <= 1.0
