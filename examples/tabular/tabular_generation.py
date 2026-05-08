"""Tabular Data Generation Example.

This script demonstrates how to generate synthetic tabular data
using the `data-gen` package, specifically learning from a real-world
Excel dataset.
"""

from pathlib import Path

import pandas as pd

from data_gen.tabular.evaluation import evaluate_tabular
from data_gen.tabular.gaussian_copula import GaussianCopulaSynthesizer


def main() -> None:
    print("=== Tabular Data Generation Example ===")

    # 1. Load the real-world dataset
    data_path = Path(__file__).parent / "data.xlsx"
    print(f"Loading data from {data_path}...")
    real_df = pd.read_excel(data_path)

    print(f"Loaded {len(real_df)} rows. Columns: {list(real_df.columns)}")

    # Optional: Provide explicit metadata for better accuracy,
    # though GaussianCopulaSynthesizer can auto-detect it.
    _metadata = {
        "columns": {
            "approach": {"type": "categorical"},
            "depth": {"type": "numerical", "subtype": "integer"},
            "seed": {"type": "numerical", "subtype": "integer"},
            "test_acc": {"type": "numerical", "subtype": "float"},
            "test_loss": {"type": "numerical", "subtype": "float"},
            "final_val_acc": {"type": "numerical", "subtype": "float"},
            "training_time": {"type": "numerical", "subtype": "float"},
            "barren_plateau": {"type": "categorical"},  # Boolean/categorical
        }
    }

    # 2. Initialize and fit the synthesizer
    print("\nFitting GaussianCopulaSynthesizer...")
    synth = GaussianCopulaSynthesizer()
    synth.fit(real_df)

    # 3. Generate synthetic data
    num_samples = 500
    print(f"\nGenerating {num_samples} synthetic rows...")
    synthetic_df = synth.generate(num_samples=num_samples)

    # 4. Save the synthetic data
    output_path = Path(__file__).parent / "synthetic_data.csv"
    synthetic_df.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")

    # 5. Evaluate the generated data
    print("\nEvaluating synthetic data quality...")
    metrics = evaluate_tabular(real_df, synthetic_df)

    print(f"Overall Quality Score: {metrics['overall_score']:.2%}")
    print("Column Shape Similarities:")
    for col, score in metrics["column_shapes"].items():
        print(f"  - {col}: {score:.2%}")


if __name__ == "__main__":
    main()
