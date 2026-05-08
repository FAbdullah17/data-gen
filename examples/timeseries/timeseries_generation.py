"""Time-Series Data Generation Example.

This script demonstrates how to generate synthetic chronological
data using the `syntharc` package, learning from a real-world CSV dataset.
"""

from pathlib import Path

import pandas as pd

from syntharc.timeseries.evaluation import evaluate_timeseries
from syntharc.timeseries.par import TimeSeriesSynthesizer


def main() -> None:
    print("=== Time-Series Data Generation Example ===")

    # 1. Load the real-world dataset
    data_path = Path(__file__).parent / "data.csv"
    print(f"Loading data from {data_path}...")
    real_df = pd.read_csv(data_path)

    print(f"Loaded {len(real_df)} rows. Columns: {list(real_df.columns)}")

    # Clean up dates for the synthesizer to process them correctly
    real_df["date"] = pd.to_datetime(real_df["date"])

    # Sort by sequence and time
    real_df = real_df.sort_values(by=["store", "date"]).reset_index(drop=True)

    # 2. Initialize and fit the synthesizer
    # Since a 'store' has multiple 'departments', the unique sequence is
    # actually a specific department within a specific store.
    # We will create a composite key for this.
    real_df["store_dept"] = real_df["store"].astype(str) + "_" + real_df["department"].astype(str)

    print("\nFitting TimeSeriesSynthesizer (PAR model)...")
    synth = TimeSeriesSynthesizer()

    # Since fitting PAR on a huge dataset can take time, let's use a sample of
    # the first 5 unique store_dept combinations.
    sequences_to_keep = real_df["store_dept"].unique()[:5]
    sample_df = real_df[real_df["store_dept"].isin(sequences_to_keep)].copy()

    print(f"Using a subset of {len(sample_df)} rows (5 store_depts) for faster training.")

    synth.fit(sample_df, sequence_key="store_dept", context_columns=["type"])

    # 3. Generate synthetic data
    num_samples = 3  # Generate 3 new synthetic store sequences
    print(f"\nGenerating {num_samples} synthetic sequences...")
    synthetic_df = synth.generate(num_samples=num_samples)

    # 4. Save the synthetic data
    output_path = Path(__file__).parent / "synthetic_timeseries.csv"
    synthetic_df.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")

    # 5. Evaluate the generated data
    print("\nEvaluating synthetic time-series quality...")
    metrics = evaluate_timeseries(sample_df, synthetic_df, sequence_key="store_dept")

    print(f"Overall Quality Score: {metrics['overall_score']:.2%}")
    print("Mean/Std Similarity:")
    for col, score in metrics["mean_std_similarity"].items():
        print(f"  - {col}: {score:.2%}")


if __name__ == "__main__":
    main()
