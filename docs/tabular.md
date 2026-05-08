# Tabular Data Generation

The `syntharc.tabular` module provides statistical and deep learning models to clone your structural tabular data (like CSVs or SQL tables).

## Available Synthesizers

### 1. `GaussianCopulaSynthesizer`
Uses mathematical copulas to map the multi-variate distribution of your data.
*   **Backend:** SDV (Synthetic Data Vault)
*   **Pros:** Very fast, works well even with small datasets (≥ 10 rows), accurately captures column correlations.
*   **Cons:** Struggles with highly complex non-linear relationships compared to deep learning methods.

### 2. `CTGANSynthesizer`
Uses a Conditional Generative Adversarial Network designed specifically for tabular data.
*   **Backend:** SDV (CTGAN)
*   **Pros:** High quality generation for complex datasets, robust to mixed data types.
*   **Cons:** Requires more data (100+ rows minimum), significantly slower to train, requires tuning epochs and batch size.

## Usage Example

```python
from syntharc.tabular.gaussian_copula import GaussianCopulaSynthesizer
from syntharc.tabular.evaluation import evaluate_tabular
import pandas as pd

# Load data
real_df = pd.read_csv("my_data.csv")

# Train & Generate
synth = GaussianCopulaSynthesizer()
synth.fit(real_df)
synthetic_df = synth.generate(num_samples=500)

# Evaluate Quality
metrics = evaluate_tabular(real_df, synthetic_df)
print(metrics)
```
