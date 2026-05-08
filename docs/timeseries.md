# Time Series Generation

The `syntharc.timeseries` module produces synthetic sequential and temporal data. It models relationships both chronologically across time increments and structurally across different columns.

## `TimeSeriesSynthesizer`

The current implementation utilizes a **Probabilistic AutoRegressive (PAR)** model via the SDV backend.

### Key Concepts

Time series generation usually contains multiple distinct tracks of time. For example, a single dataset might contain the history of `User_A` and `User_B` simultaneously.

1.  **`sequence_key`**: The column name that separates these distinct entities (e.g., `user_id` or `device_id`).
2.  **`context_columns`**: Columns that remain static for a specific sequence (e.g., a device's `hardware_manufacturing_model`).

### Usage Example

```python
from syntharc.timeseries.par import TimeSeriesSynthesizer
import pandas as pd

# Load longitudinal sequences
# Expected columns: ['user_id', 'date', 'heart_rate', 'blood_pressure']
real_ts_df = pd.read_csv("patient_history.csv", parse_dates=['date'])

# Initialize the synthesizer
ts_synth = TimeSeriesSynthesizer()

# Fit requires telling the engine how to group historical tracks together
ts_synth.fit(
    real_ts_df,
    sequence_key='user_id',
    context_columns=[]
)

# Generate new sequences mimicking chronological rhythms
synthetic_ts_df = ts_synth.generate(num_samples=5)
```
