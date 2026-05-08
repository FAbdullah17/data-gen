# Getting Started

## Installation

`data-gen` is highly modular. To keep the package lightweight, heavy dependencies (like `torch`, `transformers`, or `sdv`) are optional based on your needs.

### 1. Core Installation
Installs the dependency-free core utilities and basic pattern generators (like Markov Chains).
```bash
pip install data-gen
```

### 2. Domain-Specific Installation
If you only need a specific domain, use the bracket notation:
```bash
pip install data-gen[tabular]     # Installs SDV
pip install data-gen[text]        # Installs Transformers & Torch
pip install data-gen[image]       # Installs Albumentations & OpenCV
pip install data-gen[timeseries]  # Installs SDV Time Series components
```

### 3. Full Installation
To install all synthesizers and modules:
```bash
pip install data-gen[all]
```

### 4. Development Installation
If you plan to contribute to the package, run tests, or build documentation, install the development dependencies:
```bash
pip install data-gen[dev]
```

## The Synthesizer Lifecycle

All synthesizers in `data-gen` inherit from `BaseSynthesizer` and follow a standard 3-step lifecycle API:

```python
# 1. Initialize
from data_gen.tabular.gaussian_copula import GaussianCopulaSynthesizer
synth = GaussianCopulaSynthesizer()

# 2. Fit to real data
synth.fit(real_pandas_dataframe)

# 3. Generate synthetic data
synthetic_dataframe = synth.generate(num_samples=1000)
```

## Global Configuration

You can ensure reproducibility across all underlying engines (NumPy, PyTorch, base Random) using the core utility:

```python
from data_gen.core.utils import set_seed

set_seed(42)
```
