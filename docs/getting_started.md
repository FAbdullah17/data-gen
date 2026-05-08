# Getting Started

## Installation

`syntharc` is highly modular. To keep the package lightweight, heavy dependencies (like `torch`, `transformers`, or `sdv`) are optional based on your needs.

### 1. Core Installation
Installs the dependency-free core utilities and basic pattern generators (like Markov Chains).
```bash
pip install syntharc
```

### 2. Domain-Specific Installation
If you only need a specific domain, use the bracket notation:
```bash
pip install syntharc[tabular]     # Installs SDV
pip install syntharc[text]        # Installs Transformers & Torch
pip install syntharc[image]       # Installs Albumentations & OpenCV
pip install syntharc[timeseries]  # Installs SDV Time Series components
```

### 3. Full Installation
To install all synthesizers and modules:
```bash
pip install syntharc[all]
```

### 4. Development Installation
If you plan to contribute to the package, run tests, or build documentation, install the development dependencies:
```bash
pip install syntharc[dev]
```

## The Synthesizer Lifecycle

All synthesizers in `syntharc` inherit from `BaseSynthesizer` and follow a standard 3-step lifecycle API:

```python
# 1. Initialize
from syntharc.tabular.gaussian_copula import GaussianCopulaSynthesizer
synth = GaussianCopulaSynthesizer()

# 2. Fit to real data
synth.fit(real_pandas_dataframe)

# 3. Generate synthetic data
synthetic_dataframe = synth.generate(num_samples=1000)
```

## Global Configuration

You can ensure reproducibility across all underlying engines (NumPy, PyTorch, base Random) using the core utility:

```python
from syntharc.core.utils import set_seed

set_seed(42)
```
