# syntharc: The Grand Unified Synthetic Data Generator

[![CI](https://img.shields.io/github/actions/workflow/status/FAbdullah17/syntharc/ci.yml?branch=main)](https://github.com/FAbdullah17/syntharc/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/syntharc.svg)](https://pypi.org/project/syntharc/)
[![Python Versions](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://pypi.org/project/syntharc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`syntharc`** is a powerful, flexible, and unified Python library for generating high-quality synthetic data across multiple domains. In the modern machine learning ecosystem, acquiring high-quality, privacy-compliant, and diverse datasets is often the biggest bottleneck. `syntharc` solves this by abstracting the complexity of various generative algorithms into a single, intuitive framework.

Whether you need to anonymize sensitive tabular records, augment image operations for computer vision, simulate realistic text sequences, or replicate complex financial time-series signals. `syntharc` provides the unified API to do it all using state-of-the-art backend engines like PyTorch, Hugging Face Transformers, Albumentations, and the Synthetic Data Vault (SDV).

---

## How It Works: The Unified Lifecycle

One of the biggest hurdles in synthetic data generation is the fragmented tooling across different data types. `syntharc` forces all underlying models into a highly predictable **3-step lifecycle** inherited from our `BaseSynthesizer` architecture:

1. **`prepare(**kwargs)`**: Configures the internal environment. This is where you define structural metadata, column types, sequence keys, or image transformation pipelines to guide the generation.
2. **`fit(data)`**: Feeds your actual, real-world dataset into the generative model so it can learn patterns, probabilistic distributions, and internal representations.
3. **`generate(**kwargs)`**: Samples from the trained model to yield your brand new synthetic dataset—retaining the statistical properties of the original without exposing real user information.

---

## Key Features

*   **Tabular Data:** Synthesize fully relational datasets using Deep Learning (`CTGANSynthesizer`) or statistical modeling (`GaussianCopulaSynthesizer`).
*   **Text Generation:** Leverage the power of LLMs (`TransformerSynthesizer`), classic statistical chains (`MarkovSynthesizer`), or strict ruleings (`TemplateSynthesizer`).
*   **Image Augmentation:** Use `ImageAugmentor` to rapidly iterate through spatial and pixel-level augmentations, dynamically expanding computer vision datasets.
*   **Time-Series:** Employ the `PARSynthesizer` (Probabilistic AutoRegressive model) to safely synthesize robust sequence data over time.
*   **Evaluations:** Built-in tools for evaluating the quality, fidelity, and privacy metrics of your generated data against your source data.

---

## Installation

Install `syntharc` via pip. The base package provides the core infrastructure. We highly recommend installing the specific domain dependencies you intend to use to keep your environment lean:

```bash
# Install everything (Recommended for the full experience)
pip install "syntharc[all]"

# Or individually pick your domains:
pip install "syntharc[tabular]"
pip install "syntharc[text]"
pip install "syntharc[image]"
pip install "syntharc[timeseries]"

# For contributors and local development:
pip install "syntharc[dev]"
```

---

## Comprehensive Quick Start

Below are examples of how our unified framework elegantly handles vastly different data constraints.

### 1. Privacy-Preserving Tabular Data
Train a CTGAN model to learn the distribution of your customer data without saving real identities.
```python
import pandas as pd
from syntharc.tabular.ctgan import CTGANSynthesizer

real_data = pd.read_csv("customer_data.csv")

# 1. Initialize & Prepare metadata
synth = CTGANSynthesizer(epochs=50)
synth.prepare(metadata_dict={"primary_key": "user_id"})

# 2. Fit to real data
synth.fit(real_data)

# 3. Generate 1,000 synthetic rows!
synthetic_data = synth.generate(num_rows=1000)
print(synthetic_data.head())
```

### 2. Deep Time-Series Generation
Synthesize sequential metrics, like stock prices or IoT sensor readings, using AutoRegressive modeling.
```python
from syntharc.timeseries.par import PARSynthesizer

# 1. Initialize & Prepare
synth = PARSynthesizer(epochs=25)
# Define what makes a "sequence" (e.g., separate tracking per 'device_id')
synth.prepare(metadata_dict={
    "sequence_key": "device_id",
    "context_columns": ["region"]
})

# 2. Fit
synth.fit(sensor_dataframe)

# 3. Generate sequential data
synthetic_series = synth.generate(num_sequences=50)
```

### 3. Causal Text Generation (LLMs)
Easily utilize Hugging Face causal language models.
```python
from syntharc.text.transformer import TransformerTextGenerator

# 1. Initialize (Downloads SmolLM2-360M-Instruct by default)
synth = TransformerTextGenerator()

# 2. Prepare with an optional style context
synth.prepare(corpus="The future of synthetic data is bright and highly scalable.")

# 3. Generate structured text from instructions
text_output = synth.generate(num_samples=1, instructions="Write a short summary about data.")
print(text_output[0])
```

### 4. High-Speed Image Augmentation
Prepare a pipeline of augmentations to expand your Machine Learning dataset effortlessly.
```python
from syntharc.image.augmentor import ImageAugmentor
import cv2

# 1. Initialize rules
augmentor = ImageAugmentor()
# Uses Albumentations backend dictionary standards mapping
augmentor.prepare(config={
    'resize': (256, 256),
    'horizontal_flip': 0.5,
    'brightness_contrast': 0.2
})

# 2. Load Real Image
image = cv2.imread('dataset/cat.jpg')

# 3. Generate augmentations
# The generator seamlessly wraps Albumentations under the hood
augmented_image = augmentor.generate(num_samples=1, data=[image])
print(f"Augmented Shape: {augmented_image[0].shape}")
```

---

## Documentation

For comprehensive guides, parameter references, evaluations, and interactive Jupyter showcase notebooks, please check out the `docs/` folder:
- [Getting Started](docs/getting_started.md)
- [Tabular Generation Guide](docs/tabular.md)
- [Time-Series Generation Guide](docs/timeseries.md)
- [Text Generation Guide](docs/text.md)
- [Image Augmentation Guide](docs/image.md)
- [API Reference](docs/api_reference.md)

---

## Contributing

We love our contributors! If you're interested in adding a new generative model, fixing a bug, or improving the documentation, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions on setting up your local environment, managing dependencies, and passing our CI checks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by the syntharc team.
