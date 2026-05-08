# data-gen Documentation

Welcome to the official documentation for **data-gen**, a comprehensive Python unified framework for generating synthetic data across multiple domains.

## Overview

`data-gen` is designed to be a one-stop-shop for all your synthetic data and data augmentation needs. It abstracts away the complexity of using different machine learning models and libraries, providing a consistent API built upon a standard synthesizer lifecycle:

1. **`prepare()`**: Load resources, configure engines, or cache contextual data.
2. **`fit()`**: Learn patterns from your real-world data.
3. **`generate()`**: Synthesize new data mirroring the learned distributions.

## Supported Domains

The package natively supports four main types of data generation:

*   **[Tabular Data](tabular.md)**: Structured database or CSV rows.
*   **[Text Data](text.md)**: Natural language generation (Markov Chains & Transformers).
*   **[Image Data](image.md)**: High-performance image augmentations.
*   **[Time Series Data](timeseries.md)**: Sequential data modeling across multiple entities.

## Navigation

*   [Getting Started](getting_started.md)
*   [Tabular Synthesizers](tabular.md)
*   [Text Synthesizers](text.md)
*   [Image Augmentation](image.md)
*   [Time Series Generation](timeseries.md)
