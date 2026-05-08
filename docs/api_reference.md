# API Reference

The `data-gen` architecture relies heavily on a unified interface abstraction. All functional synthesizers inherit from `BaseSynthesizer`.

## BaseSynthesizer (`data_gen.core.base`)

`BaseSynthesizer(config: dict | None = None)`

An abstract base class that enforces method chaining and the three-stage lifecycle. Base implementations automatically handle standard logging, config tracking, and boolean state checks (`self.is_prepared`, `self.is_fitted`).

### Core Methods

*   **`prepare(data=None, **kwargs) -> self`**:
    Handles hardware setups, model downloading, template extraction, and data caching.
    *Must be called before `generate()` for models inheriting the `_lifecycle = "prepare"` class attribute.*

*   **`fit(data, **kwargs) -> self`**:
    Consumes pandas dataframes or python lists to adapt statistical curves, markov probabilities, or ML weights.
    *Must be called before `generate()` for models inheriting the `_lifecycle = "fit"` class attribute.*

*   **`generate(num_samples: int, **kwargs) -> Any`**:
    Executes the generation loop using the prior prepared/fitted state. Parameter arguments vary by synthesizer type (e.g. `seed`, `max_length`, `instructions`).

*   **`process(data, num_samples, **kwargs)`**:
    Convenience method that wraps all three. Calls `prepare()`, `fit()`, and returns the exact output of `generate()`.

## Evaluation Utilities

Every `data_gen` module contains an `evaluation.py` file exposing functions to score the synthetic distributions:
*   `data_gen.tabular.evaluation.evaluate_tabular(real, synth)`
*   `data_gen.image.evaluation.evaluate_images(real, synth)`
*   `data_gen.text.evaluation.evaluate_text(real, synth)`
*   `data_gen.timeseries.evaluation.evaluate_timeseries(real, synth)`

These functions generally return a dictionary of numeric scores denoting distance, similarity bounds, or mathematical overlaps.
