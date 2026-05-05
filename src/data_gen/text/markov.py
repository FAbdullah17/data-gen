"""Markov-chain based text synthesizer.

Uses n-gram probabilistic transitions to generate synthetic text
based on a given sample text. Zero external dependencies.
"""

from __future__ import annotations

import random
import re
from typing import Any

from data_gen.core.base import BaseSynthesizer


class MarkovTextGenerator(BaseSynthesizer):
    """Text synthesizer using Markov chains.

    Learns n-gram transition probabilities from a sample text
    and generates new text by walking the chain.

    Parameters
    ----------
    order : int
        The n-gram order (state size). A higher order yields text
        closer to the original, while a lower order is more random.
        Default is 2.
    config : dict | None
        Additional configuration passed to BaseSynthesizer.

    Examples
    --------
    >>> gen = MarkovTextGenerator(order=2)
    >>> gen.fit("The quick brown fox jumps over the lazy dog. The dog barked loudly.")
    >>> texts = gen.generate(5, max_length=100)
    """

    _lifecycle = "fit"

    def __init__(
        self,
        order: int = 2,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        if order < 1:
            raise ValueError("Markov chain order must be >= 1.")
        self.order = order
        self._chain: dict[tuple[str, ...], list[str]] = {}
        self._starts: list[tuple[str, ...]] = []

    def fit(self, data: Any, **kwargs: Any) -> MarkovTextGenerator:
        """Build the Markov chain from sample text.

        Parameters
        ----------
        data : str | list[str]
            Sample text to learn from. Can be a single string or a list of strings.
        **kwargs : Any
            Additional unused keyword arguments.

        Returns
        -------
        MarkovTextGenerator
            ``self``, for method chaining.

        Raises
        ------
        TypeError
            If *data* is not a string or list of strings.
        ValueError
            If *data* is empty or contains insufficient tokens for the requested order.
        """
        if isinstance(data, list):
            data = " ".join(str(item) for item in data)
        elif not isinstance(data, str):
            raise TypeError(f"Expected str or list[str], got {type(data).__name__}")

        if not data.strip():
            raise ValueError("Cannot fit on empty text.")

        # Simple tokenization: split by whitespace but keep basic punctuation attached to words.
        # For a more refined chain, one could use nltk/spacy, but we keep zero deps.
        tokens = [t for t in re.split(r"(\s+)", data) if t.strip()]

        if len(tokens) <= self.order:
            raise ValueError(
                f"Text too short ({len(tokens)} tokens) for order {self.order}."
            )

        self._chain.clear()
        self._starts.clear()

        # Build chain
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            next_token = tokens[i + self.order]
            
            if state not in self._chain:
                self._chain[state] = []
            self._chain[state].append(next_token)

            # A state is a valid start if it's the very beginning,
            # or if the word immediately preceding it ended with a sentence terminator.
            if i == 0 or tokens[i - 1].endswith((".", "!", "?")):
                self._starts.append(state)

        # Fallback if no natural starts found (e.g. no punctuation)
        if not self._starts:
            self._starts = list(self._chain.keys())

        self.is_fitted = True
        self._logger.info(
            "Markov chain (order %d) built with %d states.",
            self.order,
            len(self._chain),
        )
        return self

    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        *,
        max_length: int = 100,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate synthetic text sequences.

        Parameters
        ----------
        num_samples : int
            Number of text sequences to generate.
        instructions : str | None
            Unused for Markov chains. Reserved for API consistency.
        max_length : int
            Maximum number of tokens per generated sequence.
        seed : int | None
            Random seed for reproducibility.
        **kwargs : Any
            Additional unused keyword arguments.

        Returns
        -------
        list[str]
            Generated text strings.

        Raises
        ------
        RuntimeError
            If not fitted.
        """
        self._check_is_fitted()
        rng = random.Random(seed)

        if instructions:
            self._logger.warning("instructions are ignored by MarkovTextGenerator.")

        generated = []
        for _ in range(num_samples):
            # Pick a random starting state
            current_state = rng.choice(self._starts)
            output_tokens = list(current_state)

            while len(output_tokens) < max_length:
                if current_state not in self._chain:
                    break  # Dead end

                possible_next = self._chain[current_state]
                next_token = rng.choice(possible_next)
                output_tokens.append(next_token)

                # Stop if we hit a sentence-ending token and have at least some words
                # To prevent single-word sentences, we ensure length is > order + some margin
                if len(output_tokens) > self.order + 3 and next_token.endswith((".", "!", "?")):
                    # We probabilistically stop to allow for multi-sentence outputs
                    if rng.random() < 0.5:
                        break

                current_state = tuple(output_tokens[-self.order :])

            generated.append(" ".join(output_tokens))

        self._logger.info("Generated %d sequences.", num_samples)
        return generated

    def evaluate(self, real_data: Any, synthetic_data: Any) -> dict[str, Any]:
        """Evaluate synthetic text quality against original.

        Parameters
        ----------
        real_data : str | list[str]
            Original sample text.
        synthetic_data : list[str]
            Generated synthetic text.

        Returns
        -------
        dict[str, Any]
            Quality metrics from ``evaluate_text()``.
        """
        from data_gen.text.evaluation import evaluate_text

        return evaluate_text(real_data, synthetic_data)
