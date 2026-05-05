"""Template-based text synthesizer.

Extracts simple vocabulary from sample text and fills templates to
generate synthetic text. Zero external dependencies.
"""

from __future__ import annotations

import random
import re
from typing import Any

from data_gen.core.base import BaseSynthesizer

BUILTIN_TEMPLATES = {
    "reviews": [
        "The {adjective} {noun} {verb} my expectations.",
        "I was very {adjective} with this {noun}.",
        "This {noun} is {adjective} and {verb} perfectly.",
        "A truly {adjective} {noun} that {verb} well.",
    ],
    "sentences": [
        "The {noun} {verb} {adverb}.",
        "A {adjective} {noun} {verb} over the {noun}.",
        "Why did the {noun} {verb}?",
        "{noun} is {adjective}.",
    ],
    "records": [
        "User {noun} performed action {verb} with status {adjective}.",
        "Record: {noun} | State: {adjective} | Event: {verb}.",
        "System {verb} a {adjective} {noun}.",
    ],
}


class TemplateTextGenerator(BaseSynthesizer):
    """Text synthesizer using templates and extracted vocabulary.

    Learns basic vocabulary (nouns, verbs, adjectives based on simple rules)
    from sample text and injects them into predefined or custom templates.

    Parameters
    ----------
    config : dict | None
        Additional configuration passed to BaseSynthesizer.

    Examples
    --------
    >>> gen = TemplateTextGenerator()
    >>> gen.fit(["The excellent product exceeded my expectations."])
    >>> texts = gen.generate(5, instructions="reviews")
    >>> custom_texts = gen.generate(2, templates=["The {noun} was {adjective}."])
    """

    _lifecycle = "fit"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._vocab: dict[str, list[str]] = {
            "noun": [],
            "verb": [],
            "adjective": [],
            "adverb": [],
        }

    def _extract_vocab(self, text: str) -> None:
        """Naive vocabulary extraction.

        Since we have zero dependencies (no spaCy/NLTK), we use very basic
        heuristics to guess part of speech.
        """
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)]

        for word in words:
            if word in {
                "the",
                "and",
                "with",
                "this",
                "that",
                "for",
                "was",
                "is",
                "are",
            }:
                continue

            if word.endswith(("ly", "wise")):
                self._vocab["adverb"].append(word)
            elif word.endswith(("ed", "ing", "s")):
                self._vocab["verb"].append(word)
            elif word.endswith(("ful", "ous", "ish", "ive", "y", "able", "ible")):
                self._vocab["adjective"].append(word)
            else:
                # Default to noun if it doesn't match above rules
                self._vocab["noun"].append(word)

        # Deduplicate
        for k in self._vocab:
            self._vocab[k] = list(set(self._vocab[k]))

    def fit(self, data: Any, **kwargs: Any) -> TemplateTextGenerator:
        """Extract vocabulary from sample text.

        Parameters
        ----------
        data : str | list[str]
            Sample text to learn vocabulary from.
        **kwargs : Any
            Additional unused keyword arguments.

        Returns
        -------
        TemplateTextGenerator
            ``self``, for method chaining.

        Raises
        ------
        TypeError
            If *data* is not a string or list of strings.
        """
        if isinstance(data, list):
            data = " ".join(str(item) for item in data)
        elif not isinstance(data, str):
            raise TypeError(f"Expected str or list[str], got {type(data).__name__}")

        if not data.strip():
            raise ValueError("Fitted on empty text. Cannot extract vocabulary.")

        self._extract_vocab(data)

        self.is_fitted = True
        self._logger.info(
            "Extracted vocabulary: %d nouns, %d verbs, %d adjs, %d advs.",
            len(self._vocab["noun"]),
            len(self._vocab["verb"]),
            len(self._vocab["adjective"]),
            len(self._vocab["adverb"]),
        )
        return self

    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        *,
        templates: list[str] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate synthetic text by filling templates.

        Parameters
        ----------
        num_samples : int
            Number of text strings to generate.
        instructions : str | None
            Name of a built-in template category (e.g., "reviews", "sentences", "records").
            Defaults to "sentences" if no match and no custom templates provided.
        templates : list[str] | None
            Custom templates to fill. Overrides *instructions*. Example:
            ``["{noun} is {adjective}"]``
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

        if templates is None:
            if instructions is None:
                raise ValueError(
                    "Must provide either 'instructions' (category name) or 'templates' list."
                )

            # Exact match category from instructions
            category = instructions.lower()
            if category in BUILTIN_TEMPLATES:
                templates = BUILTIN_TEMPLATES[category]
            else:
                raise ValueError(
                    f"Instruction category '{instructions}' not found. "
                    f"Available categories: {list(BUILTIN_TEMPLATES.keys())}."
                )

        assert templates is not None

        generated = []
        for _ in range(num_samples):
            template = rng.choice(templates)

            # Find all placeholders like {noun}, {adjective}
            placeholders = re.findall(r"\{([a-zA-Z]+)\}", template)

            replacements = {}
            for p in placeholders:
                p_lower = p.lower()
                vocab_list = self._vocab.get(p_lower, [])
                if not vocab_list:
                    # If we don't have this token in vocab, leave the placeholder intact
                    replacements[p] = f"[{p}]"
                else:
                    replacements[p] = rng.choice(vocab_list)

            filled = template.format(**replacements)
            # Basic capitalization rule for sentences
            if filled:
                filled = filled[0].upper() + filled[1:]

            generated.append(filled)

        self._logger.info("Generated %d templated sequences.", num_samples)
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
