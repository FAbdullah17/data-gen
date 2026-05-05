"""Transformer-based text synthesizer using SmolLM2.

Requires the optional `data-gen[text]` dependencies (transformers, torch).
Downloads SmolLM2-360M-Instruct for local inference.
"""

from __future__ import annotations

from typing import Any

from data_gen.core.base import BaseSynthesizer
from data_gen.text.utils import require_transformers


class TransformerTextGenerator(BaseSynthesizer):
    """Text synthesizer using SmolLM2-360M-Instruct.

    Uses a pre-trained small language model for zero-shot text generation.
    Does not require training (uses ``prepare()`` lifecycle).

    Parameters
    ----------
    Examples
    --------
    >>> gen = TransformerTextGenerator()
    >>> gen.prepare(corpus="Here is a sample of the writing style...")
    >>> texts = gen.generate(5, instructions="write formal product descriptions")
    """

    _lifecycle = "prepare"

    def __init__(
        self,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        self.model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        self._device_override = device
        self._pipeline: Any = None
        self._context: str | None = None

    def prepare(
        self, data: Any = None, *, corpus: str | list[str] | None = None, **kwargs: Any
    ) -> TransformerTextGenerator:
        """Load the model and optionally cache sample text as style context.

        Parameters
        ----------
        data : str | list[str] | None
            Alias for corpus (for BaseSynthesizer compatibility).
        corpus : str | list[str] | None
            Optional sample text to use as context/style reference for generation.
        **kwargs : Any
            Additional unused keyword arguments.

        Returns
        -------
        TransformerTextGenerator
            ``self``, for method chaining.
        """
        require_transformers()

        import torch
        from transformers import pipeline

        if self._device_override:
            device = self._device_override
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self._logger.info(
            "Loading model %s on %s (this may take a moment to download)...",
            self.model_name,
            device,
        )

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )

        # Handle both corpus and data parameter for compatibility
        context_data = corpus if corpus is not None else data

        if context_data is not None:
            if isinstance(context_data, list):
                self._context = "\n".join(str(d) for d in context_data)
            elif isinstance(context_data, str):
                self._context = context_data
            else:
                raise TypeError(f"Expected str or list[str], got {type(context_data).__name__}")
            
            self._logger.info("Cached context/style reference (%d chars).", len(self._context))
        else:
            self._context = None

        self.is_prepared = True
        self._logger.info("Transformer preparation complete.")
        return self

    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        *,
        max_length: int = 200,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate synthetic text using the language model.

        Parameters
        ----------
        num_samples : int
            Number of text strings to generate.
        instructions : str | None
            Prompt instructions for the model (e.g., "write product reviews").
            If None, attempts generic text completion based on context.
        max_length : int
            Maximum number of tokens to generate per sequence.
        temperature : float
            Sampling temperature (higher = more random).
        top_p : float
            Nucleus sampling probability.
        seed : int | None
            Random seed. Applied via ``torch.manual_seed``.
        **kwargs : Any
            Additional kwargs passed directly to the HuggingFace pipeline.

        Returns
        -------
        list[str]
            Generated text strings.

        Raises
        ------
        RuntimeError
            If not prepared.
        """
        self._check_is_prepared()

        import torch

        if seed is not None:
            torch.manual_seed(seed)

        # Build prompt
        system_prompt = "You are a helpful text generation assistant."
        if self._context:
            system_prompt += f"\n\nPlease match the style of this text:\n{self._context}"

        user_prompt = instructions or "Write a few sentences of text."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self._logger.info("Generating %d sequences...", num_samples)
        
        generated_texts = []
        for _ in range(num_samples):
            # Pass as list of messages for chat template formatting
            output = self._pipeline(
                messages,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_full_text=False,
                **kwargs,
            )
            
            # pipeline returns list of dicts: [{'generated_text': '...'}]
            text = output[0]["generated_text"].strip()
            generated_texts.append(text)

        return generated_texts

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
