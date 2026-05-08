"""Evaluation metrics for text synthetic data.

Provides quality comparison between real and synthetic text
using vocabulary overlap, diversity, and length statistics.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def _tokenize(text: str | list[str]) -> list[str]:
    """Simple tokenization for evaluation."""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    if not isinstance(text, str):
        return []

    # lowercase and remove non-alphanumeric
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def evaluate_text(
    real_data: str | list[str],
    synthetic_data: str | list[str],
) -> dict[str, Any]:
    """Evaluate quality of synthetic text against original.

    Parameters
    ----------
    real_data : str | list[str]
        Original sample text.
    synthetic_data : str | list[str]
        Generated synthetic text.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``vocab_overlap``: Jaccard similarity of vocabulary (0 to 1).
        - ``diversity_real``: Unique word ratio in real data.
        - ``diversity_synthetic``: Unique word ratio in synthetic data.
        - ``length_similarity``: Similarity score based on mean text length.
        - ``overall_score``: Weighted average of the relevant metrics.
    """
    real_tokens = _tokenize(real_data)
    synth_tokens = _tokenize(synthetic_data)

    if not real_tokens or not synth_tokens:
        return {
            "vocab_overlap": 0.0,
            "diversity_real": 0.0,
            "diversity_synthetic": 0.0,
            "length_similarity": 0.0,
            "overall_score": 0.0,
        }

    real_vocab = set(real_tokens)
    synth_vocab = set(synth_tokens)

    # 1. Vocabulary Overlap (Jaccard similarity)
    intersection = len(real_vocab.intersection(synth_vocab))
    union = len(real_vocab.union(synth_vocab))
    vocab_overlap = float(intersection / union) if union > 0 else 0.0

    # 2. Diversity (Unique words / Total words)
    div_real = float(len(real_vocab) / len(real_tokens))
    div_synth = float(len(synth_vocab) / len(synth_tokens))

    # 3. Length Stats
    def get_lengths(data: str | list[str]) -> list[int]:
        if isinstance(data, str):
            # If it's a single string, we just evaluate the whole string's length
            return [len(data)]
        return [len(str(d)) for d in data]

    real_lengths = get_lengths(real_data)
    synth_lengths = get_lengths(synthetic_data)

    mean_r_len = np.mean(real_lengths)
    mean_s_len = np.mean(synth_lengths)

    len_diff = min(abs(mean_r_len - mean_s_len) / (abs(mean_r_len) + 1e-10), 1.0)
    len_sim = float(1.0 - len_diff)

    # 4. Overall score (weight overlap and length similarity)
    # We penalize if diversity is wildly different, but mostly overlap and length
    overall = float(np.mean([vocab_overlap, len_sim]))

    return {
        "vocab_overlap": vocab_overlap,
        "diversity_real": div_real,
        "diversity_synthetic": div_synth,
        "length_similarity": len_sim,
        "overall_score": overall,
    }
