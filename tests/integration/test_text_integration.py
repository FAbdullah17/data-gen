"""Integration tests for the text module.

Validates the full lifecycle: text preparation -> synthesis -> evaluation.
"""

from data_gen.text.evaluation import evaluate_text
from data_gen.text.markov import MarkovTextGenerator
from data_gen.text.template import TemplateTextGenerator


def test_text_lifecycle_markov() -> None:
    """Test full text generation and evaluation pipeline with Markov."""
    # 1. Provide corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaped across the sleeping dog.",
        "The lazy dog woke up and barked loudly.",
        "A loud bark startled the quick brown fox.",
    ]

    # 2. Fit Synthesizer
    gen = MarkovTextGenerator(order=2)
    gen.fit(corpus)
    assert gen.is_fitted is True

    # 3. Generate Synthetic Data
    synthetic_texts = gen.generate(num_samples=5, max_length=50, seed=42)
    assert len(synthetic_texts) == 5

    # 4. Evaluate
    metrics = evaluate_text(corpus, synthetic_texts)

    assert "overall_score" in metrics
    assert "vocab_overlap" in metrics
    assert "length_similarity" in metrics
    assert "diversity_synthetic" in metrics
    assert 0.0 <= metrics["overall_score"] <= 1.0


def test_text_lifecycle_template() -> None:
    """Test full text generation and evaluation pipeline with Template."""
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A beautiful red bird flies gracefully above the tall tree.",
        "An extremely smart student solved the hard problem quickly.",
    ]

    gen = TemplateTextGenerator()
    gen.fit(corpus)

    # Generate sentences
    synthetic_texts = gen.generate(num_samples=3, instructions="sentences")
    assert len(synthetic_texts) == 3

    metrics = evaluate_text(corpus, synthetic_texts)
    assert 0.0 <= metrics["overall_score"] <= 1.0
