"""Text Data Generation Example.

This script demonstrates how to generate synthetic text using the
`syntharc` package, learning from a real-world text file (Project Gutenberg).
"""

import re
from pathlib import Path

from syntharc.text.evaluation import evaluate_text
from syntharc.text.markov import MarkovTextGenerator


def main() -> None:
    print("=== Text Data Generation Example ===")

    # 1. Load the real-world dataset
    data_path = Path(__file__).parent / "data.txt"
    print(f"Loading data from {data_path}...")

    # Text file is huge, let's just parse the first 10,000 lines
    # and split them into valid sentences for our corpus.
    corpus = []
    with open(data_path, encoding="utf-8") as f:
        text_chunk = ""
        for i, line in enumerate(f):
            if i > 10000:
                break
            line = line.strip()
            if line:
                text_chunk += " " + line

    # Split roughly by periods to form a sentence corpus
    raw_sentences = re.split(r"(?<=[.!?]) +", text_chunk)
    # Filter out extremely short/long sentences to clean data
    corpus = [s.strip() for s in raw_sentences if 20 < len(s) < 200]

    print(f"Parsed {len(corpus)} valid sentences from the source text.")

    # 2. Initialize and fit the Markov Synthesizer
    print("\nFitting MarkovTextGenerator (Order 2)...")
    synth = MarkovTextGenerator(order=2)
    synth.fit(corpus)

    # 3. Generate synthetic text
    num_samples = 10
    print(f"\nGenerating {num_samples} synthetic sentences...")
    synthetic_texts = synth.generate(num_samples=num_samples, max_length=150, seed=42)

    # 4. Save the synthetic data
    output_path = Path(__file__).parent / "synthetic_text.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for t in synthetic_texts:
            f.write(t + "\n")
            print(f'  "{t}"')

    print(f"\nSaved synthetic data to {output_path}")

    # 5. Evaluate the generated data
    print("\nEvaluating synthetic text quality...")
    metrics = evaluate_text(corpus, synthetic_texts)

    print(f"Overall Quality Score: {metrics['overall_score']:.2%}")
    print(f"Vocabulary Overlap: {metrics['vocab_overlap']:.2%}")
    print(f"Length Similarity: {metrics['length_similarity']:.2%}")
    print(f"Synthetic Diversity: {metrics['diversity_synthetic']:.2f}")


if __name__ == "__main__":
    main()
