# Text Data Generation

The `syntharc.text` module is designed for natural language generation. It offers both lightweight statistical methods and heavy-duty Large Language Models (LLMs).

## Available Synthesizers

### 1. `MarkovTextGenerator`
A dependency-free statistical synthesizer that uses internal N-gram probabilities to construct text.
*   **Pros:** Extremely fast, mathematically predictable, requires zero ML libraries.
*   **Cons:** Struggles with long-term semantic cohesion.
*   **Parameters:** `order` (int) - The n-gram lookback length.

```python
from syntharc.text.markov import MarkovTextGenerator

generator = MarkovTextGenerator(order=2)
generator.fit(["List", "of", "real", "sentences", "from", "corpus"])
synthetic_sentences = generator.generate(num_samples=5)
```

### 2. `TemplateTextGenerator`
A strict structure-enforcing generator. Extracts vocabulary from your corpus and injects them back into strict grammar templates.

### 3. `TransformerTextGenerator`
Uses a pre-trained causal language model (`SmolLM2-360M-Instruct`) to generate highly coherent, zero-shot AI text.
*   **Dependencies:** `transformers`, `torch`
*   **Hardware:** Automatically detects and leverages `cuda` / `mps` / `cpu`.
*   **Usage:** Uses `prepare(corpus=...)` to inject stylistic context instead of `fit()`.

```python
from syntharc.text.transformer import TransformerTextGenerator

# Initialize and download/load model
transformer = TransformerTextGenerator()

# Provide real text just to use as stylistic context for the AI prompt
transformer.prepare(corpus="The sun rose over the misty London streets...")

# Generate using prompt instructions
synthetic_text = transformer.generate(
    num_samples=2,
    instructions="Write a sentence continuing the story.",
    max_length=50
)
```
