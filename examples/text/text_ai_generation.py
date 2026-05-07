"""Transformer Text Data Generation Example.

This script demonstrates how to generate synthetic text using the
deep learning `TransformerTextGenerator` (requires transformers/torch),
which uses the `.prepare()` lifecycle.
"""

from pathlib import Path

from data_gen.text.transformer import TransformerTextGenerator

def main():
    print("=== Transformer Text Generation Example ===")
    
    try:
        # 1. Initialize the synthesizer
        print("Initializing TransformerTextGenerator (This requires Transformers & PyTorch)...")
        # Transformers do not need .fit() on raw data. They come pre-trained.
        synth = TransformerTextGenerator()
        
        # 2. Prepare the synthesizer
        # This downloads the weights (if not cached) and loads them into memory/GPU.
        print("\nPreparing model weights (SmolLM2-360M-Instruct)...")
        synth.prepare()
        
        # 3. Generate synthetic text
        num_samples = 3
        # We can pass specific semantic instructions to direct the generation
        instructions = "Write a highly detailed user review for a modern smartphone."
        
        print(f"\nGenerating {num_samples} synthetic paragraphs...")
        print(f"Prompt Context: '{instructions}'")
        
        synthetic_texts = synth.generate(
            num_samples=num_samples, 
            instructions=instructions,
            max_length=150, 
            seed=42
        )
        
        # 4. Save the synthetic data
        output_path = Path(__file__).parent / "synthetic_transformer_text.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(synthetic_texts):
                f.write(f"--- Sample {i+1} ---\n{t}\n\n")
                print(f"\n--- Sample {i+1} ---\n{t}")
                
        print(f"\nSaved synthetic data to {output_path}")
        
    except ImportError as e:
        print(f"\n[!] SKIPPING TRANSFORMER EXAMPLE: {e}")
        print("To run this example, please install Transformers/Torch: pip install data-gen[text]")

if __name__ == "__main__":
    main()
