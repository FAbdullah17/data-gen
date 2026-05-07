"""Image Data Generation Example (Albumentations).

This script demonstrates how to generate synthetic image variations
using the high-performance Albumentations engine in the `data-gen` package.
Optimized for high-throughput generation (handles 50k+ images with ease).
"""

from pathlib import Path
from data_gen.image.augmentor import ImageAugmentor
from data_gen.image.evaluation import evaluate_images

def main():
    print("=== Image Data Generation Example (Albumentations) ===")
    
    # 1. Load the real-world dataset
    # We use a small sample dataset (60 images) for this demonstration.
    data_path = Path(__file__).parent / "data"
    print(f"Loading data from {data_path}...")
    
    # 2. Initialize the Augmentor
    # Intensity 'medium' uses a balance of geometric and pixel-level transforms.
    # For more complex distortions (Elastic, Grid), use intensity='heavy'.
    print("\nPreparing ImageAugmentor (Intensity: medium)...")
    aug = ImageAugmentor(intensity="medium")
    
    # .prepare() loads images into memory for high-speed access.
    aug.prepare(data_path)
    
    real_images = aug._images 
    print(f"Loaded {len(real_images)} images into memory.")
    
    # 3. Generate synthetic data
    # The augmentor uses OpenCV-backed Albumentations, allowing for 
    # generation of thousands of images in seconds.
    num_samples = 300
    print(f"\nGenerating {num_samples} augmented synthetic images...")
    
    # We specify a seed for reproducibility.
    synthetic_images = aug.generate(num_samples=num_samples, seed=42)
    
    # 4. Save the synthetic data
    output_dir = Path(__file__).parent / "synthetic_images"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving synthetic images to {output_dir}...")
    aug.save_images(synthetic_images, output_dir)
    print("Saving complete.")
    
    # 5. Evaluate the generated data
    print("\nEvaluating synthetic image quality...")
    metrics = evaluate_images(real_images, synthetic_images)
    
    print(f"Overall Quality Score: {metrics['overall_score']:.2%}")
    print(f"Pixel Stats Similarity: {metrics['pixel_stats_similarity']:.2%}")
    print(f"Structural Similarity (SSIM Proxy): {metrics['ssim_score']:.2%}")
    print(f"Diversity Score: {metrics['diversity_score']:.2%}")

if __name__ == "__main__":
    main()
