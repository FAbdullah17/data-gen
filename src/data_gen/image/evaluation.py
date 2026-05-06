"""Evaluation metrics for synthetic images.

Provides quality comparison between real and synthetic images using
pixel statistics, SSIM (Structural Similarity Index), and diversity scores.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image


def _get_pixel_stats(images: Sequence[Image.Image]) -> dict[str, float]:
    """Calculate mean and std of pixel intensities."""
    if not images:
        return {"mean": 0.0, "std": 0.0}
        
    means = []
    stds = []
    for img in images:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))
        
    return {"mean": float(np.mean(means)), "std": float(np.mean(stds))}


def _calculate_ssim_proxy(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate a simple proxy for SSIM using structural correlation.
    
    This avoids requiring skimage or other large dependencies.
    """
    # Resize to small common size and grayscale to compare structure
    size = (32, 32)
    i1 = np.array(img1.convert("L").resize(size), dtype=np.float32)
    i2 = np.array(img2.convert("L").resize(size), dtype=np.float32)
    
    mu1, mu2 = np.mean(i1), np.mean(i2)
    var1, var2 = np.var(i1), np.var(i2)
    covar = float(np.cov(i1.flatten(), i2.flatten())[0][1])
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * covar + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    )
    return float(ssim)


def evaluate_images(
    real_images: list[Image.Image],
    synthetic_images: list[Image.Image],
) -> dict[str, Any]:
    """Evaluate quality of synthetic images against originals.

    Parameters
    ----------
    real_images : list[Image.Image]
        Original sample images.
    synthetic_images : list[Image.Image]
        Generated synthetic images.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``pixel_stats_similarity``: 0 to 1 similarity of pixel distributions.
        - ``ssim_score``: Structural similarity proxy against nearest real image.
        - ``diversity_score``: internal diversity of synthetic images.
        - ``overall_score``: Weighted average.
    """
    if not real_images or not synthetic_images:
        return {
            "pixel_stats_similarity": 0.0,
            "ssim_score": 0.0,
            "diversity_score": 0.0,
            "overall_score": 0.0,
        }

    # 1. Pixel stats similarity
    r_stats = _get_pixel_stats(real_images)
    s_stats = _get_pixel_stats(synthetic_images)
    
    mean_diff = abs(r_stats["mean"] - s_stats["mean"])
    std_diff = abs(r_stats["std"] - s_stats["std"])
    pixel_sim = max(0.0, 1.0 - (mean_diff + std_diff))

    # 2. Structural Similarity (Proxy)
    # Compare each synthetic image to the most similar real image
    # (Warning: O(N*M) so we limit samples to 100 for performance while maintaining statistical validity)
    max_samples = 100
    
    # Use random state to ensure unbiased statistical sampling rather than just taking the first N
    rng = np.random.default_rng(42)
    
    sample_synth = [
        synthetic_images[i] for i in rng.choice(len(synthetic_images), min(len(synthetic_images), max_samples), replace=False)
    ]
    sample_real = [
        real_images[i] for i in rng.choice(len(real_images), min(len(real_images), max_samples), replace=False)
    ]
    
    ssims = []
    for s_img in sample_synth:
        best_ssim = -1.0
        for r_img in sample_real:
            score = _calculate_ssim_proxy(s_img, r_img)
            if score > best_ssim:
                best_ssim = score
        ssims.append(best_ssim)
        
    ssim_score = float(np.mean(ssims)) if ssims else 0.0

    # 3. Diversity Score
    # How different are synthetic images from EACH OTHER?
    div_scores = []
    n = len(sample_synth)
    if n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                # 1 - SSIM means higher diversity
                div = 1.0 - _calculate_ssim_proxy(sample_synth[i], sample_synth[j])
                div_scores.append(max(0.0, div))
        diversity = float(np.mean(div_scores))
    else:
        diversity = 0.0

    # Overall score (diversity is good, but too much means noise. pixel_sim is good)
    # We just average the basic stats for a proxy score.
    overall = float(np.mean([pixel_sim, ssim_score, diversity]))

    return {
        "pixel_stats_similarity": pixel_sim,
        "ssim_score": ssim_score,
        "diversity_score": diversity,
        "overall_score": overall,
    }
