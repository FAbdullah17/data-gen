"""Image augmentor synthesizer.

Provides fast, CPU-bound image variations using Pillow and NumPy.
Zero deep-learning dependencies required.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from data_gen.core.base import BaseSynthesizer


class ImageAugmentor(BaseSynthesizer):
    """Generates synthetic image variations using classical augmentation.

    Parameters
    ----------
    intensity : str
        Preset augmentation intensity: "light", "medium", or "heavy".
        Determines the aggressiveness of random ops applied during generation.
        Default is "medium".
    config : dict | None
        Additional configuration passed to BaseSynthesizer.

    Examples
    --------
    >>> aug = ImageAugmentor(intensity="medium")
    >>> aug.prepare("./samples/")
    >>> results = aug.generate(50)
    >>> aug.save_images(results, "./output/")
    """

    _lifecycle = "prepare"

    def __init__(
        self,
        intensity: str = "medium",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        if intensity not in {"light", "medium", "heavy"}:
            raise ValueError(f"intensity must be 'light', 'medium', or 'heavy', got '{intensity}'")
        self.intensity = intensity
        self._images: list[Image.Image] = []

    def prepare(self, data: Any, **kwargs: Any) -> ImageAugmentor:
        """Load and cache images for augmentation.

        Parameters
        ----------
        data : str | Path | list[str | Path | Image.Image]
            Source images. Can be a directory path containing images,
            a list of file paths, or a list of PIL Image objects.
        **kwargs : Any
            Additional unused arguments.

        Returns
        -------
        ImageAugmentor
            ``self``, for method chaining.
        """
        self._images.clear()

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.is_dir():
                # Load all valid images from directory
                for file_path in path.iterdir():
                    if file_path.is_file():
                        try:
                            img = Image.open(file_path).convert("RGB")
                            self._images.append(img.copy())
                            img.close()
                        except Exception:
                            pass
            elif path.is_file():
                img = Image.open(path).convert("RGB")
                self._images.append(img.copy())
                img.close()
            else:
                raise ValueError(f"Path does not exist: {path}")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, Image.Image):
                    self._images.append(item.convert("RGB").copy())
                elif isinstance(item, (str, Path)):
                    img = Image.open(item).convert("RGB")
                    self._images.append(img.copy())
                    img.close()
                else:
                    raise TypeError(f"Unsupported item type in list: {type(item)}")
        else:
            raise TypeError(f"Unsupported data type for prepare: {type(data)}")

        if not self._images:
            raise ValueError("No valid images found to prepare.")

        self.is_prepared = True
        self._logger.info("Prepared %d images.", len(self._images))
        return self

    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Generate augmented variations of the prepared images.

        Parameters
        ----------
        num_samples : int
            Number of images to generate.
        instructions : str | None
            Not actively used for augmentation. Reserved for API consistency.
        seed : int | None
            Random seed for reproducibility.
        **kwargs : Any
            Additional unused keyword arguments.

        Returns
        -------
        list[Image.Image]
            List of augmented PIL Images.
        """
        self._check_is_prepared()
        rng = random.Random(seed)

        if instructions:
            pass
        
        # Number of ops to apply based on intensity
        op_counts = {
            "light": (1, 2),
            "medium": (2, 4),
            "heavy": (4, 7),
        }
        min_ops, max_ops = op_counts[self.intensity]

        all_ops = ["rotate", "flip", "crop", "blur", "noise", "brightness", "color"]

        results = []
        for i in range(num_samples):
            # Sequentially select source images to ensure balanced augmentation
            # This is mathematically superior to random choice for dataset distribution
            base_img = self._images[i % len(self._images)].copy()

            # Randomly select operations
            num_ops = rng.randint(min_ops, max_ops)
            ops_to_apply = rng.choices(all_ops, k=num_ops)

            # Apply
            augmented = self.augment([base_img], ops=ops_to_apply, seed=rng.randint(0, 999999))[0]
            results.append(augmented)

        self._logger.info("Generated %d augmented variations.", num_samples)
        return results

    def augment(
        self,
        images: list[Image.Image],
        ops: list[str] | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Apply a specific sequence of operations to images.

        Parameters
        ----------
        images : list[Image.Image]
            Images to augment.
        ops : list[str] | None
            List of operation names to apply sequentially.
            Supported: rotate, flip, crop, blur, noise, brightness, color, pattern, elastic.
            If None, applies a random preset sequence.
        seed : int | None
            Random seed for op parameters.

        Returns
        -------
        list[Image.Image]
            Augmented images.
        """
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        if ops is None:
            # If no ops specified, apply a robust random sequence for diversity
            all_ops = ["rotate", "flip", "crop", "blur", "noise", "brightness", "color"]
            ops = rng.choices(all_ops, k=rng.randint(2, 4))

        augmented = []
        for img in images:
            img = img.copy()
            for op in ops:
                op = op.lower()
                if op == "rotate":
                    angle = rng.uniform(-30, 30)
                    # Expand to True so we don't crop corners, or False to keep size
                    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
                elif op == "flip":
                    if rng.random() > 0.5:
                        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    else:
                        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                elif op == "crop":
                    w, h = img.size
                    crop_w, crop_h = int(w * 0.8), int(h * 0.8)
                    x = rng.randint(0, w - crop_w)
                    y = rng.randint(0, h - crop_h)
                    img = img.crop((x, y, x + crop_w, y + crop_h)).resize(
                        (w, h), resample=Image.Resampling.LANCZOS
                    )
                elif op == "blur":
                    radius = rng.uniform(0.5, 2.5)
                    img = img.filter(ImageFilter.GaussianBlur(radius))
                elif op == "noise":
                    arr = np.array(img, dtype=np.float32)
                    noise = np_rng.normal(scale=rng.uniform(10, 30), size=arr.shape)
                    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr)
                elif op == "brightness":
                    factor = rng.uniform(0.5, 1.5)
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(factor)
                elif op == "color":
                    factor = rng.uniform(0.5, 1.5)
                    color_enhancer = ImageEnhance.Color(img)
                    img = color_enhancer.enhance(factor)
                elif op == "pattern":
                    # Simple grid overlay pattern
                    arr = np.array(img)
                    step = rng.randint(10, 40)
                    arr[::step, :] = 255
                    arr[:, ::step] = 255
                    img = Image.fromarray(arr)
                elif op == "elastic":
                    # Simple proxy for elastic: wavy distortion via numpy roll
                    arr = np.array(img)
                    freq = rng.uniform(0.05, 0.2)
                    amp = rng.uniform(2, 10)
                    for i in range(arr.shape[0]):
                        shift = int(amp * math.sin(2 * math.pi * i * freq))
                        arr[i] = np.roll(arr[i], shift, axis=0)
                    img = Image.fromarray(arr)
                else:
                    self._logger.warning("Unsupported augmentation operation: %s", op)
                    
            augmented.append(img)
            
        return augmented

    def save_images(self, images: Sequence[Image.Image], path: str | Path) -> None:
        """Save a list of images to a directory.

        Parameters
        ----------
        images : Sequence[Image.Image]
            Images to save.
        path : str | Path
            Directory to save images in. Created if it doesn't exist.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            out_path = out_dir / f"augmented_{i:04d}.png"
            img.save(out_path)
            
        self._logger.info("Saved %d images to %s", len(images), out_dir)

    def show_grid(self, images: Sequence[Image.Image], max_images: int = 16) -> None:
        """Display a grid of images (useful in notebooks).

        Parameters
        ----------
        images : Sequence[Image.Image]
            Images to display.
        max_images : int
            Maximum number of images to show.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._logger.error("matplotlib is required to show grids.")
            return

        display_imgs = images[:max_images]
        n = len(display_imgs)
        if n == 0:
            return

        cols = min(4, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, img in enumerate(display_imgs):
            axes[idx].imshow(img)
            axes[idx].axis("off")

        for idx in range(n, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()

    def evaluate(self, real_data: Any, synthetic_data: Any) -> dict[str, Any]:
        """Evaluate synthetic images against original.

        Parameters
        ----------
        real_data : list[Image.Image]
            Original images.
        synthetic_data : list[Image.Image]
            Generated images.

        Returns
        -------
        dict[str, Any]
            Quality metrics.
        """
        from data_gen.image.evaluation import evaluate_images

        return evaluate_images(real_data, synthetic_data)
