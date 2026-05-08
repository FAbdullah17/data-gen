"""Image augmentor synthesizer.

Provides high-performance, CPU-bound image variations using Albumentations
and OpenCV. Optimized for large-scale synthetic generation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from syntharc.core.base import BaseSynthesizer
from syntharc.image.utils import require_albumentations


class ImageAugmentor(BaseSynthesizer):
    """Generates synthetic image variations using Albumentations.

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
        self._pipeline = None

    def _build_pipeline(self, seed: int | None = None) -> Any:
        require_albumentations()
        import albumentations as A  # noqa: N812

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.intensity == "light":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3, brightness_limit=0.1, contrast_limit=0.1),
                    A.Rotate(limit=15, p=0.3),
                ]
            )
        elif self.intensity == "medium":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
                    A.Rotate(limit=30, p=0.5),
                    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.3),
                ]
            )
        else:  # heavy
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomBrightnessContrast(p=0.7, brightness_limit=0.3, contrast_limit=0.3),
                    A.Rotate(limit=45, p=0.7),
                    A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
                    A.Blur(blur_limit=5, p=0.4),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                ]
            )

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

        if isinstance(data, str | Path):
            path = Path(data)
            if path.is_dir():
                # Load all valid images from directory
                for file_path in sorted(path.iterdir()):
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
                elif isinstance(item, str | Path):
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
        if self._pipeline is None or seed is not None:
            self._pipeline = self._build_pipeline(seed=seed)

        results: list[Image.Image] = []
        for i in range(num_samples):
            # Sequentially select source images to ensure balanced augmentation
            base_img = self._images[i % len(self._images)]
            img_arr = np.array(base_img)

            assert self._pipeline is not None

            # Apply Albumentations pipeline
            augmented_arr = self._pipeline(image=img_arr)["image"]
            results.append(Image.fromarray(augmented_arr))

        self._logger.info("Generated %d augmented variations.", num_samples)
        return results

    def augment(
        self,
        images: list[Image.Image],
        ops: list[str] | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Apply augmentation to images manually.

        Note: The `ops` parameter is deprecated with the Albumentations backend.
        This method will apply the standard intensity-based pipeline.

        Parameters
        ----------
        images : list[Image.Image]
            Images to augment.
        ops : list[str] | None
            Ignored in this version. Uses configured intensity pipeline.
        seed : int | None
            Random seed for the  pipeline.

        Returns
        -------
        list[Image.Image]
            Augmented images.
        """
        if ops is not None:
            self._logger.warning(
                "The 'ops' parameter is deprecated. Using configured intensity pipeline."
            )

        pipeline = self._build_pipeline(seed=seed)

        augmented = []
        for img in images:
            img_arr = np.array(img)
            augmented_arr = pipeline(image=img_arr)["image"]
            augmented.append(Image.fromarray(augmented_arr))

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

        _fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
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
        from syntharc.image.evaluation import evaluate_images

        return evaluate_images(real_data, synthetic_data)
