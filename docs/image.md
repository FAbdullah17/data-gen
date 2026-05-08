# Image Data Augmentation

The `syntharc.image` module provides a bulk operational pipeline for mutating and augmenting image datasets. Image data doesn't rely heavily on "fitting", but rather applies parameterized transformations.

## `ImageAugmentor`

The `ImageAugmentor` strictly relies on the highly optimized `albumentations` library under the hood. It loads images directly into memory and applies mutations.

### Intensity Profiles

You can configure the behavior of the augmentor purely using the `intensity` profile:

*   **`light`**: Basic non-destructive manipulations (Brightness, contrast, horizontal flips, safe rotations).
*   **`medium`**: Moderate distortions (Gaussian Noise, heavy blurs, Affine transforms like scaling/shearing).
*   **`heavy`**: Extreme mutations (Elastic transforms, Grid Distortions, Perspective shifts).

### Usage Example

```python
from syntharc.image.augmentor import ImageAugmentor
from pathlib import Path

# 1. Initialize
aug = ImageAugmentor(intensity="medium")

# 2. Prepare (Loads images from the given directory into RAM)
image_directory = Path("my_real_images/")
aug.prepare(image_directory)

# 3. Generate and mutate
synthetic_images = aug.generate(num_samples=100)

# 4. Save to disk
output_directory = Path("synthetic_output/")
aug.save_images(synthetic_images, output_directory)
```
