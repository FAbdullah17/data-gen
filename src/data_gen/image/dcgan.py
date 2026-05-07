"""Deep Convolutional GAN (DCGAN) synthesizer.

Requires optional `data-gen[image]` dependencies (torch, torchvision).
Trains a DCGAN from scratch on a folder of images to generate novel images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_gen.core.base import BaseSynthesizer
from data_gen.image.utils import require_torch_vision


class DCGANSynthesizer(BaseSynthesizer):
    """Deep Convolutional GAN for generating synthetic images.

    Trains a simple generator and discriminator on a set of sample images
    to learn the distribution and generate entirely novel outputs.
    Requires ~50+ images to produce recognizable results.

    Parameters
    ----------
    image_size : int
        Size to resize all input images to (height and width).
        Default is 64. Must be a multiple of 16 (e.g., 32, 64, 128)
        for the standard DCGAN architecture.
    latent_dim : int
        Size of the latent vector (z). Default is 100.
    config : dict | None
        Additional configuration passed to BaseSynthesizer.

    Examples
    --------
    >>> gan = DCGANSynthesizer(image_size=64)
    >>> gan.fit("./samples/", epochs=50, batch_size=64)
    >>> results = gan.generate(20)
    """

    _lifecycle = "fit"

    def __init__(
        self,
        image_size: int = 64,
        latent_dim: int = 100,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.device = self._detect_device()
        self._generator: Any = None

    def _detect_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _build_networks(self) -> tuple[Any, Any]:
        import torch.nn as nn

        # Standard DCGAN Generator
        class Generator(nn.Module):
            def __init__(self, z_dim: int, img_size: int) -> None:
                super().__init__()
                self.init_size = img_size // 4
                self.l1 = nn.Sequential(nn.Linear(z_dim, 128 * self.init_size**2))

                self.conv_blocks = nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 128, 3, stride=1, padding=1),
                    nn.BatchNorm2d(128, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 3, 3, stride=1, padding=1),
                    nn.Tanh(),
                )

            def forward(self, z: Any) -> Any:
                out = self.l1(z)
                out = out.view(out.shape[0], 128, self.init_size, self.init_size)
                img = self.conv_blocks(out)
                return img

        # Standard DCGAN Discriminator
        class Discriminator(nn.Module):
            def __init__(self, img_size: int) -> None:
                super().__init__()

                def discriminator_block(
                    in_filters: int, out_filters: int, bn: bool = True
                ) -> list[nn.Module]:
                    block: list[nn.Module] = [
                        nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25),
                    ]
                    if bn:
                        block.append(nn.BatchNorm2d(out_filters, 0.8))
                    return block

                self.model = nn.Sequential(
                    *discriminator_block(3, 16, bn=False),
                    *discriminator_block(16, 32),
                    *discriminator_block(32, 64),
                    *discriminator_block(64, 128),
                )
                ds_size = img_size // 16
                self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

            def forward(self, img: Any) -> Any:
                out = self.model(img)
                out = out.view(out.shape[0], -1)
                validity = self.adv_layer(out)
                return validity

        generator = Generator(self.latent_dim, self.image_size).to(self.device)
        discriminator = Discriminator(self.image_size).to(self.device)
        return generator, discriminator

    def fit(self, data: Any, **kwargs: Any) -> DCGANSynthesizer:
        """Train the DCGAN on a directory of images.

        Parameters
        ----------
        data : str | Path
            Directory containing the training images.
        **kwargs : Any
            Training parameters:
            - ``epochs`` (int, default=50)
            - ``batch_size`` (int, default=64)
            - ``lr`` (float, default=0.0002)

        Returns
        -------
        DCGANSynthesizer
            ``self``, for method chaining.
        """
        require_torch_vision()

        import torch
        import torch.nn as nn
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        path = Path(str(data))
        if not path.is_dir():
            raise ValueError(f"Training data must be a directory of images. Got: {path}")

        epochs = kwargs.get("epochs", 50)
        batch_size = kwargs.get("batch_size", 64)
        lr = kwargs.get("lr", 0.0002)

        # ImageFolder expects a structure like path/class_name/image.png
        # If the user just passed a folder of images, we need a wrapper or
        # we can use a custom dataset. Let's build a quick custom dataset to be robust.

        class FlatImageFolder(torch.utils.data.Dataset):  # type: ignore[name-defined]
            def __init__(self, root: Path, transform: Any) -> None:
                self.files = []
                for p in root.iterdir():
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        self.files.append(p)
                self.transform = transform

            def __len__(self) -> int:
                return len(self.files)

            def __getitem__(self, idx: int) -> Any:
                from PIL import Image

                img = Image.open(self.files[idx]).convert("RGB")
                return self.transform(img), 0

        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        try:
            # First try standard ImageFolder (if subdirectories exist)
            dataset = datasets.ImageFolder(str(path), transform=transform)
            if len(dataset) == 0:
                dataset = FlatImageFolder(path, transform=transform)  # type: ignore[assignment]
        except (FileNotFoundError, RuntimeError):
            # Fallback to flat structure
            dataset = FlatImageFolder(path, transform=transform)  # type: ignore[assignment]

        if len(dataset) < 50:
            self._logger.warning("Very few images (%d) found. DCGAN needs 50+.", len(dataset))
        elif len(dataset) == 0:
            raise ValueError(f"No valid images found in {path}")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # Handle case where dataset is smaller than batch size
        if len(dataloader) == 0:
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

        generator, discriminator = self._build_networks()

        adversarial_loss = nn.BCELoss()
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self._logger.info(
            "Training DCGAN on %s (%d images, %d epochs)", self.device, len(dataset), epochs
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Training DCGAN...", total=epochs)

            for epoch in range(epochs):
                for _i, (imgs, _) in enumerate(dataloader):
                    valid = torch.ones((imgs.size(0), 1), device=self.device, dtype=torch.float32)
                    fake = torch.zeros((imgs.size(0), 1), device=self.device, dtype=torch.float32)

                    real_imgs = imgs.to(self.device)

                    # Train Generator
                    optimizer_g.zero_grad()
                    z = torch.randn((imgs.size(0), self.latent_dim), device=self.device)
                    gen_imgs = generator(z)
                    g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                    g_loss.backward()
                    optimizer_g.step()

                    # Train Discriminator
                    optimizer_d.zero_grad()
                    real_loss = adversarial_loss(discriminator(real_imgs), valid)
                    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                desc = f"Epoch {epoch+1}/{epochs} [G: {g_loss.item():.3f}, D: {d_loss.item():.3f}]"
                progress.update(
                    task,
                    advance=1,
                    description=desc,
                )

        self._generator = generator
        self._generator.eval()
        self.is_fitted = True
        self._logger.info("DCGAN training complete.")
        return self

    def generate(
        self,
        num_samples: int,
        instructions: str | None = None,
        *,
        seed: int | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> list[Any]:
        """Generate synthetic images using the trained generator.

        Parameters
        ----------
        num_samples : int
            Number of images to generate.
        instructions : str | None
            Ignored.
        seed : int | None
            Random seed.
        batch_size : int
            Batch size for generation.
        **kwargs : Any
            Additional unused keyword arguments.

        Returns
        -------
        list[Image.Image]
            List of generated PIL Images.

        Raises
        ------
        RuntimeError
            If not fitted.
        """
        self._check_is_fitted()
        require_torch_vision()
        import torch
        from torchvision.transforms.functional import to_pil_image

        if seed is not None:
            torch.manual_seed(seed)

        if instructions:
            pass

        results = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                b_size = min(batch_size, num_samples - i)
                z = torch.randn((b_size, self.latent_dim), device=self.device)
                gen_imgs = self._generator(z)

                # Denormalize [-1, 1] to [0, 1]
                gen_imgs = gen_imgs * 0.5 + 0.5
                gen_imgs = torch.clamp(gen_imgs, 0, 1)

                for img_tensor in gen_imgs.cpu():
                    results.append(to_pil_image(img_tensor))

        self._logger.info("Generated %d synthetic images.", num_samples)
        return results

    def save(self, path: str | Path) -> None:
        """Save the generator state dictionary."""
        self._check_is_fitted()
        require_torch_vision()
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._generator.state_dict(), path)
        self._logger.info("Saved generator state to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> DCGANSynthesizer:
        """Load a trained generator state dictionary."""
        require_torch_vision()
        import torch

        instance = cls()
        instance._generator, _ = instance._build_networks()

        instance._generator.load_state_dict(
            torch.load(path, map_location=instance.device, weights_only=True)
        )
        instance._generator.eval()
        instance.is_fitted = True
        instance._logger.info("Loaded generator state from %s", path)
        return instance

    def evaluate(self, real_data: Any, synthetic_data: Any) -> dict[str, Any]:
        """Evaluate synthetic images against original."""
        from data_gen.image.evaluation import evaluate_images

        return evaluate_images(real_data, synthetic_data)
