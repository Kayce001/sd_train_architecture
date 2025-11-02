from __future__ import annotations

import random
from typing import Callable

import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from ..config import DataConfig


def build_transforms(config: DataConfig) -> Callable[[Image.Image], torch.Tensor]:
    augmentations: list = []
    for name in config.augmentations:
        if name == "horizontal_flip":
            augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
        elif name == "color_jitter":
            augmentations.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
        elif name == "gaussian_noise":
            augmentations.append(
                lambda img, rand_fn=random.uniform: img.filter(
                    ImageFilter.GaussianBlur(radius=rand_fn(0.0, 1.0))
                )
            )

    pipeline = transforms.Compose(
        [
            transforms.Resize(
                config.image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(config.image_size),
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )
    return pipeline
