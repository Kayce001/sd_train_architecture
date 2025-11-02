from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def save_image(image: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", quality=quality, optimize=True)


def list_images(root: Path) -> Iterable[Path]:
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        yield from root.rglob(pattern)
