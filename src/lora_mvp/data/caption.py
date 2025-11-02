from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset import ImageMeta
from ..config import DataConfig


@dataclass
class Captioner:
    config: DataConfig

    def build_caption(self, meta: ImageMeta) -> str:
        return self.config.caption_template.format(label=meta.label or "object")

    def write_caption(self, meta: ImageMeta) -> Path:
        caption = self.build_caption(meta)
        target = meta.processed_path.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(caption, encoding="utf-8")
        return target
