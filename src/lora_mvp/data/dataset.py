from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

from sklearn.model_selection import train_test_split

from ..config import DataConfig
from ..logging import get_logger
from ..utils.image_io import list_images


log = get_logger(__name__)


class ImageMeta(NamedTuple):
    source_path: Path
    processed_path: Path
    split: str
    label: str | None


@dataclass
class DatasetPreparer:
    config: DataConfig

    def discover(self) -> list[Path]:
        paths = list(list_images(self.config.raw_data_dir))
        if not paths:
            raise FileNotFoundError(f"No images found in {self.config.raw_data_dir}")
        return sorted(paths)

    def split(self, paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
        items = list(paths)
        train, val = train_test_split(
            items,
            train_size=self.config.train_ratio,
            random_state=self.config.seed,
            shuffle=True,
        )
        log.info("Split dataset into %d train and %d val samples", len(train), len(val))
        return train, val

    def build_metadata(self, train: Iterable[Path], val: Iterable[Path]) -> list[ImageMeta]:
        metadata: list[ImageMeta] = []
        for split, samples in (("train", train), ("val", val)):
            for path in samples:
                label = path.parent.name
                processed_path = self.config.processed_dir / split / label / path.name
                metadata.append(
                    ImageMeta(
                        source_path=path,
                        processed_path=processed_path,
                        split=split,
                        label=label,
                    )
                )
        return metadata

    def write_manifest(self, metadata: Iterable[ImageMeta]) -> Path:
        manifest_path = self.config.processed_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "source": str(meta.source_path),
                "processed": str(meta.processed_path),
                "split": meta.split,
                "label": meta.label,
            }
            for meta in metadata
        ]
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        log.info("Wrote manifest to %s", manifest_path)
        return manifest_path
