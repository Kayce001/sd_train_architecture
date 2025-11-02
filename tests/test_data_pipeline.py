from pathlib import Path

from lora_mvp.config import DataConfig
from lora_mvp.data.transforms import build_transforms


def test_build_transforms(tmp_path: Path) -> None:
    cfg = DataConfig(
        raw_data_dir=tmp_path,
        processed_dir=tmp_path / "processed",
        image_size=256,
    )
    transform = build_transforms(cfg)
    assert transform is not None
