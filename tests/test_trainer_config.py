from pathlib import Path

from lora_mvp.config import TrainingConfig


def test_training_config_defaults(tmp_path: Path) -> None:
    cfg = TrainingConfig(output_dir=tmp_path / "out")
    assert cfg.output_dir == tmp_path / "out"
    assert cfg.mixed_precision in {"no", "fp16", "bf16"}
