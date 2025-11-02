from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


@dataclass
class DataConfig:
    raw_data_dir: Path
    processed_dir: Path
    caption_template: str = "A photo of {label}"
    train_ratio: float = 0.8
    seed: int = 42
    image_size: int = 512
    augmentations: Sequence[Literal["horizontal_flip", "color_jitter", "gaussian_noise"]] = field(
        default_factory=lambda: ("horizontal_flip",)
    )
    min_resolution: int = 256
    blur_threshold: float = 100.0


@dataclass
class TrainingConfig:
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    output_dir: Path = Path("outputs/lora")
    resolution: int = 512
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    checkpointing_steps: int = 100
    max_checkpoints: int = 3
    resume_from: Path | None = None
    seed: int = 42
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


@dataclass
class EvaluationConfig:
    prompts: tuple[str, ...] = (
        "A photo of a cat",
        "A futuristic cityscape at sunset",
    )
    num_samples: int = 4
    fid_batch_size: int = 4
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    baseline_model_path: Path | None = None


@dataclass
class ApiConfig:
    db_path: Path = Path("state/training_jobs.json")
    max_workers: int = 2
    poll_interval: float = 2.0


@dataclass
class AppConfig:
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    api: ApiConfig

    @staticmethod
    def default(base_dir: Path) -> "AppConfig":
        return AppConfig(
            data=DataConfig(
                raw_data_dir=base_dir / "data/raw",
                processed_dir=base_dir / "data/processed",
            ),
            training=TrainingConfig(output_dir=base_dir / "outputs/lora"),
            evaluation=EvaluationConfig(),
            api=ApiConfig(db_path=base_dir / "state/training_jobs.json"),
        )
