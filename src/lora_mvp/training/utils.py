from __future__ import annotations

import shutil
from pathlib import Path

from torch.optim import Optimizer
from transformers import get_scheduler

from ..config import TrainingConfig


def create_scheduler(config: TrainingConfig, optimizer: Optimizer, num_training_steps: int):
    return get_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )


def should_save(step: int, config: TrainingConfig) -> bool:
    return step % config.checkpointing_steps == 0 or step == config.max_train_steps


def rotate_checkpoints(output_dir: Path, max_to_keep: int) -> None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: path.stat().st_mtime)
    while len(checkpoints) > max_to_keep:
        old = checkpoints.pop(0)
        shutil.rmtree(old, ignore_errors=True)
