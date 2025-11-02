from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lora_mvp.config import AppConfig
from lora_mvp.data.transforms import build_transforms
from lora_mvp.training.trainer import LoRATrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LoRA training loop")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to dataset manifest")
    parser.add_argument("--steps", type=int, default=10, help="Training steps to run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_manifest = [row for row in manifest if row.get("split") == "train"]
    if not train_manifest:
        raise SystemExit("Manifest does not contain any training samples")

    cfg = AppConfig.default(Path.cwd())
    cfg.training.max_train_steps = args.steps
    cfg.training.output_dir = Path("outputs/benchmark")

    transform = build_transforms(cfg.data)
    trainer = LoRATrainingPipeline(cfg.training)

    start = time.time()
    trainer.run(train_manifest, train_manifest, transform)
    duration = time.time() - start
    print(f"Completed {args.steps} steps in {duration:.2f}s")


if __name__ == "__main__":
    main()
