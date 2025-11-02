from __future__ import annotations

import argparse
import json
import os
import shutil
import glob
from pathlib import Path

import yaml

from .config import ApiConfig, AppConfig, DataConfig, EvaluationConfig, TrainingConfig
from .data import Captioner, DatasetPreparer, QualityChecker, build_transforms
from .evaluation.compare import Evaluator
from .logging import get_logger
from .training.trainer import LoRATrainingPipeline


log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA MVP pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-data", help="Run preprocessing pipeline")
    prep.add_argument("--config", type=Path, help="Optional YAML config path")
    prep.add_argument(
        "--no-copy",
        action="store_true",
        help="Create hard links instead of copying into processed_dir",
    )

    train = sub.add_parser("train", help="Execute LoRA fine-tuning")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--config", type=Path, help="Optional YAML config path")

    eval_cmd = sub.add_parser("evaluate", help="Evaluate trained pipeline")
    eval_cmd.add_argument("--model-path", type=Path, required=True)
    eval_cmd.add_argument("--real-glob", type=str, required=True)
    eval_cmd.add_argument("--config", type=Path, help="Optional YAML config path")

    return parser.parse_args()


def _coerce_paths(payload: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = Path(value)


def load_app_config(path: Path | None) -> AppConfig:
    if path and path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        data_cfg = dict(raw.get("data", {}))
        training_cfg = dict(raw.get("training", {}))
        evaluation_cfg = dict(raw.get("evaluation", {}))
        api_cfg = dict(raw.get("api", {}))

        _coerce_paths(data_cfg, ("raw_data_dir", "processed_dir"))
        _coerce_paths(training_cfg, ("pretrained_model", "output_dir", "resume_from"))
        _coerce_paths(evaluation_cfg, ("baseline_model_path",))
        _coerce_paths(api_cfg, ("db_path",))

        return AppConfig(
            data=DataConfig(**data_cfg),
            training=TrainingConfig(**training_cfg),
            evaluation=EvaluationConfig(**evaluation_cfg),
            api=ApiConfig(**api_cfg),
        )
    return AppConfig.default(Path.cwd())


def ensure_materialized(meta, use_links: bool) -> None:
    meta.processed_path.parent.mkdir(parents=True, exist_ok=True)
    if meta.processed_path.exists():
        return
    if use_links:
        try:
            os.link(meta.source_path, meta.processed_path)
            return
        except OSError:
            log.warning("Hard link failed for %s, falling back to copy", meta.source_path)
    shutil.copy2(meta.source_path, meta.processed_path)


def main() -> None:
    args = parse_args()
    cfg = load_app_config(getattr(args, "config", None))

    if args.command == "prepare-data":
        preparer = DatasetPreparer(cfg.data)
        captioner = Captioner(cfg.data)
        checker = QualityChecker(cfg.data)

        paths = preparer.discover()
        train_paths, val_paths = preparer.split(paths)
        metadata = preparer.build_metadata(train_paths, val_paths)

        filtered: list = []
        for meta in metadata:
            report = checker.check(meta.source_path)
            if not report.passed:
                continue
            ensure_materialized(meta, use_links=args.no_copy)
            captioner.write_caption(meta)
            filtered.append(meta)

        manifest = preparer.write_manifest(filtered)
        log.info("Prepared dataset manifest at %s", manifest)

    elif args.command == "train":
        manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
        train_manifest = [row for row in manifest_data if row.get("split") == "train"]
        val_manifest = [row for row in manifest_data if row.get("split") == "val"]
        transform = build_transforms(cfg.data)
        trainer = LoRATrainingPipeline(cfg.training)
        trainer.run(train_manifest, val_manifest, transform)

    elif args.command == "evaluate":
        evaluator = Evaluator(cfg.evaluation)
        real_samples = [Path(p) for p in glob.glob(args.real_glob, recursive=True)]
        if not real_samples:
            raise SystemExit(f"No real samples found for pattern {args.real_glob}")
        result = evaluator.evaluate(args.model_path, real_samples)
        log.info("Evaluation -> FID: %.4f | CLIP: %.4f", result.fid, result.clip)


if __name__ == "__main__":
    main()
