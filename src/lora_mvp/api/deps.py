from __future__ import annotations

from pathlib import Path

import yaml

from ..config import ApiConfig, AppConfig, DataConfig, EvaluationConfig, TrainingConfig


def _coerce_paths(payload: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = Path(value)


def load_config(config_path: Path | None = None) -> AppConfig:
    if config_path and config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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

    base_dir = Path.cwd()
    return AppConfig.default(base_dir)
