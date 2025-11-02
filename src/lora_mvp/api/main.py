from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException

from ..config import AppConfig
from ..data.transforms import build_transforms
from ..evaluation.compare import Evaluator
from ..logging import get_logger
from ..training.trainer import LoRATrainingPipeline
from .deps import load_config
from .schemas import (
    EvaluateRequest,
    EvaluateResponse,
    StatusResponse,
    TrainRequest,
    TrainResponse,
    TrainingStatus,
)


log = get_logger(__name__)
app = FastAPI(title="LoRA MVP API", version="0.1.0")
_executor: ThreadPoolExecutor | None = None
_state: dict[str, dict] = {}
_state_lock = Lock()


def _get_executor(cfg: AppConfig) -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=cfg.api.max_workers)
    return _executor


def _persist_state(cfg: AppConfig) -> None:
    cfg.api.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.api.db_path.write_text(json.dumps(_state, indent=2, default=str), encoding="utf-8")


@app.on_event("startup")
def on_startup() -> None:
    cfg = load_config()
    if cfg.api.db_path.exists():
        try:
            data = json.loads(cfg.api.db_path.read_text(encoding="utf-8"))
            with _state_lock:
                _state.update(data)
        except json.JSONDecodeError:
            log.warning("Failed to parse persisted job state at %s", cfg.api.db_path)
    log.info("LoRA MVP API started")


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
async def submit_training(request: TrainRequest) -> TrainResponse:
    cfg = load_config(request.config_path)
    manifest_path = request.dataset_manifest
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")

    job_id = uuid.uuid4().hex
    with _state_lock:
        _state[job_id] = {
            "status": TrainingStatus.queued.value,
            "created_at": datetime.utcnow().isoformat(),
        }

    executor = _get_executor(cfg)
    executor.submit(_run_training, job_id, manifest_path, cfg)
    _persist_state(cfg)
    return TrainResponse(job_id=job_id, status=TrainingStatus.queued)


def _run_training(job_id: str, manifest_path: Path, cfg: AppConfig) -> None:
    with _state_lock:
        _state[job_id]["status"] = TrainingStatus.running.value
        _state[job_id]["started_at"] = datetime.utcnow().isoformat()
    _persist_state(cfg)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_manifest = [entry for entry in manifest if entry.get("split") == "train"]
        val_manifest = [entry for entry in manifest if entry.get("split") == "val"]
        transform = build_transforms(cfg.data)
        trainer = LoRATrainingPipeline(cfg.training)
        trainer.run(train_manifest, val_manifest, transform)
    except Exception as exc:  # noqa: BLE001
        log.exception("Training job %s failed", job_id)
        with _state_lock:
            _state[job_id]["status"] = TrainingStatus.failed.value
            _state[job_id]["message"] = str(exc)
            _state[job_id]["finished_at"] = datetime.utcnow().isoformat()
        _persist_state(cfg)
        return

    with _state_lock:
        _state[job_id]["status"] = TrainingStatus.succeeded.value
        _state[job_id]["finished_at"] = datetime.utcnow().isoformat()
    _persist_state(cfg)


@app.get("/status/{job_id}", response_model=StatusResponse)
async def job_status(job_id: str) -> StatusResponse:
    cfg = load_config()
    with _state_lock:
        payload = _state.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found")
    _persist_state(cfg)
    return StatusResponse(job_id=job_id, **payload)


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_model(request: EvaluateRequest) -> EvaluateResponse:
    cfg = load_config()
    model_path = request.model_path
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model path not found")

    real_samples = list(Path().glob(request.real_samples_glob))
    if not real_samples:
        raise HTTPException(status_code=400, detail="No real samples matched glob")

    evaluator = Evaluator(cfg.evaluation)
    result = evaluator.evaluate(model_path, real_samples)
    return EvaluateResponse(fid=result.fid, clip=result.clip)
