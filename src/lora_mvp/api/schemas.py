from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class TrainingStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class TrainRequest(BaseModel):
    dataset_manifest: Path
    config_path: Path | None = None


class TrainResponse(BaseModel):
    job_id: str
    status: TrainingStatus


class StatusResponse(BaseModel):
    job_id: str
    status: TrainingStatus
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class EvaluateRequest(BaseModel):
    model_path: Path
    real_samples_glob: str


class EvaluateResponse(BaseModel):
    fid: float
    clip: float
