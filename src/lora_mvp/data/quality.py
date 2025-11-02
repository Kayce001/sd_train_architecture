from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import DataConfig
from ..logging import get_logger
from ..utils.image_io import load_image


log = get_logger(__name__)


@dataclass
class QualityReport:
    path: Path
    passed: bool
    resolution_ok: bool
    blur_score: float


@dataclass
class QualityChecker:
    config: DataConfig

    def check(self, path: Path) -> QualityReport:
        image = load_image(path)
        width, height = image.size
        resolution_ok = min(width, height) >= self.config.min_resolution

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(cv_image, cv2.CV_64F).var()
        blur_ok = blur_score >= self.config.blur_threshold
        passed = resolution_ok and blur_ok

        if not passed:
            log.warning("Quality check failed for %s (blur=%.2f)", path, blur_score)

        return QualityReport(
            path=path,
            passed=passed,
            resolution_ok=resolution_ok,
            blur_score=blur_score,
        )
