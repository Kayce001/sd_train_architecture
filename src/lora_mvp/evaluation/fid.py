from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

from ..logging import get_logger
from ..utils.image_io import load_image


log = get_logger(__name__)


def compute_fid(real_paths: Iterable[Path], generated_paths: Iterable[Path], batch_size: int = 8) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        from torch import hub as torch_hub

        torch_hub.set_dir(torch_home)
        weights_cache = Path(torch_home) / "checkpoints" / "weights-inception-2015-12-05-6726825d.pth"
        manual_path = os.environ.get("FID_WEIGHTS_PATH")
        if manual_path:
            manual_file = Path(manual_path)
            if manual_file.exists():
                weights_cache.parent.mkdir(parents=True, exist_ok=True)
                if not weights_cache.exists():
                    shutil.copy2(manual_file, weights_cache)

    metric = FrechetInceptionDistance(normalize=True).to(device)
    preproc = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )

    def _update(paths: Iterable[Path], real: bool) -> None:
        batch: list[torch.Tensor] = []
        for path in paths:
            tensor = preproc(load_image(path))
            batch.append(tensor.unsqueeze(0))
            if len(batch) == batch_size:
                stacked = torch.cat(batch).to(device)
                metric.update(stacked, real=real)
                batch.clear()
        if batch:
            stacked = torch.cat(batch).to(device)
            metric.update(stacked, real=real)

    _update(real_paths, real=True)
    _update(generated_paths, real=False)
    score = metric.compute().item()
    log.info("FID score: %.4f", score)
    return score
