from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from ..config import EvaluationConfig
from ..logging import get_logger
from ..utils.image_io import save_image
from .clip_score import compute_clip_score
from .fid import compute_fid


log = get_logger(__name__)


@dataclass
class EvaluationResult:
    fid: float
    clip: float


@dataclass
class Evaluator:
    config: EvaluationConfig

    def generate_samples(self, pipeline_path: Path, prompts: Sequence[str]) -> list[Path]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            pipeline_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        outputs: list[Path] = []
        for prompt in prompts:
            result = pipe(prompt=prompt, num_inference_steps=30, num_images_per_prompt=1)
            for index, image in enumerate(result.images):
                safe_name = prompt.replace(" ", "_")
                target = pipeline_path / "eval" / f"{safe_name}_{index}.png"
                save_image(image, target)
                outputs.append(target)
        return outputs

    def evaluate(self, pipeline_path: Path, real_samples: Iterable[Path]) -> EvaluationResult:
        generated_paths = self.generate_samples(pipeline_path, self.config.prompts)
        fid_score = compute_fid(real_samples, generated_paths, batch_size=self.config.fid_batch_size)
        generated_images = [Image.open(path).convert("RGB") for path in generated_paths]
        clip_score = compute_clip_score(
            prompts=self.config.prompts,
            images=generated_images,
            model_name=self.config.clip_model,
            pretrained=self.config.clip_pretrained,
        )
        for image in generated_images:
            image.close()
        return EvaluationResult(fid=fid_score, clip=clip_score)

    def ab_test(self, model_a: Path, model_b: Path, real_samples: Iterable[Path]) -> dict[str, EvaluationResult]:
        results = {
            "model_a": self.evaluate(model_a, real_samples),
            "model_b": self.evaluate(model_b, real_samples),
        }
        log.info("A/B comparison: %s", results)
        return results
