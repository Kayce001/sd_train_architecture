from __future__ import annotations

from typing import Iterable, Sequence

import torch
from PIL import Image
from open_clip import create_model_and_transforms, tokenize

from ..logging import get_logger


log = get_logger(__name__)


def compute_clip_score(
    prompts: Sequence[str],
    images: Iterable[Image.Image],
    model_name: str,
    pretrained: str,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenized = tokenize(list(prompts)).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores: list[float] = []
        for image in images:
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).max().item()
            scores.append(similarity)

    mean_score = float(sum(scores) / len(scores)) if scores else 0.0
    log.info("Mean CLIP score: %.4f", mean_score)
    return mean_score
