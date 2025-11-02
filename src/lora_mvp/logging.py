from __future__ import annotations

import logging
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "lora_mvp") -> logging.Logger:
    global _LOGGER
    if _LOGGER:
        return _LOGGER.getChild(name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    _LOGGER = logging.getLogger("lora_mvp")
    return _LOGGER.getChild(name)
