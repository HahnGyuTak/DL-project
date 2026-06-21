"""Configuration values for the standalone MLLM image-edit demo."""

from __future__ import annotations

import os
from pathlib import Path

EFFICIENT_SAM_REPO_ID = "merve/EfficientSAM"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SD3_INPAINT_MODEL_ID = "IrohXu/stable-diffusion-3-inpainting"
SD3_BASE_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

_LOCAL_QWEN_PATH = Path("/mnt/data1/models/qwen/Qwen2.5-VL-7B-Instruct")


def resolve_qwen_model_id() -> str:
    """Prefer the server's local Qwen checkpoint, otherwise use Hugging Face."""
    override = os.getenv("QWEN_VL_MODEL_ID", "").strip()
    if override:
        return override
    if _LOCAL_QWEN_PATH.exists():
        return str(_LOCAL_QWEN_PATH)
    return "Qwen/Qwen2.5-VL-7B-Instruct"


def is_local_path(model_id: str) -> bool:
    return Path(model_id).exists()


QWEN_VL_MODEL_ID = resolve_qwen_model_id()
