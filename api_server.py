import base64
import io
import json
import threading
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import ToTensor

REPO_ID = "merve/EfficientSAM"

_MODEL = None
_DEVICE = None
_CHECKPOINT = None
_LOCK = threading.Lock()


def get_model() -> tuple[Any, torch.device, str]:
    global _MODEL, _DEVICE, _CHECKPOINT
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _DEVICE, _CHECKPOINT

        preferred = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = preferred
        model = None
        checkpoint = ""

        if preferred.type == "cuda":
            try:
                gpu_ckpt = hf_hub_download(repo_id=REPO_ID, filename="efficient_sam_s_gpu.jit")
                model = torch.jit.load(gpu_ckpt, map_location="cuda")
                checkpoint = "efficient_sam_s_gpu.jit"
            except Exception:
                device = torch.device("cpu")

        if model is None:
            cpu_ckpt = hf_hub_download(repo_id=REPO_ID, filename="efficient_sam_s_cpu.jit")
            model = torch.jit.load(cpu_ckpt, map_location="cpu")
            checkpoint = "efficient_sam_s_cpu.jit"

        model.eval()

        _MODEL = model
        _DEVICE = device
        _CHECKPOINT = checkpoint
        return _MODEL, _DEVICE, _CHECKPOINT


def resize_longest_side(image: Image.Image, target: int = 1024) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = target / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return image.resize((nw, nh)), scale


def run_inference(image: Image.Image, points: list[list[int]], labels: list[int], input_size: int) -> dict:
    model, device, checkpoint = get_model()
    resized, scale = resize_longest_side(image, input_size)

    scaled_points = np.array(
        [[int(round(x * scale)), int(round(y * scale))] for x, y in points],
        dtype=np.float32,
    )
    labels_np = np.array(labels, dtype=np.int64)

    img_tensor = ToTensor()(np.array(resized)).unsqueeze(0).to(device)
    pts_tensor = torch.tensor(scaled_points, dtype=torch.float32, device=device).view(1, 1, -1, 2)
    labels_tensor = torch.tensor(labels_np, dtype=torch.int64, device=device).view(1, 1, -1)

    with torch.no_grad():
        logits, iou = model(img_tensor, pts_tensor, labels_tensor)

    masks = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy()
    ious = iou[0, 0].detach().cpu().numpy()
    best_idx = int(np.argmax(ious))
    best_mask_small = masks[best_idx].astype(np.uint8) * 255

    mask_full = Image.fromarray(best_mask_small, mode="L").resize(image.size, resample=Image.NEAREST)
    mask_bool = np.array(mask_full) > 127

    return {
        "mask": mask_bool,
        "ious": ious,
        "best_idx": best_idx,
        "device": str(device),
        "checkpoint": checkpoint,
    }


def encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    out = base.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


app = FastAPI(title="EfficientSAM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    _, device, checkpoint = get_model()
    return {
        "ok": True,
        "model": REPO_ID,
        "device": str(device),
        "checkpoint": checkpoint,
    }


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    mode: str = Form("point"),
    points_json: str = Form("[]"),
    labels_json: str = Form("[]"),
    input_size: int = Form(1024),
) -> dict:
    try:
        points = json.loads(points_json)
        labels = json.loads(labels_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}") from e

    if mode not in {"point", "box"}:
        raise HTTPException(status_code=400, detail="mode must be 'point' or 'box'")

    if mode == "point":
        if len(points) < 1:
            raise HTTPException(status_code=400, detail="At least one point required for point mode")
        if len(points) != len(labels):
            raise HTTPException(status_code=400, detail="points and labels length mismatch")
        for label in labels:
            if int(label) not in (0, 1):
                raise HTTPException(status_code=400, detail="point labels must be 0 or 1")
        used_points = [[int(x), int(y)] for x, y in points]
        used_labels = [int(v) for v in labels]
    else:
        if len(points) != 2:
            raise HTTPException(status_code=400, detail="Box mode requires exactly 2 corner points")
        (x0, y0), (x1, y1) = points
        x_min, x_max = sorted([int(x0), int(x1)])
        y_min, y_max = sorted([int(y0), int(y1)])
        used_points = [[x_min, y_min], [x_max, y_max]]
        used_labels = [2, 3]

    result = run_inference(pil, used_points, used_labels, input_size=int(input_size))

    mask_img = Image.fromarray((result["mask"].astype(np.uint8) * 255), mode="L")
    overlay = make_overlay(pil, result["mask"])

    return {
        "overlay_png_b64": encode_png(overlay),
        "mask_png_b64": encode_png(mask_img),
        "best_idx": result["best_idx"],
        "ious": [float(v) for v in result["ious"].tolist()],
        "device": result["device"],
        "checkpoint": result["checkpoint"],
    }
