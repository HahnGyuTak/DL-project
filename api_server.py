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
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, LlavaForConditionalGeneration
from torchvision.transforms import ToTensor

REPO_ID = "merve/EfficientSAM"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

_MODEL = None
_DEVICE = None
_CHECKPOINT = None
_LOCK = threading.Lock()

_DINO_MODEL = None
_DINO_PROCESSOR = None
_DINO_DEVICE = None
_DINO_LOCK = threading.Lock()

_LLAVA_MODEL = None
_LLAVA_PROCESSOR = None
_LLAVA_DEVICE = None
_LLAVA_DTYPE = None
_LLAVA_LOCK = threading.Lock()


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


def get_grounding_dino_model() -> tuple[Any, Any, torch.device]:
    global _DINO_MODEL, _DINO_PROCESSOR, _DINO_DEVICE
    with _DINO_LOCK:
        if _DINO_MODEL is not None and _DINO_PROCESSOR is not None:
            return _DINO_MODEL, _DINO_PROCESSOR, _DINO_DEVICE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID).to(device)
        model.eval()

        _DINO_MODEL = model
        _DINO_PROCESSOR = processor
        _DINO_DEVICE = device
        return _DINO_MODEL, _DINO_PROCESSOR, _DINO_DEVICE


def get_llava_model() -> tuple[Any, Any, torch.device, torch.dtype]:
    global _LLAVA_MODEL, _LLAVA_PROCESSOR, _LLAVA_DEVICE, _LLAVA_DTYPE
    with _LLAVA_LOCK:
        if _LLAVA_MODEL is not None and _LLAVA_PROCESSOR is not None:
            return _LLAVA_MODEL, _LLAVA_PROCESSOR, _LLAVA_DEVICE, _LLAVA_DTYPE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        model.eval()

        _LLAVA_MODEL = model
        _LLAVA_PROCESSOR = processor
        _LLAVA_DEVICE = device
        _LLAVA_DTYPE = dtype
        return _LLAVA_MODEL, _LLAVA_PROCESSOR, _LLAVA_DEVICE, _LLAVA_DTYPE


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


def build_text_prompt(labels: list[str]) -> str:
    normalized: list[str] = []
    for label in labels:
        v = label.strip().lower()
        if not v:
            continue
        if not v.endswith("."):
            v += "."
        normalized.append(v)
    return " ".join(normalized)


@torch.no_grad()
def run_open_vocab_detection(
    image: Image.Image,
    text_prompt: str,
    threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> dict[str, Any]:
    model, processor, device = get_grounding_dino_model()
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=float(threshold),
        text_threshold=float(text_threshold),
        target_sizes=[image.size[::-1]],
    )
    result = results[0]

    boxes_t = result["boxes"].detach().cpu()
    scores_t = result["scores"].detach().cpu()

    if "text_labels" in result:
        raw_labels = result["text_labels"]
    else:
        raw_labels = result["labels"]

    labels: list[str] = []
    for v in raw_labels:
        if isinstance(v, str):
            labels.append(v)
        else:
            labels.append(str(int(v)))

    detections = []
    for box, score, label in zip(boxes_t.tolist(), scores_t.tolist(), labels):
        detections.append(
            {
                "label": label,
                "score": float(score),
                "box_xyxy": [float(x) for x in box],
            }
        )

    return {
        "detections": detections,
        "device": str(device),
        "model_id": GROUNDING_DINO_MODEL_ID,
    }


def make_detection_overlay(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        x0, y0, x1, y1 = det["box_xyxy"]
        label = det["label"]
        score = det["score"]
        draw.rectangle([x0, y0, x1, y1], outline=(80, 255, 80), width=3)
        text = f"{label}: {score:.3f}"
        tx = x0
        ty = max(0.0, y0 - 18.0)
        draw.rectangle([tx, ty, tx + (len(text) * 7) + 8, ty + 16], fill=(80, 255, 80))
        draw.text((tx + 4, ty + 1), text, fill=(0, 0, 0))
    return out


@torch.no_grad()
def run_llava_vqa(
    image: Image.Image,
    question: str,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
) -> dict[str, Any]:
    model, processor, device, dtype = get_llava_model()
    prompt = f"USER: <image>\n{question.strip()}\nASSISTANT:"

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    model_inputs: dict[str, Any] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if k == "pixel_values":
                model_inputs[k] = v.to(device=device, dtype=dtype)
            else:
                model_inputs[k] = v.to(device=device)
        else:
            model_inputs[k] = v

    do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-4)

    generation = model.generate(
        **model_inputs,
        **generation_kwargs,
    )

    prompt_len = model_inputs["input_ids"].shape[1]
    answer_tokens = generation[:, prompt_len:]
    if answer_tokens.shape[1] == 0:
        decoded = processor.batch_decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answer = decoded.replace(prompt, "").strip()
    else:
        answer = processor.batch_decode(
            answer_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    return {
        "answer": answer,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "model_id": LLAVA_MODEL_ID,
        "prompt": prompt,
    }


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


@app.get("/health/detector")
def health_detector() -> dict[str, Any]:
    _, _, device = get_grounding_dino_model()
    return {
        "ok": True,
        "model": GROUNDING_DINO_MODEL_ID,
        "device": str(device),
    }


@app.get("/health/vqa")
def health_vqa() -> dict[str, Any]:
    loaded = _LLAVA_MODEL is not None and _LLAVA_PROCESSOR is not None
    device = str(_LLAVA_DEVICE) if loaded and _LLAVA_DEVICE is not None else str(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return {
        "ok": True,
        "model": LLAVA_MODEL_ID,
        "loaded": loaded,
        "device": device,
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


@app.post("/detect/open-vocab")
async def detect_open_vocab(
    image: UploadFile = File(...),
    labels_csv: str = Form(""),
    text_prompt: str = Form(""),
    threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
) -> dict[str, Any]:
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}") from e

    prompt = text_prompt.strip()
    if not prompt:
        labels = [x.strip() for x in labels_csv.split(",") if x.strip()]
        if not labels:
            raise HTTPException(status_code=400, detail="labels_csv or text_prompt is required")
        prompt = build_text_prompt(labels)

    result = run_open_vocab_detection(
        image=pil,
        text_prompt=prompt,
        threshold=float(threshold),
        text_threshold=float(text_threshold),
    )
    overlay = make_detection_overlay(pil, result["detections"])

    return {
        "overlay_png_b64": encode_png(overlay),
        "detections": result["detections"],
        "num_detections": len(result["detections"]),
        "text_prompt": prompt,
        "device": result["device"],
        "model_id": result["model_id"],
        "threshold": float(threshold),
        "text_threshold": float(text_threshold),
    }


@app.post("/vqa/llava")
async def vqa_llava(
    image: UploadFile = File(...),
    question: str = Form(...),
    max_new_tokens: int = Form(128),
    temperature: float = Form(0.2),
) -> dict[str, Any]:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}") from e

    try:
        result = run_llava_vqa(
            image=pil,
            question=question,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLaVA inference failed: {e}") from e

    return {
        "answer": result["answer"],
        "question": question.strip(),
        "model_id": result["model_id"],
        "device": result["device"],
        "dtype": result["dtype"],
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }
