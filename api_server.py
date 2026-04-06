import base64
import importlib.util
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
SD3_INPAINT_MODEL_ID = "IrohXu/stable-diffusion-3-inpainting"
SD3_BASE_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

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

_SD3_INPAINT_PIPE = None
_SD3_INPAINT_DEVICE = None
_SD3_INPAINT_DTYPE = None
_SD3_INPAINT_LOCK = threading.Lock()


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


def get_sd3_inpaint_pipeline() -> tuple[Any, torch.device, torch.dtype]:
    global _SD3_INPAINT_PIPE, _SD3_INPAINT_DEVICE, _SD3_INPAINT_DTYPE
    with _SD3_INPAINT_LOCK:
        if _SD3_INPAINT_PIPE is not None:
            return _SD3_INPAINT_PIPE, _SD3_INPAINT_DEVICE, _SD3_INPAINT_DTYPE

        try:
            from diffusers import DiffusionPipeline
        except Exception as e:
            raise RuntimeError(f"diffusers import failed: {e}") from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        # IrohXu repo provides custom pipeline code, not a full diffusers model repo.
        # Load custom pipeline class from that repo, then attach it to SD3 base weights.
        pipeline_file = hf_hub_download(
            repo_id=SD3_INPAINT_MODEL_ID,
            filename="pipeline_stable_diffusion_3_inpaint.py",
        )
        spec = importlib.util.spec_from_file_location("custom_sd3_inpaint_pipeline", pipeline_file)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create module spec for custom SD3 inpaint pipeline")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pipeline_cls = getattr(mod, "StableDiffusion3InpaintPipeline", None)
        if pipeline_cls is None:
            raise RuntimeError("StableDiffusion3InpaintPipeline class not found in custom pipeline file")
        if not issubclass(pipeline_cls, DiffusionPipeline):
            raise RuntimeError("Custom SD3 inpaint pipeline class is not a DiffusionPipeline")

        pipe = pipeline_cls.from_pretrained(
            SD3_BASE_MODEL_ID,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        _SD3_INPAINT_PIPE = pipe
        _SD3_INPAINT_DEVICE = device
        _SD3_INPAINT_DTYPE = dtype
        return _SD3_INPAINT_PIPE, _SD3_INPAINT_DEVICE, _SD3_INPAINT_DTYPE


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


def _resize_for_inpaint(
    image: Image.Image,
    mask: Image.Image,
    max_side: int = 1024,
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    ow, oh = image.size
    target = max(256, int(max_side))
    scale = min(1.0, target / max(ow, oh))
    nw = max(64, int(round((ow * scale) / 64.0)) * 64)
    nh = max(64, int(round((oh * scale) / 64.0)) * 64)
    if nw == ow and nh == oh:
        return image, mask, (ow, oh)
    img_r = image.resize((nw, nh), resample=Image.LANCZOS)
    mask_r = mask.resize((nw, nh), resample=Image.NEAREST)
    return img_r, mask_r, (ow, oh)


@torch.no_grad()
def run_sd3_inpaint(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    seed: int = -1,
    max_side: int = 1024,
) -> dict[str, Any]:
    pipe, device, dtype = get_sd3_inpaint_pipeline()
    image = image.convert("RGB")
    mask = mask.convert("L").point(lambda p: 255 if p > 127 else 0)

    image_r, mask_r, original_size = _resize_for_inpaint(image, mask, max_side=max_side)

    generator = None
    if int(seed) >= 0:
        generator = torch.Generator(device=device.type).manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        image=image_r,
        mask_image=mask_r,
        num_inference_steps=max(1, int(num_inference_steps)),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images[0]

    if result.size != original_size:
        result = result.resize(original_size, resample=Image.LANCZOS)

    return {
        "image": result,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "model_id": SD3_INPAINT_MODEL_ID,
    }


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


@app.get("/health/inpaint")
def health_inpaint() -> dict[str, Any]:
    loaded = _SD3_INPAINT_PIPE is not None
    device = str(_SD3_INPAINT_DEVICE) if loaded and _SD3_INPAINT_DEVICE is not None else str(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return {
        "ok": True,
        "model": SD3_INPAINT_MODEL_ID,
        "base_model": SD3_BASE_MODEL_ID,
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


@app.post("/inpaint/sd3")
async def inpaint_sd3(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.0),
    seed: int = Form(-1),
    max_side: int = Form(1024),
) -> dict[str, Any]:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    image_content = await image.read()
    if not image_content:
        raise HTTPException(status_code=400, detail="Image file is empty")

    mask_content = await mask.read()
    if not mask_content:
        raise HTTPException(status_code=400, detail="Mask file is empty")

    try:
        image_pil = Image.open(io.BytesIO(image_content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}") from e

    try:
        mask_pil = Image.open(io.BytesIO(mask_content)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode mask: {e}") from e

    try:
        result = run_sd3_inpaint(
            image=image_pil,
            mask=mask_pil,
            prompt=prompt.strip(),
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            max_side=int(max_side),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SD3 inpaint failed: {e}") from e

    return {
        "output_png_b64": encode_png(result["image"]),
        "model_id": result["model_id"],
        "base_model_id": SD3_BASE_MODEL_ID,
        "device": result["device"],
        "dtype": result["dtype"],
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
    }
