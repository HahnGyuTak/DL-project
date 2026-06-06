import base64
import gc
import importlib.util
import io
import json
import re
import tempfile
import threading
import uuid
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFilter
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torchvision import transforms
from torchvision.transforms import ToTensor

REPO_ID = "merve/EfficientSAM"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
VQA_MODEL_ID = "/mnt/data1/models/qwen/Qwen2.5-VL-7B-Instruct"
VQA_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
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

_CHAT_SESSIONS: dict[str, dict[str, Any]] = {}
_CHAT_LOCK = threading.Lock()


def pick_best_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    count = torch.cuda.device_count()
    for idx in range(count):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            free_bytes = 0
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx
    return torch.device(f"cuda:{best_idx}")


def get_model() -> tuple[Any, torch.device, str]:
    global _MODEL, _DEVICE, _CHECKPOINT
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _DEVICE, _CHECKPOINT

        preferred = pick_best_device()
        device = preferred
        model = None
        checkpoint = ""

        if preferred.type == "cuda":
            try:
                gpu_ckpt = hf_hub_download(repo_id=REPO_ID, filename="efficient_sam_s_gpu.jit")
                model = torch.jit.load(gpu_ckpt, map_location=str(device))
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

        device = pick_best_device()
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

        device = pick_best_device()
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(
            VQA_MODEL_ID,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
            local_files_only=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VQA_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
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

        device = pick_best_device()
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        # IrohXu repo provides custom pipeline code, not a full diffusers model repo.
        # Load custom pipeline class from that repo, then attach it to SD3 base weights.
        pipeline_file = hf_hub_download(
            repo_id=SD3_INPAINT_MODEL_ID,
            filename="pipeline_stable_diffusion_3_inpaint.py",
        )
        # Patch known batch-size bug in upstream custom pipeline:
        # latent_timestep should repeat by batch*num_images_per_prompt, not batch*num_inference_steps.
        with open(pipeline_file, "r", encoding="utf-8") as f:
            source = f.read()
        buggy = "latent_timestep = timesteps[:1].repeat(batch_size * num_inference_steps)"
        fixed = "latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)"
        if buggy in source:
            source = source.replace(buggy, fixed)
        patched_file = f"{tempfile.gettempdir()}/pipeline_stable_diffusion_3_inpaint_patched.py"
        with open(patched_file, "w", encoding="utf-8") as f:
            f.write(source)

        spec = importlib.util.spec_from_file_location("custom_sd3_inpaint_pipeline", patched_file)
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



def cuda_memory_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "devices": []}

    devices = []
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        except Exception:
            free_bytes, total_bytes = 0, 0
        devices.append(
            {
                "index": idx,
                "free_mb": round(free_bytes / (1024 * 1024), 1),
                "total_mb": round(total_bytes / (1024 * 1024), 1),
                "used_mb": round((total_bytes - free_bytes) / (1024 * 1024), 1),
                "allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 * 1024), 1),
                "reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 * 1024), 1),
            }
        )
    return {"available": True, "devices": devices}


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def model_state(model_key: str) -> dict[str, Any]:
    if model_key == "segmentation":
        loaded = _MODEL is not None
        return {
            "key": model_key,
            "name": "EfficientSAM Segmentation",
            "model": REPO_ID,
            "loaded": loaded,
            "device": str(_DEVICE) if loaded and _DEVICE is not None else str(pick_best_device()),
            "checkpoint": _CHECKPOINT if loaded else None,
        }
    if model_key == "detector":
        loaded = _DINO_MODEL is not None and _DINO_PROCESSOR is not None
        return {
            "key": model_key,
            "name": "Grounding DINO Detection",
            "model": GROUNDING_DINO_MODEL_ID,
            "loaded": loaded,
            "device": str(_DINO_DEVICE) if loaded and _DINO_DEVICE is not None else str(pick_best_device()),
        }
    if model_key == "vqa":
        loaded = _LLAVA_MODEL is not None and _LLAVA_PROCESSOR is not None
        return {
            "key": model_key,
            "name": "Qwen2.5-VL VQA",
            "model": VQA_MODEL_ID,
            "loaded": loaded,
            "device": str(_LLAVA_DEVICE) if loaded and _LLAVA_DEVICE is not None else str(pick_best_device()),
            "dtype": str(_LLAVA_DTYPE).replace("torch.", "") if loaded and _LLAVA_DTYPE is not None else None,
        }
    if model_key == "inpaint":
        loaded = _SD3_INPAINT_PIPE is not None
        return {
            "key": model_key,
            "name": "SD3 Inpaint",
            "model": SD3_INPAINT_MODEL_ID,
            "base_model": SD3_BASE_MODEL_ID,
            "loaded": loaded,
            "device": str(_SD3_INPAINT_DEVICE) if loaded and _SD3_INPAINT_DEVICE is not None else str(pick_best_device()),
            "dtype": str(_SD3_INPAINT_DTYPE).replace("torch.", "") if loaded and _SD3_INPAINT_DTYPE is not None else None,
        }
    raise HTTPException(status_code=404, detail=f"Unknown model key: {model_key}")


def all_model_states() -> list[dict[str, Any]]:
    return [model_state(key) for key in ("segmentation", "detector", "vqa", "inpaint")]


def load_model_by_key(model_key: str) -> dict[str, Any]:
    if model_key == "segmentation":
        get_model()
    elif model_key == "detector":
        get_grounding_dino_model()
    elif model_key == "vqa":
        get_llava_model()
    elif model_key == "inpaint":
        get_sd3_inpaint_pipeline()
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {model_key}")
    return model_state(model_key)


def unload_model_by_key(model_key: str) -> dict[str, Any]:
    global _MODEL, _DEVICE, _CHECKPOINT
    global _DINO_MODEL, _DINO_PROCESSOR, _DINO_DEVICE
    global _LLAVA_MODEL, _LLAVA_PROCESSOR, _LLAVA_DEVICE, _LLAVA_DTYPE
    global _SD3_INPAINT_PIPE, _SD3_INPAINT_DEVICE, _SD3_INPAINT_DTYPE

    if model_key == "segmentation":
        with _LOCK:
            _MODEL = None
            _DEVICE = None
            _CHECKPOINT = None
    elif model_key == "detector":
        with _DINO_LOCK:
            _DINO_MODEL = None
            _DINO_PROCESSOR = None
            _DINO_DEVICE = None
    elif model_key == "vqa":
        with _LLAVA_LOCK:
            _LLAVA_MODEL = None
            _LLAVA_PROCESSOR = None
            _LLAVA_DEVICE = None
            _LLAVA_DTYPE = None
    elif model_key == "inpaint":
        with _SD3_INPAINT_LOCK:
            _SD3_INPAINT_PIPE = None
            _SD3_INPAINT_DEVICE = None
            _SD3_INPAINT_DTYPE = None
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {model_key}")

    release_cuda_memory()
    return model_state(model_key)

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


def _center_crop_multiple_of_64(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = image.size
    cw = max(64, (w // 64) * 64)
    ch = max(64, (h // 64) * 64)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    right = left + cw
    bottom = top + ch
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


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
    strength: float = 0.6,
    mask_expand_px: int = 12,
) -> dict[str, Any]:
    pipe, device, dtype = get_sd3_inpaint_pipeline()
    image = image.convert("RGB")
    mask = mask.convert("L").point(lambda p: 255 if p > 127 else 0)

    # Expand mask area a bit so inpainting also covers boundary/context around the object.
    if int(mask_expand_px) > 0:
        k = int(mask_expand_px) * 2 + 1
        if k % 2 == 0:
            k += 1
        k = max(3, min(k, 255))
        mask = mask.filter(ImageFilter.MaxFilter(size=k))

    if max(image.size) > int(max_side):
        image, mask, _ = _resize_for_inpaint(image, mask, max_side=max_side)

    original_size = image.size
    image_c, crop_box = _center_crop_multiple_of_64(image)
    mask_c, _ = _center_crop_multiple_of_64(mask)

    image_t = transforms.ToTensor()(image_c).unsqueeze(0).to(device=device, dtype=dtype)
    mask_t = transforms.ToTensor()(mask_c).to(device=device, dtype=dtype)

    generator = None
    if int(seed) >= 0:
        generator = torch.Generator(device=str(device)).manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        image=image_t,
        mask_image=mask_t,
        height=image_t.shape[-2],
        width=image_t.shape[-1],
        num_inference_steps=max(1, int(num_inference_steps)),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        generator=generator,
    ).images[0]

    # Paste the generated crop back into the original canvas so output size stays stable.
    left, top, right, bottom = crop_box
    canvas = image.copy()
    if result.size != (right - left, bottom - top):
        result = result.resize((right - left, bottom - top), resample=Image.LANCZOS)
    canvas.paste(result, (left, top))

    return {
        "image": canvas,
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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = inputs.to(device)
    if "pixel_values" in model_inputs:
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(dtype=dtype)

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
        "model_id": VQA_MODEL_ID,
        "prompt": prompt,
    }



def clean_short_text(value: str, max_words: int = 12) -> str:
    text = (value or "").strip()
    text = re.sub(r"^[`'\"“”]+|[`'\"“”.,:;!]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.strip()


def extract_target_label(message: str) -> str:
    text = (message or "").strip()
    if not text:
        return ""

    patterns = [
        r"[`'\"“”]?([^`'\"“”]+?)[`'\"“”]?\s*(?:을|를)\s*(?:수정|편집|바꾸|변경|교체)",
        r"(?:수정|편집|바꾸|변경|교체)\s*(?:하고\s*싶은|할)?\s*(?:대상|개체|물체|객체)?\s*[:：]?\s*([^.,!?\n]+)",
        r"(?:edit|modify|change|replace)\s+(?:the\s+)?([^.,!?\n]+)",
        r"([^.,!?\n]+?)\s*(?:을|를)?\s*수정하고\s*싶",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        label = clean_short_text(m.group(1), max_words=8)
        label = re.sub(r"^(?:이|그|저|the|a|an)\s+", "", label, flags=re.IGNORECASE).strip()
        label = re.sub(r"\s*(?:을|를|이|가|은|는)$", "", label).strip()
        if 1 <= len(label) <= 80:
            return label
    return ""


KOREAN_OBJECT_TERMS = {
    "사람": "person",
    "남자": "man",
    "여자": "woman",
    "강아지": "dog",
    "개": "dog",
    "고양이": "cat",
    "자동차": "car",
    "차": "car",
    "컵": "cup",
    "모자": "hat",
    "의자": "chair",
    "테이블": "table",
    "책상": "desk",
    "가방": "bag",
    "신발": "shoes",
    "셔츠": "shirt",
    "얼굴": "face",
}

KOREAN_ATTRIBUTE_TERMS = {
    "빨간": "red",
    "빨강": "red",
    "붉은": "red",
    "파란": "blue",
    "파랑": "blue",
    "노란": "yellow",
    "노랑": "yellow",
    "검은": "black",
    "검정": "black",
    "하얀": "white",
    "흰": "white",
    "초록": "green",
}


def translate_known_korean_phrase(value: str) -> str:
    text = clean_short_text(value, max_words=12)
    text = re.sub(r"^(?:이|그|저)\s+", "", text).strip()
    if text in KOREAN_OBJECT_TERMS:
        return KOREAN_OBJECT_TERMS[text]

    particle_stripped = re.sub(r"\s+(?:을|를|이|가|은|는)$", "", text).strip()
    if particle_stripped in KOREAN_OBJECT_TERMS:
        return KOREAN_OBJECT_TERMS[particle_stripped]

    translated = text
    for source, target in {**KOREAN_ATTRIBUTE_TERMS, **KOREAN_OBJECT_TERMS}.items():
        translated = translated.replace(source, target)
    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


def translate_target_label_for_detection(image: Image.Image, target_label: str, request: str) -> str:
    label = clean_short_text(target_label, max_words=8)
    if re.search(r"[A-Za-z]", label):
        return label

    known_label = translate_known_korean_phrase(label)
    if known_label != label:
        return known_label

    try:
        question = (
            "The user is asking to edit an object in this image. "
            f"User request: {request!r}. Target phrase: {label!r}. "
            "Return only a short English object label for object detection, no sentence, no punctuation."
        )
        result = run_llava_vqa(image=image, question=question, max_new_tokens=24, temperature=0.0)
        translated = clean_short_text(result.get("answer", ""), max_words=6)
        translated = re.sub(r"^(?:the|a|an)\s+", "", translated, flags=re.IGNORECASE).strip()
        if translated and len(translated) <= 60:
            return translated
    except Exception:
        pass
    return label


def make_chat_segmentation_overlay(
    image: Image.Image,
    mask: np.ndarray,
    detection: dict[str, Any],
    target_label: str,
) -> Image.Image:
    overlay = make_overlay(image, mask)
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = detection["box_xyxy"]
    draw.rectangle([x0, y0, x1, y1], outline=(34, 197, 94), width=4)
    text = f"{target_label}: {detection['score']:.3f}"
    tx = max(0.0, x0)
    ty = max(0.0, y0 - 22.0)
    draw.rectangle([tx, ty, tx + (len(text) * 7) + 10, ty + 19], fill=(34, 197, 94))
    draw.text((tx + 5, ty + 2), text, fill=(0, 0, 0))
    return overlay


def segment_target_from_text(image: Image.Image, target_label: str) -> dict[str, Any]:
    prompt = build_text_prompt([target_label])
    detection_result = run_open_vocab_detection(
        image=image,
        text_prompt=prompt,
        threshold=0.25,
        text_threshold=0.20,
    )
    detections = detection_result["detections"]
    if not detections:
        detection_result = run_open_vocab_detection(
            image=image,
            text_prompt=prompt,
            threshold=0.15,
            text_threshold=0.15,
        )
        detections = detection_result["detections"]
    if not detections:
        raise ValueError(f"'{target_label}' 객체를 이미지에서 찾지 못했습니다.")

    best = max(detections, key=lambda item: item["score"])
    x0, y0, x1, y1 = best["box_xyxy"]
    points = [[int(round(x0)), int(round(y0))], [int(round(x1)), int(round(y1))]]
    seg_result = run_inference(image, points, [2, 3], input_size=1024)
    mask = seg_result["mask"]
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    overlay = make_chat_segmentation_overlay(image, mask, best, target_label)
    return {
        "target_label": target_label,
        "text_prompt": prompt,
        "detection": best,
        "mask": mask_img,
        "overlay": overlay,
        "device": seg_result["device"],
        "checkpoint": seg_result["checkpoint"],
    }


def extract_replacement_label(edit_request: str) -> str:
    text = (edit_request or "").strip()
    patterns = [
        r"(?:.+?(?:을|를)\s*)?([^.,!?\n]+?)(?:으?로)\s*(?:바꿔|변경|교체|만들)",
        r"(?:replace|change)\s+(?:the\s+)?(?:selected\s+)?[^.,!?\n]+?\s+(?:with|to)\s+(?:a\s+|an\s+|the\s+)?([^.,!?\n]+)",
        r"(?:turn|make)\s+(?:the\s+)?(?:selected\s+)?[^.,!?\n]+?\s+into\s+(?:a\s+|an\s+|the\s+)?([^.,!?\n]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        label = clean_short_text(m.group(1), max_words=8)
        label = re.sub(r"^(?:이|그|저|the|a|an)\s+", "", label, flags=re.IGNORECASE).strip()
        if label:
            return label
    return ""


def build_deterministic_edit_phrase(target_label: str, edit_request: str) -> tuple[str, str]:
    replacement = extract_replacement_label(edit_request)
    if replacement:
        desired = translate_known_korean_phrase(replacement)
        return f"replace the selected {target_label} with a {desired}", desired

    normalized = translate_known_korean_phrase(edit_request)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or f"edit the selected {target_label}", ""


def prompt_preserves_required_edit(prompt: str, required_phrase: str) -> bool:
    if not required_phrase:
        return True
    required_words = [w for w in re.findall(r"[A-Za-z0-9]+", required_phrase.lower()) if len(w) > 1]
    prompt_words = set(re.findall(r"[A-Za-z0-9]+", prompt.lower()))
    return all(word in prompt_words for word in required_words)


def build_sd3_prompt_with_llava(image: Image.Image, target_label: str, edit_request: str) -> str:
    deterministic_phrase, required_phrase = build_deterministic_edit_phrase(target_label, edit_request)
    fallback = deterministic_phrase
    try:
        question = (
            "Write one concise English Stable Diffusion inpainting prompt. "
            f"Masked object: {target_label}. User edit request: {edit_request}. "
            f"The prompt must preserve this exact requested edit intent: {deterministic_phrase}. "
            "Describe the desired final appearance only. Do not mention masks or instructions."
        )
        result = run_llava_vqa(image=image, question=question, max_new_tokens=96, temperature=0.1)
        candidate = clean_short_text(result.get("answer", ""), max_words=32)
        if candidate and prompt_preserves_required_edit(candidate, required_phrase):
            fallback = candidate
    except Exception:
        pass

    return (
        f"{fallback}, preserve the original scene composition, lighting, camera angle, and background, "
        f"edit only the selected {target_label}, high quality, realistic, seamless integration"
    )


def is_approval_message(message: str) -> bool:
    text = (message or "").strip().lower()
    exact_approvals = {"응", "네", "좋아", "ㅇㅇ", "yes", "y", "ok", "okay"}
    approval_phrases = ["진행", "진행해", "승인", "수정 진행", "proceed", "go ahead"]
    return text in exact_approvals or any(phrase in text for phrase in approval_phrases)


def is_cancel_message(message: str) -> bool:
    text = (message or "").strip().lower()
    cancels = ["아니", "취소", "다시", "no", "cancel", "stop"]
    return any(word in text for word in cancels)


def public_chat_response(session: dict[str, Any], **extra: Any) -> dict[str, Any]:
    response = {
        "ok": True,
        "session_id": session["id"],
        "stage": session["stage"],
        "target_label": session.get("target_label"),
        "detection_label": session.get("detection_label"),
        "assistant_message": session.get("assistant_message", ""),
        "proposed_prompt": session.get("proposed_prompt", ""),
    }
    response.update(extra)
    return response


def store_chat_session(session: dict[str, Any]) -> None:
    with _CHAT_LOCK:
        _CHAT_SESSIONS[session["id"]] = session


def get_chat_session(session_id: str) -> dict[str, Any]:
    with _CHAT_LOCK:
        session = _CHAT_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    return session


def create_unsegmented_chat_session(image: Image.Image, message: str, assistant_message: str, stage: str) -> dict[str, Any]:
    session = {
        "id": uuid.uuid4().hex,
        "original_image": image.copy(),
        "current_image": image.copy(),
        "target_label": "",
        "detection_label": "",
        "mask": None,
        "overlay": None,
        "proposed_prompt": "",
        "stage": stage,
        "assistant_message": assistant_message,
        "history": [{"role": "user", "content": message}, {"role": "assistant", "content": assistant_message}],
    }
    store_chat_session(session)
    return session


def segment_chat_session(session: dict[str, Any], message: str) -> dict[str, Any]:
    target = extract_target_label(message)
    if not target:
        session["stage"] = "awaiting_target"
        session["assistant_message"] = "수정할 개체를 다시 알려주세요. 예: '강아지를 수정하고 싶어.'"
        return session

    detection_label = translate_target_label_for_detection(session["current_image"], target, message)
    try:
        seg = segment_target_from_text(session["current_image"], detection_label)
    except ValueError:
        session["stage"] = "awaiting_target"
        session["target_label"] = target
        session["detection_label"] = detection_label
        session["assistant_message"] = f"'{target}'를 이미지에서 찾지 못했어요. 더 구체적인 개체 이름으로 다시 알려주세요."
        return session

    session["target_label"] = target
    session["detection_label"] = detection_label
    session["mask"] = seg["mask"]
    session["overlay"] = seg["overlay"]
    session["stage"] = "awaiting_edit"
    session["assistant_message"] = f"'{target}'로 보이는 영역을 표시했어요. 어떻게 수정하고 싶으세요?"
    session["history"].append({"role": "assistant", "content": session["assistant_message"]})
    return session


def propose_chat_edit(session: dict[str, Any], message: str) -> dict[str, Any]:
    prompt = build_sd3_prompt_with_llava(
        image=session["current_image"],
        target_label=session.get("detection_label") or session.get("target_label") or "object",
        edit_request=message,
    )
    session["proposed_prompt"] = prompt
    session["pending_edit_request"] = message
    session["stage"] = "awaiting_approval"
    session["assistant_message"] = f"SD3 인페인팅 프롬프트를 이렇게 정리했어요:\n\n{prompt}\n\n수정 진행할까요?"
    session["history"].append({"role": "user", "content": message})
    session["history"].append({"role": "assistant", "content": session["assistant_message"]})
    return session

app = FastAPI(title="EfficientSAM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    state = model_state("segmentation")
    return {"ok": True, **state}


@app.get("/health/detector")
def health_detector() -> dict[str, Any]:
    state = model_state("detector")
    return {"ok": True, **state}


@app.get("/health/vqa")
def health_vqa() -> dict[str, Any]:
    state = model_state("vqa")
    return {"ok": True, **state}


@app.get("/health/inpaint")
def health_inpaint() -> dict[str, Any]:
    state = model_state("inpaint")
    return {"ok": True, **state}


@app.get("/models")
def models() -> dict[str, Any]:
    return {
        "ok": True,
        "models": all_model_states(),
        "cuda": cuda_memory_snapshot(),
    }


@app.get("/models/{model_key}")
def model_status(model_key: str) -> dict[str, Any]:
    return {
        "ok": True,
        "model": model_state(model_key),
        "cuda": cuda_memory_snapshot(),
    }


@app.post("/models/{model_key}/load")
def model_load(model_key: str) -> dict[str, Any]:
    return {
        "ok": True,
        "model": load_model_by_key(model_key),
        "cuda": cuda_memory_snapshot(),
    }


@app.post("/models/{model_key}/unload")
def model_unload(model_key: str) -> dict[str, Any]:
    return {
        "ok": True,
        "model": unload_model_by_key(model_key),
        "cuda": cuda_memory_snapshot(),
    }


@app.post("/chat/edit/sessions")
async def create_chat_edit_session(
    image: UploadFile = File(...),
    message: str = Form(...),
) -> dict[str, Any]:
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")
    if not message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}") from e

    session = create_unsegmented_chat_session(
        image=pil,
        message=message,
        assistant_message="수정할 개체를 찾는 중입니다.",
        stage="segmenting",
    )
    session = segment_chat_session(session, message)
    store_chat_session(session)

    extra: dict[str, Any] = {}
    if session.get("overlay") is not None:
        extra["overlay_png_b64"] = encode_png(session["overlay"])
    if session.get("mask") is not None:
        extra["mask_png_b64"] = encode_png(session["mask"])
    return public_chat_response(session, **extra)


@app.post("/chat/edit/sessions/{session_id}/messages")
async def chat_edit_message(
    session_id: str,
    message: str = Form(...),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.0),
    strength: float = Form(0.6),
    mask_expand_px: int = Form(12),
    seed: int = Form(-1),
    max_side: int = Form(1024),
) -> dict[str, Any]:
    if not message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    session = get_chat_session(session_id)
    stage = session.get("stage")

    if stage in {"awaiting_target", "segmenting"}:
        session = segment_chat_session(session, message)
        store_chat_session(session)
        extra: dict[str, Any] = {}
        if session.get("overlay") is not None:
            extra["overlay_png_b64"] = encode_png(session["overlay"])
        if session.get("mask") is not None:
            extra["mask_png_b64"] = encode_png(session["mask"])
        return public_chat_response(session, **extra)

    if stage == "awaiting_approval" and is_approval_message(message):
        if session.get("mask") is None or not session.get("proposed_prompt"):
            session["stage"] = "awaiting_edit"
            session["assistant_message"] = "진행할 수정 프롬프트가 없어요. 수정 내용을 다시 알려주세요."
            store_chat_session(session)
            return public_chat_response(session)
        try:
            result = run_sd3_inpaint(
                image=session["current_image"],
                mask=session["mask"],
                prompt=session["proposed_prompt"],
                negative_prompt="blurry, low quality, artifacts, distorted",
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                strength=float(strength),
                mask_expand_px=int(mask_expand_px),
                seed=int(seed),
                max_side=int(max_side),
            )
        except Exception as e:
            session["assistant_message"] = f"SD3 수정에 실패했어요. 같은 프롬프트로 다시 진행하거나 수정 요청을 바꿔주세요: {e}"
            store_chat_session(session)
            return public_chat_response(session)

        output = result["image"].convert("RGB")
        session["current_image"] = output
        if session.get("mask") is not None and session["mask"].size != output.size:
            session["mask"] = session["mask"].resize(output.size, resample=Image.NEAREST)
        session["stage"] = "completed"
        session["assistant_message"] = "수정 이미지를 생성했어요. 추가로 바꾸고 싶은 점이 있으면 이어서 말해주세요."
        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "assistant", "content": session["assistant_message"]})
        store_chat_session(session)
        return public_chat_response(
            session,
            output_png_b64=encode_png(output),
            device=result["device"],
            dtype=result["dtype"],
            model_id=result["model_id"],
        )

    if stage == "awaiting_approval" and is_cancel_message(message):
        session["stage"] = "awaiting_edit"
        session["assistant_message"] = "좋아요. 수정 방향을 다시 알려주세요."
        store_chat_session(session)
        return public_chat_response(session)

    if session.get("mask") is None:
        session["stage"] = "awaiting_target"
        session["assistant_message"] = "먼저 수정할 개체를 알려주세요. 예: '강아지를 수정하고 싶어.'"
        store_chat_session(session)
        return public_chat_response(session)

    session = propose_chat_edit(session, message)
    store_chat_session(session)
    return public_chat_response(session)


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
        raise HTTPException(status_code=500, detail=f"Qwen2.5-VL inference failed: {e}") from e

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
    strength: float = Form(0.6),
    mask_expand_px: int = Form(30),
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
            strength=float(strength),
            mask_expand_px=int(mask_expand_px),
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
        "strength": float(strength),
        "mask_expand_px": int(mask_expand_px),
        "seed": int(seed),
    }
