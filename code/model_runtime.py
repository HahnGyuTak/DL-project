"""Lazy model loading and inference for the standalone image-edit chatbot."""

from __future__ import annotations

import gc
import importlib.util
import tempfile
import threading
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFilter
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from torchvision.transforms import ToTensor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .config import (
    EFFICIENT_SAM_REPO_ID,
    GROUNDING_DINO_MODEL_ID,
    QWEN_VL_MODEL_ID,
    SD3_BASE_MODEL_ID,
    SD3_INPAINT_MODEL_ID,
    is_local_path,
)


class ModelRuntime:
    """Owns lazily loaded models so the Gradio demo needs no API server."""

    def __init__(self) -> None:
        self._sam_model: Any | None = None
        self._sam_device: torch.device | None = None
        self._sam_checkpoint = ""
        self._sam_lock = threading.Lock()

        self._dino_model: Any | None = None
        self._dino_processor: Any | None = None
        self._dino_device: torch.device | None = None
        self._dino_lock = threading.Lock()

        self._vqa_model: Any | None = None
        self._vqa_processor: Any | None = None
        self._vqa_device: torch.device | None = None
        self._vqa_dtype: torch.dtype | None = None
        self._vqa_lock = threading.Lock()

        self._inpaint_pipe: Any | None = None
        self._inpaint_device: torch.device | None = None
        self._inpaint_dtype: torch.dtype | None = None
        self._inpaint_lock = threading.Lock()

    @staticmethod
    def pick_best_device() -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")

        best_idx = 0
        best_free = -1
        for idx in range(torch.cuda.device_count()):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
            except Exception:
                free_bytes = 0
            if free_bytes > best_free:
                best_idx = idx
                best_free = free_bytes
        return torch.device(f"cuda:{best_idx}")

    def get_efficient_sam(self) -> tuple[Any, torch.device, str]:
        with self._sam_lock:
            if self._sam_model is not None and self._sam_device is not None:
                return self._sam_model, self._sam_device, self._sam_checkpoint

            device = self.pick_best_device()
            model = None
            checkpoint = ""
            if device.type == "cuda":
                try:
                    checkpoint_path = hf_hub_download(
                        repo_id=EFFICIENT_SAM_REPO_ID,
                        filename="efficient_sam_s_gpu.jit",
                    )
                    model = torch.jit.load(checkpoint_path, map_location=str(device))
                    checkpoint = "efficient_sam_s_gpu.jit"
                except Exception:
                    device = torch.device("cpu")

            if model is None:
                checkpoint_path = hf_hub_download(
                    repo_id=EFFICIENT_SAM_REPO_ID,
                    filename="efficient_sam_s_cpu.jit",
                )
                model = torch.jit.load(checkpoint_path, map_location="cpu")
                checkpoint = "efficient_sam_s_cpu.jit"

            model.eval()
            self._sam_model = model
            self._sam_device = device
            self._sam_checkpoint = checkpoint
            return model, device, checkpoint

    def get_grounding_dino(self) -> tuple[Any, Any, torch.device]:
        with self._dino_lock:
            if self._dino_model is not None and self._dino_processor is not None and self._dino_device is not None:
                return self._dino_model, self._dino_processor, self._dino_device

            device = self.pick_best_device()
            processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID).to(device)
            model.eval()
            self._dino_model = model
            self._dino_processor = processor
            self._dino_device = device
            return model, processor, device

    def get_qwen_vl(self) -> tuple[Any, Any, torch.device, torch.dtype]:
        with self._vqa_lock:
            if (
                self._vqa_model is not None
                and self._vqa_processor is not None
                and self._vqa_device is not None
                and self._vqa_dtype is not None
            ):
                return self._vqa_model, self._vqa_processor, self._vqa_device, self._vqa_dtype

            device = self.pick_best_device()
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            local_only = is_local_path(QWEN_VL_MODEL_ID)
            processor = AutoProcessor.from_pretrained(
                QWEN_VL_MODEL_ID,
                min_pixels=256 * 28 * 28,
                max_pixels=1024 * 28 * 28,
                local_files_only=local_only,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_VL_MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_only,
            ).to(device)
            model.eval()
            self._vqa_model = model
            self._vqa_processor = processor
            self._vqa_device = device
            self._vqa_dtype = dtype
            return model, processor, device, dtype

    def get_sd3_inpaint(self) -> tuple[Any, torch.device, torch.dtype]:
        with self._inpaint_lock:
            if self._inpaint_pipe is not None and self._inpaint_device is not None and self._inpaint_dtype is not None:
                return self._inpaint_pipe, self._inpaint_device, self._inpaint_dtype

            from diffusers import DiffusionPipeline

            device = self.pick_best_device()
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            pipeline_file = hf_hub_download(
                repo_id=SD3_INPAINT_MODEL_ID,
                filename="pipeline_stable_diffusion_3_inpaint.py",
            )
            with open(pipeline_file, "r", encoding="utf-8") as handle:
                source = handle.read()
            source = source.replace(
                "latent_timestep = timesteps[:1].repeat(batch_size * num_inference_steps)",
                "latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)",
            )
            patched_path = f"{tempfile.gettempdir()}/model_dock_sd3_inpaint_pipeline.py"
            with open(patched_path, "w", encoding="utf-8") as handle:
                handle.write(source)

            spec = importlib.util.spec_from_file_location("model_dock_sd3_inpaint", patched_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to load the SD3 inpainting pipeline module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            pipeline_cls = getattr(module, "StableDiffusion3InpaintPipeline", None)
            if pipeline_cls is None or not issubclass(pipeline_cls, DiffusionPipeline):
                raise RuntimeError("StableDiffusion3InpaintPipeline was not found in the downloaded pipeline")

            pipe = pipeline_cls.from_pretrained(SD3_BASE_MODEL_ID, torch_dtype=dtype).to(device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()

            self._inpaint_pipe = pipe
            self._inpaint_device = device
            self._inpaint_dtype = dtype
            return pipe, device, dtype

    @staticmethod
    def _resize_longest_side(image: Image.Image, target: int = 1024) -> tuple[Image.Image, float]:
        width, height = image.size
        scale = target / max(width, height)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return image.resize(size), scale

    @torch.no_grad()
    def segment_from_box(self, image: Image.Image, box_xyxy: list[float], input_size: int = 1024) -> dict[str, Any]:
        model, device, checkpoint = self.get_efficient_sam()
        resized, scale = self._resize_longest_side(image, input_size)
        x0, y0, x1, y1 = box_xyxy
        points = np.array(
            [
                [int(round(x0 * scale)), int(round(y0 * scale))],
                [int(round(x1 * scale)), int(round(y1 * scale))],
            ],
            dtype=np.float32,
        )
        image_tensor = ToTensor()(np.array(resized)).unsqueeze(0).to(device)
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device).view(1, 1, 2, 2)
        labels_tensor = torch.tensor([2, 3], dtype=torch.int64, device=device).view(1, 1, 2)
        logits, iou = model(image_tensor, points_tensor, labels_tensor)
        masks = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy()
        ious = iou[0, 0].detach().cpu().numpy()
        best_idx = int(np.argmax(ious))
        mask_small = Image.fromarray(masks[best_idx].astype(np.uint8) * 255, mode="L")
        mask = mask_small.resize(image.size, resample=Image.NEAREST)
        return {
            "mask": mask,
            "best_idx": best_idx,
            "ious": ious,
            "device": str(device),
            "checkpoint": checkpoint,
        }

    @torch.no_grad()
    def detect(self, image: Image.Image, text_prompt: str, threshold: float, text_threshold: float) -> dict[str, Any]:
        model, processor, device = self.get_grounding_dino()
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=float(threshold),
            text_threshold=float(text_threshold),
            target_sizes=[image.size[::-1]],
        )[0]
        labels_raw = result.get("text_labels", result.get("labels", []))
        detections = []
        for box, score, label in zip(result["boxes"].detach().cpu().tolist(), result["scores"].detach().cpu().tolist(), labels_raw):
            detections.append(
                {
                    "label": label if isinstance(label, str) else str(int(label)),
                    "score": float(score),
                    "box_xyxy": [float(value) for value in box],
                }
            )
        return {"detections": detections, "device": str(device), "model_id": GROUNDING_DINO_MODEL_ID}

    @torch.no_grad()
    def ask_qwen(self, image: Image.Image, question: str, max_new_tokens: int = 96, temperature: float = 0.1) -> dict[str, Any]:
        model, processor, device, dtype = self.get_qwen_vl()
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
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(max_new_tokens)),
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = max(float(temperature), 1e-4)
        generated = model.generate(**model_inputs, **generation_kwargs)
        answer_tokens = generated[:, model_inputs["input_ids"].shape[1] :]
        answer = processor.batch_decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return {
            "answer": answer,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "model_id": QWEN_VL_MODEL_ID,
        }

    @staticmethod
    def _resize_for_inpaint(image: Image.Image, mask: Image.Image, max_side: int) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        scale = min(1.0, max(256, int(max_side)) / max(width, height))
        new_size = (
            max(64, int(round((width * scale) / 64.0)) * 64),
            max(64, int(round((height * scale) / 64.0)) * 64),
        )
        if new_size == image.size:
            return image, mask
        return image.resize(new_size, resample=Image.LANCZOS), mask.resize(new_size, resample=Image.NEAREST)

    @staticmethod
    def _center_crop_multiple_of_64(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
        width, height = image.size
        crop_width = max(64, (width // 64) * 64)
        crop_height = max(64, (height // 64) * 64)
        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        return image.crop((left, top, left + crop_width, top + crop_height)), (left, top, left + crop_width, top + crop_height)

    @staticmethod
    def _expand_mask(mask: Image.Image, expand_px_at_1024: int) -> tuple[Image.Image, int]:
        if int(expand_px_at_1024) <= 0:
            return mask, 0
        effective_px = max(1, int(round(int(expand_px_at_1024) * max(mask.size) / 1024)))
        effective_px = min(effective_px, 127)
        return mask.filter(ImageFilter.MaxFilter(size=effective_px * 2 + 1)), effective_px

    @torch.no_grad()
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, artifacts, distorted",
        steps: int = 30,
        guidance_scale: float = 7.0,
        strength: float = 0.6,
        mask_expand_px: int = 12,
        seed: int = -1,
        max_side: int = 1024,
    ) -> dict[str, Any]:
        pipe, device, dtype = self.get_sd3_inpaint()
        image = image.convert("RGB")
        mask = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
        image, mask = self._resize_for_inpaint(image, mask, max_side=max_side)
        image_crop, crop_box = self._center_crop_multiple_of_64(image)
        mask_crop, _ = self._center_crop_multiple_of_64(mask)
        mask_crop, effective_expand_px = self._expand_mask(mask_crop, mask_expand_px)

        image_tensor = transforms.ToTensor()(image_crop).unsqueeze(0).to(device=device, dtype=dtype)
        mask_tensor = transforms.ToTensor()(mask_crop).to(device=device, dtype=dtype)
        generator = None
        if int(seed) >= 0:
            generator = torch.Generator(device=str(device)).manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt.strip() or None,
            image=image_tensor,
            mask_image=mask_tensor,
            height=image_tensor.shape[-2],
            width=image_tensor.shape[-1],
            num_inference_steps=max(1, int(steps)),
            guidance_scale=float(guidance_scale),
            strength=float(strength),
            generator=generator,
        ).images[0]

        left, top, right, bottom = crop_box
        if result.size != (right - left, bottom - top):
            result = result.resize((right - left, bottom - top), resample=Image.LANCZOS)
        canvas = image.copy()
        canvas.paste(result, (left, top))
        return {
            "image": canvas,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "model_id": SD3_INPAINT_MODEL_ID,
            "mask_expand_px_at_input": effective_expand_px,
        }

    @staticmethod
    def make_mask_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
        mask_array = np.array(mask.convert("L")) > 127
        base = np.array(image.convert("RGB"), dtype=np.float32)
        out = base.copy()
        out[mask_array] = (1.0 - alpha) * out[mask_array] + alpha * np.array([255.0, 0.0, 0.0])
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    @staticmethod
    def make_chat_overlay(image: Image.Image, mask: Image.Image, detection: dict[str, Any], target_label: str) -> Image.Image:
        overlay = ModelRuntime.make_mask_overlay(image, mask)
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = detection["box_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=(34, 197, 94), width=4)
        text = f"{target_label}: {detection['score']:.3f}"
        label_top = max(0.0, y0 - 22.0)
        draw.rectangle([x0, label_top, x0 + len(text) * 7 + 10, label_top + 19], fill=(34, 197, 94))
        draw.text((x0 + 5, label_top + 2), text, fill=(0, 0, 0))
        return overlay

    def preload_all(self) -> str:
        """Load every model used by the chatbot before the Gradio UI opens."""
        self.get_efficient_sam()
        self.get_grounding_dino()
        self.get_qwen_vl()
        self.get_sd3_inpaint()
        return self.model_status()

    def model_status(self) -> str:
        states = [
            ("EfficientSAM", self._sam_model, self._sam_device),
            ("Grounding DINO", self._dino_model, self._dino_device),
            ("Qwen2.5-VL", self._vqa_model, self._vqa_device),
            ("SD3 Inpaint", self._inpaint_pipe, self._inpaint_device),
        ]
        loaded = [f"{name}: {device}" for name, model, device in states if model is not None]
        if not loaded:
            return "로드된 모델 없음"
        return " | ".join(loaded)

    def unload_all(self) -> str:
        with self._sam_lock, self._dino_lock, self._vqa_lock, self._inpaint_lock:
            self._sam_model = None
            self._sam_device = None
            self._sam_checkpoint = ""
            self._dino_model = None
            self._dino_processor = None
            self._dino_device = None
            self._vqa_model = None
            self._vqa_processor = None
            self._vqa_device = None
            self._vqa_dtype = None
            self._inpaint_pipe = None
            self._inpaint_device = None
            self._inpaint_dtype = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return "모든 GPU 모델을 unload했습니다. 다음 작업 시 필요한 모델만 다시 로드합니다."
