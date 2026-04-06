import os
import threading
from typing import Any

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor

REPO_ID = "merve/EfficientSAM"

_MODEL = None
_MODEL_DEVICE = None
_MODEL_CHECKPOINT = None
_MODEL_LOCK = threading.Lock()


def get_model() -> tuple[Any, torch.device, str]:
    global _MODEL, _MODEL_DEVICE, _MODEL_CHECKPOINT
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _MODEL_DEVICE, _MODEL_CHECKPOINT

        preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None
        device = preferred_device
        checkpoint = ""

        if preferred_device.type == "cuda":
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
        _MODEL_DEVICE = device
        _MODEL_CHECKPOINT = checkpoint
        return _MODEL, _MODEL_DEVICE, _MODEL_CHECKPOINT


def _resize_longest_side(image: Image.Image, target_size: int = 1024) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h)), scale


def _predict_mask(image: Image.Image, points: list[list[int]], labels: list[int], input_size: int = 1024) -> dict:
    model, device, checkpoint = get_model()
    resized, scale = _resize_longest_side(image, input_size)

    scaled_points = np.array(
        [[int(round(x * scale)), int(round(y * scale))] for x, y in points],
        dtype=np.float32,
    )
    labels_np = np.array(labels, dtype=np.int64)

    img_tensor = ToTensor()(np.array(resized)).unsqueeze(0).to(device)
    pts_sampled = torch.tensor(scaled_points, dtype=torch.float32, device=device).view(1, 1, -1, 2)
    pts_labels = torch.tensor(labels_np, dtype=torch.int64, device=device).view(1, 1, -1)

    with torch.no_grad():
        predicted_logits, predicted_iou = model(img_tensor, pts_sampled, pts_labels)

    masks = (torch.sigmoid(predicted_logits[0, 0]) > 0.5).cpu().numpy()
    ious = predicted_iou[0, 0].detach().cpu().numpy()
    best_idx = int(np.argmax(ious))
    best_mask_small = masks[best_idx].astype(np.uint8) * 255

    # Bring mask back to original image resolution.
    mask_full = Image.fromarray(best_mask_small).resize(image.size, resample=Image.NEAREST)
    mask_full = np.array(mask_full) > 127

    return {
        "mask": mask_full,
        "ious": ious,
        "best_idx": best_idx,
        "device": str(device),
        "checkpoint": checkpoint,
    }


def _overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    color = np.array([255, 0, 0], dtype=np.float32)
    out = base.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _draw_prompts(image: Image.Image, points: list[list[int]], labels: list[int], box_points: list[list[int]]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for (x, y), label in zip(points, labels):
        r = 6
        color = "lime" if int(label) == 1 else "red"
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)

    if len(box_points) == 1:
        x0, y0 = box_points[0]
        r = 6
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), outline="yellow", width=3)
    elif len(box_points) >= 2:
        (x0, y0), (x1, y1) = box_points[:2]
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        draw.rectangle((x_min, y_min, x_max, y_max), outline="yellow", width=3)

    return canvas


def on_image_change(image: Image.Image):
    if image is None:
        return None, [], [], [], "이미지를 업로드하세요."
    return image, [], [], [], "이미지 로드 완료. 프롬프트 모드를 고른 뒤 클릭하세요."


def on_select(
    image: Image.Image,
    mode: str,
    point_kind: str,
    points: list[list[int]],
    point_labels: list[int],
    box_points: list[list[int]],
    evt: gr.SelectData,
):
    if image is None or evt is None or evt.index is None:
        return image, points, point_labels, box_points, "이미지를 먼저 업로드하세요."

    x, y = int(evt.index[0]), int(evt.index[1])
    points = points or []
    point_labels = point_labels or []
    box_points = box_points or []

    if mode == "point":
        label = 1 if point_kind == "foreground" else 0
        points = points + [[x, y]]
        point_labels = point_labels + [label]
        status = f"포인트 추가: ({x}, {y}), label={label}, 총 {len(points)}개"
    else:
        if len(box_points) >= 2:
            box_points = []
        box_points = box_points + [[x, y]]
        status = f"박스 코너 추가: ({x}, {y}), {len(box_points)}/2"

    preview = _draw_prompts(image, points, point_labels, box_points)
    return preview, points, point_labels, box_points, status


def on_clear(image: Image.Image):
    return image, [], [], [], "프롬프트를 초기화했습니다."


def on_undo(image: Image.Image, mode: str, points: list[list[int]], point_labels: list[int], box_points: list[list[int]]):
    points = points or []
    point_labels = point_labels or []
    box_points = box_points or []

    if mode == "point":
        if points:
            points = points[:-1]
            point_labels = point_labels[:-1]
            status = f"포인트 1개 제거. 남은 포인트: {len(points)}"
        else:
            status = "제거할 포인트가 없습니다."
    else:
        if box_points:
            box_points = box_points[:-1]
            status = f"박스 코너 1개 제거. 남은 코너: {len(box_points)}"
        else:
            status = "제거할 박스 코너가 없습니다."

    preview = _draw_prompts(image, points, point_labels, box_points) if image is not None else None
    return preview, points, point_labels, box_points, status


def on_run(
    image: Image.Image,
    mode: str,
    points: list[list[int]],
    point_labels: list[int],
    box_points: list[list[int]],
    input_size: int,
):
    if image is None:
        return None, "이미지를 먼저 업로드하세요."

    points = points or []
    point_labels = point_labels or []
    box_points = box_points or []

    if mode == "point":
        if not points:
            return None, "포인트가 없습니다. 이미지를 클릭해 포인트를 추가하세요."
        result = _predict_mask(image, points, point_labels, input_size=int(input_size))
        vis = _overlay_mask(image, result["mask"])
        vis = _draw_prompts(vis, points, point_labels, [])
        status = (
            f"완료 | device={result['device']} | checkpoint={result['checkpoint']} | "
            f"best_idx={result['best_idx']} | ious={np.round(result['ious'], 4).tolist()}"
        )
        return vis, status

    if len(box_points) < 2:
        return None, "박스 모드에서는 이미지 위에 코너 2개를 찍어주세요."

    (x0, y0), (x1, y1) = box_points[:2]
    x_min, x_max = sorted([x0, x1])
    y_min, y_max = sorted([y0, y1])
    box = [[x_min, y_min], [x_max, y_max]]
    labels = [2, 3]

    result = _predict_mask(image, box, labels, input_size=int(input_size))
    vis = _overlay_mask(image, result["mask"])
    vis = _draw_prompts(vis, [], [], box)
    status = (
        f"완료 | device={result['device']} | checkpoint={result['checkpoint']} | "
        f"best_idx={result['best_idx']} | ious={np.round(result['ious'], 4).tolist()}"
    )
    return vis, status


with gr.Blocks(title="EfficientSAM Web Demo") as demo:
    gr.Markdown("## EfficientSAM Segmentation Demo")
    gr.Markdown("이미지를 업로드한 뒤 클릭으로 Point/Box 프롬프트를 넣고 세그멘테이션을 실행하세요.")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image", sources=["upload"])
        prompt_preview = gr.Image(type="pil", label="Prompt Preview", interactive=False)
        output_image = gr.Image(type="pil", label="Segmentation Output", interactive=False)

    with gr.Row():
        mode = gr.Radio(["point", "box"], value="point", label="Prompt Mode")
        point_kind = gr.Radio(["foreground", "background"], value="foreground", label="Point Label")
        input_size = gr.Slider(512, 1536, value=1024, step=64, label="Inference Resize (long side)")

    with gr.Row():
        clear_btn = gr.Button("Clear Prompts")
        undo_btn = gr.Button("Undo")
        run_btn = gr.Button("Run Segmentation", variant="primary")

    status = gr.Textbox(label="Status", value="이미지를 업로드하세요.")

    points_state = gr.State([])
    labels_state = gr.State([])
    box_state = gr.State([])

    input_image.change(
        on_image_change,
        inputs=[input_image],
        outputs=[prompt_preview, points_state, labels_state, box_state, status],
    )

    input_image.select(
        on_select,
        inputs=[input_image, mode, point_kind, points_state, labels_state, box_state],
        outputs=[prompt_preview, points_state, labels_state, box_state, status],
    )

    clear_btn.click(
        on_clear,
        inputs=[input_image],
        outputs=[prompt_preview, points_state, labels_state, box_state, status],
    )

    undo_btn.click(
        on_undo,
        inputs=[input_image, mode, points_state, labels_state, box_state],
        outputs=[prompt_preview, points_state, labels_state, box_state, status],
    )

    run_btn.click(
        on_run,
        inputs=[input_image, mode, points_state, labels_state, box_state, input_size],
        outputs=[output_image, status],
    )


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port_env = os.getenv("GRADIO_SERVER_PORT")
    server_port = int(server_port_env) if server_port_env else None
    share = os.getenv("GRADIO_SHARE", "0") == "1"

    demo.queue(default_concurrency_limit=2).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
