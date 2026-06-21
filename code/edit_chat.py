"""Conversation state machine for text-guided segmentation and SD3 image editing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .model_runtime import ModelRuntime

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
COLOR_WORDS = {"red", "blue", "yellow", "black", "white", "green"}


@dataclass
class EditSettings:
    steps: int = 30
    guidance_scale: float = 7.0
    strength: float = 0.6
    mask_expand_px: int = 12
    seed: int = -1
    max_side: int = 1024


@dataclass
class EditChatSession:
    original_image: Image.Image
    current_image: Image.Image
    stage: str = "awaiting_target"
    target_label: str = ""
    detection_label: str = ""
    detection: dict[str, Any] | None = None
    mask: Image.Image | None = None
    overlay: Image.Image | None = None
    proposed_prompt: str = ""


@dataclass
class ChatUpdate:
    assistant_message: str
    view: str
    status: str


def clean_short_text(value: str, max_words: int = 12) -> str:
    text = (value or "").strip()
    text = re.sub(r"^[`'\"“”]+|[`'\"“”.,:;!]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return " ".join(text.split()[:max_words]).strip()


def extract_target_label(message: str) -> str:
    text = (message or "").strip()
    patterns = [
        r"[`'\"“”]?([^`'\"“”]+?)[`'\"“”]?\s*(?:을|를)\s*(?:수정|편집|바꾸|변경|교체)",
        r"(?:수정|편집|바꾸|변경|교체)\s*(?:하고\s*싶은|할)?\s*(?:대상|개체|물체|객체)?\s*[:：]?\s*([^.,!?\n]+)",
        r"(?:edit|modify|change|replace)\s+(?:the\s+)?([^.,!?\n]+)",
        r"([^.,!?\n]+?)\s*(?:을|를)?\s*수정하고\s*싶",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        label = clean_short_text(match.group(1), max_words=8)
        label = re.sub(r"^(?:이|그|저|the|a|an)\s+", "", label, flags=re.IGNORECASE).strip()
        if 1 <= len(label) <= 80:
            return label
    return ""


def translate_known_korean_phrase(value: str) -> str:
    text = clean_short_text(value, max_words=12)
    text = re.sub(r"^(?:이|그|저)\s+", "", text).strip()
    if text in KOREAN_OBJECT_TERMS:
        return KOREAN_OBJECT_TERMS[text]
    stripped = re.sub(r"\s+(?:을|를|이|가|은|는)$", "", text).strip()
    if stripped in KOREAN_OBJECT_TERMS:
        return KOREAN_OBJECT_TERMS[stripped]
    translated = text
    for source, target in {**KOREAN_ATTRIBUTE_TERMS, **KOREAN_OBJECT_TERMS}.items():
        translated = translated.replace(source, target)
    return re.sub(r"\s+", " ", translated).strip()


def build_text_prompt(label: str) -> str:
    value = label.strip().lower()
    return value if value.endswith(".") else f"{value}."


def extract_replacement_label(edit_request: str) -> str:
    text = (edit_request or "").strip()
    patterns = [
        r"(?:.+?(?:을|를)\s*)?([^.,!?\n]+?)(?:으?로)\s*(?:바꿔|변경|교체|만들|수정|편집)",
        r"(?:replace|change)\s+(?:the\s+)?(?:selected\s+)?[^.,!?\n]+?\s+(?:with|to)\s+(?:a\s+|an\s+|the\s+)?([^.,!?\n]+)",
        r"(?:turn|make)\s+(?:the\s+)?(?:selected\s+)?[^.,!?\n]+?\s+into\s+(?:a\s+|an\s+|the\s+)?([^.,!?\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        replacement = clean_short_text(match.group(1), max_words=8)
        replacement = re.sub(r"^(?:이|그|저|the|a|an)\s+", "", replacement, flags=re.IGNORECASE).strip()
        if replacement:
            return replacement
    return ""


def build_deterministic_edit_phrase(target_label: str, edit_request: str) -> tuple[str, str]:
    replacement = extract_replacement_label(edit_request)
    if replacement:
        desired = translate_known_korean_phrase(replacement)
        return f"replace the selected {target_label} with a {desired}", desired
    normalized = translate_known_korean_phrase(edit_request)
    return normalized or f"edit the selected {target_label}", ""


def prompt_preserves_required_edit(prompt: str, required_phrase: str) -> bool:
    if not required_phrase:
        return True
    required_words = set(re.findall(r"[A-Za-z0-9]+", required_phrase.lower()))
    prompt_words = set(re.findall(r"[A-Za-z0-9]+", prompt.lower()))
    if not required_words.issubset(prompt_words):
        return False
    requested_colors = required_words & COLOR_WORDS
    candidate_colors = prompt_words & COLOR_WORDS
    return not requested_colors or not (candidate_colors - requested_colors)


def is_approval_message(message: str) -> bool:
    text = (message or "").strip().lower()
    return text in {"응", "네", "좋아", "ㅇㅇ", "yes", "y", "ok", "okay"} or any(
        phrase in text for phrase in ("진행", "진행해", "승인", "수정 진행", "proceed", "go ahead")
    )


def is_cancel_message(message: str) -> bool:
    text = (message or "").strip().lower()
    return any(word in text for word in ("아니", "취소", "다시", "no", "cancel", "stop"))


class EditChatService:
    """Runs the same four-stage flow as the web chatbot without HTTP sessions."""

    def __init__(self, runtime: ModelRuntime | None = None) -> None:
        self.runtime = runtime or ModelRuntime()

    def new_session(self, image: Image.Image) -> EditChatSession:
        normalized = image.convert("RGB")
        return EditChatSession(original_image=normalized.copy(), current_image=normalized.copy())

    def get_view(self, session: EditChatSession | None, view: str) -> Image.Image | None:
        if session is None:
            return None
        if view == "Segmentation Overlay" and session.overlay is not None:
            return session.overlay
        if view == "Mask" and session.mask is not None:
            return session.mask
        return session.current_image

    def _translate_target_for_detection(self, image: Image.Image, target_label: str, request: str) -> str:
        if re.search(r"[A-Za-z]", target_label):
            return target_label
        known = translate_known_korean_phrase(target_label)
        if known != target_label:
            return known
        question = (
            "Return only a short English object label for image object detection. "
            f"Target phrase: {target_label!r}. User request: {request!r}."
        )
        try:
            answer = self.runtime.ask_qwen(image, question, max_new_tokens=24, temperature=0.0)["answer"]
            translated = re.sub(r"^(?:the|a|an)\s+", "", clean_short_text(answer, max_words=6), flags=re.IGNORECASE)
            return translated or target_label
        except Exception:
            return target_label

    def _segment_target(self, session: EditChatSession, message: str) -> ChatUpdate:
        target = extract_target_label(message)
        if not target:
            session.stage = "awaiting_target"
            return ChatUpdate("수정할 개체를 다시 알려주세요. 예: '강아지를 수정하고 싶어.'", "Current Image", "대상 개체 입력 대기")

        detection_label = self._translate_target_for_detection(session.current_image, target, message)
        prompt = build_text_prompt(detection_label)
        detection_result = self.runtime.detect(session.current_image, prompt, threshold=0.25, text_threshold=0.20)
        detections = detection_result["detections"]
        if not detections:
            detection_result = self.runtime.detect(session.current_image, prompt, threshold=0.15, text_threshold=0.15)
            detections = detection_result["detections"]
        if not detections:
            session.stage = "awaiting_target"
            return ChatUpdate(
                f"'{target}'를 이미지에서 찾지 못했어요. 더 구체적인 개체 이름으로 다시 알려주세요.",
                "Current Image",
                f"Grounding DINO: '{detection_label}' 미검출",
            )

        best = max(detections, key=lambda item: item["score"])
        segmentation = self.runtime.segment_from_box(session.current_image, best["box_xyxy"], input_size=1024)
        session.target_label = target
        session.detection_label = detection_label
        session.detection = best
        session.mask = segmentation["mask"]
        session.overlay = self.runtime.make_chat_overlay(session.current_image, session.mask, best, target)
        session.stage = "awaiting_edit"
        status = (
            f"DINO={detection_result['device']} | EfficientSAM={segmentation['device']} | "
            f"score={best['score']:.3f}"
        )
        return ChatUpdate(f"'{target}'로 보이는 영역을 표시했어요. 어떻게 수정하고 싶으세요?", "Segmentation Overlay", status)

    def _build_prompt(self, image: Image.Image, target_label: str, edit_request: str) -> str:
        deterministic_phrase, required_phrase = build_deterministic_edit_phrase(target_label, edit_request)
        proposed = deterministic_phrase
        question = (
            "Write one concise English Stable Diffusion inpainting prompt. "
            f"Masked object: {target_label}. User edit request: {edit_request}. "
            f"The prompt must preserve this exact requested edit intent: {deterministic_phrase}. "
            "Describe the desired final appearance only. Do not mention masks or instructions."
        )
        try:
            candidate = clean_short_text(
                self.runtime.ask_qwen(image, question, max_new_tokens=96, temperature=0.1)["answer"],
                max_words=32,
            )
            if candidate and prompt_preserves_required_edit(candidate, required_phrase):
                proposed = candidate
        except Exception:
            pass
        return (
            f"{proposed}, preserve the original scene composition, lighting, camera angle, and background, "
            f"edit only the selected {target_label}, high quality, realistic, seamless integration"
        )

    def _propose_edit(self, session: EditChatSession, message: str) -> ChatUpdate:
        target = session.detection_label or session.target_label or "object"
        session.proposed_prompt = self._build_prompt(session.current_image, target, message)
        session.stage = "awaiting_approval"
        return ChatUpdate(
            f"SD3 인페인팅 프롬프트를 이렇게 정리했어요:\n\n{session.proposed_prompt}\n\n수정 진행할까요?",
            "Segmentation Overlay",
            "Qwen2.5-VL 프롬프트 생성 완료",
        )

    def _apply_edit(self, session: EditChatSession, settings: EditSettings) -> ChatUpdate:
        if session.mask is None or not session.proposed_prompt:
            session.stage = "awaiting_edit"
            return ChatUpdate("진행할 수정 프롬프트가 없어요. 수정 내용을 다시 알려주세요.", "Current Image", "수정 요청 대기")
        result = self.runtime.inpaint(
            image=session.current_image,
            mask=session.mask,
            prompt=session.proposed_prompt,
            steps=settings.steps,
            guidance_scale=settings.guidance_scale,
            strength=settings.strength,
            mask_expand_px=settings.mask_expand_px,
            seed=settings.seed,
            max_side=settings.max_side,
        )
        session.current_image = result["image"].convert("RGB")
        if session.mask.size != session.current_image.size:
            session.mask = session.mask.resize(session.current_image.size, resample=Image.NEAREST)
        session.overlay = self.runtime.make_mask_overlay(session.current_image, session.mask)
        session.stage = "completed"
        return ChatUpdate(
            "수정 이미지를 생성했어요. 추가로 바꾸고 싶은 점이 있으면 이어서 말해주세요.",
            "Current Image",
            f"SD3={result['device']} | mask 확장={result['mask_expand_px_at_input']}px (1024 기준)",
        )

    def process_message(self, session: EditChatSession, message: str, settings: EditSettings) -> ChatUpdate:
        text = (message or "").strip()
        if not text:
            return ChatUpdate("메시지를 입력해주세요.", "Current Image", "입력 대기")
        if session.stage in {"awaiting_target", "segmenting"}:
            return self._segment_target(session, text)
        if session.stage == "awaiting_approval":
            if is_approval_message(text):
                return self._apply_edit(session, settings)
            if is_cancel_message(text):
                session.stage = "awaiting_edit"
                return ChatUpdate("좋아요. 수정 방향을 다시 알려주세요.", "Segmentation Overlay", "수정 요청 대기")
        if session.mask is None:
            session.stage = "awaiting_target"
            return ChatUpdate("먼저 수정할 개체를 알려주세요. 예: '강아지를 수정하고 싶어.'", "Current Image", "대상 개체 입력 대기")
        return self._propose_edit(session, text)
