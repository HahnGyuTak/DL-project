"""Conversation state machine for MLLM-guided segmentation and SD3 image editing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .intent_parser import ChatIntent, IntentParseError, QwenIntentParser, prompt_preserves_intent
from .model_runtime import ModelRuntime


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
    pending_intent: ChatIntent | None = None


@dataclass
class ChatUpdate:
    assistant_message: str
    view: str
    status: str


def clean_short_text(value: str, max_words: int = 32) -> str:
    text = " ".join((value or "").strip().split())
    text = re.sub(r"^[`'\"“”]+|[`'\"“”.,:;!]+$", "", text)
    return " ".join(text.split()[:max_words]).strip()


class EditChatService:
    """Runs the image-edit chat flow using structured Qwen intents, not regex routing."""

    def __init__(self, runtime: ModelRuntime | None = None) -> None:
        self.runtime = runtime or ModelRuntime()
        self.intent_parser = QwenIntentParser(self.runtime)

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

    def _parse_intent(self, session: EditChatSession, message: str, forced_action: str | None) -> ChatIntent:
        if forced_action is not None:
            return ChatIntent(action=forced_action)
        return self.intent_parser.parse(
            image=session.current_image,
            stage=session.stage,
            message=message,
            selected_target=session.detection_label or None,
        )

    def _segment_target(self, session: EditChatSession, intent: ChatIntent) -> ChatUpdate:
        if intent.action != "select_target" or not intent.target_en:
            session.stage = "awaiting_target"
            return ChatUpdate(
                "수정할 개체를 알려주세요. 예: '고양이' 또는 '강아지를 수정하고 싶어.'",
                "Current Image",
                "대상 개체 입력 대기",
            )

        detection_label = intent.target_en
        display_label = intent.target_display or detection_label
        prompt = f"{detection_label}."
        detection_result = self.runtime.detect(session.current_image, prompt, threshold=0.25, text_threshold=0.20)
        detections = detection_result["detections"]
        if not detections:
            detection_result = self.runtime.detect(session.current_image, prompt, threshold=0.15, text_threshold=0.15)
            detections = detection_result["detections"]
        if not detections:
            session.stage = "awaiting_target"
            return ChatUpdate(
                f"'{display_label}'를 이미지에서 찾지 못했어요. 더 구체적인 개체 이름으로 다시 알려주세요.",
                "Current Image",
                f"Grounding DINO: '{detection_label}' 미검출",
            )

        best = max(detections, key=lambda item: item["score"])
        segmentation = self.runtime.segment_from_box(session.current_image, best["box_xyxy"], input_size=1024)
        session.target_label = display_label
        session.detection_label = detection_label
        session.detection = best
        session.mask = segmentation["mask"]
        session.overlay = self.runtime.make_chat_overlay(session.current_image, session.mask, best, display_label)
        session.stage = "awaiting_edit"
        status = (
            f"DINO={detection_result['device']} | EfficientSAM={segmentation['device']} | "
            f"score={best['score']:.3f}"
        )
        return ChatUpdate(
            f"'{display_label}'로 보이는 영역을 표시했어요. 어떻게 수정하고 싶으세요?",
            "Segmentation Overlay",
            status,
        )

    def _build_prompt(self, image: Image.Image, target_label: str, intent: ChatIntent) -> str:
        base_prompt = intent.edit_en or f"edit the selected {target_label}"
        question = (
            "Write one concise English Stable Diffusion inpainting prompt. "
            f"Masked object: {target_label}. Structured requested edit: {intent.edit_en}. "
            f"Required attributes: {intent.attributes}. "
            "Describe the desired final appearance only. Do not mention masks or instructions."
        )
        try:
            candidate = clean_short_text(
                self.runtime.ask_qwen(image, question, max_new_tokens=96, temperature=0.1)["answer"],
                max_words=32,
            )
            if candidate and prompt_preserves_intent(candidate, intent, target_label):
                base_prompt = candidate
        except Exception:
            pass
        return (
            f"{base_prompt}, preserve the original scene composition, lighting, camera angle, and background, "
            f"edit only the selected {target_label}, high quality, realistic, seamless integration"
        )

    def _propose_edit(self, session: EditChatSession, intent: ChatIntent) -> ChatUpdate:
        if intent.action != "edit" or not intent.edit_en:
            session.stage = "awaiting_edit"
            return ChatUpdate(
                "어떻게 수정하고 싶은지 알려주세요. 예: '검정 고양이로 바꿔줘.'",
                "Segmentation Overlay",
                "수정 요청 입력 대기",
            )

        target = session.detection_label or session.target_label or "object"
        session.pending_intent = intent
        session.proposed_prompt = self._build_prompt(session.current_image, target, intent)
        session.stage = "awaiting_approval"
        return ChatUpdate(
            f"SD3 인페인팅 프롬프트를 이렇게 정리했어요:\n\n{session.proposed_prompt}\n\n수정 진행할까요?",
            "Segmentation Overlay",
            "Qwen2.5-VL structured intent + prompt 생성 완료",
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

    def process_message(
        self,
        session: EditChatSession,
        message: str,
        settings: EditSettings,
        forced_action: str | None = None,
    ) -> ChatUpdate:
        text = (message or "").strip()
        if not text and forced_action is None:
            return ChatUpdate("메시지를 입력해주세요.", "Current Image", "입력 대기")

        try:
            intent = self._parse_intent(session, text, forced_action)
        except IntentParseError:
            return ChatUpdate(
                "요청을 이해하지 못했어요. 수정할 대상이나 수정 내용을 조금 더 구체적으로 알려주세요.",
                "Current Image",
                "Qwen JSON intent 파싱 실패",
            )

        if session.stage in {"awaiting_target", "segmenting"}:
            return self._segment_target(session, intent)

        if session.stage == "awaiting_approval":
            if intent.action == "approve":
                return self._apply_edit(session, settings)
            if intent.action == "cancel":
                session.stage = "awaiting_edit"
                return ChatUpdate("좋아요. 수정 방향을 다시 알려주세요.", "Segmentation Overlay", "수정 요청 대기")

        if session.mask is None:
            session.stage = "awaiting_target"
            return ChatUpdate(
                "먼저 수정할 개체를 알려주세요. 예: '고양이' 또는 '강아지를 수정하고 싶어.'",
                "Current Image",
                "대상 개체 입력 대기",
            )

        return self._propose_edit(session, intent)
