"""Gradio UI for the standalone MLLM image-edit chatbot."""

from __future__ import annotations

from typing import Generator

import gradio as gr
from PIL import Image

from .edit_chat import EditChatService, EditChatSession, EditSettings

THINKING_MESSAGE = "● ● ●"
CSS = """
.gradio-container { max-width: 1440px !important; background: radial-gradient(circle at top left, #f4e5c0, transparent 34%), #f7faf8; }
#demo-title { letter-spacing: -0.04em; }
#chat-panel { border: 1px solid #d4ddd8; border-radius: 18px; background: #fcfdfb; }
#image-panel { border: 1px solid #d4ddd8; border-radius: 18px; background: #fcfdfb; }
"""


def create_demo(service: EditChatService | None = None) -> gr.Blocks:
    service = service or EditChatService()

    def on_image_upload(image: Image.Image | None):
        if image is None:
            return None, [], None, gr.update(value="Current Image"), gr.update(visible=False), "이미지를 업로드하세요."
        session = service.new_session(image)
        history = [{"role": "assistant", "content": "이미지를 받았어요. 수정하고 싶은 개체를 말해주세요."}]
        return session, history, session.current_image, gr.update(value="Current Image"), gr.update(visible=False), "이미지 로드 완료"

    def render_view(view: str, session: EditChatSession | None):
        return service.get_view(session, view)

    def process_message(
        message: str,
        session: EditChatSession | None,
        history: list[dict[str, str]] | None,
        view: str,
        steps: int,
        guidance: float,
        strength: float,
        mask_expand: int,
        seed: int,
        max_side: int,
    ) -> Generator[tuple, None, None]:
        history = list(history or [])
        text = (message or "").strip()
        if not text:
            yield "", session, history, service.get_view(session, view), gr.update(value=view), gr.update(visible=False), "메시지를 입력해주세요."
            return
        if session is None:
            history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": "이미지를 먼저 업로드해주세요."}])
            yield "", None, history, None, gr.update(value="Current Image"), gr.update(visible=False), "이미지 입력 대기"
            return

        history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": THINKING_MESSAGE}])
        yield "", session, history, service.get_view(session, view), gr.update(value=view), gr.update(visible=False), "모델 작업 중..."

        settings = EditSettings(
            steps=int(steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            mask_expand_px=int(mask_expand),
            seed=int(seed),
            max_side=int(max_side),
        )
        try:
            update = service.process_message(session, text, settings)
            history[-1] = {"role": "assistant", "content": update.assistant_message}
            approve_visible = session.stage == "awaiting_approval"
            yield (
                "",
                session,
                history,
                service.get_view(session, update.view),
                gr.update(value=update.view),
                gr.update(visible=approve_visible),
                update.status,
            )
        except Exception as error:
            history[-1] = {"role": "assistant", "content": f"작업에 실패했어요: {error}"}
            yield "", session, history, service.get_view(session, view), gr.update(value=view), gr.update(visible=False), "오류 발생"

    def approve_edit(
        session: EditChatSession | None,
        history: list[dict[str, str]] | None,
        view: str,
        steps: int,
        guidance: float,
        strength: float,
        mask_expand: int,
        seed: int,
        max_side: int,
    ) -> Generator[tuple, None, None]:
        yield from process_message(
            "진행",
            session,
            history,
            view,
            steps,
            guidance,
            strength,
            mask_expand,
            seed,
            max_side,
        )

    with gr.Blocks(title="MLLM Image Edit Chatbot", css=CSS) as demo:
        gr.Markdown("# MLLM Image Edit Chatbot", elem_id="demo-title")
        gr.Markdown("이미지 + 자연어 요청으로 `Grounding DINO → EfficientSAM → Qwen2.5-VL → SD3 Inpaint` 과정을 대화형으로 실행한다.")

        session_state = gr.State(value=None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=11, elem_id="image-panel"):
                image_input = gr.Image(label="Input Image", type="pil", sources=["upload"])
                view_selector = gr.Radio(
                    ["Current Image", "Segmentation Overlay", "Mask"],
                    value="Current Image",
                    label="Segmentation View",
                )
                image_view = gr.Image(label="Current Image", type="pil", interactive=False)
            with gr.Column(scale=9, elem_id="chat-panel"):
                chatbot = gr.Chatbot(label="MLLM Edit Chat", height=590, type="messages")
                message_box = gr.Textbox(
                    label="Message",
                    placeholder="예: 강아지를 수정하고 싶어.",
                    lines=2,
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    approve_button = gr.Button("수정 진행", visible=False)
                status = gr.Markdown("이미지를 업로드하세요.")

        with gr.Accordion("Advanced SD3 Settings", open=False):
            with gr.Row():
                steps = gr.Slider(10, 60, value=30, step=1, label="Inference steps")
                guidance = gr.Slider(1.0, 12.0, value=7.0, step=0.5, label="Guidance scale")
                strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Strength")
            with gr.Row():
                mask_expand = gr.Slider(0, 64, value=12, step=1, label="Mask expansion at 1024px")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1: random)")
                max_side = gr.Slider(512, 1536, value=1024, step=64, label="Max input side")

        with gr.Row():
            model_status = gr.Textbox(label="GPU Model Status", value="로드된 모델 없음", interactive=False)
            refresh_status = gr.Button("모델 상태 확인")
            unload_models = gr.Button("GPU 모델 Unload")

        image_input.change(
            on_image_upload,
            inputs=[image_input],
            outputs=[session_state, chatbot, image_view, view_selector, approve_button, status],
        )
        view_selector.change(render_view, inputs=[view_selector, session_state], outputs=[image_view])
        send_inputs = [
            message_box,
            session_state,
            chatbot,
            view_selector,
            steps,
            guidance,
            strength,
            mask_expand,
            seed,
            max_side,
        ]
        send_outputs = [message_box, session_state, chatbot, image_view, view_selector, approve_button, status]
        send_button.click(process_message, inputs=send_inputs, outputs=send_outputs)
        message_box.submit(process_message, inputs=send_inputs, outputs=send_outputs)
        approve_button.click(
            approve_edit,
            inputs=send_inputs[1:],
            outputs=send_outputs,
        )
        refresh_status.click(lambda: service.runtime.model_status(), outputs=[model_status])
        unload_models.click(service.runtime.unload_all, outputs=[model_status])

    return demo
