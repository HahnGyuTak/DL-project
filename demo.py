"""Launch the standalone MLLM image-edit chatbot without FastAPI or Cloudflare."""

from __future__ import annotations

import os

from code.edit_chat import EditChatService
from code.gradio_ui import create_demo
from code.model_runtime import ModelRuntime


def main() -> None:
    runtime = ModelRuntime()
    print("Loading EfficientSAM, Grounding DINO, Qwen2.5-VL, and SD3 Inpaint...")
    try:
        print(f"Model preload complete: {runtime.preload_all()}")
    except Exception:
        runtime.unload_all()
        raise

    demo = create_demo(EditChatService(runtime=runtime))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port_value = os.getenv("GRADIO_SERVER_PORT", "").strip()
    # Let Gradio select an open local port unless the user explicitly pins one.
    server_port = int(port_value) if port_value else None
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    demo.queue(default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    main()
