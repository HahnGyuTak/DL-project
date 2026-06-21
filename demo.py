"""Launch the standalone MLLM image-edit chatbot without FastAPI or Cloudflare."""

from __future__ import annotations

import os

from code.gradio_ui import create_demo


def main() -> None:
    demo = create_demo()
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    demo.queue(default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    main()
