# Standalone MLLM Image Edit Demo

`demo.py`는 FastAPI, Cloudflare Pages 없이 로컬에서 직접 실행하는 Gradio 기반 이미지 편집 챗봇이다.

## Run

```bash
pip install -r requirements.txt
python demo.py
```

실행 로그에 출력된 포트를 확인한 뒤 브라우저에서 `http://127.0.0.1:<port>`를 연다. 기본적으로 7860부터 비어 있는 포트를 자동 선택한다.

UI를 열기 전에 EfficientSAM, Grounding DINO, Qwen2.5-VL, SD3 Inpaint를 모두 GPU에 preload한다. 따라서 첫 실행은 모델 로딩 때문에 시간이 걸리지만, UI가 열린 뒤에는 모델 로딩 대기 없이 작업을 시작할 수 있다.

포트나 외부 접속 주소를 바꾸려면 다음 환경 변수를 사용한다.

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python demo.py
```

## Chat Flow

1. 이미지를 업로드한다.
2. `강아지를 수정하고 싶어.`처럼 수정할 개체를 말한다.
3. Grounding DINO와 EfficientSAM이 대상 mask 및 overlay를 만든다.
4. 수정 요청을 입력하면 Qwen2.5-VL이 SD3용 inpainting prompt를 제안한다.
5. `수정 진행` 버튼 또는 `진행` 메시지로 승인하면 SD3 Inpaint가 결과를 만든다.
6. 추가 요청을 입력하면 현재 결과 이미지를 기준으로 같은 과정을 반복한다.

## Model Sources

- EfficientSAM: `merve/EfficientSAM`
- Open-vocabulary detection: `IDEA-Research/grounding-dino-tiny`
- MLLM: local `/mnt/data1/models/qwen/Qwen2.5-VL-7B-Instruct` if available, otherwise `Qwen/Qwen2.5-VL-7B-Instruct`
- Inpainting: `IrohXu/stable-diffusion-3-inpainting` + `stabilityai/stable-diffusion-3-medium-diffusers`

`QWEN_VL_MODEL_ID` 환경 변수로 Qwen 모델 경로 또는 Hugging Face 모델 ID를 바꿀 수 있다. SD3 base model access requires any Hugging Face access approval required by the model card.
