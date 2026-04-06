# EfficientSAM Web Demo

`merve/EfficientSAM` 기반 포인트/박스 프롬프트 세그멘테이션 웹 데모입니다.

## 1) 로컬 실행

```bash
cd /mnt/data/DL-project
pip install -r requirements.txt
python app.py
```

브라우저에서 `http://localhost:7860` 접속.

## 2) 사용 방법

- 이미지를 업로드
- `Prompt Mode` 선택
  - `point`: `Point Label`을 `foreground/background`로 바꿔가며 클릭
  - `box`: 이미지에 두 번 클릭해서 박스 코너 지정
- `Run Segmentation` 클릭

## 3) Hugging Face Spaces 배포

1. Hugging Face에서 `Gradio` Space 생성
2. 이 폴더의 파일 업로드
   - `app.py`
   - `requirements.txt`
3. Space가 자동 빌드되면 바로 웹 데모로 사용 가능

## 참고

- 앱은 시작 시 GPU 사용 가능하면 `efficient_sam_s_gpu.jit`를 로드하고,
  실패하면 자동으로 CPU 체크포인트로 폴백합니다.
