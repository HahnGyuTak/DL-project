# Model Dock From-Scratch Demo

외부 Hugging Face 데모 iframe 없이, 이 레포 코드로 직접 동작하는 구성입니다.

## 구성

- `api_server.py`: FastAPI 백엔드
  - `/segment`: EfficientSAM 세그멘테이션
  - `/detect/open-vocab`: Grounding DINO open-vocabulary detection
  - `/vqa/llava`: LLaVA 1.5 7B 기반 VQA
- `web/`: 정적 데모 프론트엔드 (Cloudflare Pages 배포 대상)
  - `index.html`: Segmentation 페이지
  - `detection.html`: Detection 페이지
  - `vqa.html`: LLaVA VQA 페이지
  - 상단 탭으로 페이지 전환

## 1) 백엔드 실행

```bash
cd /mnt/data/DL-project
pip install -r requirements_api.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

헬스체크:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/detector
curl http://127.0.0.1:8000/health/vqa
```

## 2) 프론트 실행 (로컬 미리보기)

```bash
cd /mnt/data/DL-project
python -m http.server 8080 -d web
```

브라우저에서 `http://127.0.0.1:8080` 접속 후,
- API URL에 `http://127.0.0.1:8000` 입력
- Segmentation 탭: 이미지 업로드 + 포인트/박스 클릭 + Run
- Detection 탭: 이미지 업로드 + labels 입력 + Run
- LLaVA VQA 탭: 이미지 업로드 + 질문 입력 + Ask LLaVA

## 3) Cloudflare Pages 배포

```bash
cd /mnt/data/DL-project
npx wrangler pages deploy web --project-name model-dock --branch main
```

주의:
- Pages는 정적 호스팅이므로 `api_server.py`는 별도 서버에서 실행되어야 합니다.
- 배포 페이지에서 API URL을 해당 서버 주소로 입력해야 동작합니다.
