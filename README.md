# DL-project: From-Scratch Vision Demo Suite

이 저장소는 Hugging Face 데모 iframe 없이, 로컬 백엔드 + 정적 웹으로 아래 기능을 제공합니다.

- EfficientSAM 세그멘테이션
- Grounding DINO Open-vocabulary Detection
- LLaVA VQA
- Segmentation + SD3 Inpainting

---

## 1. 요구사항

- OS: Linux 권장 (Ubuntu 계열)
- Python 3.10+
- `pip`
- (선택) NVIDIA GPU + CUDA 환경
- (선택) Cloudflare Pages 배포 시 `node`, `npm`, `wrangler`

참고:
- GPU가 있으면 서버가 자동으로 여유 GPU를 선택합니다.
- 모델은 첫 실행 시 Hugging Face에서 다운로드되므로 시간이 오래 걸릴 수 있습니다.

---

## 2. 클론

```bash
git clone https://github.com/HahnGyuTak/DL-project.git
cd DL-project
```

---

## 3. Python 환경 준비

가상환경 사용을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_api.txt
```

---

## 4. 백엔드 서버 실행

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

백그라운드 실행:

```bash
nohup python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 > uvicorn_api.log 2>&1 &
```

서버 중지:

```bash
pkill -f "uvicorn api_server:app"
```

---

## 5. 프론트 실행

새 터미널에서:

```bash
python -m http.server 8080 -d web
```

브라우저 접속:

- 로컬: `http://127.0.0.1:8080`
- 같은 네트워크의 다른 PC: `http://<서버사설IP>:8080`

---

## 6. 헬스체크

백엔드 정상 여부:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/detector
curl http://127.0.0.1:8000/health/vqa
curl http://127.0.0.1:8000/health/inpaint
```

---

## 7. 웹 사용 순서

웹 상단 탭에서 페이지를 전환합니다.

### 7.1 EfficientSAM Segmentation

1. 이미지 업로드
2. 포인트/박스 프롬프트 입력
3. `Run Segmentation`

### 7.2 Grounding DINO Detection

1. 이미지 업로드
2. 라벨 입력 (`cat, remote control, sofa` 형태)
3. `Run Detection`

### 7.3 LLaVA VQA

1. 이미지 업로드
2. 질문 입력
3. `Ask LLaVA`

### 7.4 Seg + SD3 Inpaint

1. 이미지 업로드
2. 세그멘테이션 수행 (`Run Segmentation`)
3. `Segmentation View`에서 `Overlay / Auto Mask / Segmented Crop` 클릭 전환
4. 편집 프롬프트 입력
5. `Run SD3 Inpaint`

추가 옵션:
- `Mask Expand(px)`: 세그멘테이션 마스크 경계 확장

---

## 8. 자주 발생하는 문제

### `TypeError: Failed to fetch`

- API URL이 잘못됐거나, 브라우저의 HTTPS/HTTP 혼합 정책에 막힌 경우입니다.
- `https://...pages.dev`에서 API를 `http://...`로 호출하면 차단될 수 있습니다.

### 첫 요청이 너무 느림

- 정상입니다. 모델 다운로드/초기 로딩 시간입니다.
- 특히 LLaVA/SD3는 첫 요청이 오래 걸립니다.

### SD3 inpaint 실패/메모리 부족

- 다른 프로세스가 GPU 메모리를 점유 중일 수 있습니다.
- `nvidia-smi`로 확인 후 정리하거나, 입력 해상도/steps를 낮추세요.

### 포트 충돌

- 8000 또는 8080이 이미 사용 중이면 다른 포트로 실행하세요.

---

## 9. Cloudflare Pages 배포(선택)

정적 프론트만 배포됩니다. (모델 추론 백엔드는 별도 서버 필요)

```bash
npm i
npx wrangler pages deploy web --project-name model-dock --branch main
```

배포 후에도 웹에서 `API URL`은 실제 백엔드 주소로 설정해야 합니다.

---

## 10. 참고 문서

- 웹 데모 요약: `README_WEB_DEMO.md`
- Cloudflare 배포 가이드: `README_CLOUDFLARE.md`
