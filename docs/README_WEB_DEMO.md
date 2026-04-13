# DL-project: Web Demo

`pip install -r requirements_api.txt`

## 1. 백엔드 서버 실행

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

## 2. 프론트 실행

새 터미널에서:

```bash
python -m http.server 8080 -d web
```

브라우저 접속:

- 로컬: `http://127.0.0.1:8080`
- 같은 네트워크의 다른 PC: `http://<서버사설IP>:8080`

---

## 3. 헬스체크

백엔드 정상 여부:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/detector
curl http://127.0.0.1:8000/health/vqa
curl http://127.0.0.1:8000/health/inpaint
```


## 4. 자주 발생하는 문제

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

## 5. Cloudflare Pages 배포(선택)

정적 프론트만 배포됩니다. (모델 추론 백엔드는 별도 서버 필요)

```bash
npm i
npx wrangler pages deploy web --project-name model-dock --branch main
```

배포 후에도 웹에서 `API URL`은 실제 백엔드 주소로 설정해야 합니다.

[Cloudflare Pages 배포 세부 가이드라인](./README_CLOUDFLARE.md)
