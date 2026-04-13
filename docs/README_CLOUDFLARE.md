# Cloudflare Pages 배포 가이드

## 프로젝트 구조

- `web/`: 배포할 정적 사이트
- `web/index.html`, `web/styles.css`, `web/app.js`

## 1) Cloudflare 로그인

```bash
wrangler login
```

## 2) Pages 프로젝트 생성 (최초 1회)

```bash
wrangler pages project create model-dock --production-branch main
```

## 3) 배포

```bash
wrangler pages deploy web --project-name model-dock
```

성공하면 `https://<project>.pages.dev` URL이 발급됩니다.

## 4) 업데이트 배포

```bash
wrangler pages deploy web --project-name model-dock
```

## 참고

- 이 페이지는 정적 프론트엔드입니다.
- 실제 모델 추론 서버(Gradio, HF Space, API)는 별도 URL로 연결합니다.
- Model Dock 페이지에서 데모 URL을 등록/수정해 즉시 통합할 수 있습니다.
