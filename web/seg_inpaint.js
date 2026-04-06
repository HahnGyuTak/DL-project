const API_KEY = "efficientsam_api_url_v1";
const DEFAULT_API_URL = "http://127.0.0.1:8000";

const els = {
  apiUrl: document.getElementById("apiUrl"),
  saveApiBtn: document.getElementById("saveApiBtn"),
  healthBtn: document.getElementById("healthBtn"),
  modeSelect: document.getElementById("modeSelect"),
  labelSelect: document.getElementById("labelSelect"),
  inputSize: document.getElementById("inputSize"),
  imageInput: document.getElementById("imageInput"),
  undoBtn: document.getElementById("undoBtn"),
  clearBtn: document.getElementById("clearBtn"),
  runSegBtn: document.getElementById("runSegBtn"),
  status: document.getElementById("status"),
  inputCanvas: document.getElementById("inputCanvas"),
  segResultImg: document.getElementById("segResultImg"),
  maskPreviewImg: document.getElementById("maskPreviewImg"),
  segCropGrid: document.getElementById("segCropGrid"),
  editPrompt: document.getElementById("editPrompt"),
  negativePrompt: document.getElementById("negativePrompt"),
  maxSideInput: document.getElementById("maxSideInput"),
  stepsInput: document.getElementById("stepsInput"),
  guidanceInput: document.getElementById("guidanceInput"),
  seedInput: document.getElementById("seedInput"),
  maskExpandInput: document.getElementById("maskExpandInput"),
  runInpaintBtn: document.getElementById("runInpaintBtn"),
  inpaintResultImg: document.getElementById("inpaintResultImg"),
};

const state = {
  imageFile: null,
  imageObj: null,
  points: [],
  labels: [],
  boxPoints: [],
  maskPngB64: "",
};

const ctx = els.inputCanvas.getContext("2d");

function setStatus(msg) {
  els.status.textContent = msg;
}

function looksLikeIpv4(host) {
  return /^(?:\d{1,3}\.){3}\d{1,3}$/.test(host);
}

function normalizeApiUrl(raw) {
  const v = (raw || "").trim();
  if (!v) return "";
  if (/^https?:\/\//i.test(v)) return v.replace(/\/+$/, "");

  const hostPort = v.split("/")[0];
  const host = hostPort.split(":")[0].toLowerCase();
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";
  const hasPort = /:\d+$/.test(hostPort);
  const isIp = looksLikeIpv4(host);
  const scheme = isLocal || hasPort || isIp ? "http" : "https";
  return `${scheme}://${v}`.replace(/\/+$/, "");
}

function getApiBase() {
  return normalizeApiUrl(els.apiUrl.value) || DEFAULT_API_URL;
}

function saveApiBase() {
  const base = getApiBase();
  localStorage.setItem(API_KEY, base);
  els.apiUrl.value = base;
  setStatus(`API URL 저장됨: ${base}`);
}

function assertMixedContentSafe(base) {
  const api = new URL(base);
  const pageIsHttps = window.location.protocol === "https:";
  const apiIsHttp = api.protocol === "http:";
  const isLoopback = api.hostname === "localhost" || api.hostname === "127.0.0.1";
  if (pageIsHttps && apiIsHttp && !isLoopback) {
    throw new Error(
      "HTTPS 페이지에서 HTTP API 호출은 브라우저가 차단됩니다. API를 HTTPS로 열거나, 로컬 http 페이지에서 실행하세요."
    );
  }
}

function drawCanvas() {
  const img = state.imageObj;
  if (!img) {
    ctx.clearRect(0, 0, els.inputCanvas.width, els.inputCanvas.height);
    return;
  }

  els.inputCanvas.width = img.width;
  els.inputCanvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  for (let i = 0; i < state.points.length; i++) {
    const [x, y] = state.points[i];
    const label = state.labels[i];
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = label === 1 ? "#28c76f" : "#ef4444";
    ctx.stroke();
  }

  if (state.boxPoints.length === 1) {
    const [x, y] = state.boxPoints[0];
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#f59e0b";
    ctx.stroke();
  }

  if (state.boxPoints.length >= 2) {
    const [p0, p1] = state.boxPoints;
    const x = Math.min(p0[0], p1[0]);
    const y = Math.min(p0[1], p1[1]);
    const w = Math.abs(p1[0] - p0[0]);
    const h = Math.abs(p1[1] - p0[1]);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
  }
}

function clearPrompts() {
  state.points = [];
  state.labels = [];
  state.boxPoints = [];
  drawCanvas();
}

function clearDerivedOutputs() {
  state.maskPngB64 = "";
  els.segResultImg.removeAttribute("src");
  els.maskPreviewImg.removeAttribute("src");
  els.inpaintResultImg.removeAttribute("src");
  clearSegmentCrops();
}

function toCanvasCoords(evt) {
  const rect = els.inputCanvas.getBoundingClientRect();
  const scaleX = els.inputCanvas.width / rect.width;
  const scaleY = els.inputCanvas.height / rect.height;
  const x = Math.round((evt.clientX - rect.left) * scaleX);
  const y = Math.round((evt.clientY - rect.top) * scaleY);
  return [x, y];
}

function clearSegmentCrops() {
  els.segCropGrid.innerHTML = "";
}

function showSegmentCropEmpty(msg) {
  clearSegmentCrops();
  const p = document.createElement("p");
  p.className = "crop-empty";
  p.textContent = msg;
  els.segCropGrid.appendChild(p);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed"));
    img.src = url;
  });
}

function computeMaskBBox(maskData, w, h, threshold = 127) {
  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const v = maskData[i];
      if (v > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return {
    x: minX,
    y: minY,
    w: maxX - minX + 1,
    h: maxY - minY + 1,
  };
}

async function renderSegmentCrop(maskPngB64) {
  if (!state.imageObj) {
    showSegmentCropEmpty("원본 이미지를 먼저 업로드하세요.");
    return;
  }

  if (!maskPngB64) {
    showSegmentCropEmpty("마스크 결과가 없습니다.");
    return;
  }

  const src = state.imageObj;
  const w = src.width;
  const h = src.height;
  const maskImg = await loadImage(`data:image/png;base64,${maskPngB64}`);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = w;
  maskCanvas.height = h;
  const maskCtx = maskCanvas.getContext("2d");
  maskCtx.drawImage(maskImg, 0, 0, w, h);

  const maskData = maskCtx.getImageData(0, 0, w, h).data;
  const box = computeMaskBBox(maskData, w, h, 127);
  if (!box) {
    showSegmentCropEmpty("세그멘테이션 영역을 찾지 못했습니다.");
    return;
  }

  const objectCanvas = document.createElement("canvas");
  objectCanvas.width = box.w;
  objectCanvas.height = box.h;
  const objCtx = objectCanvas.getContext("2d");
  objCtx.drawImage(src, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);

  const srcImageData = objCtx.getImageData(0, 0, box.w, box.h);
  const srcPx = srcImageData.data;
  const maskPx = maskCtx.getImageData(box.x, box.y, box.w, box.h).data;
  for (let i = 0; i < box.w * box.h; i++) {
    srcPx[i * 4 + 3] = maskPx[i * 4];
  }
  objCtx.putImageData(srcImageData, 0, 0);

  const outCanvas = document.createElement("canvas");
  const maxSide = 220;
  const scale = Math.min(1, maxSide / Math.max(box.w, box.h));
  outCanvas.width = Math.max(1, Math.round(box.w * scale));
  outCanvas.height = Math.max(1, Math.round(box.h * scale));
  const outCtx = outCanvas.getContext("2d");
  outCtx.fillStyle = "#ffffff";
  outCtx.fillRect(0, 0, outCanvas.width, outCanvas.height);
  outCtx.drawImage(objectCanvas, 0, 0, outCanvas.width, outCanvas.height);

  clearSegmentCrops();
  const card = document.createElement("article");
  card.className = "crop-card";
  const meta = document.createElement("p");
  meta.className = "crop-meta";
  meta.textContent = "1. segmented object";
  card.appendChild(meta);
  card.appendChild(outCanvas);
  els.segCropGrid.appendChild(card);
}

function b64ToFile(base64, filename, mimeType = "image/png") {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return new File([bytes], filename, { type: mimeType });
}

els.inputCanvas.addEventListener("click", (evt) => {
  if (!state.imageObj) return;
  const mode = els.modeSelect.value;
  const [x, y] = toCanvasCoords(evt);

  if (mode === "point") {
    state.points.push([x, y]);
    state.labels.push(parseInt(els.labelSelect.value, 10));
    setStatus(`point 추가: (${x}, ${y}), 총 ${state.points.length}개`);
  } else {
    if (state.boxPoints.length >= 2) state.boxPoints = [];
    state.boxPoints.push([x, y]);
    setStatus(`box corner 추가: (${x}, ${y}) ${state.boxPoints.length}/2`);
  }
  drawCanvas();
});

els.imageInput.addEventListener("change", () => {
  const file = els.imageInput.files?.[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    state.imageFile = file;
    state.imageObj = img;
    clearPrompts();
    clearDerivedOutputs();
    drawCanvas();
    setStatus(`이미지 로드됨: ${file.name} (${img.width}x${img.height})`);
  };
  img.src = URL.createObjectURL(file);
});

els.undoBtn.addEventListener("click", () => {
  if (els.modeSelect.value === "point") {
    if (state.points.length > 0) {
      state.points.pop();
      state.labels.pop();
      setStatus(`point 제거, 남은 ${state.points.length}개`);
    }
  } else {
    if (state.boxPoints.length > 0) {
      state.boxPoints.pop();
      setStatus(`box corner 제거, 남은 ${state.boxPoints.length}개`);
    }
  }
  drawCanvas();
});

els.clearBtn.addEventListener("click", () => {
  clearPrompts();
  setStatus("프롬프트 초기화 완료");
});

els.saveApiBtn.addEventListener("click", () => {
  saveApiBase();
});

els.healthBtn.addEventListener("click", async () => {
  const base = getApiBase();
  try {
    assertMixedContentSafe(base);
    const [segRes, inpRes] = await Promise.all([
      fetch(`${base}/health`),
      fetch(`${base}/health/inpaint`),
    ]);
    if (!segRes.ok) throw new Error(`segment health HTTP ${segRes.status}`);
    if (!inpRes.ok) throw new Error(`inpaint health HTTP ${inpRes.status}`);
    const segData = await segRes.json();
    const inpData = await inpRes.json();
    setStatus(
      `health ok | seg_device=${segData.device} | inpaint_device=${inpData.device} | inpaint_loaded=${inpData.loaded}`
    );
  } catch (e) {
    setStatus(`health 실패: ${e}`);
  }
});

els.runSegBtn.addEventListener("click", async () => {
  if (!state.imageFile) {
    setStatus("이미지를 먼저 업로드하세요.");
    return;
  }

  const mode = els.modeSelect.value;
  if (mode === "point" && state.points.length < 1) {
    setStatus("point 모드에서는 최소 1개 포인트가 필요합니다.");
    return;
  }
  if (mode === "box" && state.boxPoints.length !== 2) {
    setStatus("box 모드에서는 코너 2개가 필요합니다.");
    return;
  }

  const points = mode === "point" ? state.points : state.boxPoints;
  const labels = mode === "point" ? state.labels : [];

  const base = getApiBase();
  const fd = new FormData();
  fd.append("image", state.imageFile);
  fd.append("mode", mode);
  fd.append("points_json", JSON.stringify(points));
  fd.append("labels_json", JSON.stringify(labels));
  fd.append("input_size", String(parseInt(els.inputSize.value, 10) || 1024));

  setStatus("세그멘테이션 실행 중...");

  try {
    assertMixedContentSafe(base);
    const res = await fetch(`${base}/segment`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} ${err}`);
    }

    const data = await res.json();
    state.maskPngB64 = data.mask_png_b64 || "";
    els.segResultImg.src = `data:image/png;base64,${data.overlay_png_b64}`;
    els.maskPreviewImg.src = `data:image/png;base64,${data.mask_png_b64}`;
    els.inpaintResultImg.removeAttribute("src");
    await renderSegmentCrop(state.maskPngB64);

    setStatus(
      `세그멘테이션 완료 | device=${data.device} | checkpoint=${data.checkpoint} | best_idx=${data.best_idx}`
    );
  } catch (e) {
    if (e instanceof TypeError) {
      setStatus("실패: 네트워크 연결 오류입니다. API URL, 포트 오픈, CORS/HTTPS 설정을 확인하세요.");
      return;
    }
    setStatus(`실패: ${e}`);
  }
});

els.runInpaintBtn.addEventListener("click", async () => {
  if (!state.imageFile) {
    setStatus("이미지를 먼저 업로드하세요.");
    return;
  }
  if (!state.maskPngB64) {
    setStatus("먼저 세그멘테이션을 실행해서 마스크를 생성하세요.");
    return;
  }

  const prompt = (els.editPrompt.value || "").trim();
  if (!prompt) {
    setStatus("Edit Prompt를 입력하세요.");
    return;
  }

  const base = getApiBase();
  const fd = new FormData();
  fd.append("image", state.imageFile);
  fd.append("mask", b64ToFile(state.maskPngB64, "mask.png"));
  fd.append("prompt", prompt);
  fd.append("negative_prompt", (els.negativePrompt.value || "").trim());
  fd.append("num_inference_steps", String(parseInt(els.stepsInput.value || "30", 10)));
  fd.append("guidance_scale", String(parseFloat(els.guidanceInput.value || "7.0")));
  fd.append("seed", String(parseInt(els.seedInput.value || "-1", 10)));
  fd.append("mask_expand_px", String(parseInt(els.maskExpandInput.value || "12", 10)));
  fd.append("max_side", String(parseInt(els.maxSideInput.value || "1024", 10)));

  setStatus("SD3 인페인팅 실행 중... (첫 요청은 모델 로딩으로 오래 걸릴 수 있음)");

  try {
    assertMixedContentSafe(base);
    const res = await fetch(`${base}/inpaint/sd3`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} ${err}`);
    }
    const data = await res.json();
    els.inpaintResultImg.src = `data:image/png;base64,${data.output_png_b64}`;
    setStatus(`인페인팅 완료 | model=${data.model_id} | device=${data.device} | dtype=${data.dtype}`);
  } catch (e) {
    if (e instanceof TypeError) {
      setStatus("실패: 네트워크 연결 오류입니다. API URL, 포트 오픈, CORS/HTTPS 설정을 확인하세요.");
      return;
    }
    setStatus(`실패: ${e}`);
  }
});

(function init() {
  const saved = localStorage.getItem(API_KEY) || DEFAULT_API_URL;
  els.apiUrl.value = saved;
  clearSegmentCrops();
  setStatus("이미지를 업로드하고 세그멘테이션 후, 프롬프트로 SD3 인페인팅을 실행하세요.");
})();
