const API_KEY = "efficientsam_api_url_v1";
const DEFAULT_API_URL = "http://127.0.0.1:8000";
const EMPTY_IMAGE_DATA_URL = "data:image/gif;base64,R0lGODlhAgABAIABAP///wAAACwAAAAAAQABAAACAkQBADs=";
const DEFAULT_ASPECT_RATIO = "2 / 1";

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
  runBtn: document.getElementById("runBtn"),
  status: document.getElementById("status"),
  inputCanvas: document.getElementById("inputCanvas"),
  resultImg: document.getElementById("resultImg"),
  segCropGrid: document.getElementById("segCropGrid"),
};

const state = {
  imageFile: null,
  imageObj: null,
  points: [],
  labels: [],
  boxPoints: [],
};

const ctx = els.inputCanvas.getContext("2d");

function setStatus(msg) {
  els.status.textContent = msg;
}

function setEmptyImage(imgEl) {
  imgEl.src = EMPTY_IMAGE_DATA_URL;
}

function setImageBoxAspect(imgEl, width, height) {
  if (width > 0 && height > 0) {
    imgEl.style.aspectRatio = `${width} / ${height}`;
    return;
  }
  imgEl.style.aspectRatio = DEFAULT_ASPECT_RATIO;
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

  // Draw point prompts
  for (let i = 0; i < state.points.length; i++) {
    const [x, y] = state.points[i];
    const label = state.labels[i];
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = label === 1 ? "#28c76f" : "#ef4444";
    ctx.stroke();
  }

  // Draw box prompt
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

  // Canvas compositing uses source alpha, so grayscale mask colors alone do not cut out.
  // Apply mask luminance directly to alpha channel for true cutout.
  const srcImageData = objCtx.getImageData(0, 0, box.w, box.h);
  const srcPx = srcImageData.data;
  const maskPx = maskCtx.getImageData(box.x, box.y, box.w, box.h).data;
  for (let i = 0; i < box.w * box.h; i++) {
    const maskVal = maskPx[i * 4]; // 0..255
    srcPx[i * 4 + 3] = maskVal; // alpha
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

  const boxInfo = document.createElement("p");
  boxInfo.className = "crop-box";
  boxInfo.textContent = `box: [${box.x}, ${box.y}, ${box.x + box.w}, ${box.y + box.h}]`;
  card.appendChild(boxInfo);

  els.segCropGrid.appendChild(card);
}

function clearPrompts() {
  state.points = [];
  state.labels = [];
  state.boxPoints = [];
  drawCanvas();
}

function toCanvasCoords(evt) {
  const rect = els.inputCanvas.getBoundingClientRect();
  const scaleX = els.inputCanvas.width / rect.width;
  const scaleY = els.inputCanvas.height / rect.height;

  const x = Math.round((evt.clientX - rect.left) * scaleX);
  const y = Math.round((evt.clientY - rect.top) * scaleY);
  return [x, y];
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
    setImageBoxAspect(els.resultImg, img.width, img.height);
    setEmptyImage(els.resultImg);
    clearSegmentCrops();
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
    const res = await fetch(`${base}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus(`health ok | device=${data.device} | checkpoint=${data.checkpoint}`);
  } catch (e) {
    setStatus(`health 실패: ${e}`);
  }
});

els.runBtn.addEventListener("click", async () => {
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
    const res = await fetch(`${base}/segment`, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} ${err}`);
    }

    const data = await res.json();
    els.resultImg.src = `data:image/png;base64,${data.overlay_png_b64}`;
    await renderSegmentCrop(data.mask_png_b64);
    setStatus(
      `완료 | device=${data.device} | checkpoint=${data.checkpoint} | best_idx=${data.best_idx} | ious=${JSON.stringify(data.ious)}`
    );
  } catch (e) {
    clearSegmentCrops();
    if (e instanceof TypeError) {
      setStatus(
        "실패: 네트워크 연결 오류입니다. API URL, 포트 오픈, CORS/HTTPS 설정을 확인하세요."
      );
      return;
    }
    setStatus(`실패: ${e}`);
  }
});

(function init() {
  const saved = localStorage.getItem(API_KEY) || DEFAULT_API_URL;
  els.apiUrl.value = saved;
  setImageBoxAspect(els.resultImg, 0, 0);
  setEmptyImage(els.resultImg);
  clearSegmentCrops();
  setStatus("이미지를 업로드하고 프롬프트를 클릭하세요.");
})();
