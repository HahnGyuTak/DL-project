const API_KEY = "efficientsam_api_url_v1";
const DEFAULT_API_URL = "http://127.0.0.1:8000";
const EMPTY_IMAGE_DATA_URL = "data:image/gif;base64,R0lGODlhAgABAIABAP///wAAACwAAAAAAQABAAACAkQBADs=";
const DEFAULT_ASPECT_RATIO = "2 / 1";

const els = {
  apiUrl: document.getElementById("apiUrl"),
  saveApiBtn: document.getElementById("saveApiBtn"),
  healthBtn: document.getElementById("healthBtn"),
  unloadBtn: document.getElementById("unloadBtn"),
  modeSelect: document.getElementById("modeSelect"),
  labelSelect: document.getElementById("labelSelect"),
  inputSize: document.getElementById("inputSize"),
  imageInput: document.getElementById("imageInput"),
  undoBtn: document.getElementById("undoBtn"),
  clearBtn: document.getElementById("clearBtn"),
  runSegBtn: document.getElementById("runSegBtn"),
  status: document.getElementById("status"),
  inputCanvas: document.getElementById("inputCanvas"),
  viewOverlayBtn: document.getElementById("viewOverlayBtn"),
  viewMaskBtn: document.getElementById("viewMaskBtn"),
  viewCutoutBtn: document.getElementById("viewCutoutBtn"),
  segViewImg: document.getElementById("segViewImg"),
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
  currentView: "overlay",
  viewImages: {
    overlay: "",
    mask: "",
    cutout: "",
  },
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

function setUnloadStatus(data) {
  const unloaded = Object.entries(data.unloaded || {})
    .filter(([, didUnload]) => didUnload)
    .map(([name]) => name)
    .join(", ") || "none";
  const before = (data.cuda_memory_before || []).reduce((sum, gpu) => sum + (gpu.used_mib || 0), 0);
  const after = (data.cuda_memory_after || []).reduce((sum, gpu) => sum + (gpu.used_mib || 0), 0);
  const freed = Math.max(0, before - after);
  setStatus(`모델 언로드 완료 | unloaded=${unloaded} | freed≈${freed.toFixed(1)} MiB`);
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
  state.viewImages.overlay = "";
  state.viewImages.mask = "";
  state.viewImages.cutout = "";
  setEmptyImage(els.segViewImg);
  setEmptyImage(els.inpaintResultImg);
}

function toCanvasCoords(evt) {
  const rect = els.inputCanvas.getBoundingClientRect();
  const scaleX = els.inputCanvas.width / rect.width;
  const scaleY = els.inputCanvas.height / rect.height;
  const x = Math.round((evt.clientX - rect.left) * scaleX);
  const y = Math.round((evt.clientY - rect.top) * scaleY);
  return [x, y];
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed"));
    img.src = url;
  });
}

function setActiveView(view) {
  state.currentView = view;
  const map = {
    overlay: els.viewOverlayBtn,
    mask: els.viewMaskBtn,
    cutout: els.viewCutoutBtn,
  };
  Object.values(map).forEach((btn) => btn.classList.remove("active"));
  if (map[view]) map[view].classList.add("active");
}

function renderSegView() {
  const src = state.viewImages[state.currentView] || "";
  if (!src) {
    setEmptyImage(els.segViewImg);
    return;
  }
  els.segViewImg.src = src;
}

async function buildFullSizeCutout(maskPngB64) {
  if (!state.imageObj || !maskPngB64) return "";
  const src = state.imageObj;
  const w = src.width;
  const h = src.height;
  const maskImg = await loadImage(`data:image/png;base64,${maskPngB64}`);

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = w;
  srcCanvas.height = h;
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.drawImage(src, 0, 0, w, h);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = w;
  maskCanvas.height = h;
  const maskCtx = maskCanvas.getContext("2d");
  maskCtx.drawImage(maskImg, 0, 0, w, h);

  const srcImageData = srcCtx.getImageData(0, 0, w, h);
  const srcPx = srcImageData.data;
  const maskPx = maskCtx.getImageData(0, 0, w, h).data;

  const outCanvas = document.createElement("canvas");
  outCanvas.width = w;
  outCanvas.height = h;
  const outCtx = outCanvas.getContext("2d");
  outCtx.fillStyle = "#ffffff";
  outCtx.fillRect(0, 0, w, h);
  const outData = outCtx.getImageData(0, 0, w, h);
  const outPx = outData.data;

  for (let i = 0; i < w * h; i++) {
    const m = maskPx[i * 4];
    if (m > 127) {
      outPx[i * 4] = srcPx[i * 4];
      outPx[i * 4 + 1] = srcPx[i * 4 + 1];
      outPx[i * 4 + 2] = srcPx[i * 4 + 2];
      outPx[i * 4 + 3] = 255;
    }
  }

  outCtx.putImageData(outData, 0, 0);
  return outCanvas.toDataURL("image/png");
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
    setImageBoxAspect(els.segViewImg, img.width, img.height);
    setImageBoxAspect(els.inpaintResultImg, img.width, img.height);
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

els.viewOverlayBtn.addEventListener("click", () => {
  setActiveView("overlay");
  renderSegView();
});

els.viewMaskBtn.addEventListener("click", () => {
  setActiveView("mask");
  renderSegView();
});

els.viewCutoutBtn.addEventListener("click", () => {
  setActiveView("cutout");
  renderSegView();
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

els.unloadBtn.addEventListener("click", async () => {
  const base = getApiBase();
  try {
    assertMixedContentSafe(base);
    setStatus("GPU 모델 언로드 중...");
    const res = await fetch(`${base}/admin/unload-models`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setUnloadStatus(await res.json());
  } catch (e) {
    setStatus(`모델 언로드 실패: ${e}`);
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
    state.viewImages.overlay = `data:image/png;base64,${data.overlay_png_b64}`;
    state.viewImages.mask = `data:image/png;base64,${data.mask_png_b64}`;
    state.viewImages.cutout = await buildFullSizeCutout(state.maskPngB64);
    setActiveView("overlay");
    renderSegView();
    setEmptyImage(els.inpaintResultImg);

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
  setActiveView("overlay");
  setImageBoxAspect(els.segViewImg, 0, 0);
  setImageBoxAspect(els.inpaintResultImg, 0, 0);
  setEmptyImage(els.segViewImg);
  setEmptyImage(els.inpaintResultImg);
  renderSegView();
  setStatus("이미지를 업로드하고 세그멘테이션 후, 프롬프트로 SD3 인페인팅을 실행하세요.");
})();
