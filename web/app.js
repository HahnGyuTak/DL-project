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
  runBtn: document.getElementById("runBtn"),
  status: document.getElementById("status"),
  inputCanvas: document.getElementById("inputCanvas"),
  resultImg: document.getElementById("resultImg"),
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

function normalizeApiUrl(raw) {
  const v = (raw || "").trim();
  if (!v) return "";
  return /^https?:\/\//i.test(v) ? v : `https://${v}`;
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
    els.resultImg.removeAttribute("src");
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
    setStatus(
      `완료 | device=${data.device} | checkpoint=${data.checkpoint} | best_idx=${data.best_idx} | ious=${JSON.stringify(data.ious)}`
    );
  } catch (e) {
    setStatus(`실패: ${e}`);
  }
});

(function init() {
  const saved = localStorage.getItem(API_KEY) || DEFAULT_API_URL;
  els.apiUrl.value = saved;
  setStatus("이미지를 업로드하고 프롬프트를 클릭하세요.");
})();
