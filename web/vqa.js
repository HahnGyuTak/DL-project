const API_KEY = "efficientsam_api_url_v1";
const DEFAULT_API_URL = "http://localhost:8000";
const EMPTY_IMAGE_DATA_URL = "data:image/gif;base64,R0lGODlhAgABAIABAP///wAAACwAAAAAAQABAAACAkQBADs=";

const els = {
  apiUrl: document.getElementById("apiUrl"),
  saveApiBtn: document.getElementById("saveApiBtn"),
  healthBtn: document.getElementById("healthBtn"),
  unloadBtn: document.getElementById("unloadBtn"),
  status: document.getElementById("status"),
  imageInput: document.getElementById("imageInput"),
  questionInput: document.getElementById("questionInput"),
  maxTokensInput: document.getElementById("maxTokensInput"),
  temperatureInput: document.getElementById("temperatureInput"),
  askBtn: document.getElementById("askBtn"),
  preview: document.getElementById("vqaImagePreview"),
  answerBox: document.getElementById("answerBox"),
};

const state = {
  imageFile: null,
};

function setStatus(msg) {
  els.status.textContent = msg;
}

function setEmptyImage(imgEl) {
  imgEl.src = EMPTY_IMAGE_DATA_URL;
}

function looksLikeIpv4(host) {
  return /^(?:\d{1,3}\.){3}\d{1,3}$/.test(host);
}

function normalizeApiUrl(raw) {
  const v = (raw || "").trim();
  if (!v) return "";
  if (/^https?:\/\//i.test(v)) {
    const url = new URL(v);
    if (url.hostname === "127.0.0.1" || url.hostname === "0.0.0.0") {
      url.hostname = "localhost";
    }
    return url.toString().replace(/\/+$/, "");
  }

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

function clearAnswer() {
  els.answerBox.textContent = "";
}

els.imageInput.addEventListener("change", () => {
  const file = els.imageInput.files?.[0];
  if (!file) return;
  state.imageFile = file;
  els.preview.src = URL.createObjectURL(file);
  clearAnswer();
  setStatus(`이미지 로드됨: ${file.name}`);
});

els.saveApiBtn.addEventListener("click", () => {
  saveApiBase();
});

els.healthBtn.addEventListener("click", async () => {
  const base = getApiBase();
  try {
    assertMixedContentSafe(base);
    const res = await fetch(`${base}/health/vqa`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus(
      `health ok | model=${data.model} | device=${data.device} | loaded=${data.loaded}`
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

els.askBtn.addEventListener("click", async () => {
  if (!state.imageFile) {
    setStatus("이미지를 먼저 업로드하세요.");
    return;
  }

  const question = (els.questionInput.value || "").trim();
  if (!question) {
    setStatus("질문을 입력하세요.");
    return;
  }

  const maxNewTokens = parseInt(els.maxTokensInput?.value || "128", 10);
  const temperature = parseFloat(els.temperatureInput?.value || "0.2");

  const fd = new FormData();
  fd.append("image", state.imageFile);
  fd.append("question", question);
  fd.append("max_new_tokens", String(maxNewTokens));
  fd.append("temperature", String(temperature));

  const base = getApiBase();
  setStatus("LLaVA 답변 생성 중...");
  clearAnswer();

  try {
    assertMixedContentSafe(base);
    const res = await fetch(`${base}/vqa/llava`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} ${err}`);
    }

    const data = await res.json();
    els.answerBox.textContent = data.answer || "(empty answer)";
    setStatus(`완료 | model=${data.model_id} | device=${data.device} | dtype=${data.dtype}`);
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
  setEmptyImage(els.preview);
  setStatus("이미지를 업로드하고 질문을 입력한 뒤 Ask LLaVA를 누르세요.");
})();
