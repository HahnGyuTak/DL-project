const API_KEY = "efficientsam_api_url_v1";
const DEFAULT_API_URL = "http://127.0.0.1:8000";
const EMPTY_IMAGE_DATA_URL = "data:image/gif;base64,R0lGODlhAgABAIABAP///wAAACwAAAAAAQABAAACAkQBADs=";
const DEFAULT_ASPECT_RATIO = "2 / 1";

const els = {
  apiUrl: document.getElementById("apiUrl"),
  saveApiBtn: document.getElementById("saveApiBtn"),
  healthBtn: document.getElementById("healthBtn"),
  status: document.getElementById("status"),
  imageInput: document.getElementById("imageInput"),
  chatImageView: document.getElementById("chatImageView"),
  viewCurrentBtn: document.getElementById("viewCurrentBtn"),
  viewOverlayBtn: document.getElementById("viewOverlayBtn"),
  viewMaskBtn: document.getElementById("viewMaskBtn"),
  chatMessages: document.getElementById("chatMessages"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  sendBtn: document.getElementById("sendBtn"),
  approveBtn: document.getElementById("approveBtn"),
};

const state = {
  imageFile: null,
  sessionId: "",
  stage: "idle",
  currentView: "current",
  images: {
    current: "",
    overlay: "",
    mask: "",
  },
};

function setStatus(msg) {
  els.status.textContent = msg;
}

function setImageBoxAspect(width, height) {
  if (width > 0 && height > 0) {
    els.chatImageView.style.aspectRatio = `${width} / ${height}`;
    return;
  }
  els.chatImageView.style.aspectRatio = DEFAULT_ASPECT_RATIO;
}

function setEmptyImage() {
  els.chatImageView.src = EMPTY_IMAGE_DATA_URL;
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
    throw new Error("HTTPS 페이지에서 HTTP API 호출은 브라우저가 차단됩니다. API를 HTTPS로 열거나, 로컬 http 페이지에서 실행하세요.");
  }
}

function addMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `chat-msg ${role}`;
  msg.textContent = text;
  els.chatMessages.appendChild(msg);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function addThinkingMessage() {
  const msg = document.createElement("div");
  msg.className = "chat-msg assistant typing-indicator";
  msg.setAttribute("role", "status");
  msg.setAttribute("aria-label", "모델이 작업 중입니다");
  msg.innerHTML = `
    <span class="typing-dots" aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </span>
  `;
  els.chatMessages.appendChild(msg);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return msg;
}

function removeThinkingMessage(msg) {
  if (msg?.parentNode) msg.remove();
}

function setActiveView(view) {
  state.currentView = view;
  const map = {
    current: els.viewCurrentBtn,
    overlay: els.viewOverlayBtn,
    mask: els.viewMaskBtn,
  };
  Object.values(map).forEach((btn) => btn.classList.remove("active"));
  map[view]?.classList.add("active");
  renderImageView();
}

function renderImageView() {
  const src = state.images[state.currentView] || state.images.current || "";
  if (!src) {
    setEmptyImage();
    return;
  }
  els.chatImageView.src = src;
}

function updateFromResponse(data) {
  state.sessionId = data.session_id || state.sessionId;
  state.stage = data.stage || state.stage;

  if (data.overlay_png_b64) {
    state.images.overlay = `data:image/png;base64,${data.overlay_png_b64}`;
  }
  if (data.mask_png_b64) {
    state.images.mask = `data:image/png;base64,${data.mask_png_b64}`;
  }
  if (data.output_png_b64) {
    state.images.current = `data:image/png;base64,${data.output_png_b64}`;
    setActiveView("current");
  } else if (data.overlay_png_b64) {
    setActiveView("overlay");
  } else {
    renderImageView();
  }

  if (data.assistant_message) addMessage("assistant", data.assistant_message);
  els.approveBtn.hidden = state.stage !== "awaiting_approval";
  setStatus(`stage=${state.stage}${data.target_label ? ` | target=${data.target_label}` : ""}`);
}

async function postMessage(message) {
  const base = getApiBase();
  assertMixedContentSafe(base);

  const fd = new FormData();
  fd.append("message", message);

  let url = "";
  if (!state.sessionId) {
    if (!state.imageFile) throw new Error("이미지를 먼저 업로드하세요.");
    fd.append("image", state.imageFile);
    url = `${base}/chat/edit/sessions`;
  } else {
    url = `${base}/chat/edit/sessions/${state.sessionId}/messages`;
  }

  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} ${err}`);
  }
  return res.json();
}

async function submitMessage(message, options = {}) {
  const text = (message || "").trim();
  if (!text) return;
  if (!options.silentUser) addMessage("user", text);
  els.chatInput.value = "";
  els.sendBtn.disabled = true;
  els.approveBtn.disabled = true;
  setStatus("챗봇 처리 중...");

  let thinkingMsg = null;
  try {
    thinkingMsg = addThinkingMessage();
    const data = await postMessage(text);
    removeThinkingMessage(thinkingMsg);
    thinkingMsg = null;
    updateFromResponse(data);
  } catch (e) {
    removeThinkingMessage(thinkingMsg);
    thinkingMsg = null;
    if (e instanceof TypeError) {
      setStatus("실패: 네트워크 연결 오류입니다. API URL, 포트 오픈, CORS/HTTPS 설정을 확인하세요.");
    } else {
      setStatus(`실패: ${e}`);
    }
  } finally {
    removeThinkingMessage(thinkingMsg);
    els.sendBtn.disabled = false;
    els.approveBtn.disabled = false;
  }
}

els.imageInput.addEventListener("change", () => {
  const file = els.imageInput.files?.[0];
  if (!file) return;

  state.imageFile = file;
  state.sessionId = "";
  state.stage = "idle";
  state.images.overlay = "";
  state.images.mask = "";
  state.images.current = URL.createObjectURL(file);
  els.chatMessages.innerHTML = "";
  els.approveBtn.hidden = true;

  const probe = new Image();
  probe.onload = () => setImageBoxAspect(probe.width, probe.height);
  probe.onerror = () => setImageBoxAspect(0, 0);
  probe.src = state.images.current;

  setActiveView("current");
  addMessage("assistant", "이미지를 받았어요. 수정하고 싶은 개체를 말해주세요.");
  setStatus(`이미지 로드됨: ${file.name}`);
});

els.saveApiBtn.addEventListener("click", saveApiBase);

els.healthBtn.addEventListener("click", async () => {
  const base = getApiBase();
  try {
    assertMixedContentSafe(base);
    const res = await fetch(`${base}/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const loaded = (data.models || []).filter((m) => m.loaded).map((m) => `${m.key}:${m.device}`).join(", ") || "none";
    setStatus(`health ok | loaded=${loaded}`);
  } catch (e) {
    setStatus(`health 실패: ${e}`);
  }
});

els.chatForm.addEventListener("submit", (evt) => {
  evt.preventDefault();
  submitMessage(els.chatInput.value);
});

els.approveBtn.addEventListener("click", () => {
  addMessage("user", "진행");
  submitMessage("진행", { silentUser: true });
});

els.viewCurrentBtn.addEventListener("click", () => setActiveView("current"));
els.viewOverlayBtn.addEventListener("click", () => setActiveView("overlay"));
els.viewMaskBtn.addEventListener("click", () => setActiveView("mask"));

(function init() {
  const saved = localStorage.getItem(API_KEY) || DEFAULT_API_URL;
  els.apiUrl.value = saved;
  setImageBoxAspect(0, 0);
  setEmptyImage();
  addMessage("assistant", "이미지를 업로드하고 '강아지를 수정하고 싶어'처럼 시작해보세요.");
  setStatus("MLLM 편집 챗봇 준비 완료");
})();
