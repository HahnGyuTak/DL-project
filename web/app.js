const STORAGE_KEY = "efficientsam_demo_url_v1";
const DEFAULT_URL = "https://yunyangx-efficientsam.hf.space";

const demoUrlInput = document.getElementById("demoUrl");
const saveBtn = document.getElementById("saveBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const viewerEl = document.getElementById("viewer");
const openNewTabEl = document.getElementById("openNewTab");

function normalizeUrl(raw) {
  const v = (raw || "").trim();
  if (!v) return "";
  return /^https?:\/\//i.test(v) ? v : `https://${v}`;
}

function getSavedUrl() {
  return localStorage.getItem(STORAGE_KEY) || DEFAULT_URL;
}

function setSavedUrl(url) {
  localStorage.setItem(STORAGE_KEY, url);
}

function renderViewer(url) {
  if (!url) {
    viewerEl.innerHTML = `<div class="empty">백엔드 URL을 입력하면 데모가 표시됩니다.</div>`;
    openNewTabEl.href = "#";
    return;
  }

  viewerEl.innerHTML = `<iframe src="${url}" title="EfficientSAM Demo" loading="lazy" referrerpolicy="no-referrer"></iframe>`;
  openNewTabEl.href = url;
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function applyUrl(raw, persist = true) {
  const url = normalizeUrl(raw);
  if (!url) {
    setStatus("URL이 비어 있습니다.");
    return;
  }

  try {
    new URL(url);
  } catch {
    setStatus("유효한 URL 형식이 아닙니다.");
    return;
  }

  if (persist) setSavedUrl(url);
  demoUrlInput.value = url;
  renderViewer(url);
  setStatus(`현재 연결 URL: ${url}`);
}

saveBtn.addEventListener("click", () => {
  applyUrl(demoUrlInput.value, true);
});

resetBtn.addEventListener("click", () => {
  applyUrl(DEFAULT_URL, true);
});

demoUrlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    applyUrl(demoUrlInput.value, true);
  }
});

applyUrl(getSavedUrl(), false);
