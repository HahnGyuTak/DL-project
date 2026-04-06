const API_KEY = "efficientsam_api_url_v1";
const DEFAULT_API_URL = "http://127.0.0.1:8000";

const els = {
  apiUrl: document.getElementById("apiUrl"),
  saveApiBtn: document.getElementById("saveApiBtn"),
  healthBtn: document.getElementById("healthBtn"),
  labelsInput: document.getElementById("labelsInput"),
  thresholdInput: document.getElementById("thresholdInput"),
  textThresholdInput: document.getElementById("textThresholdInput"),
  imageInput: document.getElementById("imageInput"),
  runBtn: document.getElementById("runBtn"),
  status: document.getElementById("status"),
  inputImg: document.getElementById("inputImg"),
  resultImg: document.getElementById("resultImg"),
  detList: document.getElementById("detList"),
};

const state = {
  imageFile: null,
};

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

function setDetectionList(detections) {
  if (!detections || detections.length === 0) {
    els.detList.textContent = "(no detections)";
    return;
  }

  const rows = detections.map((d, i) => {
    const b = d.box_xyxy.map((x) => x.toFixed(1)).join(", ");
    return `${i + 1}. ${d.label} | score=${d.score.toFixed(4)} | box=[${b}]`;
  });
  els.detList.textContent = rows.join("\n");
}

els.imageInput.addEventListener("change", () => {
  const file = els.imageInput.files?.[0];
  if (!file) return;
  state.imageFile = file;
  const url = URL.createObjectURL(file);
  els.inputImg.src = url;
  els.resultImg.removeAttribute("src");
  els.detList.textContent = "";
  setStatus(`이미지 로드됨: ${file.name}`);
});

els.saveApiBtn.addEventListener("click", () => {
  saveApiBase();
});

els.healthBtn.addEventListener("click", async () => {
  const base = getApiBase();
  try {
    assertMixedContentSafe(base);
    const res = await fetch(`${base}/health/detector`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus(`health ok | model=${data.model} | device=${data.device}`);
  } catch (e) {
    setStatus(`health 실패: ${e}`);
  }
});

els.runBtn.addEventListener("click", async () => {
  if (!state.imageFile) {
    setStatus("이미지를 먼저 업로드하세요.");
    return;
  }

  const labelsCsv = (els.labelsInput.value || "").trim();
  if (!labelsCsv) {
    setStatus("labels를 입력하세요. 예: cat, remote control, sofa");
    return;
  }

  const threshold = parseFloat(els.thresholdInput.value || "0.35");
  const textThreshold = parseFloat(els.textThresholdInput.value || "0.25");

  const fd = new FormData();
  fd.append("image", state.imageFile);
  fd.append("labels_csv", labelsCsv);
  fd.append("threshold", String(threshold));
  fd.append("text_threshold", String(textThreshold));

  const base = getApiBase();
  setStatus("디텍션 실행 중...");

  try {
    assertMixedContentSafe(base);
    const res = await fetch(`${base}/detect/open-vocab`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} ${err}`);
    }

    const data = await res.json();
    els.resultImg.src = `data:image/png;base64,${data.overlay_png_b64}`;
    setDetectionList(data.detections);
    setStatus(
      `완료 | model=${data.model_id} | device=${data.device} | detections=${data.num_detections} | prompt="${data.text_prompt}"`
    );
  } catch (e) {
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
  setStatus("이미지를 업로드하고 labels를 입력한 뒤 Run Detection을 누르세요.");
})();
