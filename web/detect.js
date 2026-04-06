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
    setStatus(`실패: ${e}`);
  }
});

(function init() {
  const saved = localStorage.getItem(API_KEY) || DEFAULT_API_URL;
  els.apiUrl.value = saved;
  setStatus("이미지를 업로드하고 labels를 입력한 뒤 Run Detection을 누르세요.");
})();
