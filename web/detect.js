const API_KEY = "efficientsam_api_url_v1";
const DEFAULT_API_URL = "http://127.0.0.1:8000";
const EMPTY_IMAGE_DATA_URL = "data:image/gif;base64,R0lGODlhAgABAIABAP///wAAACwAAAAAAQABAAACAkQBADs=";
const DEFAULT_ASPECT_RATIO = "2 / 1";

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
  cropGrid: document.getElementById("cropGrid"),
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

function clearCrops() {
  els.cropGrid.innerHTML = "";
}

function clipBox(box, imgW, imgH) {
  const x0 = Math.min(box[0], box[2]);
  const y0 = Math.min(box[1], box[3]);
  const x1 = Math.max(box[0], box[2]);
  const y1 = Math.max(box[1], box[3]);

  let sx = Math.max(0, Math.floor(x0));
  let sy = Math.max(0, Math.floor(y0));
  let ex = Math.min(imgW, Math.ceil(x1));
  let ey = Math.min(imgH, Math.ceil(y1));

  if (ex <= sx) ex = Math.min(imgW, sx + 1);
  if (ey <= sy) ey = Math.min(imgH, sy + 1);

  const sw = Math.max(1, ex - sx);
  const sh = Math.max(1, ey - sy);
  return { sx, sy, sw, sh };
}

async function ensureInputImageLoaded() {
  if (els.inputImg.complete && els.inputImg.naturalWidth > 0) return;
  await new Promise((resolve, reject) => {
    const onLoad = () => {
      els.inputImg.removeEventListener("load", onLoad);
      els.inputImg.removeEventListener("error", onError);
      resolve();
    };
    const onError = () => {
      els.inputImg.removeEventListener("load", onLoad);
      els.inputImg.removeEventListener("error", onError);
      reject(new Error("input image load failed"));
    };
    els.inputImg.addEventListener("load", onLoad, { once: true });
    els.inputImg.addEventListener("error", onError, { once: true });
  });
}

async function renderCrops(detections) {
  clearCrops();
  if (!detections || detections.length === 0) {
    const empty = document.createElement("p");
    empty.className = "crop-empty";
    empty.textContent = "검출된 객체가 없습니다.";
    els.cropGrid.appendChild(empty);
    return;
  }

  await ensureInputImageLoaded();
  const imgW = els.inputImg.naturalWidth;
  const imgH = els.inputImg.naturalHeight;

  detections.forEach((det, idx) => {
    const { sx, sy, sw, sh } = clipBox(det.box_xyxy, imgW, imgH);
    const card = document.createElement("article");
    card.className = "crop-card";

    const meta = document.createElement("p");
    meta.className = "crop-meta";
    meta.textContent = `${idx + 1}. ${det.label} (${det.score.toFixed(3)})`;
    card.appendChild(meta);

    const canvas = document.createElement("canvas");
    const maxSide = 220;
    const scale = Math.min(1, maxSide / Math.max(sw, sh));
    canvas.width = Math.max(1, Math.round(sw * scale));
    canvas.height = Math.max(1, Math.round(sh * scale));

    const ctx = canvas.getContext("2d");
    ctx.drawImage(els.inputImg, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    card.appendChild(canvas);

    const boxInfo = document.createElement("p");
    boxInfo.className = "crop-box";
    boxInfo.textContent = `box: [${sx}, ${sy}, ${sx + sw}, ${sy + sh}]`;
    card.appendChild(boxInfo);

    els.cropGrid.appendChild(card);
  });
}

els.imageInput.addEventListener("change", () => {
  const file = els.imageInput.files?.[0];
  if (!file) return;
  state.imageFile = file;
  const inputUrl = URL.createObjectURL(file);
  const probeUrl = URL.createObjectURL(file);
  els.inputImg.src = inputUrl;
  els.inputImg.onload = () => URL.revokeObjectURL(inputUrl);
  els.inputImg.onerror = () => URL.revokeObjectURL(inputUrl);

  const probe = new Image();
  probe.onload = () => {
    setImageBoxAspect(els.inputImg, probe.width, probe.height);
    setImageBoxAspect(els.resultImg, probe.width, probe.height);
    URL.revokeObjectURL(probeUrl);
  };
  probe.onerror = () => {
    setImageBoxAspect(els.inputImg, 0, 0);
    setImageBoxAspect(els.resultImg, 0, 0);
    URL.revokeObjectURL(probeUrl);
  };
  probe.src = probeUrl;

  setEmptyImage(els.resultImg);
  els.detList.textContent = "";
  clearCrops();
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
    await renderCrops(data.detections);
    setStatus(
      `완료 | model=${data.model_id} | device=${data.device} | detections=${data.num_detections} | prompt="${data.text_prompt}"`
    );
  } catch (e) {
    clearCrops();
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
  setImageBoxAspect(els.inputImg, 0, 0);
  setImageBoxAspect(els.resultImg, 0, 0);
  setEmptyImage(els.inputImg);
  setEmptyImage(els.resultImg);
  clearCrops();
  setStatus("이미지를 업로드하고 labels를 입력한 뒤 Run Detection을 누르세요.");
})();
