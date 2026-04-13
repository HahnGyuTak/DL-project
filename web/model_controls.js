(() => {
  const API_KEY = "efficientsam_api_url_v1";
  const DEFAULT_API_URL = "http://127.0.0.1:8000";
  const MODEL_LABELS = {
    segmentation: "SAM",
    detector: "DINO",
    vqa: "LLaVA",
    inpaint: "SD3",
  };

  const panel = document.getElementById("modelControlPanel");
  if (!panel) return;

  const statusEl = document.getElementById("status");
  const apiInput = document.getElementById("apiUrl");
  const modelKeys = (panel.dataset.models || "segmentation,detector,vqa,inpaint")
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
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
    return normalizeApiUrl(apiInput?.value) || localStorage.getItem(API_KEY) || DEFAULT_API_URL;
  }

  function assertMixedContentSafe(base) {
    const api = new URL(base);
    const pageIsHttps = window.location.protocol === "https:";
    const apiIsHttp = api.protocol === "http:";
    const isLoopback = api.hostname === "localhost" || api.hostname === "127.0.0.1";
    if (pageIsHttps && apiIsHttp && !isLoopback) {
      throw new Error("HTTPS 페이지에서는 HTTP API 호출이 차단됩니다. HTTPS 터널 API URL을 사용하세요.");
    }
  }

  function memoryText(cuda) {
    if (!cuda?.available || !cuda.devices?.length) return "CUDA unavailable";
    return cuda.devices.map((gpu) => `cuda:${gpu.index} ${gpu.used_mb}MB`).join(" · ");
  }

  function render(models) {
    panel.innerHTML = "";
    panel.setAttribute("aria-label", "GPU model controls");

    for (const key of modelKeys) {
      const model = models.find((item) => item.key === key) || { key, loaded: false, device: "" };
      const button = document.createElement("button");
      button.type = "button";
      button.className = `model-toggle ${model.loaded ? "on" : "off"}`;
      button.setAttribute("aria-pressed", model.loaded ? "true" : "false");
      button.title = `${MODEL_LABELS[key] || key} ${model.loaded ? "unload" : "load"}`;

      const label = document.createElement("span");
      label.className = "model-toggle-name";
      label.textContent = MODEL_LABELS[key] || key;

      const state = document.createElement("span");
      state.className = "model-toggle-state";
      state.textContent = model.loaded ? "ON" : "OFF";

      button.append(label, state);
      if (model.loaded) {
        const device = document.createElement("span");
        device.className = "model-toggle-device";
        device.textContent = model.device || "loaded";
        button.appendChild(device);
      }

      button.addEventListener("click", () => mutateModel(key, model.loaded ? "unload" : "load"));
      panel.appendChild(button);
    }
  }

  async function refreshModels(silent = false) {
    const base = getApiBase();
    try {
      assertMixedContentSafe(base);
      const res = await fetch(`${base}/models`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render(data.models || []);
      if (!silent) setStatus(`모델 상태 업데이트 완료 | ${memoryText(data.cuda)}`);
    } catch (e) {
      panel.innerHTML = `<span class="model-toggle-error">model off?</span>`;
      if (!silent) setStatus(`모델 상태 실패: ${e}`);
    }
  }

  async function mutateModel(key, action) {
    const base = getApiBase();
    const label = MODEL_LABELS[key] || key;
    try {
      assertMixedContentSafe(base);
      setStatus(`${label} ${action === "load" ? "로드" : "언로드"} 중...`);
      const res = await fetch(`${base}/models/${key}/${action}`, { method: "POST" });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(`HTTP ${res.status} ${err}`);
      }
      const data = await res.json();
      await refreshModels(true);
      setStatus(`${label} ${action === "load" ? "ON" : "OFF"} 완료 | ${memoryText(data.cuda)}`);
    } catch (e) {
      setStatus(`${label} ${action === "load" ? "로드" : "언로드"} 실패: ${e}`);
    }
  }

  panel.addEventListener("dblclick", () => refreshModels(false));
  refreshModels(true);
})();
