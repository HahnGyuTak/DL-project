(() => {
  const API_KEY = "efficientsam_api_url_v1";
  const DEFAULT_API_URL = "http://127.0.0.1:8000";
  const MODEL_LABELS = {
    segmentation: "EfficientSAM",
    detector: "Grounding DINO",
    vqa: "LLaVA",
    inpaint: "SD3 Inpaint",
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
    return cuda.devices
      .map((gpu) => `GPU${gpu.index}: used ${gpu.used_mb}MB / reserved ${gpu.reserved_mb}MB`)
      .join(" · ");
  }

  function render(models, cuda) {
    panel.innerHTML = "";
    const heading = document.createElement("div");
    heading.className = "model-control-head";
    heading.innerHTML = `<strong>GPU Model Controls</strong><span>${memoryText(cuda)}</span>`;
    panel.appendChild(heading);

    const grid = document.createElement("div");
    grid.className = "model-control-grid";

    for (const key of modelKeys) {
      const model = models.find((item) => item.key === key) || { key, loaded: false, device: "unknown" };
      const card = document.createElement("article");
      card.className = "model-control-card";

      const title = document.createElement("div");
      title.className = "model-control-title";
      title.textContent = MODEL_LABELS[key] || model.name || key;

      const badge = document.createElement("span");
      badge.className = `model-badge ${model.loaded ? "on" : "off"}`;
      const onLabel = String(model.device || "").startsWith("cuda") ? "GPU ON" : "CPU ON";
      badge.textContent = model.loaded ? onLabel : "OFF";

      const meta = document.createElement("p");
      meta.className = "model-control-meta";
      meta.textContent = `${model.device || "device?"}${model.dtype ? ` · ${model.dtype}` : ""}`;

      const actions = document.createElement("div");
      actions.className = "model-control-actions";

      const loadBtn = document.createElement("button");
      loadBtn.type = "button";
      loadBtn.className = "small";
      loadBtn.textContent = "Load";
      loadBtn.disabled = !!model.loaded;
      loadBtn.addEventListener("click", () => mutateModel(key, "load"));

      const unloadBtn = document.createElement("button");
      unloadBtn.type = "button";
      unloadBtn.className = "ghost small danger-soft";
      unloadBtn.textContent = "Unload";
      unloadBtn.disabled = !model.loaded;
      unloadBtn.addEventListener("click", () => mutateModel(key, "unload"));

      actions.append(loadBtn, unloadBtn);
      card.append(title, badge, meta, actions);
      grid.appendChild(card);
    }

    panel.appendChild(grid);
  }

  async function refreshModels(silent = false) {
    const base = getApiBase();
    try {
      assertMixedContentSafe(base);
      const res = await fetch(`${base}/models`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render(data.models || [], data.cuda);
      if (!silent) setStatus(`모델 상태 업데이트 완료 | ${memoryText(data.cuda)}`);
    } catch (e) {
      panel.innerHTML = `<p class="model-control-error">모델 상태를 불러오지 못함: ${e}</p>`;
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
      setStatus(`${label} ${action === "load" ? "로드" : "언로드"} 완료 | ${memoryText(data.cuda)}`);
    } catch (e) {
      setStatus(`${label} ${action === "load" ? "로드" : "언로드"} 실패: ${e}`);
    }
  }

  panel.addEventListener("dblclick", () => refreshModels(false));
  refreshModels(true);
})();
