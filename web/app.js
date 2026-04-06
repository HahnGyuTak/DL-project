const STORAGE_KEY = "model_dock_custom_models_v1";

const baseModels = [
  {
    id: "efficientsam-hf",
    name: "EfficientSAM Demo",
    task: "segmentation",
    desc: "이미지 업로드 후 point/box 프롬프트로 세그멘테이션",
    demoUrl: "https://yunyangx-efficientsam.hf.space",
    immutable: true,
  },
  {
    id: "groundingdino-slot",
    name: "Grounding DINO Slot",
    task: "open-vocabulary detection",
    desc: "추후 URL만 입력해 바로 연결할 수 있는 슬롯",
    demoUrl: "",
    immutable: true,
  },
];

const els = {
  modelList: document.getElementById("modelList"),
  modelCount: document.getElementById("modelCount"),
  activeModelBadge: document.getElementById("activeModelBadge"),
  viewer: document.getElementById("viewer"),
  searchInput: document.getElementById("searchInput"),
  modelForm: document.getElementById("modelForm"),
  nameInput: document.getElementById("nameInput"),
  taskInput: document.getElementById("taskInput"),
  urlInput: document.getElementById("urlInput"),
  descInput: document.getElementById("descInput"),
  addPresetBtn: document.getElementById("addPresetBtn"),
  clearCustomBtn: document.getElementById("clearCustomBtn"),
};

let activeModelId = null;
let customModels = loadCustomModels();

function loadCustomModels() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveCustomModels() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(customModels));
}

function allModels() {
  return [...baseModels, ...customModels];
}

function filteredModels() {
  const q = els.searchInput.value.trim().toLowerCase();
  if (!q) return allModels();
  return allModels().filter((m) =>
    `${m.name} ${m.task} ${m.desc}`.toLowerCase().includes(q)
  );
}

function escapeHtml(str) {
  return (str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderViewer(model) {
  if (!model || !model.demoUrl) {
    els.viewer.className = "viewer empty";
    els.viewer.innerHTML = `
      <div class="empty-state">
        <h3>데모 URL이 아직 없습니다</h3>
        <p>오른쪽 Add / Edit Model 영역에서 URL을 등록하면 바로 열립니다.</p>
      </div>
    `;
    els.activeModelBadge.textContent = model ? `${model.name} (URL 없음)` : "선택 없음";
    return;
  }

  els.viewer.className = "viewer";
  els.viewer.innerHTML = `<iframe src="${model.demoUrl}" title="${escapeHtml(model.name)}" loading="lazy" referrerpolicy="no-referrer"></iframe>`;
  els.activeModelBadge.textContent = `${model.name} · ${model.task}`;
}

function onDeleteModel(id) {
  customModels = customModels.filter((m) => m.id !== id);
  saveCustomModels();

  if (activeModelId === id) {
    activeModelId = null;
    renderViewer(null);
  }

  renderModelList();
}

function renderModelList() {
  const models = filteredModels();
  els.modelCount.textContent = `${models.length} models`;

  if (!models.length) {
    els.modelList.innerHTML = `<p class="model-task">검색 결과가 없습니다.</p>`;
    return;
  }

  els.modelList.innerHTML = models
    .map((model) => {
      const activeClass = model.id === activeModelId ? "active" : "";
      const hasUrl = !!model.demoUrl;
      const state = hasUrl ? "LIVE" : "SET URL";
      const removeButton = model.immutable
        ? ""
        : `<button class="btn danger" data-action="delete" data-id="${model.id}">삭제</button>`;

      return `
        <article class="model-card ${activeClass}" data-action="select" data-id="${model.id}">
          <div class="model-top">
            <span class="model-name">${escapeHtml(model.name)}</span>
            <span class="pill ${hasUrl ? "" : "muted"}">${state}</span>
          </div>
          <div class="model-task">${escapeHtml(model.task)}</div>
          <div class="model-desc">${escapeHtml(model.desc || "설명 없음")}</div>
          <div class="model-actions">${removeButton}</div>
        </article>
      `;
    })
    .join("");
}

function selectModelById(id) {
  const model = allModels().find((m) => m.id === id);
  activeModelId = model ? id : null;
  renderModelList();
  renderViewer(model || null);
}

function generateId() {
  return `custom-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

els.modelList.addEventListener("click", (e) => {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;

  const actionEl = target.closest("[data-action]");
  if (!actionEl) return;

  const action = actionEl.getAttribute("data-action");
  const id = actionEl.getAttribute("data-id");
  if (!id) return;

  if (action === "delete") {
    e.stopPropagation();
    onDeleteModel(id);
    return;
  }

  if (action === "select") {
    selectModelById(id);
  }
});

els.searchInput.addEventListener("input", () => {
  renderModelList();
});

els.modelForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const name = els.nameInput.value.trim();
  const task = els.taskInput.value.trim();
  const demoUrl = els.urlInput.value.trim();
  const desc = els.descInput.value.trim();

  if (!name || !task || !demoUrl) {
    alert("모델 이름, 태스크, URL을 입력해주세요.");
    return;
  }

  customModels.unshift({
    id: generateId(),
    name,
    task,
    demoUrl,
    desc,
    immutable: false,
  });

  saveCustomModels();
  renderModelList();
  selectModelById(customModels[0].id);
  els.modelForm.reset();
});

els.clearCustomBtn.addEventListener("click", () => {
  const ok = confirm("커스텀 모델을 모두 지울까요?");
  if (!ok) return;
  customModels = [];
  saveCustomModels();
  activeModelId = null;
  renderModelList();
  renderViewer(null);
});

els.addPresetBtn.addEventListener("click", () => {
  const exists = customModels.some((m) => m.demoUrl === "http://localhost:7861");
  if (!exists) {
    customModels.unshift({
      id: generateId(),
      name: "EfficientSAM (Local Server)",
      task: "segmentation",
      demoUrl: "http://localhost:7861",
      desc: "로컬/서버에서 실행 중인 EfficientSAM Gradio 앱 연결",
      immutable: false,
    });
    saveCustomModels();
  }

  renderModelList();
  const model = customModels.find((m) => m.demoUrl === "http://localhost:7861");
  if (model) selectModelById(model.id);
});

renderModelList();
selectModelById(baseModels[0].id);
