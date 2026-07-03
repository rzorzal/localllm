"use strict";
// Model Manager SPA — vanilla JS, talks to the token-guarded /admin/* API.
// The admin token is injected in-memory by the host webview as window.__ADMIN_TOKEN__.

const TOKEN = window.__ADMIN_TOKEN__ || "";
const view = document.getElementById("view");
const crumbs = document.getElementById("crumbs");
const activeEl = document.getElementById("active");
const toastEl = document.getElementById("toast");

let families = [];          // FamilyView[]
let pollTimer = null;
// Re-renders the pane the user is currently on, re-resolving from the latest
// `families` so a background refresh (or external hot-swap) updates badges
// without losing the user's place. Set by each render*() below.
let currentView = renderFamilies;
// JSON of the last-rendered families, so the background tick only re-renders
// (and risks scroll reset) when the data actually changed.
let lastFamiliesJson = "";

function authHeaders(extra) {
  return Object.assign({ "x-admin-token": TOKEN }, extra || {});
}

async function api(method, path, body) {
  const opts = { method, headers: authHeaders(body ? { "content-type": "application/json" } : {}) };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.status === 401 ? "Unauthorized — restart the app"
            : res.status === 409 ? "Conflict (in use / switch in progress)"
            : `Request failed (${res.status})`;
    try { const j = await res.json(); if (j && j.error) msg = j.error; } catch (_) {}
    throw new Error(msg);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res.text();
}

function toast(msg, isErr) {
  toastEl.textContent = msg;
  toastEl.className = "toast" + (isErr ? " err" : "");
  toastEl.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { toastEl.hidden = true; }, 4000);
}

const STATUS_LABEL = { in_use: "In use", downloaded: "Downloaded", needs_download: "Needs download" };
const FIT_LABEL = { fits: "Fits", tight: "Tight", wont_fit: "Won't fit" };
const gb = (mb) => (mb / 1024).toFixed(1) + " GB";

async function refresh() {
  families = await api("GET", "/admin/models");
  lastFamiliesJson = JSON.stringify(families);
  await refreshStatusHeader();
}

// Background sync: re-fetch the catalog and, if it changed, re-render whatever
// pane is open so an external switch (or a switch finished elsewhere) is
// reflected. Skips while a switch we started is actively polling.
async function tick() {
  if (pollTimer) return;
  let next;
  try { next = await api("GET", "/admin/models"); }
  catch (_) { return; }
  await refreshStatusHeader();
  const j = JSON.stringify(next);
  if (j !== lastFamiliesJson) {
    lastFamiliesJson = j;
    families = next;
    currentView();
  }
}

async function refreshStatusHeader() {
  try {
    const s = await api("GET", "/admin/model/status");
    const cur = s.current ? `${s.current.repo.split("/").pop()} / ${s.current.file}` : "—";
    if (s.state === "switching") {
      activeEl.className = "active switching";
      activeEl.innerHTML = `switching… <b>${s.phase}</b> ${s.progress || 0}%`;
    } else {
      activeEl.className = "active";
      activeEl.innerHTML = `Active: <b>${cur}</b>`;
    }
    return s;
  } catch (e) { activeEl.textContent = e.message; return null; }
}

function setCrumbs(parts) {
  crumbs.innerHTML = "";
  parts.forEach((p, i) => {
    if (i) crumbs.append(document.createTextNode(" / "));
    if (p.onClick) { const a = document.createElement("a"); a.textContent = p.label; a.onclick = p.onClick; crumbs.append(a); }
    else crumbs.append(document.createTextNode(p.label));
  });
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

// ---- Pane 0: Config landing ----
const NAV_CARDS = [
  { route: "/models", icon: "◈", title: "Models", desc: "Escolher, baixar e configurar modelos locais" },
  { route: "/tools", icon: "⛭", title: "Tools", desc: "Filtrar tools por cliente" },
  { route: "/budget", icon: "$", title: "Budget", desc: "Teto de gasto cloud por dia" },
  { route: "/dashboard", icon: "▤", title: "Dashboard", desc: "Roteamento local↔cloud e tokens economizados" },
];

function renderConfig() {
  currentView = renderConfig;
  setCrumbs([{ label: "Config" }]);
  const shell = el("div", "config-home");
  shell.append(el("div", "config-lead", "Configuração"));

  const grid = el("div", "navgrid");
  NAV_CARDS.forEach((c, i) => {
    const card = el("div", "navcard");
    card.style.setProperty("--i", i);
    card.append(el("div", "navicon", c.icon));
    const body = el("div", "navbody");
    body.append(el("h3", null, c.title));
    body.append(el("div", "navdesc", c.desc));
    card.append(body);
    card.append(el("div", "navgo", "→"));
    card.onclick = () => navigate(c.route);
    grid.append(card);
  });
  shell.append(grid);

  view.innerHTML = "";
  view.append(shell);
  renderRoutingSelector(shell);
  renderThresholdConfig(shell);
  renderSmartHistoryToggle(shell);
  renderIntegrationToggle(shell);
}

// Balanced-profile difficulty cutoff, as a percentage. GET/POST /admin/threshold.
// Higher % = harder to escalate = more local; lower = more cloud.
async function renderThresholdConfig(container) {
  const panel = el("div", "intpanel");
  panel.append(el("div", "intpanel-title", "Limiar do Balanced"));
  panel.append(el("div", "intpanel-sub",
    "Dificuldade do pedido acima deste ponto vai para a cloud. Maior % = mais local; menor % = mais cloud. Usado no perfil Balanced."));

  const row = el("div", "thresh-row");
  const range = el("input", "thresh-range");
  range.type = "range"; range.min = 0; range.max = 100; range.step = 1;
  const num = el("div", "thresh-num");
  const pct = el("span", "thresh-pct", "—");
  num.append(pct, el("span", "thresh-unit", "%"));
  row.append(range, num);
  panel.append(row);
  const status = el("div", "intpanel-status");
  panel.append(status);
  container.append(panel);

  const paint = (p) => { range.value = p; pct.textContent = p; };
  try { const r = await api("GET", "/admin/threshold"); paint(r.percent); }
  catch (e) { status.textContent = e.message; return; }

  let busy = false;
  range.oninput = () => { pct.textContent = range.value; };
  range.onchange = async () => {
    if (busy) return;
    busy = true; range.disabled = true;
    try {
      const r = await api("POST", "/admin/threshold", { percent: Number(range.value) });
      paint(r.percent); toast(`Limiar em ${r.percent}%`);
    } catch (e) { toast(e.message, true); }
    finally { busy = false; range.disabled = false; }
  };
}

// Global smart-history filter toggle. GET/POST /admin/history-filter.
async function renderSmartHistoryToggle(container) {
  const panel = el("div", "intpanel");
  const head = el("div", "intpanel-head");
  const txt = el("div", "intpanel-txt");
  txt.append(el("div", "intpanel-title", "Filtro inteligente de histórico"));
  txt.append(el("div", "intpanel-sub",
    "Ao cortar o histórico, seleciona os turnos mais relevantes ao pedido atual (BM25 + MMR) em vez de só os mais recentes. Desligado = recência."));
  head.append(txt);
  const sw = el("button", "switch");
  sw.setAttribute("role", "switch");
  sw.append(el("span", "switch-knob"));
  head.append(sw);
  panel.append(head);
  container.append(panel);

  let enabled = false, busy = false;
  const paint = (on) => {
    enabled = !!on;
    sw.classList.toggle("on", enabled);
    sw.setAttribute("aria-checked", enabled ? "true" : "false");
  };
  try { const r = await api("GET", "/admin/history-filter"); paint(r.enabled); }
  catch (_) {}
  sw.onclick = async () => {
    if (busy) return;
    busy = true; sw.classList.add("busy");
    try { const r = await api("POST", "/admin/history-filter", { enabled: !enabled }); paint(r.enabled); toast("Filtro atualizado"); }
    catch (e) { toast(e.message, true); }
    finally { busy = false; sw.classList.remove("busy"); }
  };
}

// Routing profile selector (moved from the tray). GET current + options,
// POST on click to apply + persist live.
async function renderRoutingSelector(container) {
  const panel = el("div", "intpanel");
  panel.append(el("div", "intpanel-title", "Roteamento"));
  panel.append(el("div", "intpanel-sub",
    "Como o localllm decide entre modelo local e cloud. Aplica na hora e persiste."));
  const seg = el("div", "segmented");
  panel.append(seg);
  container.append(panel);

  let data;
  try { data = await api("GET", "/admin/routing"); }
  catch (e) { seg.append(el("div", "intpanel-status", e.message)); return; }

  const paint = (current) => {
    seg.innerHTML = "";
    (data.options || []).forEach((o) => {
      const b = el("button", "seg-btn" + (o.value === current ? " on" : ""), o.label);
      b.onclick = async () => {
        if (o.value === current) return;
        seg.querySelectorAll("button").forEach((x) => x.disabled = true);
        try {
          const r = await api("POST", "/admin/routing", { profile: o.value });
          paint(r.current); toast("Roteamento atualizado");
        } catch (e) { toast(e.message, true); paint(current); }
      };
      seg.append(b);
    });
  };
  paint(data.current);
}

// Route-apps integration toggle (moved from the tray). Reads GET /admin/integrations,
// flips via POST. The tray shows the same state read-only.
async function renderIntegrationToggle(container) {
  const panel = el("div", "intpanel");
  const head = el("div", "intpanel-head");
  const txt = el("div", "intpanel-txt");
  txt.append(el("div", "intpanel-title", "Rotear apps pelo localllm"));
  txt.append(el("div", "intpanel-sub",
    "Clientes (Claude Code, Codex) passam por este servidor. Ao sair do localllm, o roteamento é removido automaticamente."));
  head.append(txt);

  const sw = el("button", "switch");
  sw.setAttribute("role", "switch");
  sw.append(el("span", "switch-knob"));
  head.append(sw);
  panel.append(head);

  const status = el("div", "intpanel-status", "carregando…");
  panel.append(status);

  let enabled = false;
  let busy = false;
  const paint = (st) => {
    enabled = !!st.enabled;
    sw.classList.toggle("on", enabled);
    sw.setAttribute("aria-checked", enabled ? "true" : "false");
    const wired = (st.wired || []).join(", ");
    status.className = "intpanel-status" + (enabled ? " on" : "");
    status.textContent = enabled
      ? `Ligado — wired: ${wired || "nenhum cliente encontrado"}`
      : "Desligado — apps vão direto ao provider";
  };

  try { paint(await api("GET", "/admin/integrations")); }
  catch (e) { status.textContent = e.message; }

  sw.onclick = async () => {
    if (busy) return;
    busy = true; sw.classList.add("busy");
    try { paint(await api("POST", "/admin/integrations", { enabled: !enabled })); toast("Integração atualizada"); }
    catch (e) { toast(e.message, true); }
    finally { busy = false; sw.classList.remove("busy"); }
  };

  container.append(panel);
}

// Firestore-style drilldown selection, preserved across background refreshes.
let selFamilyName = null;
let selModelFile = null;

// ---- Models: three-column drilldown (Família → Modelo → Detalhe) ----
function renderFamilies() {
  currentView = renderFamilies;
  setCrumbs([{ label: "Config", onClick: renderConfig }, { label: "Models" }]);

  const fam = families.find((f) => f.family === selFamilyName) || null;
  const model = fam ? fam.models.find((m) => m.file === selModelFile) || null : null;

  const cols = el("div", "columns");

  // Column 1 — families
  const c1 = el("div", "col");
  c1.append(el("div", "colhead", "Família"));
  const c1b = el("div", "colbody");
  families.forEach((f) => {
    const inUse = f.models.some((m) => m.status === "in_use");
    const rec = f.models.some((m) => m.recommended);
    const it = el("div", "colitem" + (f.family === selFamilyName ? " sel" : ""));
    const left = el("div", "colitem-main");
    left.append(el("span", "colitem-name", f.family));
    left.append(el("span", "colitem-sub", `${f.models.length} variante${f.models.length === 1 ? "" : "s"}`));
    it.append(left);
    const tags = el("div", "colitem-tags");
    if (inUse) tags.append(el("span", "pip pip-use", ""));
    if (rec) tags.append(el("span", "pip pip-rec", ""));
    tags.append(el("span", "chev", "›"));
    it.append(tags);
    it.onclick = () => { selFamilyName = f.family; selModelFile = null; renderFamilies(); };
    c1b.append(it);
  });
  c1.append(c1b);
  cols.append(c1);

  // Column 2 — models in the selected family
  const c2 = el("div", "col");
  c2.append(el("div", "colhead", "Modelo"));
  const c2b = el("div", "colbody");
  if (!fam) c2b.append(el("div", "colempty", "escolha uma família"));
  else fam.models.forEach((m) => {
    const it = el("div", "colitem" + (m.file === selModelFile ? " sel" : ""));
    if (m.recommended) it.classList.add("rec");
    const left = el("div", "colitem-main");
    left.append(el("span", "colitem-name", m.display_name));
    left.append(el("span", "colitem-sub", `${m.params} · ${m.quant} · ${gb(m.size_mb)}`));
    it.append(left);
    const tags = el("div", "colitem-tags");
    tags.append(el("span", "badge " + m.status, STATUS_LABEL[m.status] || m.status));
    tags.append(el("span", "chev", "›"));
    it.append(tags);
    it.onclick = () => { selModelFile = m.file; renderFamilies(); };
    c2b.append(it);
  });
  c2.append(c2b);
  cols.append(c2);

  // Column 3 — detail of the selected model
  const c3 = el("div", "col col-detail");
  c3.append(el("div", "colhead", "Detalhe"));
  const c3b = el("div", "colbody");
  if (!model) c3b.append(el("div", "colempty", fam ? "escolha um modelo" : "← comece pela família"));
  else c3b.append(buildDetail(fam, model));
  c3.append(c3b);
  cols.append(c3);

  view.innerHTML = ""; view.append(cols);
}

// Build the detail/actions element for a model (rendered into column 3).
// Internal saves call currentView() (= renderFamilies) to refresh in place,
// preserving the current family/model selection.
function buildDetail(fam, m) {
  const wrap = el("div", "detail");
  wrap.append(el("h2", null, m.display_name + (m.recommended ? " ★" : "")));
  const specs = el("div", "specs");
  specs.innerHTML = `
    <span>Parameters</span><b>${m.params}</b>
    <span>Quantization</span><b>${m.quant}</b>
    <span>Download size</span><b>${gb(m.size_mb)}</b>
    <span>Est. RAM</span><b>~${gb(m.est_ram_mb)} (${FIT_LABEL[m.fit] || m.fit})</b>
    <span>Status</span><b>${STATUS_LABEL[m.status] || m.status}</b>
    <span>Repo</span><b>${m.repo}</b>`;
  wrap.append(specs);

  // --- Context window control (per-model ctx override) ---
  const ctxBox = el("div", "ctxbox");
  ctxBox.append(el("div", "ctxtitle", "Context window"));
  if (m.ctx_max === 0) {
    ctxBox.append(el("div", "ctxnote", "não cabe nesta máquina"));
  } else {
    const row = el("div", "ctxrow");
    const input = el("input", "ctxinput");
    input.type = "number";
    input.min = m.ctx_min;
    input.max = m.ctx_max;
    input.step = 256;
    input.value = m.ctx_current;
    row.append(input);
    row.append(el("span", "ctxbounds", `min ${m.ctx_min} · max ${m.ctx_max}`));
    ctxBox.append(row);
    ctxBox.append(el("div", "ctxhint", `suggested default: ${m.ctx_default}`));

    const ctxActions = el("div", "actions");
    const saveBtn = el("button", "btn primary", "Salvar");
    const defBtn = el("button", "btn", `Usar padrão (${m.ctx_default})`);
    const valid = () => {
      const v = Number(input.value);
      return Number.isInteger(v) && v >= m.ctx_min && v <= m.ctx_max
        && v % 256 === 0 && v !== m.ctx_current;
    };
    saveBtn.disabled = true;
    // Already on the default → nothing to reset.
    defBtn.disabled = m.ctx_current === m.ctx_default;
    input.oninput = () => { saveBtn.disabled = !valid(); };
    saveBtn.onclick = () => saveCtx(m, Number(input.value), wrap);
    defBtn.onclick = () => saveCtx(m, 0, wrap);
    ctxActions.append(saveBtn, defBtn);
    ctxBox.append(ctxActions);
  }
  wrap.append(ctxBox);

  // --- KV cache quantization ---
  const kvBox = el("div", "ctxbox");
  kvBox.append(el("div", "ctxtitle", "KV cache"));
  const kvSel = el("select", "kvselect");
  [["q8", "Q8 · metade da RAM (padrão)"], ["f16", "F16 · máxima qualidade"], ["q4", "Q4 · menor RAM"]]
    .forEach(([val, label]) => {
      const opt = el("option", "", label);
      opt.value = val;
      if (val === m.kv_current) opt.selected = true;
      kvSel.append(opt);
    });
  kvBox.append(kvSel);
  kvBox.append(el("div", "ctxhint", `recomendado: ${m.kv_default}`));
  wrap.append(kvBox);

  // --- History window (turns) ---
  const histBox = el("div", "ctxbox");
  histBox.append(el("div", "ctxtitle", "Histórico (turnos)"));
  const histInput = el("input", "ctxinput");
  histInput.type = "number";
  histInput.min = 0;
  histInput.placeholder = "todos";
  if (m.history_turns_current != null) histInput.value = m.history_turns_current;
  histBox.append(histInput);
  histBox.append(el("div", "ctxhint", "0 ou vazio = manter tudo"));
  wrap.append(histBox);

  // --- Advanced: GPU layers (offload) ---
  const adv = el("details", "advbox");
  adv.append(el("summary", "advsummary", "Avançado"));
  const gpuInput = el("input", "ctxinput");
  gpuInput.type = "number";
  gpuInput.min = 0;
  gpuInput.placeholder = "todas";
  if (m.gpu_layers_current != null) gpuInput.value = m.gpu_layers_current;
  adv.append(el("div", "ctxtitle", "Camadas na GPU"));
  adv.append(gpuInput);
  adv.append(el("div", "ctxhint", "vazio = todas na GPU. Menos = tira pressão da Metal, porém mais lento."));
  wrap.append(adv);

  // --- Quant variant ---
  const qBox = el("div", "ctxbox");
  qBox.append(el("div", "ctxtitle", "Quantização"));
  const qSel = el("select", "kvselect");
  (m.variants || []).forEach(v => {
    const status = v.status === "downloaded" ? "✓ baixado" : `⇩ ${gb(v.size_mb)}`;
    const opt = el("option", "", `${v.quant} · ~${gb(v.est_ram_mb)} · ${FIT_LABEL[v.fit] || v.fit} · ${status}`);
    opt.value = v.quant;
    if (v.selected) opt.selected = true;
    qSel.append(opt);
  });
  qBox.append(qSel);
  qBox.append(el("div", "ctxhint", `atual: ${m.quant_selected}`));
  wrap.append(qBox);

  // --- Save profile button ---
  const profActions = el("div", "actions");
  const saveProfBtn = el("button", "btn primary", "Salvar perfil");
  saveProfBtn.onclick = () => saveProfile(m, {
    kv_type: kvSel.value,
    history_turns: histInput.value === "" ? 0 : Number(histInput.value),
    gpu_layers: gpuInput.value === "" ? 4294967295 : Number(gpuInput.value), // u32::MAX = clear
    quant: qSel.value,
  }, wrap);
  profActions.append(saveProfBtn);
  wrap.append(profActions);

  const actions = el("div", "actions");
  const switchBtn = el("button", "btn primary", m.status === "in_use" ? "Active" : "Switch to this model");
  switchBtn.disabled = m.status === "in_use";
  switchBtn.onclick = () => doSwitch(m, wrap);
  actions.append(switchBtn);

  const delBtn = el("button", "btn danger", "Delete");
  delBtn.disabled = m.status !== "downloaded"; // can't delete in-use or not-downloaded
  delBtn.title = m.status === "in_use" ? "Switch away first" : m.status === "needs_download" ? "Not downloaded" : "";
  // Inline two-click confirm — window.confirm() is a no-op in the wry webview.
  let delArmed = false, delTimer = null;
  delBtn.onclick = () => {
    if (!delArmed) {
      delArmed = true;
      delBtn.textContent = "Confirmar delete?";
      delBtn.classList.add("armed");
      delTimer = setTimeout(() => {
        delArmed = false; delBtn.textContent = "Delete"; delBtn.classList.remove("armed");
      }, 4000);
      return;
    }
    clearTimeout(delTimer);
    delArmed = false;
    doDelete(m);
  };
  actions.append(delBtn);
  wrap.append(actions);

  return wrap;
}

async function doSwitch(m, wrap) {
  try {
    await api("POST", "/admin/model", { repo: m.repo, file: m.file });
  } catch (e) { return toast(e.message, true); }
  // progress UI
  const prog = el("div", "progress");
  const bar = el("div", "bar"); const fill = el("div", "fill"); bar.append(fill);
  const phase = el("div", "phase", "starting…");
  prog.append(bar, phase); wrap.append(prog);
  startPolling(fill, phase);
}

// Polls the switch/reload status. `onDone(s)` (optional) overrides the default
// terminal behaviour (toast "Model switched" + navigate to families) so a ctx
// reload can keep the user on the detail pane with its own message.
function startPolling(fill, phase, onDone) {
  stopPolling();
  pollTimer = setInterval(async () => {
    const s = await refreshStatusHeader();
    if (!s) return;
    if (fill) fill.style.width = (s.progress || 0) + "%";
    if (phase) phase.textContent = `${s.phase} — ${s.progress || 0}%`;
    if (s.state !== "switching") {
      stopPolling();
      await refresh().catch(() => {});
      if (onDone) { onDone(s); return; }
      if (s.state === "error") toast(s.error || "switch failed", true);
      else toast("Model switched");
      renderFamilies();
    }
  }, 1000);
}
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

async function saveCtx(m, ctx, wrap) {
  let res;
  try {
    res = await api("POST", "/admin/model/ctx", { repo: m.repo, file: m.file, ctx });
  } catch (e) {
    return toast(e.message, true);
  }
  if (res && res.reloading) {
    // Active model is reloading at the new ctx — show the same progress UI as a switch.
    toast("recarregando…");
    const prog = el("div", "progress");
    const bar = el("div", "bar"); const fill = el("div", "fill"); bar.append(fill);
    const phase = el("div", "phase", "reloading…");
    prog.append(bar, phase); wrap.append(prog);
    startPolling(fill, phase, (s) => {
      toast(s.state === "error" ? (s.error || "reload failed") : "Contexto aplicado", s.state === "error");
      currentView();
    });
  } else {
    toast(res && res.cleared ? "Voltou ao padrão" : "Contexto salvo");
    await refresh().catch(() => {});
    currentView();
  }
}

async function saveProfile(m, fields, wrap) {
  let res;
  try {
    res = await api("POST", "/admin/model/profile", { repo: m.repo, file: m.file, ...fields });
  } catch (e) {
    return toast(`Falha ao salvar perfil: ${e.message || e}`, true);
  }
  if (res && res.reloading) {
    // Active model is reloading at the new profile — show the same progress UI as a ctx reload.
    toast("recarregando…");
    const prog = el("div", "progress");
    const bar = el("div", "bar"); const fill = el("div", "fill"); bar.append(fill);
    const phase = el("div", "phase", "reloading…");
    prog.append(bar, phase); wrap.append(prog);
    startPolling(fill, phase, (s) => {
      toast(s.state === "error" ? (s.error || "reload failed") : "Perfil aplicado", s.state === "error");
      currentView();
    });
  } else {
    toast("Perfil salvo");
    await refresh().catch(() => {});
    currentView();
  }
}

async function doDelete(m) {
  try {
    const r = await api("DELETE", "/admin/models", { repo: m.repo, file: m.file });
    toast(r && r.deleted ? "Deleted" : "Nothing to delete");
    await refresh(); renderFamilies();
  } catch (e) { toast(e.message, true); }
}

// ---- Pane 4: tools filter (text + status filters, collapsible descriptions) ----
async function renderTools() {
  currentView = renderTools;
  setCrumbs([
    { label: "Config", onClick: renderConfig },
    { label: "Tools" },
  ]);
  let data;
  try {
    data = await api("GET", "/admin/tools");
  } catch (e) {
    view.innerHTML = ""; view.append(el("div", "detail", `Falha ao carregar: ${e.message || e}`));
    return;
  }
  const wrap = el("div", "detail tools-wrap");
  wrap.append(el("h2", null, "Filtro de tools por cliente"));
  wrap.append(el("div", "ctxhint", "Desmarcar remove a tool do que o modelo recebe. Aplica no próximo request."));

  const rows = []; // { el, name, desc, blocked, setDesc }

  // Controls: text search + status filter + expand-all.
  const controls = el("div", "tools-controls");
  const search = el("input", "tools-search");
  search.type = "search";
  search.placeholder = "filtrar por nome ou descrição…";
  controls.append(search);
  const statusSeg = el("div", "segmented");
  let statusFilter = "all";
  const statusBtns = {};
  [["all", "Todas"], ["allowed", "Permitidas"], ["blocked", "Bloqueadas"]].forEach(([v, l]) => {
    const b = el("button", "seg-btn" + (v === "all" ? " on" : ""), l);
    b.onclick = () => { statusFilter = v; Object.entries(statusBtns).forEach(([k, btn]) => btn.classList.toggle("on", k === v)); repaint(); };
    statusBtns[v] = b; statusSeg.append(b);
  });
  controls.append(statusSeg);
  const expandBtn = el("button", "btn", "Expandir descrições");
  let allExpanded = false;
  expandBtn.onclick = () => {
    allExpanded = !allExpanded;
    expandBtn.textContent = allExpanded ? "Recolher descrições" : "Expandir descrições";
    rows.forEach(r => r.setDesc(allExpanded));
  };
  controls.append(expandBtn);
  wrap.append(controls);

  const surfaces = Object.keys(data);
  if (surfaces.length === 0) {
    wrap.append(el("div", "ctxnote", "Nenhuma tool descoberta ainda. Envie um request de um cliente (Claude Code / Codex) e recarregue."));
  }
  surfaces.forEach(surface => {
    const { seen = [], disabled = [], descriptions = {} } = data[surface];
    const box = el("div", "ctxbox");
    box.append(el("div", "ctxtitle", surface));
    if (seen.length === 0 && disabled.length === 0) {
      box.append(el("div", "ctxnote", "envie um request deste cliente para descobrir as tools"));
    }
    const boxes = [];
    const addRow = (name, blocked, notSeen) => {
      const row = el("div", "toolrow" + (blocked ? " blocked" : ""));
      const head = el("div", "toolhead");
      const lbl = el("label", "tool-lbl");
      const cb = el("input", "toolcb");
      cb.type = "checkbox";
      cb.checked = !blocked; // checked = enabled
      cb.dataset.name = name;
      lbl.append(cb, el("span", "toolname", name));
      head.append(lbl);
      if (blocked) head.append(el("span", "toolflag", "bloqueada"));
      if (notSeen) head.append(el("span", "toolmuted", "não vista agora"));
      const desc = descriptions[name] || "";
      let descEl = null, toggle = null;
      if (desc) {
        toggle = el("button", "desc-toggle", "descrição ▾");
        head.append(toggle);
        descEl = el("div", "tooldesc hidden", desc);
      }
      row.append(head);
      if (descEl) row.append(descEl);
      box.append(row);
      boxes.push(cb);
      const setDesc = (open) => {
        if (!descEl) return;
        descEl.classList.toggle("hidden", !open);
        toggle.textContent = open ? "descrição ▴" : "descrição ▾";
      };
      if (toggle) toggle.onclick = (e) => { e.preventDefault(); setDesc(descEl.classList.contains("hidden")); };
      rows.push({ el: row, name, desc, blocked, setDesc });
    };
    seen.forEach(name => addRow(name, disabled.includes(name), false));
    // Blocked tools the agent isn't currently sending stay listed as blocked.
    disabled.filter(n => !seen.includes(n)).forEach(name => addRow(name, true, true));
    if (boxes.length) {
      const actions = el("div", "actions");
      const saveBtn = el("button", "btn primary", "Salvar");
      saveBtn.onclick = () => saveToolFilter(surface, boxes);
      actions.append(saveBtn);
      box.append(actions);
    }
    wrap.append(box);
  });

  // Live filter across all surfaces (visual only — saving still reads every box).
  const repaint = () => {
    const q = search.value.trim().toLowerCase();
    rows.forEach(r => {
      const matchText = !q || r.name.toLowerCase().includes(q) || r.desc.toLowerCase().includes(q);
      const matchStatus = statusFilter === "all"
        || (statusFilter === "blocked" && r.blocked)
        || (statusFilter === "allowed" && !r.blocked);
      r.el.style.display = (matchText && matchStatus) ? "" : "none";
    });
  };
  search.oninput = repaint;

  view.innerHTML = ""; view.append(wrap);
}

async function saveToolFilter(surface, boxes) {
  const disabled = boxes.filter(cb => !cb.checked).map(cb => cb.dataset.name);
  try {
    await api("POST", "/admin/tools", { surface, disabled });
  } catch (e) {
    return toast(`Falha ao salvar: ${e.message || e}`, true);
  }
  toast("Filtro salvo");
  renderTools();
}

// ---- Pane: Dashboard (routing + tokens saved) ----
const PERIOD_LABEL = { hour: "Última hora", day: "Hoje", month: "Este mês" };

// Compact number formatting: 1240000 → "1.24M", 12000 → "12.0k".
function fmtNum(n) {
  n = n || 0;
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}
function pctSaved(b) {
  const all = b.tokens_if_all_cloud || 0;
  return all === 0 ? 0 : Math.round((b.tokens_saved / all) * 100);
}
function pctLocal(b) {
  const total = (b.local_count || 0) + (b.cloud_count || 0);
  return total === 0 ? 0 : Math.round((b.local_count / total) * 100);
}

async function renderDashboard() {
  currentView = renderDashboard;
  setCrumbs([{ label: "Config", onClick: renderConfig }, { label: "Dashboard" }]);
  let d;
  try { d = await api("GET", "/admin/dashboard"); }
  catch (e) { view.innerHTML = ""; view.append(el("div", "detail", e.message)); return; }

  const wrap = el("div", "dash");

  // Toolbar: clear-data action. Inline two-click confirm — window.confirm() is
  // unreliable inside the wry webview (no JS dialog), so the first click arms
  // the button and the second click actually clears + reloads.
  const bar = el("div", "dash-bar");
  bar.append(el("div", "dash-bar-title", "Dashboard"));
  const barActions = el("div", "dash-bar-actions");
  const refreshBtn = el("button", "btn", "↻ Atualizar");
  refreshBtn.onclick = () => renderDashboard();
  barActions.append(refreshBtn);
  const exportBtn = el("button", "btn", "⇩ Exportar CSV");
  exportBtn.onclick = () => window.open(`/admin/export?format=csv&token=${encodeURIComponent(TOKEN)}`, "_blank");
  barActions.append(exportBtn);
  const clearBtn = el("button", "btn danger", "Limpar dados");
  let armed = false, armTimer = null;
  clearBtn.onclick = async () => {
    if (!armed) {
      armed = true;
      clearBtn.textContent = "Confirmar limpeza?";
      clearBtn.classList.add("armed");
      armTimer = setTimeout(() => {
        armed = false; clearBtn.textContent = "Limpar dados"; clearBtn.classList.remove("armed");
      }, 4000);
      return;
    }
    clearTimeout(armTimer);
    armed = false;
    clearBtn.disabled = true;
    try {
      await api("DELETE", "/admin/dashboard");
      toast("Dados limpos");
      renderDashboard();
    } catch (e) {
      toast(e.message, true);
      clearBtn.disabled = false;
      clearBtn.textContent = "Limpar dados";
      clearBtn.classList.remove("armed");
    }
  };
  barActions.append(clearBtn);
  bar.append(barActions);
  wrap.append(bar);

  const m = d.month || { tokens_saved: 0, tokens_if_all_cloud: 0, local_count: 0, cloud_count: 0 };

  // Hero — monthly savings
  const hero = el("div", "dash-hero");
  const heroL = el("div", "dash-hero-main");
  heroL.append(el("div", "dash-kicker", "Tokens economizados este mês"));
  heroL.append(el("div", "dash-big", fmtNum(m.tokens_saved)));
  heroL.append(el("div", "dash-sub",
    `de ${fmtNum(m.tokens_if_all_cloud)} se tudo fosse para a cloud`));
  hero.append(heroL);
  const heroR = el("div", "dash-hero-side");
  const savedPct = pctSaved(m);
  heroR.append(el("div", "dash-ring-num", savedPct + "%"));
  heroR.append(el("div", "dash-ring-cap", "economia"));
  const ring = el("div", "dash-ring");
  ring.style.setProperty("--pct", savedPct);
  ring.append(heroR);
  hero.append(ring);
  wrap.append(hero);

  // Period cards
  const cards = el("div", "dash-cards");
  ["hour", "day", "month"].forEach((k) => {
    const b = d[k] || { local_count: 0, cloud_count: 0, tokens_saved: 0, tokens_if_all_cloud: 0 };
    const card = el("div", "dash-card");
    card.append(el("div", "dash-card-head", PERIOD_LABEL[k]));
    const saved = el("div", "dash-card-num");
    saved.append(document.createTextNode(fmtNum(b.tokens_saved)));
    saved.append(el("span", "dash-card-unit", " salvos"));
    card.append(saved);
    card.append(el("div", "dash-card-alt", `${fmtNum(b.tokens_if_all_cloud)} se tudo cloud`));
    // local vs cloud ratio bar
    const bar = el("div", "ratio");
    const loc = el("div", "ratio-local");
    loc.style.width = pctLocal(b) + "%";
    bar.append(loc);
    card.append(bar);
    card.append(el("div", "dash-card-split",
      `${b.local_count} local · ${b.cloud_count} cloud`));
    card.append(el("div", "dash-card-split",
      `$ ${(b.cost_saved_usd || 0).toFixed(2)} economizado`));
    cards.append(card);
  });
  wrap.append(cards);

  // Latency by route (from post-generation outcomes)
  const L = d.local_latency || { avg_ttft_ms: 0, avg_tok_s: 0, n: 0 };
  const C = d.cloud_latency || { avg_ttft_ms: 0, avg_tok_s: 0, n: 0 };
  const lat = el("div", "dash-card");
  lat.append(el("div", "dash-card-head", "LATÊNCIA (30d)"));
  lat.append(el("div", "dash-card-alt",
    `local: TTFT ${L.avg_ttft_ms}ms · ${(L.avg_tok_s || 0).toFixed(0)} tok/s (${L.n})`));
  lat.append(el("div", "dash-card-alt",
    `cloud: TTFT ${C.avg_ttft_ms}ms · ${(C.avg_tok_s || 0).toFixed(0)} tok/s (${C.n})`));
  wrap.append(lat);

  // Fallback windows — why routing went local (budget or provider)
  if (d.windows && d.windows.length) {
    const box = el("div", "fallback-box");
    box.append(el("div", "fallback-title", "Períodos em local"));
    d.windows.slice().reverse().forEach((w) => {
      const when = w.start_ts === w.end_ts
        ? fmtDateTime(w.start_ts)
        : `${fmtDateTime(w.start_ts)} – ${fmtDateTime(w.end_ts)}`;
      const why = w.kind === "budget"
        ? "budget diário estourado"
        : `cloud indisponível (${w.reason})`;
      const line = el("div", "fallback-line " + (w.kind === "budget" ? "budget" : "provider"));
      line.textContent = `${when} · ${why} · ${w.count} pedido(s) atendido(s) local · você seguiu trabalhando`;
      box.append(line);
    });
    wrap.append(box);
  }

  // Recent decisions table (with filters, score popover, prompt row-expand)
  const panel = el("div", "dash-table-wrap");
  panel.append(el("div", "dash-table-title", "Decisões recentes"));
  const recent = d.recent || [];
  if (recent.length === 0) {
    panel.append(el("div", "colempty", "sem requisições ainda — envie um prompt por um cliente"));
  } else {
    renderDecisionsTable(panel, recent);
  }
  wrap.append(panel);

  view.innerHTML = ""; view.append(wrap);
}

// Local date+time from unix seconds: "DD/MM HH:MM:SS".
function fmtDateTime(ts) {
  const dt = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(dt.getDate())}/${p(dt.getMonth() + 1)} ${p(dt.getHours())}:${p(dt.getMinutes())}:${p(dt.getSeconds())}`;
}

// datetime-local string (for filter inputs) at a given ts, or "".
function toLocalInput(ts) {
  const dt = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${dt.getFullYear()}-${p(dt.getMonth() + 1)}-${p(dt.getDate())}T${p(dt.getHours())}:${p(dt.getMinutes())}`;
}

// Human explanation of a routing score from its logged breakdown.
function scoreExplainNode(e) {
  const box = el("div", "score-pop");
  if (e.last_turn_tok == null || e.n_messages == null) {
    box.append(el("div", "score-pop-line", "Detalhe indisponível (registro antigo)."));
    box.append(el("div", "score-pop-line muted", `score = ${e.score.toFixed(3)}`));
    return box;
  }
  const turn = Math.min(1, e.last_turn_tok / 2000);
  const depth = Math.min(1, e.n_messages / 20);
  const diff = 0.8 * turn + 0.2 * depth;
  const line = (html) => box.append(el("div", "score-pop-line", html));
  box.append(el("div", "score-pop-title", "Como o score foi calculado"));
  line(`dificuldade = 0.8·turn + 0.2·depth`);
  line(`turn = min(1, ${e.last_turn_tok}/2000) = <b>${turn.toFixed(2)}</b>`);
  line(`depth = min(1, ${e.n_messages}/20) = <b>${depth.toFixed(2)}</b>`);
  line(`= 0.8·${turn.toFixed(2)} + 0.2·${depth.toFixed(2)} = <b>${diff.toFixed(3)}</b>`);
  if (e.threshold != null) {
    line(`limite (ajustado ao modelo${e.capability_b ? ` ${e.capability_b}B` : ""}) = <b>${e.threshold.toFixed(2)}</b>`);
  }
  if (e.ctx_window) {
    const fill = e.prompt_tok / e.ctx_window;
    line(`preenchimento do contexto = ${fmtNum(e.prompt_tok)}/${fmtNum(e.ctx_window)} = ${(fill * 100).toFixed(0)}%`);
  }
  const why = el("div", "score-pop-why");
  if (e.reason === "ContextOverflow") {
    why.textContent = "→ prompt maior que o contexto local → cloud.";
  } else if (e.dest === "cloud") {
    why.textContent = `→ score ${diff.toFixed(2)} acima do limite ${e.threshold != null ? e.threshold.toFixed(2) : ""} → cloud.`;
  } else {
    why.textContent = `→ score ${diff.toFixed(2)} abaixo do limite → local.`;
  }
  box.append(why);
  return box;
}

function renderDecisionsTable(panel, recent) {
  // Filters
  const controls = el("div", "dash-filters");
  const search = el("input", "tools-search");
  search.type = "search";
  search.placeholder = "filtrar surface, motivo ou prompt…";
  controls.append(search);
  const destSeg = el("div", "segmented");
  let destFilter = "all";
  const destBtns = {};
  [["all", "Todos"], ["local", "Local"], ["cloud", "Cloud"]].forEach(([v, l]) => {
    const b = el("button", "seg-btn" + (v === "all" ? " on" : ""), l);
    b.onclick = () => { destFilter = v; Object.entries(destBtns).forEach(([k, btn]) => btn.classList.toggle("on", k === v)); repaint(); };
    destBtns[v] = b; destSeg.append(b);
  });
  controls.append(destSeg);
  const fromIn = el("input", "dt-input"); fromIn.type = "datetime-local"; fromIn.title = "de";
  const toIn = el("input", "dt-input"); toIn.type = "datetime-local"; toIn.title = "até";
  controls.append(el("span", "dt-lbl", "de")); controls.append(fromIn);
  controls.append(el("span", "dt-lbl", "até")); controls.append(toIn);
  fromIn.oninput = repaint; toIn.oninput = repaint;
  panel.append(controls);

  const table = el("table", "dash-table");
  const thead = el("thead");
  thead.innerHTML = "<tr><th>quando</th><th>surface</th><th>destino</th><th>motivo</th><th>score</th><th>prompt tok</th></tr>";
  table.append(thead);
  const tbody = el("tbody");

  // one shared popover
  let pop = null;
  const closePop = () => { if (pop) { pop.remove(); pop = null; } };
  document.addEventListener("click", closePop);

  const rows = [];
  recent.forEach((e) => {
    const tr = el("tr", "drow");
    tr.append(el("td", "muted nowrap", fmtDateTime(e.ts)));
    tr.append(el("td", null, e.surface));
    const dest = el("td");
    dest.append(el("span", "destpill dest-" + e.dest, e.dest));
    tr.append(dest);
    tr.append(el("td", "muted", e.reason || "—"));
    const sc = el("td");
    const scWrap = el("div", "scorecell scoreclick");
    const scBar = el("div", "scorebar");
    scBar.style.width = Math.round(Math.min(1, e.score) * 100) + "%";
    scWrap.append(el("span", "scoretxt", e.score.toFixed(2)));
    const scTrack = el("div", "scoretrack"); scTrack.append(scBar); scWrap.append(scTrack);
    scWrap.onclick = (ev) => {
      ev.stopPropagation();
      closePop();
      pop = scoreExplainNode(e);
      document.body.append(pop);
      // Position after it's in the DOM so we can measure real size and keep it
      // fully on-screen: clamp horizontally, flip above the row if it would
      // overflow the bottom, then clamp vertically.
      const r = scWrap.getBoundingClientRect();
      const pr = pop.getBoundingClientRect();
      const M = 8; // viewport margin
      const left = Math.max(M, Math.min(r.left, window.innerWidth - pr.width - M));
      let top = r.bottom + 6;
      if (top + pr.height + M > window.innerHeight) {
        top = r.top - pr.height - 6; // flip above
      }
      top = Math.max(M, Math.min(top, window.innerHeight - pr.height - M));
      pop.style.left = left + "px";
      pop.style.top = top + "px";
    };
    sc.append(scWrap);
    tr.append(sc);
    tr.append(el("td", "muted", fmtNum(e.prompt_tok)));

    // Expandable prompt row.
    const detail = el("tr", "drow-detail hidden");
    const dcell = el("td"); dcell.colSpan = 6;
    if (e.prompt_snippet) {
      dcell.append(el("div", "prompt-label", "Prompt (último turno)"));
      dcell.append(el("div", "prompt-box", e.prompt_snippet));
    } else {
      dcell.append(el("div", "muted", "sem prompt registrado para esta linha"));
    }
    detail.append(dcell);
    tr.onclick = () => detail.classList.toggle("hidden");

    tbody.append(tr); tbody.append(detail);
    rows.push({ tr, detail, e });
  });
  table.append(tbody);
  panel.append(table);

  function repaint() {
    const q = search.value.trim().toLowerCase();
    const fromTs = fromIn.value ? new Date(fromIn.value).getTime() / 1000 : null;
    const toTs = toIn.value ? new Date(toIn.value).getTime() / 1000 : null;
    rows.forEach(({ tr, detail, e }) => {
      const hay = `${e.surface} ${e.reason || ""} ${e.prompt_snippet || ""}`.toLowerCase();
      const matchText = !q || hay.includes(q);
      const matchDest = destFilter === "all" || e.dest === destFilter;
      const matchFrom = fromTs == null || e.ts >= fromTs;
      const matchTo = toTs == null || e.ts <= toTs;
      const show = matchText && matchDest && matchFrom && matchTo;
      tr.style.display = show ? "" : "none";
      if (!show) detail.classList.add("hidden");
      if (!show) detail.style.display = "none"; else detail.style.display = "";
    });
  }
}

// ---- Pane: Budget (daily cloud-spend cap) ----
async function renderBudget() {
  currentView = renderBudget;
  setCrumbs([{ label: "Config", onClick: renderConfig }, { label: "Budget" }]);
  const wrap = el("div", "detail");
  wrap.append(el("h2", null, "Budget diário"));
  wrap.append(el("div", "ctxhint", "Teto de gasto cloud por dia (estimado pela tabela de preços). Ao estourar, roteia tudo local até o dia virar (UTC). Zera sozinho no dia seguinte."));

  const panel = el("div", "intpanel");
  const head = el("div", "intpanel-head");
  const txt = el("div", "intpanel-txt");
  txt.append(el("div", "intpanel-title", "Ativar budget"));
  txt.append(el("div", "intpanel-sub", "Quando ligado e o gasto do dia atinge o teto, força local."));
  head.append(txt);
  const sw = el("button", "switch"); sw.setAttribute("role", "switch");
  sw.append(el("span", "switch-knob"));
  head.append(sw);
  panel.append(head);

  const row = el("div", "budget-row");
  const input = el("input", "ctxinput"); input.type = "number"; input.min = 0; input.step = 0.5;
  const saveBtn = el("button", "btn primary", "Salvar");
  row.append(el("span", "dt-lbl", "Teto $/dia"), input, saveBtn);
  panel.append(row);

  const readout = el("div", "budget-readout");
  panel.append(readout);
  wrap.append(panel);
  view.innerHTML = ""; view.append(wrap);

  let enabled = false;
  const paint = (b) => {
    enabled = !!b.enabled;
    sw.classList.toggle("on", enabled);
    sw.setAttribute("aria-checked", enabled ? "true" : "false");
    input.value = b.daily_usd;
    readout.className = "budget-readout" + (b.over ? " over" : "");
    readout.textContent = `gasto hoje $${(b.spent_today || 0).toFixed(2)} / $${(b.daily_usd || 0).toFixed(2)} · resta $${(b.remaining || 0).toFixed(2)}` + (b.over ? " · ESTOUROU (local)" : "");
  };
  try { paint(await api("GET", "/admin/budget")); }
  catch (e) { readout.textContent = e.message; return; }

  sw.onclick = async () => {
    try { paint(await api("POST", "/admin/budget", { enabled: !enabled })); toast("Budget atualizado"); }
    catch (e) { toast(e.message, true); }
  };
  saveBtn.onclick = async () => {
    const v = Number(input.value);
    try { paint(await api("POST", "/admin/budget", { daily_usd: isNaN(v) ? 0 : v })); toast("Teto salvo"); }
    catch (e) { toast(e.message, true); }
  };
}

// ---- hash router ----
function routeFromHash() {
  const h = (location.hash || "#/config").replace(/^#/, "");
  if (h.startsWith("/models")) return renderFamilies();
  if (h.startsWith("/tools")) return renderTools();
  if (h.startsWith("/budget")) return renderBudget();
  if (h.startsWith("/dashboard")) return renderDashboard();
  return renderConfig();
}
function navigate(route) {
  if (location.hash === "#" + route) routeFromHash();
  else location.hash = route; // triggers hashchange → routeFromHash
}
window.addEventListener("hashchange", () => routeFromHash());

// ---- boot ----
(async function boot() {
  try { await refresh(); routeFromHash(); }
  catch (e) { view.innerHTML = `<div class="loading">${e.message}</div>`; }
  // Keep header + grid live: catches switches started elsewhere and stale
  // badges after the window was hidden then reopened from the tray.
  setInterval(() => { tick().catch(() => {}); }, 3000);
  // The window is hidden (not destroyed) on close, so reopening fires focus
  // rather than a reload — refresh immediately so it never shows stale state.
  window.addEventListener("focus", () => { tick().catch(() => {}); });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) tick().catch(() => {});
  });
})();
