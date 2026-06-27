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

// ---- Pane 1: families ----
function renderFamilies() {
  currentView = renderFamilies;
  setCrumbs([{ label: "Models" }]);
  const grid = el("div", "grid");
  families.forEach((fam) => {
    const recommended = fam.models.some((m) => m.recommended);
    const inUse = fam.models.some((m) => m.status === "in_use");
    const card = el("div", "card");
    card.append(el("h3", null, fam.family));
    card.append(el("div", "meta", `${fam.models.length} model${fam.models.length === 1 ? "" : "s"}`));
    const row = el("div", "row");
    if (inUse) row.append(el("span", "badge in_use", "In use"));
    if (recommended) row.append(el("span", "badge fits", "★ Recommended"));
    card.append(row);
    card.onclick = () => renderModels(fam);
    grid.append(card);
  });
  view.innerHTML = ""; view.append(grid);
}

// ---- Pane 2: models in a family ----
function renderModels(fam) {
  const famName = fam.family;
  currentView = () => {
    const f = families.find((x) => x.family === famName);
    f ? renderModels(f) : renderFamilies();
  };
  setCrumbs([{ label: "Models", onClick: renderFamilies }, { label: fam.family }]);
  const grid = el("div", "grid");
  fam.models.forEach((m) => {
    const card = el("div", "card");
    if (m.recommended) card.append(el("div", "ribbon", "RECOMMENDED"));
    card.append(el("h3", null, m.display_name));
    card.append(el("div", "meta", `${m.params} · ${m.quant} · ${gb(m.size_mb)}`));
    const row = el("div", "row");
    row.append(el("span", "badge " + m.status, STATUS_LABEL[m.status] || m.status));
    row.append(el("span", "badge " + m.fit, `${FIT_LABEL[m.fit] || m.fit} · ~${gb(m.est_ram_mb)} RAM`));
    card.append(row);
    card.onclick = () => renderDetail(fam, m);
    grid.append(card);
  });
  view.innerHTML = ""; view.append(grid);
}

// ---- Pane 3: detail + actions ----
function renderDetail(fam, m) {
  const famName = fam.family, file = m.file;
  currentView = () => {
    const f = families.find((x) => x.family === famName);
    const mm = f && f.models.find((x) => x.file === file);
    mm ? renderDetail(f, mm) : (f ? renderModels(f) : renderFamilies());
  };
  setCrumbs([
    { label: "Models", onClick: renderFamilies },
    { label: fam.family, onClick: () => renderModels(fam) },
    { label: m.display_name },
  ]);
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

  const actions = el("div", "actions");
  const switchBtn = el("button", "btn primary", m.status === "in_use" ? "Active" : "Switch to this model");
  switchBtn.disabled = m.status === "in_use";
  switchBtn.onclick = () => doSwitch(m, wrap);
  actions.append(switchBtn);

  const delBtn = el("button", "btn danger", "Delete");
  delBtn.disabled = m.status !== "downloaded"; // can't delete in-use or not-downloaded
  delBtn.title = m.status === "in_use" ? "Switch away first" : m.status === "needs_download" ? "Not downloaded" : "";
  delBtn.onclick = () => doDelete(m);
  actions.append(delBtn);
  wrap.append(actions);

  view.innerHTML = ""; view.append(wrap);
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

function startPolling(fill, phase) {
  stopPolling();
  pollTimer = setInterval(async () => {
    const s = await refreshStatusHeader();
    if (!s) return;
    if (fill) fill.style.width = (s.progress || 0) + "%";
    if (phase) phase.textContent = `${s.phase} — ${s.progress || 0}%`;
    if (s.state !== "switching") {
      stopPolling();
      if (s.state === "error") toast(s.error || "switch failed", true);
      else toast("Model switched");
      await refresh().catch(() => {});
      renderFamilies();
    }
  }, 1000);
}
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

async function doDelete(m) {
  if (!confirm(`Delete ${m.display_name} from disk?`)) return;
  try {
    const r = await api("DELETE", "/admin/models", { repo: m.repo, file: m.file });
    toast(r && r.deleted ? "Deleted" : "Nothing to delete");
    await refresh(); renderFamilies();
  } catch (e) { toast(e.message, true); }
}

// ---- boot ----
(async function boot() {
  try { await refresh(); renderFamilies(); }
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
