# Model Manager Ctx Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Context window" control to the Model Manager detail pane so the user can see a model's ctx bounds and set/clear its per-model context.

**Architecture:** Frontend-only. Extend `renderDetail` in `src/manager_ui/app.js` with a bounded number input + Salvar / Usar padrão buttons that POST to the existing `/admin/model/ctx` endpoint, plus matching CSS. Reuses the SPA's `el()`, `api()`, `toast()`, `startPolling()`, `refresh()`, `currentView()` helpers.

**Tech Stack:** Vanilla JS SPA (`src/manager_ui/app.js` + `style.css`), no build step, no framework.

## Global Constraints

- No backend change. The endpoint `POST /admin/model/ctx` and the catalog fields (`ctx_min`, `ctx_default`, `ctx_max`, `ctx_current`) already exist from sub-project 2.
- `manager_ui` is a static SPA with **no automated test harness** in this repo. Verification is by reading + a manual smoke test (documented below). Do not add a JS test framework.
- Reuse existing helpers verbatim: `el(tag, cls, html)`, `api(method, path, body)` (adds `x-admin-token`, returns parsed JSON on 2xx, throws `Error(server.error)` on non-2xx), `toast(msg, isErr)`, `refresh()` (re-fetches `/admin/models` into `families`), `currentView()` (re-renders the current pane, re-resolving the model from `families`), `startPolling(fill, phase)` (the reload/switch poller).
- Endpoint response shapes (exactly one): `202 {reloading:true}` (active model reloading), `200 {saved:true}`, `200 {cleared:true}`, or a non-2xx whose body `{error}` carries the range message.
- Input step = `256` (matches the fit module's rounding). Bounds: `min = ctx_min`, `max = ctx_max`.
- Won't-fit is `ctx_max === 0`: disable the control, show "não cabe nesta máquina".
- Match the existing card/badge/button styling; buttons use the existing `btn`, `btn primary` classes.

---

### Task 1: Context-window control on the detail pane

**Files:**
- Modify: `src/manager_ui/app.js` (add the control inside `renderDetail`, add `saveCtx`)
- Modify: `src/manager_ui/style.css` (styling for `.ctxbox` and children)

**Interfaces:**
- Consumes: `el`, `api`, `toast`, `refresh`, `currentView`, `startPolling` (existing in app.js); the model object `m` with `ctx_min/ctx_default/ctx_max/ctx_current` (from `/admin/models`).
- Produces: none (leaf UI).

- [ ] **Step 1: Add the control to `renderDetail`**

In `src/manager_ui/app.js`, inside `renderDetail(fam, m)`, immediately AFTER `wrap.append(specs);` (and before the `const actions = ...` block), insert:

```js
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
      return Number.isInteger(v) && v >= m.ctx_min && v <= m.ctx_max && v !== m.ctx_current;
    };
    saveBtn.disabled = true;
    input.oninput = () => { saveBtn.disabled = !valid(); };
    saveBtn.onclick = () => saveCtx(m, Number(input.value), wrap);
    defBtn.onclick = () => saveCtx(m, 0, wrap);
    ctxActions.append(saveBtn, defBtn);
    ctxBox.append(ctxActions);
  }
  wrap.append(ctxBox);
```

- [ ] **Step 2: Add the `saveCtx` helper**

In `src/manager_ui/app.js`, add this function next to `doSwitch` (e.g. right after the `doSwitch` function):

```js
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
    startPolling(fill, phase);
  } else {
    toast(res && res.cleared ? "Voltou ao padrão" : "Contexto salvo");
    await refresh().catch(() => {});
    currentView();
  }
}
```

- [ ] **Step 3: Add the CSS**

In `src/manager_ui/style.css`, append (adjust color tokens to match the file's existing palette — reuse the same variables/hex the other blocks use; the structure below is what matters):

```css
.ctxbox {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--border, #2a2a2a);
}
.ctxtitle {
  font-weight: 600;
  margin-bottom: 8px;
}
.ctxrow {
  display: flex;
  align-items: center;
  gap: 10px;
}
.ctxinput {
  width: 8em;
  padding: 6px 8px;
  font: inherit;
  background: var(--input-bg, #1b1b1b);
  color: inherit;
  border: 1px solid var(--border, #333);
  border-radius: 6px;
}
.ctxbounds {
  opacity: 0.7;
  font-size: 0.9em;
}
.ctxhint {
  opacity: 0.6;
  font-size: 0.85em;
  margin: 6px 0 10px;
}
.ctxnote {
  opacity: 0.7;
  font-style: italic;
}
```

(If `style.css` does not use CSS variables, substitute the concrete colors the neighbouring rules use — read the file first and match, don't introduce new tokens.)

- [ ] **Step 4: Verify it loads + serves (compile/serve check)**

`app.js` and `style.css` are `include_str!`-embedded and served by the running binary. Confirm the crate still builds (the files are embedded, so a syntax mistake in JS won't fail the Rust build, but confirm nothing else broke) and that the endpoints are wired:

```
MISTRALRS_METAL_PRECOMPILE=0 cargo build
node --check src/manager_ui/app.js
```
Expected: `cargo build` clean; `node --check` reports no syntax error (if `node` is unavailable, visually re-read the inserted JS for balanced braces/parens and correct helper names — there is no bundler to catch it).

- [ ] **Step 5: Manual smoke test (acceptance)**

Run the app against a real model (Qwen 3B, already cached) and exercise the control in a browser at `http://127.0.0.1:31415/manager`:

```
MISTRALRS_METAL_PRECOMPILE=0 ./target/debug/localllm \
  --model-id Qwen/Qwen2.5-3B-Instruct-GGUF --gguf-file qwen2.5-3b-instruct-q4_k_m.gguf \
  --ctx-len 32768 --port 31415 &
```
Then, in the Model Manager window (or a browser with the admin token injected — the tray window injects it; for a plain browser you can read the token from `~/Library/Application Support/localllm/admin-token` and set `window.__ADMIN_TOKEN__` in the console):
- Drill Models → Qwen2.5 → a NON-active model (e.g. 7B): the "Context window" block shows `min 2048 · max <N>`, suggested default 8192, input pre-filled to `ctx_current`.
- Change the input to an in-range value → **Salvar** enables → click → toast "Contexto salvo"; the pane re-renders with the new `ctx_current`.
- Click **Usar padrão** → toast "Voltou ao padrão".
- Drill into the **active** Qwen 3B, change ctx, **Salvar** → toast "recarregando…" + progress bar; after the reload the pane recovers and the header shows the model active again.
- Confirm a won't-fit model (e.g. Qwen2.5 32B on 16 GB) shows "não cabe nesta máquina" with no input.
Record the observed results in your report.

- [ ] **Step 6: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(manager-ui): per-model context-window control on the detail pane"
```

---

## Notes for the executor
- This is the final sub-project of the "ctx-aware model fit" feature; it is UI-only. There is no unit test to write — the endpoint and catalog fields are already tested (sub-project 2). Your evidence is the `node --check` pass + the manual smoke test in Step 5.
- Do not restructure the SPA or introduce a framework/build step. Match the existing vanilla-JS `el()`/`api()` idiom exactly.
- Apply the `frontend-design` skill's judgment for spacing/visual polish so the control reads as part of the existing detail pane, but keep the markup minimal (the helpers only build elements — no templates).
