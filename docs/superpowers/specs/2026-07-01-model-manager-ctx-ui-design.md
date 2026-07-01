# Model Manager page — per-model ctx control — design

**Date:** 2026-07-01
**Status:** approved (design)
**Sub-project:** 3 of 3 in the "ctx-aware model fit" feature. Frontend-only.
Consumes sub-project 2's catalog fields (`ctx_min/ctx_default/ctx_max/
ctx_current`) and endpoint (`POST /admin/model/ctx`). No backend change.

## Why

Sub-projects 1–2 fit context to the device and expose a per-model ctx override
with a KV-aware catalog + an endpoint. Nothing surfaces this to the user. This
sub-project adds a **Context-window control** to the Model Manager detail pane so
a non-technical user can see each model's fit and set/clear its context within
the allowed bounds.

## Scope

### In scope
- A "Context window" section on the model **detail pane**
  (`renderDetail` in `src/manager_ui/app.js`): shows `ctx_current`, the
  `min`/`max` bounds, and the suggested `ctx_default`; a bounded number input; a
  **Salvar** button and a **Usar padrão (N)** button.
- Wiring to `POST /admin/model/ctx` via the existing `api()` helper, with the
  three response shapes handled (`202 reloading` / `200 saved` / `200 cleared` /
  `400`).
- Won't-fit state (`ctx_max == 0`): the control is disabled with a "não cabe
  nesta máquina" note.
- Supporting CSS in `src/manager_ui/style.css`, matching the existing SPA style.

### Out of scope (YAGNI)
- Any backend change (the endpoint + catalog fields already exist).
- Editing ctx from the family/model **list** panes — the control lives only on
  the detail pane (one clear place).
- A slider (decided: number input).
- Automated frontend tests — `manager_ui` is static and has no test harness in
  this repo (consistent with sub-projects earlier in the feature); verification
  is by reading + a manual smoke test.

## Components

All in `src/manager_ui/app.js`, extending the existing Firestore-style drilldown
(`renderFamilies` → `renderModels` → `renderDetail`). Reuses the existing
helpers: `el(tag, cls, html)`, `api(method, path, body)` (adds auth headers,
maps errors), `toast(msg, isErr)`, `refresh()`, `startPolling(fill, phase)`
(the switch/reload poller), and `renderDetail` re-render on tick.

### The control (in `renderDetail(fam, m)`)
Appended after the existing specs block, before/with the action buttons:

```
Context window
  [ <number input> ]   min {ctx_min} · max {ctx_max}
  suggested default: {ctx_default}
  [ Salvar ]   [ Usar padrão ({ctx_default}) ]
```

- Input: `<input type="number" min={ctx_min} max={ctx_max} step={STEP} value={ctx_current}>`
  where `STEP = 256` (matches the fit module's rounding).
- **Salvar** is disabled unless the input value differs from `ctx_current` AND is
  within `[ctx_min, ctx_max]` (re-check on `input` event). The server revalidates.
- **Usar padrão (N)** is disabled when there is no override in effect
  (`ctx_current == ctx_default` is a good-enough proxy; it always POSTs `ctx=0`
  which is idempotent regardless).
- Won't-fit (`ctx_max == 0`): render the section but disable the input + both
  buttons and show "não cabe nesta máquina" instead of the bounds line.

### Data flow

`saveCtx(m, ctx)` calls `api("POST", "/admin/model/ctx", { repo: m.repo, file: m.file, ctx })`.
Response handling (the endpoint returns exactly one of):
- **202** `{reloading:true}` — the model was active and is reloading. `toast("recarregando…")`, then `startPolling(...)` (the existing reload/switch poller) so the pane updates when the reload finishes.
- **200** `{saved:true}` — `toast("Contexto salvo")` + `refresh()` so `ctx_current` re-renders.
- **200** `{cleared:true}` — `toast("Voltou ao padrão")` + `refresh()`.
- **400** — `api()` throws; the catch shows `toast(err, true)` with the server's range message.

"Usar padrão" calls `saveCtx(m, 0)`.

The existing background `tick()` continues to keep the pane fresh (re-render on
catalog change), so an external switch or reload elsewhere is reflected.

## Error handling
- Client-side: **Salvar** stays disabled until the value is a valid in-range
  change; this prevents most 400s. The server is still authoritative and its
  400 range message is surfaced via `toast`.
- `api()` already maps 401 (auth) and 409 (conflict / switch in progress) to a
  toast; a 409 on the reload path shows "switch in progress" — the user retries.
- A won't-fit model can't reach the input (disabled), so no bad value is sent.

## Testing
`manager_ui` is a static SPA with no automated test harness in this repo (the
same as sub-projects 1–3's earlier UI work). Verification:
- **By reading:** the control reads the sub-project-2 fields, the three response
  shapes are each handled, the disabled/won't-fit states are correct, and no
  helper is misused.
- **Manual smoke test (the acceptance check):** open `/manager`, drill into a
  model, confirm the Context-window control shows the right bounds; set an
  in-range value → toast + updated `ctx_current`; set an out-of-range value is
  blocked client-side; "Usar padrão" clears; for the **active** model, saving
  shows "recarregando…" and the pane recovers after the reload.
Apply the `frontend-design` skill for visual polish so the control matches the
existing card/badge/button styling.

## Files touched
- `src/manager_ui/app.js` — the Context-window control in `renderDetail` +
  `saveCtx` helper.
- `src/manager_ui/style.css` — styling for the control (matching existing).

## Follow-up
None — this completes the 3-sub-project "ctx-aware model fit" feature. (Deferred
cross-cutting minors from sub-projects 1–2 remain logged for a future cleanup
pass: `gpu_present` hardcoded true; heuristic-vs-exact ctx_train display gap;
weights_mb single-shard; memory-bound clamp not yet live-verified.)
