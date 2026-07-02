# Config Nav + Dashboard — Design

Date: 2026-07-02

## Summary

Restructure the tray + Model Manager SPA around a **Config** hub, move the
"route apps" integration toggle out of the tray into the Config page, add a
**Dashboard** view backed by a new persistent routing log, and hide webview
windows on blur to save memory. Quitting localllm un-wires integrations from
agent client configs.

Four sub-projects (A–D) plus a shared integrations API. Each is independently
shippable; suggested order A → B → integrations API → D → C (Dashboard last,
since its UI is co-designed with the user).

## Context (current state)

- **Tray** (`src/tray.rs`, macOS): spawns the axum server on a background
  thread, drives the tao event loop + `tray-icon` menu on the main thread.
  Opens a single `wry` webview window pointed at `/manager`. Menu today:
  status/info lines, Routing submenu, **"Route apps through localllm"
  CheckMenuItem** + wired line, Open Logs, Open Model Manager, Quit.
  Quit = `std::process::exit(0)` (no cleanup).
- **SPA** (`src/manager_ui/`, vanilla JS): home = Models grid, breadcrumb
  drilldown Models → family → detail; a Tools pane. Talks to token-guarded
  `/admin/*`. Polls every 3s + on focus/visibilitychange.
- **Routing** (`src/route/`, `src/server.rs`): `route_decision()` computes
  `Decision` (Local / Cloud(reason) / …) + difficulty `score` + est prompt
  tokens per request, **logged via tracing only — not persisted**. Three API
  handlers (anthropic, openai chat, openai responses) each call it.
- **Usage** (`src/usage.rs`): session-atomic counters only (no history).
- **Settings** (`src/settings.rs`): JSON file persists profile, integrations
  state, per-model exec profiles, per-model ctx, per-surface tool filters.
  **No persisted "active model"** — boot model = `Config::model_id`.
- **Integrations** (`src/integrations/`): `enable_all(port)` / `disable_all()`
  wire/unwire agent client configs (Claude Code, Codex); state persisted via
  `settings::{load,save}_integrations`.

## Sub-project A — Config landing + SPA navigation

**Goal:** SPA root becomes a **Config** hub; each area reachable as a card and
as a hash route so the tray can deep-link.

- Root view `renderConfig()`: cards for **Models** and **Tools**, plus the
  **Route-apps toggle** + wired-state line (see Integrations API), and a link
  to **Dashboard**.
- Breadcrumb root renames Models → **Config**.
- Hash routing: `#/config`, `#/models`, `#/tools`, `#/dashboard`. On load and on
  `hashchange`, dispatch to the matching render fn. Existing drilldown
  (family/detail) keeps working under `#/models`.
- Reopening the window (after blur-hide) restores the route from the hash.

**Isolation:** navigation is a thin router over existing `render*()` fns; no
change to the admin API for Models/Tools.

## Sub-project B — Tray restructure + model persistence + Quit cleanup

**Tray menu changes** (`src/tray.rs`):

- Replace "Open Model Manager" with a **Config submenu** containing **Models**,
  **Tools**, **Dashboard**. Each click opens the single shared window (open if
  absent, else show+focus) and navigates it to the route
  (`#/models` / `#/tools` / `#/dashboard`). Navigation on an already-open window:
  set the webview URL hash (reload to `…/manager#/route`) or eval
  `location.hash='#/route'`.
- **Remove** the "Route apps through localllm" CheckMenuItem. Keep the wired
  line as **read-only state**, refreshed each poll tick from live integration
  state (`settings::load_integrations`) rather than from a local toggle handle.
- Keep status/model/ctx/kv/backend/Routing/Open Logs/Quit.

**Model persistence** (`src/settings.rs`, `src/server.rs`, boot):

- New settings key `active_model: Option<{repo, file}>`.
- In `handle_admin_switch`, on a **successful** switch, save `active_model`.
- Boot resolution order: **explicit CLI/env `--model` > saved `active_model` >
  `Config::model_id` default**. Implement as a pure fn
  `resolve_active_model(cli, saved, default) -> ModelSpec` (unit-tested).

**Un-wire on Quit** (`src/tray.rs`):

- Quit handler, before `process::exit(0)`: if `load_integrations().enabled`,
  run `disable_all(&priors, &injectors)`, save the resulting (disabled) state.
  Agents fall back to provider-direct.
- Optional **boot reconcile**: if persisted state says enabled but this is a
  fresh launch after a crash, re-run `enable_all` so wiring matches the running
  port. (Nice-to-have; can defer.)

## Integrations API (shared, enables the moved toggle)

`src/server.rs`, token-guarded `/admin`:

- `GET /admin/integrations` → `{ enabled, wired: [client_id…] }`.
- `POST /admin/integrations {enabled: bool}` → runs `enable_all(port)` or
  `disable_all()`, saves state, returns the new state. Reuses
  `crate::integrations` + `settings::save_integrations`.

SPA Config page renders the toggle from `GET`, flips via `POST`, updates the
wired line. Tray's read-only line reflects the same persisted state.

## Sub-project C — Dashboard (routing log + view)

**Persistence — `src/route_log.rs` (new):**

- Append-only **JSONL** at `~/.localllm/routing-log.jsonl`. One line per request:

  ```json
  {"ts":1719900000,"surface":"claude-code","dest":"local",
   "reason":null,"score":0.12,"prompt_tok":260,"completion_tok":90}
  ```

  `dest` ∈ {`local`,`cloud`}; `reason` = RouteReason (null when local);
  `score` = `difficulty_score`; token counts are actual when known.
- **Rolling retention:** on boot and once/day, drop lines with
  `ts` older than ~1 month (30d). Pure fn `prune(lines, now, max_age)` —
  unit-tested. Rewrite file atomically (temp + rename).
- Writes happen at the decision point in each of the 3 API handlers in
  `server.rs`, after token counts are known. Local: use
  `result.{prompt,completion}_tokens`. Cloud: use provider usage from the proxied
  response when parseable, else est prompt tokens + best-effort completion (may be
  0 for streaming; note as a known limitation).
- Best-effort + non-blocking: log-write failure never fails a request.

**Rollup + API:**

- In-memory rollup over the loaded log: buckets for **this hour / today /
  this month**, each with `{local_count, cloud_count, tokens_saved,
  tokens_if_all_cloud}` where:
  - `tokens_saved` = Σ(prompt+completion) of **local** requests in the bucket.
  - `tokens_if_all_cloud` = Σ(prompt+completion) of **all** requests in the bucket.
  - (actual cloud spend = tokens_if_all_cloud − tokens_saved, derivable.)
- Pure fn `rollup(entries, now) -> Rollups` — unit-tested for bucket boundaries
  and the tokens math.
- `GET /admin/dashboard` → `{ hour, day, month, recent: [entry…] }` where
  `recent` is the last N (~50) entries for the per-prompt where/why/score table.

**Dashboard SPA view (`#/dashboard`):**

- **Visual layout is co-designed with the user via the brainstorming visual
  companion at implementation time.** The data contract above is fixed; the
  look (cards, table, any charts) is decided with the user before building.
  This is an intentional checkpoint, not an unresolved TBD.
- Reads `GET /admin/dashboard`; renders totals (hour/day/month: local vs cloud,
  tokens saved, tokens-if-all-cloud) + a recent-decisions table (surface, dest,
  reason, score, tokens).

**App-log rotation:** apply the same line-age retention to the app log
(`/tmp/localllm.log`, `LOCALLLM_LOG`): keep ~1 week, drop older, on boot.

## Sub-project D — Auto-hide window on blur

`src/tray.rs` event loop:

- Handle `WindowEvent::Focused(false)` for the shared window → `set_visible(false)`
  (hide, not destroy — same as current CloseRequested behavior).
- SPA already pauses background work on `visibilitychange`; keep that. On
  reopen (`Focused(true)` / show from tray) the SPA refreshes.
- **Risk:** transient focus loss (system dialog, tray menu interaction) can hide
  the window unexpectedly. Accept per user request; revisit if annoying.

## Testing

- `resolve_active_model` order (CLI > saved > default).
- `route_log::prune` age boundary; atomic rewrite.
- `rollup` bucket boundaries (hour/day/month edges) + tokens_saved /
  tokens_if_all_cloud math.
- JSON shape of `GET /admin/dashboard` and `GET /admin/integrations`.
- Integration toggle round-trip (enable → wired non-empty → disable → empty).

## Out of scope (YAGNI)

- Dollar-cost estimates (raw tokens only, per user).
- Per-provider price weighting.
- Charts beyond what the co-design settles on.
- Windows/Linux tray runtime (macOS only today).
