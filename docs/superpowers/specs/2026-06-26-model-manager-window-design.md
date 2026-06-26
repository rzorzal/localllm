# Model Manager window + cross-platform tray (design)

Date: 2026-06-26
Status: Proposed

## Goal

A real desktop **window** — opened from the menu-bar tray — that lets the user
browse the model catalog (Firestore-style **family → model → params/quant**
drilldown), see each model's status / estimated RAM / fit / a recommended pick,
and **switch** to a model (with download progress) or **delete** a cached one.
And: make the whole tray **cross-platform** (macOS / Linux / Windows).

This is **sub-project 3 of 3** of the model-picker. It consumes sub-projects 1
(hot-swap `/admin/model` + `/admin/model/status`) and 2 (catalog
`/admin/models` + delete).

## Decisions (from brainstorming)

- **UI delivery:** the SPA is embedded in the binary and served by axum at
  `GET /manager`; the webview opens `http://127.0.0.1:PORT/manager` (same origin
  as the API → no CORS). The admin token is injected **in-memory** via a wry
  initialization script (`window.__ADMIN_TOKEN__`); the page never reads the
  token file. (Rejected: `with_html` / custom protocol → cross-origin to the API.)
- **Window mechanics:** one **on-demand single-instance** `tao::Window` +
  `wry::WebView`, created lazily from the existing tray event loop; re-open shows
  /focuses it; close hides it (the agent stays alive). (Rejected: recreate-each-
  open; external browser tab — can't inject the token, worse security.)
- **Cross-platform:** everything cross-platform, **including the tray**. The
  SPA + `/manager` route + token flow are platform-agnostic; the tray/window
  shell uses cross-platform crates (`tao`/`wry`/`tray-icon`) with a thin
  `platform` abstraction for the few OS-specific calls.
- **Verification reality:** macOS is fully built+run+verified here. Linux
  (GTK/WebKitGTK + libappindicator) and Windows (WebView2) are
  compiled-best-effort and **flagged for on-device verification** — no platform
  is left hard-gated to macOS.
- **v1 actions:** Switch (downloads-if-needed + activates, with progress) and
  Delete. No download-only action (a model becomes "downloaded, not in use"
  naturally after switching away). Pre-download is a small later add.
- **Preserve everything in the current tray:** status 🟡/🟢, URL (click→copy),
  Model/Context/KV/Backend lines, the Routing submenu (+ persistence), Open
  Logs, Quit — all kept; "Open Model Manager" is added.

## Architecture

```
tray (cross-platform: tao + tray-icon)         [src/tray.rs — de-macOS-gated]
  • existing menu preserved + "Open Model Manager"
  └ open ─► one tao::Window + wry::WebView      [src/tray/window.rs — new]
              url = http://127.0.0.1:PORT/manager
              init script: window.__ADMIN_TOKEN__ = "<hex>"
                    │
axum                                            [src/server.rs + src/manager_ui/]
  GET /manager(.js/.css) ─► embedded SPA (no auth; static)
  SPA fetch (x-admin-token) ─►
     GET /admin/models · POST /admin/model · GET /admin/model/status
     DELETE /admin/models
```

### Module changes

- `src/tray.rs` — remove the blanket `#[cfg(target_os="macos")]`; `pub fn
  run_tray(cfg, admin_token)` is the cross-platform entry. The existing menu is
  rebuilt unchanged (all current items). A small internal `platform` module
  (cfg-dispatched) holds: `copy_to_clipboard`, `open_path`,
  `set_activation_policy`, bundle-launch detection.
- `src/tray/window.rs` (new) — `ModelManagerWindow { window, _webview }`;
  single-instance open/show/focus; `CloseRequested` → hide; token-injection
  init script; `manager_init_script(token) -> String` (pure, testable).
- `src/manager_ui/` (new) — `index.html`, `app.js`, `style.css`; vanilla
  HTML/CSS/JS (no framework/build step), built with the `frontend-design` skill.
  Baked in via `include_str!`.
- `src/server.rs` — `GET /manager`, `/manager/app.js`, `/manager/style.css`
  (static, correct content-types).
- `src/main.rs` — `--tray` (or macOS bundle launch) → `tray::run_tray(...)` on
  every OS; headless fallback unchanged.
- `Cargo.toml` — move `tao`/`tray-icon`/`png` from `[target.macos]` to plain
  `[dependencies]`; add `wry` (matched to tao 0.35).

## The window (`src/tray/window.rs`)

- The event-loop closure holds `manager_window: Option<ModelManagerWindow>`.
- Open (menu id match, or re-click): `Some` → `set_visible(true)+set_focus()`;
  `None` → build `WindowBuilder` (title "localllm — Model Manager", 900×640) +
  `WebViewBuilder` with `.with_url("http://127.0.0.1:{port}/manager")` and
  `.with_initialization_script(manager_init_script(&token))`.
- Close: `WindowEvent::CloseRequested` for the manager window → `set_visible(false)`
  (hide, keep the instance). Quit stays via the tray.
- `port` + `admin_token` captured into the event loop at `run_tray` start;
  `run_tray` gains `admin_token: Arc<str>` (same token `router` got).
- Created/used only on the main thread (tao/wry requirement). Build failure
  (missing WebView2/WebKitGTK) → log + tray notification; app keeps running.

`manager_init_script(token: &str) -> String` returns
`format!("window.__ADMIN_TOKEN__='{token}';")` — the token is a 32-hex string
(no quotes/escapes), injected before page load.

## The SPA (`src/manager_ui/`)

- **Pane 1 — families**: from `GET /admin/models`; counts + active/recommended hints.
- **Pane 2 — models in family**: cards with `params`, `quant`, est-RAM + fit
  badge (Fits/Tight/Won't-fit), status badge (In use / Downloaded / Needs
  download), ⭐ on the recommended one.
- **Pane 3 — detail + actions**:
  - **Switch**: `POST /admin/model {repo,file}` → poll `GET /admin/model/status`
    ~1s → progress bar (draining → downloading % → loading → ready) → refresh on
    ready; 409 → "switch already in progress".
  - **Delete** (Downloaded, not In-use): `DELETE /admin/models {repo,file}` →
    confirm → refresh; In-use → disabled.
- **Header**: active model + live "switching…" indicator (polls status during a
  switch; rest of UI read-only meanwhile, matching the server's 503).
- **Auth**: every `/admin/*` fetch sends `x-admin-token: window.__ADMIN_TOKEN__`.
- Consumes exactly `FamilyView[]` (sub-project 2) + `SwitchStatus` (sub-project 1).

## Cross-platform tray refactor

- De-gate `tray.rs`; rebuild the existing menu as-is on all OSes via the same
  `tao` loop + `tray-icon` menu + `MenuEvent` polling + ready→🟢 flip + Routing
  submenu/persistence + status/info lines; add "Open Model Manager".
- `platform` module: `copy_to_clipboard` (macOS `pbcopy` · Linux
  `wl-copy`/`xclip` · Windows `clip`); `open_path` (macOS `open`/`zed` · Linux
  `xdg-open` · Windows `start`); `set_activation_policy` (macOS Accessory · else
  no-op); bundle detection (macOS `__CFBundleIdentifier` · else `--tray`).
- Linux needs GTK + libappindicator + WebKitGTK (system deps); Windows needs
  WebView2 runtime — documented.

## Error handling

| Case | Handling |
|---|---|
| `/manager` static | infallible; unknown sub-path → 404 |
| window build fails (no WebView2/WebKitGTK) | log + tray notification; app keeps running; no panic |
| SPA fetch non-2xx | inline toast (401→restart, 409→in-progress/in-use, 5xx→message) |
| switch in progress | SPA disables Switch/Delete (mirrors server 503); re-enable on ready/error |
| re-open window | show/focus existing — never a second instance |
| clipboard/open tool missing | platform helper logs + no-ops |

## Testing

- **Headless (verified here):** integration tests for `GET /manager`(`.js`/`.css`)
  → 200 + content-type + non-empty; `manager_init_script` unit test (contains
  token, valid JS); full existing suite green; `cargo build`+`clippy` clean on macOS.
- **Manual (macOS, here):** open window, drilldown, switch (watch progress),
  delete a cached model, re-open (single instance), close (hides) — recorded in
  the report.
- **Deferred on-device (flagged):** Linux (GTK/WebKitGTK/libappindicator) and
  Windows (WebView2) build+run of the window shell. SPA + routes + token are
  platform-agnostic and covered headless.

## Non-goals

- Download-without-activating (pre-download) — later.
- A JS framework / build step (vanilla, baked-in).
- On-device Linux/Windows verification in this round (flagged for follow-up).
- Routing using local-model capability (separate enhancement, next).

## Risks

- Per-platform shell can only be macOS-verified here → Linux/Windows may need
  small fixes on-device; mitigated by using cross-platform crates + isolating
  OS calls in `platform`.
- `tao`/`wry` version alignment with tao 0.35 → pin a compatible `wry` at
  implementation; verify the window opens on macOS before proceeding.
- Webview adds binary size + (Linux/Windows) a system runtime dep → acceptable
  for a desktop tool; documented.
