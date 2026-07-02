# Tool Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user discover and mute per-client tool schemas (keyed by API surface) so noisy agentic clients send fewer tools to the local model — filtered at request time, no rebuild.

**Architecture:** A persisted per-surface blocklist (`settings.tool_filters`) plus an in-memory per-surface discovery registry (`AppState.tool_registry`). A pure `filter_tools` drops blocklisted tools from the normalized request in the same choke point as sub-1's history truncation; a `shape_tools` helper records seen tools and applies the filter for each of the three API handlers. Admin endpoints expose discovery + let the user set the blocklist; a manager "Tools" view drives it.

**Tech Stack:** Rust, axum, serde, vanilla JS.

## Global Constraints

- Depends on sub-1/2 lineage — branch from `feat/quant-tier`, NOT `main`.
- Client key is the API surface, EXACTLY one of: `"anthropic"` (`/v1/messages`), `"openai"` (`/v1/chat/completions`), `"openai-responses"` (`/v1/responses`).
- Persisted config is a BLOCKLIST (disabled tool names) per surface; a tool NOT in the blocklist passes; new tools default enabled.
- Filtering is request-time and LOCAL-only; the cloud reverse-proxy path is never touched. No model reload / no rebuild.
- New serde fields on persisted structs MUST be `#[serde(default)]`; settings load/save must never panic or block startup.
- `filter_tools` name match is case-sensitive exact on `ToolSpec.name`.
- The registry records the LATEST sorted-unique tool-name set per surface (replace, not union).
- Run `cargo test` after each task; keep it green.

---

### Task 1: `tool_filters` persistence in settings

**Files:**
- Modify: `src/settings.rs` (`Settings` field + helpers)
- Test: `src/settings.rs` tests

**Interfaces:**
- Produces: `pub fn load_tool_filter(surface: &str) -> Vec<String>`; `pub fn save_tool_filter(surface: &str, disabled: &[String]) -> anyhow::Result<()>`.

- [ ] **Step 1: Write the failing test**

Add to `src/settings.rs` tests:

```rust
#[test]
fn tool_filter_round_trips_and_clears() {
    with_temp_settings(|| {
        assert!(load_tool_filter("anthropic").is_empty());
        save_tool_filter("anthropic", &["Read".to_string(), "Glob".to_string()]).unwrap();
        assert_eq!(load_tool_filter("anthropic"), vec!["Read".to_string(), "Glob".to_string()]);
        // other surface unaffected
        assert!(load_tool_filter("openai").is_empty());
        // empty list clears
        save_tool_filter("anthropic", &[]).unwrap();
        assert!(load_tool_filter("anthropic").is_empty());
    });
}

#[test]
fn saving_tool_filter_preserves_profile() {
    with_temp_settings(|| {
        save_profile(Profile::MaxQuality).unwrap();
        save_tool_filter("openai", &["Foo".to_string()]).unwrap();
        assert_eq!(load_profile(), Profile::MaxQuality);
        assert_eq!(load_tool_filter("openai"), vec!["Foo".to_string()]);
    });
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib settings::tool_filter settings::saving_tool_filter 2>&1 | tail -20`
Expected: FAIL — `load_tool_filter`/`save_tool_filter` undefined.

- [ ] **Step 3: Implement the field and helpers**

Add to the `Settings` struct (with the other `#[serde(default)]` fields):

```rust
    #[serde(default)]
    tool_filters: std::collections::BTreeMap<String, Vec<String>>,
```

Add the helpers near the other per-key settings functions:

```rust
/// Load a surface's disabled-tool blocklist, or an empty list if unset.
pub fn load_tool_filter(surface: &str) -> Vec<String> {
    load_settings().tool_filters.get(surface).cloned().unwrap_or_default()
}

/// Persist a surface's blocklist (empty list clears it), preserving the rest.
pub fn save_tool_filter(surface: &str, disabled: &[String]) -> anyhow::Result<()> {
    let mut s = load_settings();
    if disabled.is_empty() {
        s.tool_filters.remove(surface);
    } else {
        s.tool_filters.insert(surface.to_string(), disabled.to_vec());
    }
    save_settings(&s)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib settings:: 2>&1 | tail -20`
Expected: PASS (new + all existing settings tests).

- [ ] **Step 5: Commit**

```bash
git add src/settings.rs
git commit -m "feat(settings): per-surface tool blocklist persistence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `filter_tools` pure function

**Files:**
- Modify: `src/api/common.rs` (add `filter_tools`)
- Test: `src/api/common.rs` tests

**Interfaces:**
- Consumes: `ToolSpec`.
- Produces: `pub fn filter_tools(tools: Vec<ToolSpec>, disabled: &[String]) -> Vec<ToolSpec>`.

- [ ] **Step 1: Write the failing tests**

Add to `src/api/common.rs` tests:

```rust
fn tool(name: &str) -> ToolSpec {
    ToolSpec { name: name.into(), description: "d".into(), parameters: serde_json::json!({}) }
}

#[test]
fn filter_tools_drops_named_and_preserves_order() {
    let tools = vec![tool("Bash"), tool("Read"), tool("Glob"), tool("Edit")];
    let out = filter_tools(tools, &["Read".to_string(), "Glob".to_string()]);
    let names: Vec<_> = out.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["Bash".to_string(), "Edit".to_string()]);
}

#[test]
fn filter_tools_empty_disabled_is_passthrough() {
    let tools = vec![tool("Bash"), tool("Read")];
    assert_eq!(filter_tools(tools.clone(), &[]), tools);
}

#[test]
fn filter_tools_unknown_name_is_noop() {
    let tools = vec![tool("Bash")];
    assert_eq!(filter_tools(tools.clone(), &["Nope".to_string()]), tools);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib api::common 2>&1 | tail -20`
Expected: FAIL — `filter_tools` undefined.

- [ ] **Step 3: Implement `filter_tools`**

Add to `src/api/common.rs`:

```rust
/// Drop tools whose name is in `disabled` (case-sensitive exact match),
/// preserving the order of the rest. `disabled` empty = passthrough.
pub fn filter_tools(tools: Vec<ToolSpec>, disabled: &[String]) -> Vec<ToolSpec> {
    if disabled.is_empty() {
        return tools;
    }
    tools.into_iter().filter(|t| !disabled.iter().any(|d| d == &t.name)).collect()
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib api::common 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/common.rs
git commit -m "feat(api): filter_tools drops blocklisted tools

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Discovery registry + `shape_tools` wired into the three handlers

**Files:**
- Modify: `src/server.rs` (`AppState` field + init, `shape_tools`, 3 call sites)
- Test: `src/server.rs` / router tests

**Interfaces:**
- Consumes: `crate::settings::load_tool_filter`, `crate::api::common::filter_tools`.
- Produces: `AppState.tool_registry: Arc<Mutex<BTreeMap<String, Vec<String>>>>`; `fn shape_tools(state: &AppState, surface: &str, req: &mut ChatRequest)`.

- [ ] **Step 1: Add the `tool_registry` field and initialize it**

In `src/server.rs`, add to the `AppState` struct (after `kv_kind`):

```rust
    /// Per-surface discovery of the tool names seen on recent requests
    /// (surface -> latest sorted-unique names). In-memory; re-discovered on restart.
    pub tool_registry: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, Vec<String>>>>,
```

Find the `AppState { … }` construction inside `router()` (grep `let state = Arc::new(AppState {`) and add the field init:

```rust
        tool_registry: std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new())),
```

- [ ] **Step 2: Write a failing integration test**

Add a router test (mirror sub-1's `active_model_history_turns_truncates_request` harness — a `RecordingGen` capturing the request, `LOCALLLM_SETTINGS` temp file, admin not needed for `/v1/*`). Save a blocklist for the `anthropic` surface, POST a `/v1/messages` request carrying tools `["Bash","Read"]`, and assert the generator received tools with `Read` removed:

```rust
#[tokio::test]
async fn anthropic_request_drops_blocklisted_tool() {
    // Arrange: temp settings with save_tool_filter("anthropic", &["Read"]).
    // A RecordingGen that stores req.tools names. Router built around it.
    // Act: POST /v1/messages with tools Bash+Read.
    // Assert: recorded tool names == ["Bash"]; and the registry now has
    //   "anthropic" -> ["Bash","Read"] (sorted-unique seen), i.e. discovery recorded
    //   the pre-filter set.
}
```

(Record `seen` BEFORE filtering, so the registry reflects what the client actually sent, not the filtered subset.)

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test --lib anthropic_request_drops_blocklisted_tool 2>&1 | tail -20`
Expected: FAIL — no filtering/registry yet.

- [ ] **Step 4: Implement `shape_tools` and call it in the three handlers**

Add the helper near `apply_history_window` in `src/server.rs`:

```rust
/// Record the request's tool names into the discovery registry for `surface`
/// (latest sorted-unique set), then drop any blocklisted tools before local
/// inference. Local-only; the cloud path forwards raw bytes untouched.
fn shape_tools(state: &AppState, surface: &str, req: &mut ChatRequest) {
    // 1) record seen (pre-filter) — reflects what the client actually sent
    {
        let mut names: Vec<String> = req.tools.iter().map(|t| t.name.clone()).collect();
        names.sort();
        names.dedup();
        let mut reg = state.tool_registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.insert(surface.to_string(), names);
    }
    // 2) filter disabled
    let disabled = crate::settings::load_tool_filter(surface);
    if !disabled.is_empty() {
        let tools = std::mem::take(&mut req.tools);
        req.tools = crate::api::common::filter_tools(tools, &disabled);
    }
}
```

Call it in each handler immediately AFTER the existing `apply_history_window(&state, &mut internal);` line, passing the surface constant:

- `handle_oai_chat` (~server.rs:784): `shape_tools(&state, "openai", &mut internal);`
- `handle_oai_responses` (~server.rs:971): `shape_tools(&state, "openai-responses", &mut internal);`
- `handle_anth_messages` (~server.rs:1069): `shape_tools(&state, "anthropic", &mut internal);`

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test --lib anthropic_request_drops_blocklisted_tool 2>&1 | tail -20`
Expected: PASS. Then `cargo test 2>&1 | tail -5` — full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): per-surface tool discovery + request-time filtering

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `GET`/`POST /admin/tools` endpoints

**Files:**
- Modify: `src/server.rs` (routes + handlers)
- Test: `src/server.rs` / router tests

**Interfaces:**
- Consumes: `AppState.tool_registry`, `crate::settings::{load_tool_filter, save_tool_filter}`.
- Produces: `GET /admin/tools`, `POST /admin/tools`.

- [ ] **Step 1: Write the failing tests**

Add router tests (admin-guarded; mirror the ctx/profile endpoint test harness with `x-admin-token` + temp settings):

```rust
#[tokio::test]
async fn get_tools_merges_registry_and_settings() {
    // Arrange: a state whose tool_registry has "anthropic" -> ["Bash","Read"],
    //   and settings blocklist "anthropic" -> ["Read"].
    // Act: GET /admin/tools with admin token.
    // Assert: 200, body["anthropic"]["seen"] == ["Bash","Read"],
    //   body["anthropic"]["disabled"] == ["Read"].
}

#[tokio::test]
async fn post_tools_persists_and_rejects_unknown_surface() {
    // POST /admin/tools {surface:"openai", disabled:["Foo"]} → 200,
    //   load_tool_filter("openai") == ["Foo"].
    // POST {surface:"bogus", disabled:[]} → 400.
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib get_tools_merges post_tools_persists 2>&1 | tail -20`
Expected: FAIL — routes/handlers absent.

- [ ] **Step 3: Add the routes and handlers**

Register next to the other admin routes in `router()`/`build_router` (near `.route("/admin/model/profile", …)`):

```rust
        .route("/admin/tools", get(handle_tools_get).post(handle_tools_set))
```

Add the handlers:

```rust
/// The three known API surfaces.
const KNOWN_SURFACES: [&str; 3] = ["anthropic", "openai", "openai-responses"];

/// GET /admin/tools — per-surface { seen, disabled } (token-guarded).
async fn handle_tools_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let reg = state.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut out = serde_json::Map::new();
    for surface in KNOWN_SURFACES {
        let seen = reg.get(surface).cloned().unwrap_or_default();
        let disabled = crate::settings::load_tool_filter(surface);
        if seen.is_empty() && disabled.is_empty() {
            continue;
        }
        out.insert(surface.to_string(), json!({ "seen": seen, "disabled": disabled }));
    }
    Json(serde_json::Value::Object(out)).into_response()
}

#[derive(serde::Deserialize)]
struct SetToolsBody {
    surface: String,
    #[serde(default)]
    disabled: Vec<String>,
}

/// POST /admin/tools — set a surface's blocklist (token-guarded).
async fn handle_tools_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: SetToolsBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    };
    if !KNOWN_SURFACES.contains(&body.surface.as_str()) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "unknown surface"}))).into_response();
    }
    match crate::settings::save_tool_filter(&body.surface, &body.disabled) {
        Ok(()) => Json(json!({"saved": true})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}
```

Ensure `get` and `post` are imported from `axum::routing` (the file already uses `post`; add `get` if not already imported).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib get_tools_merges post_tools_persists 2>&1 | tail -20`
Expected: PASS. Then `cargo test 2>&1 | tail -5` — full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): GET/POST /admin/tools discovery + blocklist endpoints

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Manager "Tools" view

**Files:**
- Modify: `src/manager_ui/app.js` (nav entry + `renderTools`)
- Modify: `src/manager_ui/style.css` (checkbox list styling)
- Test: `cargo build` (JS embedded); manual runtime verification

**Interfaces:**
- Consumes: `GET /admin/tools`, `POST /admin/tools`; existing `el`, `api`, `toast`, `setCrumbs` helpers.
- Produces: UI only.

- [ ] **Step 1: Add a nav entry that routes to the Tools view**

In `src/manager_ui/app.js`, the model views set breadcrumbs via `setCrumbs([...])` and the root view is `renderFamilies()`. Add a "Tools" entry reachable from the root crumb. In `renderFamilies` (or wherever the top-level crumbs are set), add a trailing crumb/button:

```js
  // in the root view's crumbs, add a Tools link:
  setCrumbs([
    { label: "Models", onClick: renderFamilies },
    { label: "Tools", onClick: renderTools },
  ]);
```

(Match the actual crumb API already used — `setCrumbs([{label,onClick}])`. If the root sets a single "Models" crumb, add "Tools" beside it.)

- [ ] **Step 2: Implement `renderTools`**

```js
async function renderTools() {
  setCrumbs([
    { label: "Models", onClick: renderFamilies },
    { label: "Tools" },
  ]);
  let data;
  try {
    data = await api("GET", "/admin/tools");
  } catch (e) {
    view.innerHTML = ""; view.append(el("div", "detail", `Falha ao carregar: ${e.message || e}`));
    return;
  }
  const wrap = el("div", "detail");
  wrap.append(el("h2", null, "Filtro de tools por cliente"));
  wrap.append(el("div", "ctxhint", "Desmarcar remove a tool do que o modelo recebe. Aplica no próximo request (sem rebuild). Cuidado: desabilitar uma tool que o cliente usa remove essa capacidade."));

  const surfaces = Object.keys(data);
  if (surfaces.length === 0) {
    wrap.append(el("div", "ctxnote", "Nenhuma tool descoberta ainda. Envie um request de um cliente (Claude Code / Codex) e recarregue."));
  }
  surfaces.forEach(surface => {
    const { seen = [], disabled = [] } = data[surface];
    const box = el("div", "ctxbox");
    box.append(el("div", "ctxtitle", surface));
    if (seen.length === 0) {
      box.append(el("div", "ctxnote", "envie um request deste cliente para descobrir as tools"));
    }
    const boxes = [];
    seen.forEach(name => {
      const row = el("label", "toolrow");
      const cb = el("input", "toolcb");
      cb.type = "checkbox";
      cb.checked = !disabled.includes(name); // checked = enabled
      cb.dataset.name = name;
      row.append(cb, el("span", "toolname", name));
      box.append(row);
      boxes.push(cb);
    });
    // include any disabled-but-not-currently-seen tools so they can be re-enabled
    disabled.filter(n => !seen.includes(n)).forEach(name => {
      const row = el("label", "toolrow");
      const cb = el("input", "toolcb");
      cb.type = "checkbox"; cb.checked = false; cb.dataset.name = name;
      row.append(cb, el("span", "toolname", `${name} (não visto agora)`));
      box.append(row); boxes.push(cb);
    });
    if (boxes.length) {
      const actions = el("div", "actions");
      const saveBtn = el("button", "btn primary", "Salvar");
      saveBtn.onclick = () => saveToolFilter(surface, boxes);
      actions.append(saveBtn);
      box.append(actions);
    }
    wrap.append(box);
  });
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
```

(Adjust `view` to the actual container variable the other render functions append into — grep `view.innerHTML` / `view.append` in app.js to match.)

- [ ] **Step 3: Style the checkbox list**

Add to `src/manager_ui/style.css`:

```css
.toolrow { display: flex; align-items: center; gap: 8px; padding: 3px 0; cursor: pointer; }
.toolcb { width: 16px; height: 16px; }
.toolname { font-size: 13px; }
```

(Use the existing palette; if the file uses CSS vars for borders/bg like `.ctxbox`, match them.)

- [ ] **Step 4: Verify build + suite**

Run: `cargo build 2>&1 | tail -5` (embeds the updated JS/CSS) then `cargo test 2>&1 | tail -5`.
Expected: build clean, suite green. Note in the report that manual GUI verification (open the manager, open Tools, uncheck a tool, save, confirm the next request drops it) is pending for the user.

- [ ] **Step 5: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): per-client tool filter view

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** Client key (surface constants) → T3 call sites; discovery registry → T3; persistence (blocklist) → T1; `filter_tools` → T2; wiring + record-seen-before-filter → T3; endpoints → T4; UI → T5. No-rebuild is inherent (request-time filter, no reload path touched). All spec sections mapped.
- **Record-before-filter:** T3 explicitly records `seen` from the pre-filter tool set so discovery reflects what the client sent, not the filtered subset. The integration test asserts this.
- **Surface keys** are the exact strings `"anthropic"`/`"openai"`/`"openai-responses"` in the handlers (T3), the endpoint validation `KNOWN_SURFACES` (T4), and are echoed by the UI from `GET /admin/tools` (T5) — consistent.
- **Lock safety:** every `tool_registry.lock()` uses `.unwrap_or_else(|e| e.into_inner())` so a poisoned lock never panics the request path or the endpoint.
- **No signature change to `router()`** — the registry is created inside `router()`, so `lib.rs` callers are untouched.
- **Cloud path untouched:** `shape_tools` mutates only the in-memory `internal` request on the local path; the cloud reverse-proxy uses raw bytes (unchanged), matching sub-1's history-truncation boundary.
