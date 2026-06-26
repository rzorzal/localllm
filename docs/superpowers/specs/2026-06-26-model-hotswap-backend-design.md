# Hot-swappable model backend (design)

Date: 2026-06-26
Status: Proposed

## Goal

Let the running `localllm` server **switch the active local model at runtime**, with no process restart, driven by token-authenticated localhost control endpoints. This is **sub-project 1 of 3** for the model-picker feature; it is the foundation the picker UI needs — without it nothing the UI offers can take effect.

The other two sub-projects (each its own spec → plan → build) are:
- **2 — Model catalog + recommendation + status:** curated family→model→quant data, machine-RAM detection, per-model status, RAM/CPU estimates.
- **3 — The window:** a `wry` webview opened from the tray, **built with the `frontend-design` skill, Firestore-style drilldown** (family → model → params), status badges, "recommended" tag, resource costs. Drives sub-projects 1 + 2.

## Constraints (carried from the project)

- **Single self-contained binary**, plug-and-play. No subprocess-per-model.
- **Metal single-context rule:** this codebase already crashed when a second llama.cpp context was created while the first was alive ("redefinition of as_bits" shader error). The engine therefore keeps exactly **one** persistent context. A hot-swap must never have two contexts resident at once.
- Builds on the local/cloud router branch: `AppState` currently holds `gen: Arc<dyn Generator>`, fixed at startup; `LlamaEngine::load(model_id, gguf_files, ctx_len, kv_type, kv_cache_dir)` owns one model in a worker thread; `download::ensure_model(repo, files)` downloads to `<sys_cache>/localllm/<repo>/<file>`, cache-hitting if present.

## Decisions (from brainstorming)

- **Apply mechanism:** hot-swap, no restart.
- **Switch gap behavior:** requests during a switch get an immediate **503 + `Retry-After`** ("model switching"), not blocked.
- **Control surface:** localhost HTTP — `POST /admin/model`, `GET /admin/model/status`.
- **Auth:** all `/admin/*` require an **admin token** (`X-Admin-Token` header).
- **Swap mechanism:** **state machine + in-flight gate** (drain → drop old → build new → swap), with a **builder closure** seam for testability. (Rejected: arc-swap-without-drain — risks two Metal contexts; subprocess-per-model — breaks single-binary.)
- **v1 scope:** a switch changes the **model only** (repo + file). `ctx_len` / `kv_type` / `kv_cache_dir` stay as configured (YAGNI).

## Architecture

```
                AppState { manager: Arc<ModelManager>, admin_token, policy, usage, … }
                                   │
 /v1/* handlers ─► manager.is_switching()? ─yes─► 503 + Retry-After
                                   │ no
                                   ▼
                manager.generate()/generate_stream()  (impl Generator)
                   • inflight += 1 → delegate to current engine → inflight -= 1
                                   │
 POST /admin/model      (token) ─► manager.start_switch(spec)  (spawns switch task)
 GET  /admin/model/status (token)─► manager.status() → {state, current, target, phase, progress, error}
                                   │
            switch task: Switching → drain(inflight==0) → drop old (frees Metal ctx)
                       → download(progress) → build new → swap → Ready
                       (on failure → restore previous model)
```

New module **`src/model_manager.rs`**. Touched: `src/server.rs` (AppState, handler gate, `/admin/*` routes, token guard), `src/lib.rs` (build the real builder closure + initial engine + token), `src/config.rs` (`--admin-token`), `src/download.rs` (`ensure_model_with_progress`), `Cargo.toml` (`arc-swap`).

## Components

### `ModelManager`

Fields:
- `engine: arc_swap::ArcSwapOption<dyn Generator>` — the current engine, lock-free reads. `None` only briefly between drop and (re)build; the `switching` flag means no request ever observes `None`.
- `current: Mutex<ModelSpec>` — the model currently serving.
- `switching: AtomicBool` — one switch at a time; gates the request path.
- `inflight: AtomicUsize` — active `/v1/*` requests, for draining.
- `phase: AtomicU8` + `progress: AtomicU8` + `error: Mutex<Option<String>>` + `target: Mutex<Option<ModelSpec>>` — status fields.
- `builder: Box<dyn Fn(ModelSpec) -> BoxFuture<'static, anyhow::Result<Arc<dyn Generator>>> + Send + Sync>` — builds an engine for a spec. Production: download-with-progress + `LlamaEngine::load`. Tests: instant `FakeGen`.

Implements `Generator`:
```
generate(req):
    if switching.load() { bail!(SWITCHING)  }      // backstop; handler 503s first
    inflight += 1
    let e = engine.load_full();                     // Option<Arc<dyn Generator>>
    let r = match e { Some(e) => e.generate(req).await, None => bail!(SWITCHING) };
    inflight -= 1
    r
```
(`generate_stream` mirrors this; the inflight decrement happens when the stream is fully consumed.)

Types:
```
pub struct ModelSpec { pub repo: String, pub file: String }
pub enum SwitchPhase { Idle, Draining, Downloading, Loading }   // serialized as snake_case
pub struct SwitchStatus {
    pub state: &'static str,         // "ready" | "switching" | "error"
    pub current: ModelSpec,
    pub target: Option<ModelSpec>,
    pub phase: SwitchPhase,
    pub progress: u8,                // 0..100
    pub error: Option<String>,
}
```

### Switch task (`start_switch(spec)`)

1. `switching.compare_exchange(false,true)`; if already true → `Err(AlreadySwitching)` → 409.
2. `phase = Draining`; wait for `inflight == 0`, bounded ≤ 30s. **If the wait times out → abort: nothing torn down yet, set `switching=false`, stay `Ready` on the old model, record `error`.**
3. Take the old engine out (`engine.store(None)`); ensure it is the sole owner and **drop it** → worker thread + Metal context freed. (Safe: `inflight==0` means no outstanding `load_full` clone.)
4. `phase = Downloading/Loading`; run `builder(spec)` (download with progress → `LlamaEngine::load`).
5. On `Ok(new)`: `engine.store(Some(new))`, `*current = spec`, `phase=Idle`, clear `error`, `switching=false` → `Ready`.
6. On `Err(e)`: **restore** — rebuild the previous `current` via `builder` (its GGUF is cached → fast). On restore `Ok` → `Ready` on old model, `status.error = e`, `usage::notify("Model switch failed", …)`. On restore `Err` → `state = Error`; all `/v1/*` 503 (degraded; needs restart).

### Download progress (`src/download.rs`)

```
pub async fn ensure_model_with_progress(
    repo: &str, files: &[String],
    on_progress: impl Fn(u64, Option<u64>),   // (bytes_done, total_from_Content-Length)
) -> anyhow::Result<Vec<PathBuf>>
```
Same streaming loop as `ensure_model`, calling `on_progress` per chunk (throttled to ~1% pct change). Cache-hit files report 100% immediately. `ensure_model` becomes a no-op-callback wrapper (existing call sites unchanged).

### Control endpoints (`src/server.rs`)

- `POST /admin/model` — body `{"repo":…,"file":…}` → `202 {"state":"switching"}` | `400` (bad body) | `409` (in progress) | `401`.
- `GET /admin/model/status` → `200 SwitchStatus` | `401`.
- Both call `check_admin(&headers, &state)` first.

### Admin token

- `--admin-token <token>` (config). If unset, **generate a random token** at startup (32 hex chars).
- `AppState.admin_token: Arc<str>`.
- Written to `<config-dir>/localllm/admin-token` with **0600** perms (clients — tray/window/curl — read it). Log only that the file was written, never the value.
- Guard: `/admin/*` require `X-Admin-Token` equal to the token, **constant-time compared** (XOR-accumulate over equal-length bytes; length mismatch → fail), no new dependency. Mismatch/missing → `401`.
- `/v1/*` stay unauthenticated (unchanged).

## The 503 gate

At the top of both `/v1/*` handlers, after parse, before `route_decision`:
```rust
if state.manager.is_switching() {
    return (StatusCode::SERVICE_UNAVAILABLE, [("Retry-After","5")],
            Json(json!({"error":"model switching, retry shortly"}))).into_response();
}
```
Placed before routing so a cloud-routed request during a switch also 503s (the switch is a whole-server state). Cheap atomic read.

## Error handling (summary)

| Case | Handling |
|---|---|
| Drain timeout (stuck in-flight) | Abort before drop; stay Ready on old; `error` set |
| Download/load failure | Old already dropped → restore previous model (cached, fast); `error` + notify |
| Restore also fails | `state=Error`; all `/v1/*` → 503 (needs restart) |
| Concurrent switch | 409 |
| Missing/invalid token | 401 (constant-time) |
| Bad request body | 400 |
| Request races past the gate into `None` engine | `generate` bails `SWITCHING` (backstop) |

## Testing (all headless via the builder seam)

- `FakeGen` variants with distinct output → assert which engine is live before/after a switch.
- `FakeSlowGen` (awaits a release channel) → assert `/v1/*` returns **503 + Retry-After** while switching, then **200** after release.
- Happy path → `status` progresses `switching → ready`; engine swapped; `current` updated.
- **409** on a second `start_switch` mid-switch.
- **401** on `/admin/*` without/with wrong token; success with the right token; constant-time-compare unit test.
- **Drain** → an in-flight request delays the drop until it finishes (controllable gen); phase `Draining` observed.
- **Failure + restore** → builder errors for the target → `status.error` set **and `/v1/*` still served by the old engine**.
- **Download progress** → `ensure_model_with_progress` against a wiremock body with `Content-Length` → callback reports increasing pct to 100; cache-hit → immediate 100.

## Non-goals (this sub-project)

- The model catalog / recommendation / status metadata (sub-project 2).
- The GUI window (sub-project 3).
- Changing `ctx_len`/`kv_type` during a switch.
- Auth on `/v1/*`.

## Risks

- **Metal context teardown timing** — the design's drain→drop→build ordering is precisely to keep one context; the drop step must verify sole ownership before dropping. Mitigated by `inflight==0` + the lock-free `load_full` lifetime being only the call duration.
- **Restore failure leaves no engine** — surfaced as `state=Error` + 503; a restart recovers. Acceptable for v1.
- **`/admin/*` trust model** — localhost + token; any local process with the 0600 file can switch. Acceptable for a single-user local tool; documented.
