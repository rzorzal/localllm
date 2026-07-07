//! HTTP router and handlers.
//!
//! Exposes four endpoints:
//!   POST /v1/chat/completions  — OpenAI Chat Completions API (streaming + non-streaming)
//!   POST /v1/messages          — Anthropic Messages API (streaming + non-streaming)
//!   GET  /v1/models            — OpenAI model list (reports the configured model id)
//!   GET  /health               — liveness probe
//!
//! When `stream: true` is set in the request body, the handler returns a
//! `text/event-stream` SSE response built from the engine's `generate_stream`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::BoxStream;
use serde_json::json;

use axum::body::Bytes;
use axum::http::HeaderMap;

use crate::api::anthropic::AnthRequest;
use crate::api::common::{ChatRequest, ChatResult, StreamDelta};
use crate::api::openai::{OaiChatRequest, OaiModelInfo, OaiModelList};

/// Constant-time byte comparison: false on length mismatch, otherwise XOR-
/// accumulate so the timing does not depend on where the first difference is.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Resolve the admin token: the CLI value, else a fresh random 32-hex string.
pub fn resolve_admin_token(cli: Option<String>) -> String {
    cli.unwrap_or_else(|| {
        let u = uuid::Uuid::new_v4();
        // 32 hex chars (no dashes)
        u.simple().to_string()
    })
}

/// Best-effort: write the admin token to `<config-dir>/localllm/admin-token`
/// with 0600 perms so the tray/window/CLI can read it. Logs only that it wrote
/// the file, never the value.
pub fn write_admin_token_file(token: &str) {
    let Some(dir) = dirs::config_dir() else {
        return;
    };
    let path = dir.join("localllm").join("admin-token");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    write_token_to(&path, token);
}

/// Core of `write_admin_token_file`: atomically creates the file at `path` with
/// mode 0600 (unix) before writing, so the token is never world-readable even
/// briefly. Extracted for testability. Best-effort — no panics.
fn write_token_to(path: &std::path::Path, token: &str) {
    // Remove any pre-existing file so create+mode applies cleanly.
    let _ = std::fs::remove_file(path);
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
        {
            Ok(mut f) => match f.write_all(token.as_bytes()) {
                Ok(()) => tracing::info!("admin token written to {}", path.display()),
                Err(e) => tracing::warn!("could not write admin token to {}: {e}", path.display()),
            },
            Err(e) => tracing::warn!("could not create admin token file {}: {e}", path.display()),
        }
    }
    #[cfg(not(unix))]
    {
        match std::fs::write(path, token) {
            Ok(()) => tracing::info!("admin token written to {}", path.display()),
            Err(e) => tracing::warn!("could not write admin token to {}: {e}", path.display()),
        }
    }
}

/// The webview initialization script that injects the admin token in-memory so
/// the SPA can authenticate against /admin/* without reading the token file.
/// `token` is a 32-hex string (no quotes/escapes), safe in a JS string literal.
pub fn manager_init_script(token: &str) -> String {
    // JSON-encode the token so any value (even a custom --admin-token with
    // quotes/backslashes) yields a valid, injection-safe JS string literal.
    let encoded = serde_json::to_string(token).unwrap_or_else(|_| "\"\"".to_string());
    format!("window.__ADMIN_TOKEN__={encoded};")
}

/// GET /manager — the Model Manager SPA (static, no auth).
async fn handle_manager_page() -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        [("content-type", "text/html; charset=utf-8")],
        include_str!("manager_ui/index.html"),
    )
        .into_response()
}

/// GET /manager/app.js
async fn handle_manager_js() -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        [("content-type", "text/javascript; charset=utf-8")],
        include_str!("manager_ui/app.js"),
    )
        .into_response()
}

/// GET /manager/style.css
async fn handle_manager_css() -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        [("content-type", "text/css; charset=utf-8")],
        include_str!("manager_ui/style.css"),
    )
        .into_response()
}

/// Generate a short per-request id (e.g. `req-1a2b3c4d`) used to correlate the
/// start/finish log lines of a single prompt when several run concurrently.
fn new_request_id() -> String {
    let u = uuid::Uuid::new_v4();
    format!("req-{}", &u.simple().to_string()[..8])
}

/// Summarize an internal request for the start log line.
fn request_summary(req: &ChatRequest) -> (usize, usize) {
    (req.messages.len(), req.tools.len())
}

/// Re-ask detector tunables — named so Fase D2 can calibrate them.
const REASK_JACCARD: f64 = 0.6;
const REASK_WINDOW_SECS: i64 = 120;
const REASK_BUFFER_CAP: usize = 8;

/// One recent LOCAL decision's prompt fingerprint for re-ask detection.
#[derive(Debug, Clone)]
pub struct RecentPrompt {
    pub rid: String,
    pub ts: i64,
    pub tokens: std::collections::BTreeSet<String>,
}

/// Scan newest-first for a prior prompt whose token set is Jaccard-similar
/// (> REASK_JACCARD) within REASK_WINDOW_SECS. Returns the matched rid.
fn detect_reask(
    buf: &std::collections::VecDeque<RecentPrompt>,
    tokens: &std::collections::BTreeSet<String>,
    now: i64,
) -> Option<String> {
    if tokens.is_empty() {
        return None;
    }
    for old in buf {
        if now - old.ts >= REASK_WINDOW_SECS || old.tokens.is_empty() {
            continue;
        }
        let inter = tokens.intersection(&old.tokens).count() as f64;
        let union = tokens.union(&old.tokens).count() as f64;
        if union > 0.0 && inter / union > REASK_JACCARD {
            return Some(old.rid.clone());
        }
    }
    None
}

/// Decide whether the circuit breaker forces a cloud-bound decision to local.
/// Returns the tripping `DegradeReason` when the raw decision is cloud, budget
/// did not already force local, and the breaker gate is closed to cloud (Open,
/// cooldown not elapsed). Consulting `gate` may consume a half-open probe, so
/// only call this once per decision and only for cloud-bound decisions.
fn breaker_gate_block(
    breaker: &crate::breaker::CircuitBreaker,
    decision: &crate::route::Decision,
    budget_forced: bool,
    now: u64,
) -> Option<crate::usage::DegradeReason> {
    if budget_forced {
        return None;
    }
    if !matches!(decision, crate::route::Decision::Cloud(_)) {
        return None;
    }
    match breaker.gate(now) {
        crate::breaker::Gate::Block { reason, .. } => Some(reason),
        crate::breaker::Gate::Allow => None,
    }
}

/// Returns the routing decision plus the estimated prompt token count (so the
/// caller can record cloud usage without recomputing).
fn route_decision(
    state: &AppState,
    internal: &ChatRequest,
    headers: &axum::http::HeaderMap,
    rid: &str,
    surface: &str,
) -> (crate::route::Decision, usize) {
    let has_cloud_creds =
        headers.contains_key("x-api-key") || headers.contains_key("authorization");
    let prompt_tokens = crate::route::estimate_prompt_tokens(internal);
    let last_turn_tokens = crate::route::estimate_last_turn_tokens(internal);
    let active = state.manager.status().current;
    let local_capability_b = crate::catalog::active_params_b(&active.repo, &active.file);
    let signals = crate::route::Signals {
        prompt_tokens,
        local_ctx_window: state.local_ctx_window,
        n_tools: internal.tools.len(),
        n_messages: internal.messages.len(),
        last_turn_tokens,
        has_cloud_creds,
        local_capability_b,
    };
    let policy = *state.policy.read().unwrap();
    let raw_decision = crate::route::decide(&signals, &policy);
    // Budget cap: if enabled and today's cloud spend is over the daily limit,
    // force a cloud decision back to local (until the UTC day rolls over).
    let now = crate::route_log::now_secs();
    let (budget_enabled, budget_daily) = crate::settings::load_budget();
    let budget_forced = budget_enabled
        && matches!(raw_decision, crate::route::Decision::Cloud(_))
        && state.budget.is_over(now, budget_daily);
    let decision = if budget_forced {
        crate::route::Decision::Local
    } else {
        raw_decision
    };

    // Circuit breaker: if cloud recently failed, skip it during the cooldown and
    // serve local. Budget takes precedence (already forced above). A half-open
    // probe is consumed here only when the decision is cloud-bound.
    let breaker_block = breaker_gate_block(&state.breaker, &decision, budget_forced, now as u64);
    let decision = if breaker_block.is_some() {
        crate::route::Decision::Local
    } else {
        decision
    };

    // Log WHY this decision was made: the difficulty score and its inputs, the
    // capability-adjusted threshold, and the outcome — so the log explains each
    // local-vs-cloud choice.
    let score = crate::route::difficulty_score(&signals);
    let threshold = crate::route::effective_threshold(&policy, local_capability_b);
    tracing::info!(
        target: "localllm::req",
        "{rid} route: {decision:?} score={score:.3} threshold={threshold:.3} \
         (last_turn_tok={last_turn_tokens} msgs={} prompt_tok={prompt_tokens} ctx_window={} fill={:.2} tools={} cap_b={local_capability_b:.1} creds={has_cloud_creds})",
        signals.n_messages,
        signals.local_ctx_window,
        if signals.local_ctx_window == 0 { 1.0 } else { (prompt_tokens as f64 / signals.local_ctx_window as f64).min(1.0) },
        signals.n_tools,
    );

    // Persist the routing decision (where + why + score + prompt size) for the
    // dashboard. Completion tokens are unknown at decision time → None. Records
    // the DECISION, not the eventual outcome (a cloud decision that degrades to
    // local still reads "cloud"). Best-effort — never fails the request.
    let (dest, reason) = if budget_forced {
        ("local", Some("BudgetExceeded".to_string()))
    } else if breaker_block.is_some() {
        ("local", Some("CloudDown".to_string()))
    } else {
        match decision {
            crate::route::Decision::Cloud(r) => ("cloud", Some(format!("{r:?}"))),
            _ => ("local", None),
        }
    };
    // Latest-turn prompt text (the ask only, not full history). Kept near-whole
    // so the dashboard can show the full prompt in its scrollable box; capped
    // generously to bound the route-log file size.
    const PROMPT_SNIPPET_MAX: usize = 8000;
    let prompt_snippet = internal
        .messages
        .iter()
        .rev()
        .find(|m| m.role == crate::api::common::Role::User)
        .and_then(|m| m.text.clone())
        .map(|t| {
            let t = t.trim();
            if t.chars().count() > PROMPT_SNIPPET_MAX {
                format!(
                    "{}…",
                    t.chars().take(PROMPT_SNIPPET_MAX).collect::<String>()
                )
            } else {
                t.to_string()
            }
        });
    crate::route_log::append(&crate::route_log::RouteEntry {
        ts: now,
        rid: rid.to_string(),
        surface: surface.to_string(),
        dest: dest.to_string(),
        reason,
        score,
        prompt_tok: prompt_tokens as u64,
        completion_tok: None,
        last_turn_tok: Some(last_turn_tokens as u64),
        n_messages: Some(signals.n_messages as u64),
        ctx_window: Some(signals.local_ctx_window as u64),
        threshold: Some(threshold),
        capability_b: Some(local_capability_b as f64),
        local_model: Some(crate::settings::model_ctx_key(&active.repo, &active.file)),
        prompt_snippet,
        ..Default::default()
    });

    // Re-ask detection: judge the PREVIOUS local decision if this request is a
    // near-duplicate sent shortly after it, then remember this request when it
    // itself routes local. Uses the latest user turn only.
    {
        let last_user_text = internal
            .messages
            .iter()
            .rev()
            .find(|m| m.role == crate::api::common::Role::User)
            .and_then(|m| m.text.clone())
            .unwrap_or_default();
        let tokens: std::collections::BTreeSet<String> =
            crate::history_select::tokenize(&last_user_text)
                .into_iter()
                .filter(|t| t.len() >= 2)
                .collect();
        let mut map = state
            .recent_prompts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let buf = map.entry(surface.to_string()).or_default();
        if let Some(prev_rid) = detect_reask(buf, &tokens, now) {
            crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
                rid: prev_rid,
                ts: now,
                signal: "reask".to_string(),
            });
        }
        if !matches!(decision, crate::route::Decision::Cloud(_)) {
            buf.push_front(RecentPrompt {
                rid: rid.to_string(),
                ts: now,
                tokens,
            });
            buf.truncate(REASK_BUFFER_CAP);
        }
    }

    (decision, prompt_tokens)
}

/// Record a request's post-generation outcome. `cost_saved_usd` is what a LOCAL
/// request would have cost on cloud (0 for cloud requests). Best-effort.
fn record_outcome(
    rid: &str,
    dest: &str,
    model: Option<&str>,
    prompt_tok: u64,
    completion_tok: Option<u64>,
    ttft_ms: Option<u64>,
    gen_ms: Option<u64>,
    finish: Option<crate::api::common::FinishReason>,
) {
    let cost_saved_usd = if dest == "local" {
        crate::pricing::price_for(model.unwrap_or("")).cost(prompt_tok, completion_tok.unwrap_or(0))
    } else {
        0.0
    };
    crate::route_log::append_outcome(&crate::route_log::OutcomeEntry {
        rid: rid.to_string(),
        ts: crate::route_log::now_secs(),
        completion_tok,
        ttft_ms,
        gen_ms,
        cost_saved_usd,
    });
    // Weak-local signal: a local answer that hit the length cap and was served
    // as-is (cascade escalations never reach this path with Length — they
    // escalate instead).
    if dest == "local" && finish == Some(crate::api::common::FinishReason::Length) {
        crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
            rid: rid.to_string(),
            ts: crate::route_log::now_secs(),
            signal: "truncated".to_string(),
        });
    }
}

/// Stable short label for a degrade reason, logged so the dashboard can group
/// provider-fallback windows.
fn degrade_reason_label(r: crate::usage::DegradeReason) -> &'static str {
    match r {
        crate::usage::DegradeReason::Auth => "Auth",
        crate::usage::DegradeReason::Quota => "Quota",
        crate::usage::DegradeReason::ServerError => "ServerError",
        crate::usage::DegradeReason::Offline => "Offline",
    }
}

/// Log that a cloud request fell back to local with the given provider reason,
/// so the dashboard can render "cloud unavailable HH:MM–HH:MM" windows.
fn log_degrade_fallback(
    rid: &str,
    surface: &str,
    model: Option<&str>,
    prompt_tok: u64,
    reason: crate::usage::DegradeReason,
) {
    crate::route_log::append(&crate::route_log::RouteEntry {
        ts: crate::route_log::now_secs(),
        rid: rid.to_string(),
        surface: surface.to_string(),
        dest: "local".to_string(),
        prompt_tok,
        model: model.map(|m| m.to_string()),
        degrade_reason: Some(degrade_reason_label(reason).to_string()),
        ..Default::default()
    });
}

/// Record a successful cloud call and fire the one-shot high-usage alert if the
/// session just crossed the threshold. Also clears the degrade gate so a later
/// failure notifies again.
fn record_cloud_success(state: &AppState, est_prompt_tokens: usize) {
    state.usage.note_success();
    if state
        .breaker
        .on_success(crate::route_log::now_secs() as u64)
    {
        crate::usage::notify("localllm — cloud recovered", "Cloud back — resuming.");
    }
    // Charge the budget with a prompt-only estimate. The model id is not known
    // here (raw passthrough), so this uses the fallback price — a safety-cap
    // estimate, not an exact invoice.
    let cost = crate::pricing::price_for("").cost(est_prompt_tokens as u64, 0);
    state.budget.note_cloud_cost(cost);
    if state
        .usage
        .record_cloud_call(est_prompt_tokens, state.cloud_token_alert)
    {
        crate::usage::notify(
            "localllm — high cloud usage",
            "High cloud token use this session — consider the Save tokens profile.",
        );
    }
}

/// Handle a degrade signal: notify once, then decide whether local can serve.
/// Returns `Some(response)` when the caller must return it (cannot fall back to
/// local because the prompt overflows the window); `None` when the caller should
/// proceed to the local path.
fn handle_degrade(
    state: &AppState,
    reason: crate::usage::DegradeReason,
    route_reason: crate::route::RouteReason,
) -> Option<axum::response::Response> {
    if state.usage.note_degrade() {
        crate::usage::notify("localllm — cloud degraded", reason.message());
    }
    state
        .breaker
        .on_failure(crate::route_log::now_secs() as u64, reason);
    if route_reason == crate::route::RouteReason::ContextOverflow {
        // Prompt cannot fit the local window → no local fallback.
        Some(crate::cloud::degrade_error(reason))
    } else {
        None
    }
}

/// Resolve the active model's history window: saved profile → catalog rec → None.
fn resolve_history_turns(active: &crate::model_manager::ModelSpec) -> Option<u32> {
    let key = crate::settings::model_ctx_key(&active.repo, &active.file);
    let saved = crate::settings::load_model_profile(&key).history_turns;
    let rec = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == active.repo && e.file == active.file)
        .and_then(|e| e.rec_history_turns);
    saved.or(rec)
}

/// Record the request's tool names into the discovery registry for `surface`
/// (latest sorted-unique set), then drop any blocklisted tools before local
/// inference. Local-only; the cloud path forwards raw bytes untouched.
fn shape_tools(state: &AppState, surface: &str, req: &mut ChatRequest) {
    // 1) record seen (pre-filter) — reflects what the client actually sent.
    // Cumulative: tools are only ever ADDED to the seen set, never removed — a
    // tool that stops being sent stays listed so it (and its block) persist if
    // it ever returns. Persist on growth so the Tools view survives a restart.
    {
        // Capture descriptions (seeded from disk the first time this surface is
        // touched this session), then persist on growth so the Tools view can
        // expand descriptions right after a restart, before any new request.
        let descs_grown: Option<std::collections::BTreeMap<String, String>> = {
            let mut descs = state.tool_descs.lock().unwrap_or_else(|e| e.into_inner());
            let entry = descs
                .entry(surface.to_string())
                .or_insert_with(|| crate::settings::load_tool_descs(surface));
            let before = entry.len();
            for t in &req.tools {
                if !t.description.is_empty() {
                    entry.insert(t.name.clone(), t.description.clone());
                }
            }
            if entry.len() != before {
                Some(entry.clone())
            } else {
                None
            }
        };
        if let Some(descs) = descs_grown {
            if let Err(e) = crate::settings::save_tool_descs(surface, &descs) {
                tracing::warn!("failed to persist tool descriptions for {surface}: {e}");
            }
        }
        // Union the request's tool names into the seen set (seeded from the
        // persisted set the first time this surface is touched this session).
        let grown: Option<Vec<String>> = {
            let mut reg = state
                .tool_registry
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let entry = reg
                .entry(surface.to_string())
                .or_insert_with(|| crate::settings::load_tool_seen(surface));
            let before = entry.len();
            for t in &req.tools {
                if !entry.contains(&t.name) {
                    entry.push(t.name.clone());
                }
            }
            entry.sort();
            entry.dedup();
            if entry.len() != before {
                Some(entry.clone())
            } else {
                None
            }
        };
        if let Some(seen) = grown {
            if let Err(e) = crate::settings::save_tool_seen(surface, &seen) {
                tracing::warn!("failed to persist seen tools for {surface}: {e}");
            }
        }
    }
    // 2) filter disabled
    let disabled = crate::settings::load_tool_filter(surface);
    if !disabled.is_empty() {
        let tools = std::mem::take(&mut req.tools);
        req.tools = crate::api::common::filter_tools(tools, &disabled);
    }
}

/// Trim a request's history to the active model's window, in place. When the
/// global smart-history toggle is on, selects turns by relevance (BM25+MMR)
/// instead of pure recency; otherwise falls back to recency truncation.
fn apply_history_window(state: &AppState, req: &mut ChatRequest) {
    let active = state.manager.status().current;
    let keep = resolve_history_turns(&active);
    if let Some(n) = keep {
        let msgs = std::mem::take(&mut req.messages);
        req.messages = if crate::settings::load_smart_history() {
            crate::history_select::select_history_smart(msgs, n)
        } else {
            crate::api::common::truncate_history(msgs, Some(n))
        };
    }
}

/// Bridge a local generation result into the cascade decision.
///
/// - `Ok(result)` and (cascade off OR result is strong) → returns `Ok(result)`;
///   the caller proceeds with the local result as before.
/// - `Ok(result)` weak (length-truncated) AND `want_cascade` → escalates: on
///   Relayed records success and returns `Err(cloud response)`; on Degrade keeps
///   the already-computed local result.
/// - `Err(gen error)` → if `want_cascade`, escalate to cloud; otherwise return
///   `Err(500)`.
///
/// Consumes `raw` (the original request bytes) because escalation reverse-proxies
/// it. Only ever called on the buffered/non-stream paths — never mid-stream.
#[allow(clippy::too_many_arguments)]
async fn cascade_or_result(
    want_cascade: bool,
    gen_result: anyhow::Result<ChatResult>,
    provider: crate::cloud::Provider,
    upstream_path: &str,
    headers: &HeaderMap,
    raw: Bytes,
    state: &AppState,
    est_prompt_tokens: usize,
    rid: &str,
    api: &str,
) -> Result<ChatResult, axum::response::Response> {
    use axum::response::IntoResponse;
    match gen_result {
        Ok(result) => {
            if want_cascade && crate::route::is_weak_result(&result) {
                tracing::info!(target: "localllm::req", "{rid} [{api}] cascade: weak local (length) → escalating to cloud");
                crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
                    rid: rid.to_string(),
                    ts: crate::route_log::now_secs(),
                    signal: "cascade".to_string(),
                });
                match crate::cloud::forward(
                    provider,
                    upstream_path,
                    headers,
                    raw,
                    Some(crate::cloud::RelayMeter {
                        rid: rid.to_string(),
                    }),
                )
                .await
                {
                    crate::cloud::ForwardOutcome::Relayed(resp) => {
                        record_cloud_success(state, est_prompt_tokens);
                        Err(resp)
                    }
                    crate::cloud::ForwardOutcome::Degrade(d) => {
                        if state.usage.note_degrade() {
                            crate::usage::notify("localllm — cloud degraded", d.message());
                        }
                        state
                            .breaker
                            .on_failure(crate::route_log::now_secs() as u64, d);
                        tracing::warn!(target: "localllm::req", "{rid} [{api}] cascade cloud degraded ({d:?}) → keeping local result");
                        Ok(result) // serve the (truncated) local answer we already have
                    }
                }
            } else {
                Ok(result)
            }
        }
        Err(e) => {
            if want_cascade {
                tracing::warn!(target: "localllm::req", "{rid} [{api}] cascade: local generate failed ({e}) → escalating to cloud");
                crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
                    rid: rid.to_string(),
                    ts: crate::route_log::now_secs(),
                    signal: "cascade".to_string(),
                });
                match crate::cloud::forward(
                    provider,
                    upstream_path,
                    headers,
                    raw,
                    Some(crate::cloud::RelayMeter {
                        rid: rid.to_string(),
                    }),
                )
                .await
                {
                    crate::cloud::ForwardOutcome::Relayed(resp) => {
                        record_cloud_success(state, est_prompt_tokens);
                        Err(resp)
                    }
                    crate::cloud::ForwardOutcome::Degrade(d) => {
                        if state.usage.note_degrade() {
                            crate::usage::notify("localllm — cloud degraded", d.message());
                        }
                        state
                            .breaker
                            .on_failure(crate::route_log::now_secs() as u64, d);
                        Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("local generation failed and cloud degraded ({d:?})")}))).into_response())
                    }
                }
            } else {
                tracing::error!(target: "localllm::req", "{rid} [{api}] 500 generate: {e}");
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generator trait — decouples HTTP handlers from the real Engine
// ---------------------------------------------------------------------------

/// Async-trait wrapper so handlers can hold `Arc<dyn Generator>` and tests
/// can inject a `FakeGen` without loading the real model.
#[async_trait::async_trait]
pub trait Generator: Send + Sync {
    async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult>;

    /// Stream inference. Returns a boxed stream of `StreamDelta` so the trait
    /// remains object-safe (`impl Stream` is not dyn-compatible).
    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>>;
}

// ---------------------------------------------------------------------------
// Router state — bundles the generator with the configured model id
// ---------------------------------------------------------------------------

/// Shared state injected into every handler via axum's `State` extractor.
#[derive(Clone)]
pub struct AppState {
    /// The swappable inference backend (also the Generator the handlers call).
    pub manager: Arc<crate::model_manager::ModelManager>,
    /// The model identifier reported by `GET /v1/models` and used in responses.
    pub model_id: String,
    /// Active routing policy, shared with the tray (writer) — read per request.
    pub policy: Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    /// Local model's usable context window (config `ctx_len`), for the ctx gate.
    pub local_ctx_window: usize,
    /// Per-session cloud usage counters + notification gates.
    pub usage: std::sync::Arc<crate::usage::Usage>,
    /// Session cloud-token total that triggers the one-shot high-usage alert.
    pub cloud_token_alert: usize,
    /// Token required on /admin/* endpoints.
    pub admin_token: Arc<str>,
    /// Total physical RAM in MB (read once at startup), for catalog fit/recommend.
    pub total_ram_mb: u64,
    /// Requested context ceiling passed to catalog_view for per-model ctx bounds.
    pub requested_ctx_ceiling: u32,
    /// KV cache kind used for catalog RAM estimates.
    pub kv_kind: crate::fit::KvKind,
    /// Per-surface discovery of the tool names seen on recent requests
    /// (surface -> latest sorted-unique names). In-memory; re-discovered on restart.
    pub tool_registry:
        std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, Vec<String>>>>,
    /// Per-surface tool descriptions seen this session (surface -> name -> desc).
    /// In-memory only; re-captured as requests carry them. Rendered in the Tools
    /// view when available.
    pub tool_descs: std::sync::Arc<
        std::sync::Mutex<
            std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>>,
        >,
    >,
    /// TCP port this server listens on — needed to (re)wire agent client configs
    /// via `POST /admin/integrations`.
    pub port: u16,
    /// Daily cloud-spend tracker for the budget cap.
    pub budget: std::sync::Arc<crate::budget::Budget>,
    /// Global cloud circuit breaker (skip cloud during cooldown after failures).
    pub breaker: std::sync::Arc<crate::breaker::CircuitBreaker>,
    /// Per-surface ring of recent LOCAL decisions (rid, ts, prompt token set)
    /// for re-ask detection. In-memory; cap REASK_BUFFER_CAP per surface.
    pub recent_prompts: std::sync::Arc<
        std::sync::Mutex<
            std::collections::BTreeMap<String, std::collections::VecDeque<RecentPrompt>>,
        >,
    >,
}

// Implement Generator for Engine by delegating to its inherent methods.
#[async_trait::async_trait]
impl Generator for crate::engine::Engine {
    async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult> {
        crate::engine::Engine::generate(self, req).await
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        crate::engine::Engine::generate_stream(self, req).await
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Wire all routes onto a pre-built `Arc<AppState>`. Extracted from `router()`
/// so tests can inject a custom state (e.g. with a seeded tool registry).
fn build_router_inner(state: Arc<AppState>) -> Router {
    // Large prompts must reach the routing layer to be forwarded to cloud;
    // 64 MB ≈ ~16 M chars, giving ample headroom for over-window requests.
    // axum's default is 2 MB, which would reject them with 413 before routing.
    Router::new()
        .route("/v1/chat/completions", post(handle_oai_chat))
        .route("/v1/messages", post(handle_anth_messages))
        .route("/v1/responses", post(handle_oai_responses))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .route("/admin/model", post(handle_admin_switch))
        .route("/admin/model/status", get(handle_admin_status))
        .route("/admin/model/ctx", post(handle_model_set_ctx))
        .route("/admin/model/profile", post(handle_model_set_profile))
        .route(
            "/admin/models",
            get(handle_models_catalog).delete(handle_model_delete),
        )
        .route("/admin/tools", get(handle_tools_get).post(handle_tools_set))
        .route(
            "/admin/integrations",
            get(handle_integrations_get).post(handle_integrations_set),
        )
        .route(
            "/admin/routing",
            get(handle_routing_get).post(handle_routing_set),
        )
        .route(
            "/admin/threshold",
            get(handle_threshold_get).post(handle_threshold_set),
        )
        .route(
            "/admin/budget",
            get(handle_budget_get).post(handle_budget_set),
        )
        .route("/admin/breaker", get(handle_breaker_get))
        .route("/admin/breaker/reset", post(handle_breaker_reset))
        .route(
            "/admin/history-filter",
            get(handle_history_filter_get).post(handle_history_filter_set),
        )
        .route(
            "/admin/dashboard",
            get(handle_dashboard).delete(handle_dashboard_clear),
        )
        .route("/admin/export", get(handle_export))
        .route("/metrics", get(handle_metrics))
        .route("/manager", get(handle_manager_page))
        .route("/manager/app.js", get(handle_manager_js))
        .route("/manager/style.css", get(handle_manager_css))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

/// Build the axum Router with all four endpoints wired to the given generator,
/// the configured model id, the shared routing policy, the local context
/// window used by the context gate, and the session usage tracker.
#[allow(clippy::too_many_arguments)]
pub fn router(
    manager: Arc<crate::model_manager::ModelManager>,
    model_id: String,
    policy: Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    local_ctx_window: usize,
    usage: std::sync::Arc<crate::usage::Usage>,
    cloud_token_alert: usize,
    admin_token: Arc<str>,
    total_ram_mb: u64,
    requested_ctx_ceiling: u32,
    kv_kind: crate::fit::KvKind,
    port: u16,
    breaker: std::sync::Arc<crate::breaker::CircuitBreaker>,
) -> Router {
    let state = Arc::new(AppState {
        manager,
        model_id,
        policy,
        local_ctx_window,
        usage,
        cloud_token_alert,
        admin_token,
        total_ram_mb,
        requested_ctx_ceiling,
        kv_kind,
        tool_registry: std::sync::Arc::new(
            std::sync::Mutex::new(std::collections::BTreeMap::new()),
        ),
        tool_descs: std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new())),
        port,
        budget: {
            let b = std::sync::Arc::new(crate::budget::Budget::new());
            b.seed_from_log(&crate::route_log::read_all(), crate::route_log::now_secs());
            b
        },
        breaker,
        recent_prompts: std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeMap::new(),
        )),
    });
    build_router_inner(state)
}

/// GET /admin/history-filter — global smart-history toggle state (token-guarded).
async fn handle_history_filter_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    Json(json!({ "enabled": crate::settings::load_smart_history() })).into_response()
}

#[derive(serde::Deserialize)]
struct HistoryFilterSetBody {
    enabled: bool,
}

/// POST /admin/history-filter {enabled} — set the global smart-history toggle.
async fn handle_history_filter_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: HistoryFilterSetBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    if let Err(e) = crate::settings::save_smart_history(body.enabled) {
        tracing::warn!("failed to persist smart-history toggle: {e}");
    }
    Json(json!({ "enabled": body.enabled })).into_response()
}

/// GET /admin/routing — current routing profile + selectable options (token-guarded).
async fn handle_routing_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let current = crate::settings::load_profile();
    let options: Vec<_> = crate::route::Profile::ALL
        .iter()
        .map(|p| json!({ "value": p, "label": p.label() }))
        .collect();
    Json(json!({ "current": current, "options": options })).into_response()
}

#[derive(serde::Deserialize)]
struct RoutingSetBody {
    profile: crate::route::Profile,
}

/// POST /admin/routing {profile} — apply + persist the routing profile (token-guarded).
async fn handle_routing_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: RoutingSetBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    // Resolve applies the user's Balanced threshold override, if any.
    *state.policy.write().unwrap_or_else(|e| e.into_inner()) =
        crate::settings::resolve_policy(body.profile);
    if let Err(e) = crate::settings::save_profile(body.profile) {
        tracing::warn!("failed to persist routing profile: {e}");
    }
    Json(json!({ "current": body.profile })).into_response()
}

/// GET /admin/threshold — the Balanced difficulty cutoff as a percent (token-guarded).
async fn handle_threshold_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let t = crate::settings::load_balanced_threshold();
    Json(json!({ "percent": (t * 100.0).round() as i64, "value": t })).into_response()
}

#[derive(serde::Deserialize)]
struct ThresholdSetBody {
    percent: f64,
}

/// POST /admin/threshold {percent} — set the Balanced difficulty cutoff (0–100).
/// Persists and, if Balanced is the active profile, updates the live policy.
async fn handle_threshold_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: ThresholdSetBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    let t = (body.percent / 100.0).clamp(0.0, 1.0);
    if let Err(e) = crate::settings::save_balanced_threshold(t) {
        tracing::warn!("failed to persist balanced threshold: {e}");
    }
    // Live-apply only when Balanced is active; other profiles keep their knobs.
    if crate::settings::load_profile() == crate::route::Profile::Balanced {
        state
            .policy
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .escalation_threshold = t;
    }
    Json(json!({ "percent": (t * 100.0).round() as i64, "value": t })).into_response()
}

/// GET /admin/budget — budget cap config + today's spend (token-guarded).
async fn handle_budget_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let (enabled, daily_usd) = crate::settings::load_budget();
    let spent = state.budget.spent_today(crate::route_log::now_secs());
    let remaining = (daily_usd - spent).max(0.0);
    let over = enabled && daily_usd > 0.0 && spent >= daily_usd;
    Json(json!({ "enabled": enabled, "daily_usd": daily_usd,
        "spent_today": spent, "remaining": remaining, "over": over }))
    .into_response()
}

#[derive(serde::Deserialize)]
struct BudgetSetBody {
    enabled: Option<bool>,
    daily_usd: Option<f64>,
}

/// POST /admin/budget {enabled?, daily_usd?} — update the budget cap (token-guarded).
async fn handle_budget_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: BudgetSetBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    let (cur_en, cur_usd) = crate::settings::load_budget();
    let enabled = body.enabled.unwrap_or(cur_en);
    let daily_usd = body.daily_usd.unwrap_or(cur_usd).max(0.0);
    if let Err(e) = crate::settings::save_budget(enabled, daily_usd) {
        tracing::warn!("failed to persist budget: {e}");
    }
    handle_budget_get(State(state), headers).await
}

/// GET /admin/breaker — current circuit-breaker status (token-guarded).
async fn handle_breaker_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let s = state.breaker.snapshot(crate::route_log::now_secs() as u64);
    Json(json!({
        "state": s.state,
        "reason": s.reason,
        "next_probe_secs": s.next_probe_secs,
    }))
    .into_response()
}

/// POST /admin/breaker/reset — manually force-close the breaker (retry cloud now).
async fn handle_breaker_reset(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    state.breaker.reset(crate::route_log::now_secs() as u64);
    Json(json!({ "ok": true, "state": "closed" })).into_response()
}

/// GET /admin/integrations — current wiring state (token-guarded).
async fn handle_integrations_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let st = crate::settings::load_integrations();
    let wired: Vec<String> = st.priors.keys().cloned().collect();
    Json(json!({ "enabled": st.enabled, "wired": wired })).into_response()
}

#[derive(serde::Deserialize)]
struct IntegrationsSetBody {
    enabled: bool,
}

/// POST /admin/integrations {enabled} — wire or unwire agent clients (token-guarded).
async fn handle_integrations_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: IntegrationsSetBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    let injectors = crate::integrations::injectors_default();
    let new_state = if body.enabled {
        let outcome = crate::integrations::enable_all(state.port, &injectors);
        crate::settings::IntegrationState {
            enabled: !outcome.priors.is_empty(),
            priors: outcome.priors,
        }
    } else {
        let mut st = crate::settings::load_integrations();
        let summary = crate::integrations::disable_all(&st.priors, &injectors);
        let failed: std::collections::HashSet<&String> =
            summary.failed.iter().map(|(id, _)| id).collect();
        st.priors.retain(|id, _| failed.contains(id));
        st.enabled = !st.priors.is_empty();
        st
    };
    let _ = crate::settings::save_integrations(&new_state);
    let wired: Vec<String> = new_state.priors.keys().cloned().collect();
    Json(json!({ "enabled": new_state.enabled, "wired": wired })).into_response()
}

/// GET /admin/dashboard — routing rollups + recent decisions (token-guarded).
async fn handle_dashboard(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let entries = crate::route_log::read_all();
    let balanced = crate::settings::load_profile() == crate::route::Profile::Balanced;
    let dash =
        crate::route_log::build_dashboard(&entries, crate::route_log::now_secs(), 50, balanced);
    Json(dash).into_response()
}

/// DELETE /admin/dashboard — clear all routing history (token-guarded).
async fn handle_dashboard_clear(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let cleared = crate::route_log::clear();
    Json(json!({ "cleared": cleared })).into_response()
}

/// Flatten the log to CSV (decisions + outcomes, one row each).
fn export_csv(lines: &[crate::route_log::LogLine]) -> String {
    use crate::route_log::LogLine;
    let mut out = String::from(
        "kind,rid,ts,surface,dest,reason,degrade_reason,model,prompt_tok,completion_tok,ttft_ms,gen_ms,cost_saved_usd\n");
    let esc = |s: &str| {
        if s.contains(',') || s.contains('"') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    };
    for l in lines {
        match l {
            LogLine::Decision(d) => out.push_str(&format!(
                "d,{},{},{},{},{},{},{},{},,,,\n",
                esc(&d.rid),
                d.ts,
                esc(&d.surface),
                esc(&d.dest),
                esc(d.reason.as_deref().unwrap_or("")),
                esc(d.degrade_reason.as_deref().unwrap_or("")),
                esc(d.model.as_deref().unwrap_or("")),
                d.prompt_tok
            )),
            LogLine::Outcome(o) => out.push_str(&format!(
                "o,{},{},,,,,,,{},{},{},{}\n",
                esc(&o.rid),
                o.ts,
                o.completion_tok.map(|v| v.to_string()).unwrap_or_default(),
                o.ttft_ms.map(|v| v.to_string()).unwrap_or_default(),
                o.gen_ms.map(|v| v.to_string()).unwrap_or_default(),
                o.cost_saved_usd
            )),
            LogLine::Feedback(f) => out.push_str(&format!(
                "f,{},{},,,{},,,,,,,\n",
                esc(&f.rid),
                f.ts,
                esc(&f.signal)
            )),
        }
    }
    out
}

#[derive(serde::Deserialize)]
struct ExportQuery {
    format: Option<String>,
    token: Option<String>,
}

/// GET /admin/export?format=jsonl|csv&token=… — download the routing log.
/// Token comes via query (browser downloads can't set headers).
async fn handle_export(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<ExportQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !constant_time_eq(
        q.token.as_deref().unwrap_or("").as_bytes(),
        state.admin_token.as_bytes(),
    ) {
        return (StatusCode::UNAUTHORIZED, "invalid admin token").into_response();
    }
    let lines = crate::route_log::read_all();
    let (body, ct, disp) = if q.format.as_deref() == Some("csv") {
        (
            export_csv(&lines),
            "text/csv",
            "attachment; filename=\"routing-log.csv\"",
        )
    } else {
        let jsonl = lines
            .iter()
            .filter_map(|l| serde_json::to_string(l).ok())
            .collect::<Vec<_>>()
            .join("\n");
        (
            jsonl,
            "application/x-ndjson",
            "attachment; filename=\"routing-log.jsonl\"",
        )
    };
    (
        [
            (axum::http::header::CONTENT_TYPE, ct),
            (axum::http::header::CONTENT_DISPOSITION, disp),
        ],
        body,
    )
        .into_response()
}

/// Render 30-day routing counters in Prometheus text format.
fn metrics_text(lines: &[crate::route_log::LogLine], now: i64) -> String {
    let d = crate::route_log::build_dashboard(lines, now, 0, false);
    format!(
        "# HELP localllm_requests_total Requests by destination (30d)\n\
         # TYPE localllm_requests_total counter\n\
         localllm_requests_total{{dest=\"local\"}} {}\n\
         localllm_requests_total{{dest=\"cloud\"}} {}\n\
         # HELP localllm_cost_saved_usd_total USD saved by local routing (30d)\n\
         # TYPE localllm_cost_saved_usd_total counter\n\
         localllm_cost_saved_usd_total {:.6}\n\
         # HELP localllm_ttft_ms Average TTFT by route (30d)\n\
         # TYPE localllm_ttft_ms gauge\n\
         localllm_ttft_ms{{dest=\"local\"}} {}\n\
         localllm_ttft_ms{{dest=\"cloud\"}} {}\n",
        d.month.local_count,
        d.month.cloud_count,
        d.month.cost_saved_usd,
        d.local_latency.avg_ttft_ms,
        d.cloud_latency.avg_ttft_ms
    )
}

/// GET /metrics — Prometheus text (token-guarded via header; scrapers set it).
async fn handle_metrics(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let text = metrics_text(&crate::route_log::read_all(), crate::route_log::now_secs());
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        text,
    )
        .into_response()
}

/// Verify the `X-Admin-Token` header against the configured token (constant-time).
/// Returns `Some(401 response)` to reject, `None` when the token is valid.
fn check_admin(headers: &HeaderMap, state: &AppState) -> Option<axum::response::Response> {
    use axum::response::IntoResponse;
    let provided = headers
        .get("x-admin-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if constant_time_eq(provided.as_bytes(), state.admin_token.as_bytes()) {
        None
    } else {
        Some(
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "missing or invalid admin token"})),
            )
                .into_response(),
        )
    }
}

#[derive(serde::Deserialize)]
struct AdminSwitchBody {
    repo: String,
    file: String,
    #[serde(default)]
    quant: Option<String>,
}

#[derive(serde::Deserialize)]
struct AdminCtxBody {
    repo: String,
    file: String,
    ctx: u32,
}

#[derive(serde::Deserialize)]
struct SetProfileBody {
    repo: String,
    file: String,
    #[serde(default)]
    ctx: Option<u32>,
    #[serde(default)]
    kv_type: Option<crate::config::KvType>,
    #[serde(default)]
    gpu_layers: Option<u32>,
    #[serde(default)]
    history_turns: Option<u32>,
    #[serde(default)]
    quant: Option<String>,
}

#[derive(serde::Deserialize)]
struct SetToolsBody {
    surface: String,
    #[serde(default)]
    disabled: Vec<String>,
}

/// A model `file` must be a bare filename (no path components), so it can never
/// escape the cache directory when joined into `download::cache_path`. Rejects
/// `""`, `..`, and anything containing `/` or `\`.
fn is_safe_model_file(file: &str) -> bool {
    !file.is_empty() && std::path::Path::new(file).file_name() == Some(std::ffi::OsStr::new(file))
}

/// POST /admin/model — start a model switch (token-guarded).
async fn handle_admin_switch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: AdminSwitchBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    if body.repo.is_empty() || !is_safe_model_file(&body.file) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "repo and a valid (non-path) file are required"})),
        )
            .into_response();
    }
    let spec = crate::model_manager::ModelSpec {
        repo: body.repo,
        file: body.file,
        quant: body.quant,
    };
    let (save_repo, save_file, save_quant) =
        (spec.repo.clone(), spec.file.clone(), spec.quant.clone());
    match state.manager.start_switch(spec) {
        Ok(()) => {
            if let Err(e) =
                crate::settings::save_active_model(&save_repo, &save_file, save_quant.as_deref())
            {
                tracing::warn!("failed to persist active model: {e}");
            }
            (StatusCode::ACCEPTED, Json(json!({"state": "switching"}))).into_response()
        }
        Err(crate::model_manager::SwitchError::AlreadySwitching) => (
            StatusCode::CONFLICT,
            Json(json!({"error": "switch already in progress"})),
        )
            .into_response(),
    }
}

/// GET /admin/model/status — report switch progress (token-guarded).
async fn handle_admin_status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    Json(state.manager.status()).into_response()
}

/// GET /admin/models — the annotated catalog for this machine (token-guarded).
async fn handle_models_catalog(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let active = state.manager.status().current;
    let view = crate::catalog::catalog_view(
        crate::catalog::CATALOG,
        state.total_ram_mb,
        state.requested_ctx_ceiling,
        state.kv_kind,
        Some(&active),
        |r, f| crate::download::cache_path(r, f).exists(),
        |r, f| crate::settings::load_model_ctx(&crate::settings::model_ctx_key(r, f)),
        |r, f| crate::settings::load_model_profile(&crate::settings::model_ctx_key(r, f)),
    );
    Json(view).into_response()
}

/// DELETE /admin/models — free disk for a downloaded model (token-guarded).
/// Refuses the in-use model.
async fn handle_model_delete(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: AdminSwitchBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    if body.repo.is_empty() || !is_safe_model_file(&body.file) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "repo and a valid (non-path) file are required"})),
        )
            .into_response();
    }
    let active = state.manager.status().current;
    if active.repo == body.repo && active.file == body.file {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "cannot delete the model in use"})),
        )
            .into_response();
    }
    let entry = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == body.repo && e.file == body.file);
    let files: Vec<String> = match (&body.quant, entry) {
        (Some(q), Some(e)) => {
            crate::catalog::files_for_quant(e, q).unwrap_or_else(|| vec![body.file.clone()])
        }
        _ => vec![body.file.clone()],
    };
    let mut any = false;
    for f in &files {
        any |= crate::download::delete_cached(&body.repo, f).unwrap_or(false);
    }
    Json(json!({"deleted": any})).into_response()
}

/// POST /admin/model/ctx — set (or clear, ctx=0) a model's per-model context.
/// Validates against the model's [min,max]; reloads the model if it is active.
async fn handle_model_set_ctx(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: AdminCtxBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    if body.repo.is_empty() || !is_safe_model_file(&body.file) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "repo and a valid (non-path) file are required"})),
        )
            .into_response();
    }
    let Some(entry) = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == body.repo && e.file == body.file)
    else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "unknown model"})),
        )
            .into_response();
    };

    let key = crate::settings::model_ctx_key(&body.repo, &body.file);
    let cleared = body.ctx == 0;
    if cleared {
        if let Err(e) = crate::settings::clear_model_ctx(&key) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    } else {
        let budget_mb = crate::fit::device_budget_mb(state.total_ram_mb, true);
        let kv_per_token = crate::fit::est_kv_bytes_per_token(entry.params_b, state.kv_kind);
        let bounds =
            crate::fit::ctx_bounds(entry.size_mb, kv_per_token, budget_mb, entry.ctx_train);
        if bounds.max == 0 || body.ctx < bounds.min || body.ctx > bounds.max {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!("ctx must be in the range {}..{}", bounds.min, bounds.max)
                })),
            )
                .into_response();
        }
        if let Err(e) = crate::settings::save_model_ctx(&key, body.ctx) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    }

    // Reload if this is the active model so the new ctx takes effect now.
    let active = state.manager.status().current;
    if active.repo == body.repo && active.file == body.file {
        let spec = crate::model_manager::ModelSpec {
            repo: body.repo,
            file: body.file,
            quant: None,
        };
        match state.manager.start_switch(spec) {
            Ok(()) => (StatusCode::ACCEPTED, Json(json!({"reloading": true}))).into_response(),
            Err(crate::model_manager::SwitchError::AlreadySwitching) => (
                StatusCode::CONFLICT,
                Json(json!({"error": "a switch is already in progress"})),
            )
                .into_response(),
        }
    } else if cleared {
        (StatusCode::OK, Json(json!({"cleared": true}))).into_response()
    } else {
        (StatusCode::OK, Json(json!({"saved": true}))).into_response()
    }
}

/// POST /admin/model/profile — set a subset of a model's execution profile.
/// Each field is optional; `ctx == 0` clears, `gpu_layers == u32::MAX` clears,
/// `history_turns == 0` clears. Reloads the model if it is the active one.
async fn handle_model_set_profile(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: SetProfileBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    if body.repo.is_empty() || !is_safe_model_file(&body.file) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "repo and a valid (non-path) file are required"})),
        )
            .into_response();
    }
    let Some(entry) = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == body.repo && e.file == body.file)
    else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "unknown model"})),
        )
            .into_response();
    };

    let key = crate::settings::model_ctx_key(&body.repo, &body.file);
    let mut prof = crate::settings::load_model_profile(&key);

    if let Some(ctx) = body.ctx {
        if ctx == 0 {
            prof.ctx = None;
        } else {
            // Effective KV kind: incoming body > existing profile > entry rec > global default.
            let eff_kv_kind = match body
                .kv_type
                .as_ref()
                .or(prof.kv_type.as_ref())
                .or(entry.rec_kv.as_ref())
            {
                Some(crate::config::KvType::Q4) => crate::fit::KvKind::Q4,
                Some(crate::config::KvType::F16) => crate::fit::KvKind::F16,
                Some(crate::config::KvType::Q8) | None => state.kv_kind,
            };
            let budget_mb = crate::fit::device_budget_mb(state.total_ram_mb, true);
            let kv_per_token = crate::fit::est_kv_bytes_per_token(entry.params_b, eff_kv_kind);
            let bounds =
                crate::fit::ctx_bounds(entry.size_mb, kv_per_token, budget_mb, entry.ctx_train);
            if bounds.max == 0 || ctx < bounds.min || ctx > bounds.max {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": format!("ctx must be in the range {}..{}", bounds.min, bounds.max)
                    })),
                )
                    .into_response();
            }
            prof.ctx = Some(ctx);
        }
    }

    if let Some(k) = body.kv_type {
        prof.kv_type = Some(k);
    }
    if let Some(g) = body.gpu_layers {
        prof.gpu_layers = if g == u32::MAX { None } else { Some(g) };
    }
    if let Some(h) = body.history_turns {
        prof.history_turns = if h == 0 { None } else { Some(h) };
    }

    if let Some(q) = body.quant {
        let ok = crate::catalog::variants_for(entry)
            .iter()
            .any(|v| v.quant.eq_ignore_ascii_case(&q));
        if !ok {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("unknown quant '{q}' for this model")})),
            )
                .into_response();
        }
        prof.quant = Some(q);
    }

    if let Err(e) = crate::settings::save_model_profile(&key, &prof) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response();
    }

    // Reload if this is the active model so the new profile takes effect now.
    let active = state.manager.status().current;
    if active.repo == body.repo && active.file == body.file {
        let spec = crate::model_manager::ModelSpec {
            repo: body.repo,
            file: body.file,
            quant: prof.quant.clone(),
        };
        match state.manager.start_switch(spec) {
            Ok(()) => (StatusCode::ACCEPTED, Json(json!({"reloading": true}))).into_response(),
            Err(crate::model_manager::SwitchError::AlreadySwitching) => (
                StatusCode::CONFLICT,
                Json(json!({"error": "a switch is already in progress"})),
            )
                .into_response(),
        }
    } else {
        (StatusCode::OK, Json(json!({"saved": true}))).into_response()
    }
}

/// The three known API surfaces (must stay in sync with shape_tools call-sites).
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
    let reg = state
        .tool_registry
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let descs = state
        .tool_descs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let mut out = serde_json::Map::new();
    for surface in KNOWN_SURFACES {
        // In-memory registry is the live source once a request has arrived this
        // session; fall back to the persisted set so tools show right after boot.
        let seen = match reg.get(surface) {
            Some(s) => s.clone(),
            None => crate::settings::load_tool_seen(surface),
        };
        let disabled = crate::settings::load_tool_filter(surface);
        if seen.is_empty() && disabled.is_empty() {
            continue;
        }
        // Live map once a request has arrived this session; else the persisted
        // set so descriptions expand right after boot.
        let descriptions = match descs.get(surface) {
            Some(d) => d.clone(),
            None => crate::settings::load_tool_descs(surface),
        };
        out.insert(
            surface.to_string(),
            json!({ "seen": seen, "disabled": disabled, "descriptions": descriptions }),
        );
    }
    Json(serde_json::Value::Object(out)).into_response()
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
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    if !KNOWN_SURFACES.contains(&body.surface.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "unknown surface"})),
        )
            .into_response();
    }
    match crate::settings::save_tool_filter(&body.surface, &body.disabled) {
        Ok(()) => Json(json!({"saved": true})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/chat/completions
async fn handle_oai_chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use futures::StreamExt;
    use uuid::Uuid;

    let rid = new_request_id();

    let req: OaiChatRequest = match serde_json::from_slice(&raw) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] 400 bad json: {e}");
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };
    let model = req.model.clone();
    let stream_flag = req.stream.unwrap_or(false);

    let internal = match crate::api::openai::to_internal(req) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] 400 bad request: {e}");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response();
        }
    };
    let mut internal = internal;
    apply_history_window(&state, &mut internal);
    shape_tools(&state, "openai", &mut internal);

    // --- Routing decision ---
    if state.manager.is_errored() {
        // The previous model could not be restored — a restart is required.
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "model in error state — restart required"})),
        )
            .into_response();
    }
    if state.manager.is_switching() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("Retry-After", "5")],
            Json(json!({"error": "model switching, retry shortly"})),
        )
            .into_response();
    }

    let (decision, est_prompt_tokens) = route_decision(&state, &internal, &headers, &rid, "openai");
    let want_cascade = match decision {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [openai] route=cloud reason={reason:?}");
            match crate::cloud::forward(
                crate::cloud::Provider::OpenAI,
                "/v1/chat/completions",
                &headers,
                raw.clone(),
                Some(crate::cloud::RelayMeter {
                    rid: rid.to_string(),
                }),
            )
            .await
            {
                crate::cloud::ForwardOutcome::Relayed(resp) => {
                    record_cloud_success(&state, est_prompt_tokens);
                    return resp;
                }
                crate::cloud::ForwardOutcome::Degrade(d) => {
                    if let Some(resp) = handle_degrade(&state, d, reason) {
                        return resp;
                    }
                    tracing::warn!(target: "localllm::req", "{rid} [openai] cloud degraded ({d:?}) → serving local");
                    log_degrade_fallback(&rid, "openai", Some(&model), est_prompt_tokens as u64, d);
                    false
                }
            }
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] route=local (cloud wanted but no creds/disallowed)");
            false
        }
        crate::route::Decision::LocalThenCascade => true,
        crate::route::Decision::Local => false,
    };

    let (n_msgs, n_tools) = request_summary(&internal);
    tracing::info!(target: "localllm::req", "{rid} [openai] start: model={model} msgs={n_msgs} tools={n_tools} stream={stream_flag}");
    let started_at = Instant::now();

    if stream_flag && n_tools > 0 {
        // Buffered streaming: tool-bearing requests are generated fully (reusing
        // the tested non-streaming tool-call path) then replayed as SSE chunks.
        // Incremental streaming cannot frame tool_calls safely.
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let result = match cascade_or_result(
            want_cascade,
            state.manager.generate(internal).await,
            crate::cloud::Provider::OpenAI,
            "/v1/chat/completions",
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "openai",
        )
        .await
        {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 {
            result.completion_tokens as f64 / secs
        } else {
            0.0
        };
        tracing::info!(target: "localllm::req", "{rid} [openai] done (buffered stream): finish={:?} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.completion_tokens);
        record_outcome(
            &rid,
            "local",
            Some(&model),
            est_prompt_tokens as u64,
            Some(result.completion_tokens as u64),
            None,
            Some(started_at.elapsed().as_millis() as u64),
            Some(result.finish_reason.clone()),
        );
        let mut lines = crate::api::openai::stream_chunks_from_result(&result, &id, &model);
        lines.push("[DONE]".to_string());
        let sse_stream = futures::stream::iter(lines.into_iter().map(|l| {
            let data = l.strip_prefix("data: ").unwrap_or(&l).to_string();
            Ok::<Event, Infallible>(Event::default().data(data))
        }));
        return Sse::new(sse_stream).into_response();
    }

    if stream_flag {
        // Streaming path: return SSE
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let id_clone = id.clone();
        let model_clone = model.clone();
        let rid_stream = rid.clone();

        let delta_stream = match state.manager.generate_stream(internal).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(target: "localllm::req", "{rid} [openai] 500 stream init: {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response();
            }
        };

        // Scan state: (started, completion_tok_estimate, first_delta_at). `started`
        // threads the "role":"assistant" flag per the OpenAI streaming spec; the
        // other two capture latency (TTFT) and an estimated completion count so we
        // can record the outcome when the stream finishes.
        let est_prompt_tokens = est_prompt_tokens;
        let sse_stream = delta_stream
            .scan((false, 0u64, None::<Instant>), move |st, result| {
                let (started, ctok, first_at) = st;
                let was_started = *started;
                let event: Result<Event, Infallible> = match result {
                    Ok(delta) => {
                        if first_at.is_none() { *first_at = Some(Instant::now()); }
                        if let Some(t) = &delta.text {
                            *ctok += crate::route::estimate_text_tokens(t) as u64;
                        }
                        if delta.done {
                            let secs = started_at.elapsed().as_secs_f64();
                            tracing::info!(target: "localllm::req", "{rid_stream} [openai] done (stream): {secs:.1}s");
                            let ttft = first_at.map(|f| (f - started_at).as_millis() as u64);
                            record_outcome(&rid_stream, "local", Some(&model_clone),
                                est_prompt_tokens as u64, Some(*ctok), ttft,
                                Some(started_at.elapsed().as_millis() as u64),
                                delta.finish_reason.clone());
                        }
                        let line = crate::api::openai::stream_chunk(&delta, &id_clone, &model_clone, was_started);
                        *started = true;
                        // line is "data: {json}", strip the "data: " prefix for Event::default().data()
                        let data = line.strip_prefix("data: ").unwrap_or(&line);
                        Ok(Event::default().data(data.to_string()))
                    }
                    Err(e) => {
                        tracing::error!(target: "localllm::req", "{rid_stream} [openai] stream error: {e}");
                        // Emit an error event; client will see it.
                        Ok(Event::default().data(format!("[ERROR] {e}")))
                    }
                };
                futures::future::ready(Some(event))
            })
            // Append the [DONE] sentinel after all deltas
            .chain(futures::stream::once(async {
                Ok::<Event, Infallible>(Event::default().data("[DONE]"))
            }));

        Sse::new(sse_stream).into_response()
    } else {
        // Non-streaming path
        let result = match cascade_or_result(
            want_cascade,
            state.manager.generate(internal).await,
            crate::cloud::Provider::OpenAI,
            "/v1/chat/completions",
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "openai",
        )
        .await
        {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 {
            result.completion_tokens as f64 / secs
        } else {
            0.0
        };
        tracing::info!(target: "localllm::req", "{rid} [openai] done: finish={:?} prompt_tok={} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.prompt_tokens, result.completion_tokens);
        record_outcome(
            &rid,
            "local",
            Some(&model),
            est_prompt_tokens as u64,
            Some(result.completion_tokens as u64),
            None,
            Some(started_at.elapsed().as_millis() as u64),
            Some(result.finish_reason.clone()),
        );
        let resp = crate::api::openai::from_internal(result, &model);
        Json(serde_json::to_value(resp).unwrap()).into_response()
    }
}

/// POST /v1/responses  (OpenAI Responses API)
async fn handle_oai_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use uuid::Uuid;

    let rid = new_request_id();

    let req: crate::api::openai_responses::RespRequest = match serde_json::from_slice(&raw) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [responses] 400 bad json: {e}");
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };
    let model = req.model.clone();
    let stream_flag = req.stream.unwrap_or(false);

    let internal = match crate::api::openai_responses::to_internal(req) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [responses] 400 bad request: {e}");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response();
        }
    };
    let mut internal = internal;
    apply_history_window(&state, &mut internal);
    shape_tools(&state, "openai-responses", &mut internal);

    if state.manager.is_errored() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "model in error state — restart required"})),
        )
            .into_response();
    }
    if state.manager.is_switching() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("Retry-After", "5")],
            Json(json!({"error": "model switching, retry shortly"})),
        )
            .into_response();
    }

    let (decision, est_prompt_tokens) =
        route_decision(&state, &internal, &headers, &rid, "openai-responses");
    let want_cascade = match decision {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [responses] route=cloud reason={reason:?}");
            match crate::cloud::forward(
                crate::cloud::Provider::OpenAI,
                "/v1/responses",
                &headers,
                raw.clone(),
                Some(crate::cloud::RelayMeter {
                    rid: rid.to_string(),
                }),
            )
            .await
            {
                crate::cloud::ForwardOutcome::Relayed(resp) => {
                    record_cloud_success(&state, est_prompt_tokens);
                    return resp;
                }
                crate::cloud::ForwardOutcome::Degrade(d) => {
                    if let Some(resp) = handle_degrade(&state, d, reason) {
                        return resp;
                    }
                    tracing::warn!(target: "localllm::req", "{rid} [responses] cloud degraded ({d:?}) → serving local");
                    log_degrade_fallback(
                        &rid,
                        "openai-responses",
                        Some(&model),
                        est_prompt_tokens as u64,
                        d,
                    );
                    false
                }
            }
        }
        crate::route::Decision::LocalNoCreds => false,
        crate::route::Decision::LocalThenCascade => true,
        crate::route::Decision::Local => false,
    };

    let started_at = Instant::now();
    let resp_id = format!("resp_{}", Uuid::new_v4());

    // Buffered for both stream and non-stream: generate the full result, then
    // either render the object or replay it as the Responses SSE sequence.
    let result = match cascade_or_result(
        want_cascade,
        state.manager.generate(internal).await,
        crate::cloud::Provider::OpenAI,
        "/v1/responses",
        &headers,
        raw,
        &state,
        est_prompt_tokens,
        &rid,
        "responses",
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let secs = started_at.elapsed().as_secs_f64();
    let tps = if secs > 0.0 {
        result.completion_tokens as f64 / secs
    } else {
        0.0
    };
    tracing::info!(target: "localllm::req", "{rid} [responses] done: finish={:?} prompt_tok={} completion_tok={} {secs:.1}s {tps:.1} tok/s",
        result.finish_reason, result.prompt_tokens, result.completion_tokens);
    record_outcome(
        &rid,
        "local",
        Some(&model),
        est_prompt_tokens as u64,
        Some(result.completion_tokens as u64),
        None,
        Some(started_at.elapsed().as_millis() as u64),
        Some(result.finish_reason.clone()),
    );

    if stream_flag {
        let events =
            crate::api::openai_responses::stream_events_from_result(&result, &resp_id, &model);
        let sse_stream = futures::stream::iter(
            events
                .into_iter()
                .map(|(ty, data)| Ok::<Event, Infallible>(Event::default().event(ty).data(data))),
        );
        Sse::new(sse_stream).into_response()
    } else {
        let v = crate::api::openai_responses::from_internal(result, &model);
        Json(v).into_response()
    }
}

/// POST /v1/messages
async fn handle_anth_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use futures::StreamExt;

    let rid = new_request_id();

    let req: AnthRequest = match serde_json::from_slice(&raw) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] 400 bad json: {e}");
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };
    let model = req.model.clone();
    let stream_flag = req.stream.unwrap_or(false);

    let internal = match crate::api::anthropic::to_internal(req) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] 400 bad request: {e}");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response();
        }
    };
    let mut internal = internal;
    apply_history_window(&state, &mut internal);
    shape_tools(&state, "anthropic", &mut internal);

    // --- Routing decision ---
    if state.manager.is_errored() {
        // The previous model could not be restored — a restart is required.
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "model in error state — restart required"})),
        )
            .into_response();
    }
    if state.manager.is_switching() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("Retry-After", "5")],
            Json(json!({"error": "model switching, retry shortly"})),
        )
            .into_response();
    }

    let (decision, est_prompt_tokens) =
        route_decision(&state, &internal, &headers, &rid, "anthropic");
    let want_cascade = match decision {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [anthropic] route=cloud reason={reason:?}");
            match crate::cloud::forward(
                crate::cloud::Provider::Anthropic,
                "/v1/messages",
                &headers,
                raw.clone(),
                Some(crate::cloud::RelayMeter {
                    rid: rid.to_string(),
                }),
            )
            .await
            {
                crate::cloud::ForwardOutcome::Relayed(resp) => {
                    record_cloud_success(&state, est_prompt_tokens);
                    return resp;
                }
                crate::cloud::ForwardOutcome::Degrade(d) => {
                    if let Some(resp) = handle_degrade(&state, d, reason) {
                        return resp;
                    }
                    tracing::warn!(target: "localllm::req", "{rid} [anthropic] cloud degraded ({d:?}) → serving local");
                    log_degrade_fallback(
                        &rid,
                        "anthropic",
                        Some(&model),
                        est_prompt_tokens as u64,
                        d,
                    );
                    false
                }
            }
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] route=local (cloud wanted but no creds/disallowed)");
            false
        }
        crate::route::Decision::LocalThenCascade => true,
        crate::route::Decision::Local => false,
    };

    let (n_msgs, n_tools) = request_summary(&internal);
    tracing::info!(target: "localllm::req", "{rid} [anthropic] start: model={model} msgs={n_msgs} tools={n_tools} stream={stream_flag}");
    let started_at = Instant::now();

    if stream_flag && n_tools > 0 {
        // Buffered streaming: tool-bearing requests (e.g. every Claude Code turn)
        // are generated fully then replayed as a correct tool_use/text SSE
        // sequence. Incremental streaming cannot frame tool_use blocks safely.
        let result = match cascade_or_result(
            want_cascade,
            state.manager.generate(internal).await,
            crate::cloud::Provider::Anthropic,
            "/v1/messages",
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "anthropic",
        )
        .await
        {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 {
            result.completion_tokens as f64 / secs
        } else {
            0.0
        };
        tracing::info!(target: "localllm::req", "{rid} [anthropic] done (buffered stream): finish={:?} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.completion_tokens);
        record_outcome(
            &rid,
            "local",
            Some(&model),
            est_prompt_tokens as u64,
            Some(result.completion_tokens as u64),
            None,
            Some(started_at.elapsed().as_millis() as u64),
            Some(result.finish_reason.clone()),
        );
        let events = crate::api::anthropic::stream_events_from_result(&result, &model);
        let sse_stream = futures::stream::iter(events.into_iter().map(|e_str| {
            let mut lines = e_str.splitn(2, '\n');
            let event_type = lines
                .next()
                .unwrap_or("")
                .strip_prefix("event: ")
                .unwrap_or("")
                .to_string();
            let data = lines
                .next()
                .unwrap_or("")
                .strip_prefix("data: ")
                .unwrap_or("")
                .to_string();
            Ok::<Event, Infallible>(Event::default().event(event_type).data(data))
        }));
        return Sse::new(sse_stream).into_response();
    }

    if stream_flag {
        // Streaming path: return Anthropic SSE protocol
        let rid_stream = rid.clone();
        let delta_stream = match state.manager.generate_stream(internal).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(target: "localllm::req", "{rid} [anthropic] 500 stream init: {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response();
            }
        };

        // Collect all events, tracking whether the first chunk was sent.
        // We use scan to thread `started` state through the stream.
        let sse_stream = delta_stream
            .scan(false, move |started, result| {
                let was_started = *started;
                let events: Vec<Result<Event, Infallible>> = match result {
                    Ok(delta) => {
                        if delta.done {
                            let secs = started_at.elapsed().as_secs_f64();
                            tracing::info!(target: "localllm::req", "{rid_stream} [anthropic] done (stream): {secs:.1}s");
                        }
                        let event_strs = crate::api::anthropic::stream_events(&delta, was_started);
                        *started = true;
                        event_strs
                            .into_iter()
                            .map(|e_str| {
                                // Each event string is "event: <type>\ndata: {json}"
                                // Parse it into axum's Event type.
                                let mut lines = e_str.splitn(2, '\n');
                                let event_line = lines.next().unwrap_or("");
                                let data_line = lines.next().unwrap_or("");
                                let event_type = event_line.strip_prefix("event: ").unwrap_or("");
                                let data = data_line.strip_prefix("data: ").unwrap_or("");
                                Ok(Event::default().event(event_type).data(data.to_string()))
                            })
                            .collect()
                    }
                    Err(e) => {
                        tracing::error!(target: "localllm::req", "{rid_stream} [anthropic] stream error: {e}");
                        vec![Ok(Event::default().data(format!("[ERROR] {e}")))]
                    }
                };
                futures::future::ready(Some(futures::stream::iter(events)))
            })
            .flatten();

        Sse::new(sse_stream).into_response()
    } else {
        // Non-streaming path
        let result = match cascade_or_result(
            want_cascade,
            state.manager.generate(internal).await,
            crate::cloud::Provider::Anthropic,
            "/v1/messages",
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "anthropic",
        )
        .await
        {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 {
            result.completion_tokens as f64 / secs
        } else {
            0.0
        };
        tracing::info!(target: "localllm::req", "{rid} [anthropic] done: finish={:?} prompt_tok={} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.prompt_tokens, result.completion_tokens);
        record_outcome(
            &rid,
            "local",
            Some(&model),
            est_prompt_tokens as u64,
            Some(result.completion_tokens as u64),
            None,
            Some(started_at.elapsed().as_millis() as u64),
            Some(result.finish_reason.clone()),
        );
        let resp = crate::api::anthropic::from_internal(result, &model);
        Json(serde_json::to_value(resp).unwrap()).into_response()
    }
}

/// GET /v1/models — reports the configured model id.
async fn handle_models(State(state): State<Arc<AppState>>) -> Json<OaiModelList> {
    Json(OaiModelList {
        object: "list".to_string(),
        data: vec![OaiModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            owned_by: "localllm".to_string(),
        }],
    })
}

/// GET /health
async fn handle_health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// Test-only: build a router whose tool registry is pre-seeded, so tests can
/// verify that `GET /admin/tools` merges in-memory `seen` with saved `disabled`.
#[cfg(test)]
fn make_seeded_test_router(
    tool_registry: std::sync::Arc<
        std::sync::Mutex<std::collections::BTreeMap<String, Vec<String>>>,
    >,
) -> Router {
    use crate::model_manager::{EngineBuilder, ModelManager, ModelSpec};
    let policy = Arc::new(std::sync::RwLock::new(
        crate::route::Profile::default().policy(),
    ));
    let usage = std::sync::Arc::new(crate::usage::Usage::new());
    let builder: EngineBuilder = Box::new(|_spec| {
        Box::pin(async {
            Ok(Arc::new(crate::test_support::TaggedGen("switched")) as Arc<dyn Generator>)
        })
    });
    let manager = ModelManager::new(
        Arc::new(crate::test_support::TaggedGen("test")) as Arc<dyn Generator>,
        ModelSpec {
            repo: "test".into(),
            file: "test".into(),
            quant: None,
        },
        builder,
    );
    build_router_inner(Arc::new(AppState {
        manager,
        model_id: "test-model".to_string(),
        policy,
        local_ctx_window: 1000,
        usage,
        cloud_token_alert: 200_000,
        admin_token: Arc::from("test-token"),
        total_ram_mb: 16384,
        requested_ctx_ceiling: 32768,
        kv_kind: crate::fit::KvKind::Q8,
        tool_registry,
        tool_descs: std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new())),
        port: 31415,
        budget: std::sync::Arc::new(crate::budget::Budget::new()),
        breaker: std::sync::Arc::new(crate::breaker::CircuitBreaker::new()),
        recent_prompts: std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeMap::new(),
        )),
    }))
}

#[cfg(test)]
mod tests {
    #[test]
    fn breaker_gate_block_forces_local_only_for_cloud_when_open() {
        use crate::route::{Decision, RouteReason};
        let b = crate::breaker::CircuitBreaker::new();
        // Closed → never blocks.
        assert_eq!(
            super::breaker_gate_block(&b, &Decision::Cloud(RouteReason::Difficulty), false, 0),
            None
        );
        // Open → blocks a cloud decision with the tripping reason.
        b.on_failure(0, crate::usage::DegradeReason::Quota);
        // now=5 is still within the base cooldown (30s), so the breaker is Open, not half-open
        assert_eq!(
            super::breaker_gate_block(&b, &Decision::Cloud(RouteReason::Difficulty), false, 5),
            Some(crate::usage::DegradeReason::Quota)
        );
        // Open but the decision is Local → nothing to block.
        assert_eq!(
            super::breaker_gate_block(&b, &Decision::Local, false, 5),
            None
        );
        // Budget already forced local → breaker defers (budget precedence).
        assert_eq!(
            super::breaker_gate_block(&b, &Decision::Cloud(RouteReason::Difficulty), true, 5),
            None
        );
    }

    #[test]
    fn record_outcome_saves_cost_only_for_local() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-ro-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);
        super::record_outcome(
            "r1",
            "local",
            Some("claude-sonnet-4-6"),
            1_000_000,
            Some(1_000_000),
            Some(10),
            Some(1000),
            None,
        );
        super::record_outcome(
            "r2",
            "cloud",
            Some("claude-sonnet-4-6"),
            1_000_000,
            Some(1_000_000),
            Some(10),
            Some(1000),
            None,
        );
        let lines = crate::route_log::read_all();
        let cost = |rid: &str| {
            lines
                .iter()
                .find_map(|l| match l {
                    crate::route_log::LogLine::Outcome(o) if o.rid == rid => Some(o.cost_saved_usd),
                    _ => None,
                })
                .unwrap()
        };
        assert!((cost("r1") - 18.0).abs() < 1e-6); // sonnet: 1M*3 + 1M*15 = 18
        assert_eq!(cost("r2"), 0.0); // cloud saves nothing
        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metrics_text_has_counters() {
        let now = 1_000i64;
        let lines = vec![
            crate::route_log::LogLine::Decision(crate::route_log::RouteEntry {
                ts: now,
                rid: "r".into(),
                dest: "local".into(),
                prompt_tok: 10,
                ..Default::default()
            }),
            crate::route_log::LogLine::Outcome(crate::route_log::OutcomeEntry {
                rid: "r".into(),
                ts: now,
                cost_saved_usd: 1.5,
                ..Default::default()
            }),
        ];
        let text = super::metrics_text(&lines, now);
        assert!(text.contains("localllm_requests_total{dest=\"local\"} 1"));
        assert!(text.contains("localllm_cost_saved_usd_total"));
    }

    #[test]
    fn export_csv_has_header_and_rows() {
        let lines = vec![crate::route_log::LogLine::Decision(
            crate::route_log::RouteEntry {
                ts: 10,
                rid: "r1".into(),
                surface: "openai".into(),
                dest: "local".into(),
                prompt_tok: 5,
                ..Default::default()
            },
        )];
        let csv = super::export_csv(&lines);
        assert!(csv.starts_with("kind,rid,ts,surface,dest,reason,degrade_reason,model,prompt_tok"));
        assert!(csv.contains("d,r1,10,openai,local"));
    }

    #[test]
    fn degrade_reason_label_is_stable() {
        use crate::usage::DegradeReason;
        assert_eq!(super::degrade_reason_label(DegradeReason::Quota), "Quota");
        assert_eq!(super::degrade_reason_label(DegradeReason::Auth), "Auth");
        assert_eq!(
            super::degrade_reason_label(DegradeReason::ServerError),
            "ServerError"
        );
        assert_eq!(
            super::degrade_reason_label(DegradeReason::Offline),
            "Offline"
        );
    }

    #[test]
    fn manager_init_script_injects_token() {
        let s = super::manager_init_script("abc123");
        assert_eq!(s, "window.__ADMIN_TOKEN__=\"abc123\";");
        assert!(s.contains("abc123"));
        // A token with a quote stays valid JS (escaped), not broken.
        let s2 = super::manager_init_script("a'b\"c");
        assert!(s2.starts_with("window.__ADMIN_TOKEN__="));
        assert!(s2.ends_with(";"));
        assert!(s2.contains("\\\"")); // the double-quote is escaped
    }

    #[test]
    fn constant_time_eq_matches_and_rejects() {
        assert!(super::constant_time_eq(b"abc", b"abc"));
        assert!(!super::constant_time_eq(b"abc", b"abd"));
        assert!(!super::constant_time_eq(b"abc", b"abcd")); // length mismatch
        assert!(!super::constant_time_eq(b"", b"x"));
    }

    #[test]
    fn resolve_admin_token_uses_cli_else_random() {
        assert_eq!(super::resolve_admin_token(Some("z".into())), "z");
        let r = super::resolve_admin_token(None);
        assert_eq!(r.len(), 32); // random 32-hex
        assert_ne!(super::resolve_admin_token(None), r); // different each call
    }

    /// GET /admin/tools — merges registry (seen) with settings (disabled).
    #[tokio::test]
    async fn get_tools_merges_registry_and_settings() {
        use axum::body::Body;
        use http_body_util::BodyExt;
        use std::sync::Arc;
        use tower::ServiceExt;

        let _guard = crate::settings::SETTINGS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let settings_file = std::env::temp_dir().join(format!(
            "localllm-srvtest-tools-get-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::env::set_var("LOCALLLM_SETTINGS", &settings_file);

        // Seed settings: "anthropic" blocklist = ["Read"]
        crate::settings::save_tool_filter("anthropic", &["Read".to_string()]).unwrap();

        // Seed the in-memory registry: "anthropic" -> ["Bash", "Read"]
        let mut initial_registry = std::collections::BTreeMap::new();
        initial_registry.insert(
            "anthropic".to_string(),
            vec!["Bash".to_string(), "Read".to_string()],
        );
        let tool_registry = Arc::new(std::sync::Mutex::new(initial_registry));

        // Build the router with the seeded registry via the test-only helper.
        let app = super::make_seeded_test_router(tool_registry);

        let request = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/tools")
            .header("x-admin-token", "test-token")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();

        assert_eq!(status, 200, "expected 200, got {status}: {body}");
        let anthropic = body.get("anthropic").expect("anthropic key in response");
        assert_eq!(anthropic["seen"], serde_json::json!(["Bash", "Read"]));
        assert_eq!(anthropic["disabled"], serde_json::json!(["Read"]));

        let _ = std::fs::remove_file(&settings_file);
        std::env::remove_var("LOCALLLM_SETTINGS");
    }

    /// POST /admin/tools — persists blocklist and rejects unknown surface.
    #[tokio::test]
    async fn post_tools_persists_and_rejects_unknown_surface() {
        use axum::body::Body;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let _guard = crate::settings::SETTINGS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let settings_file = std::env::temp_dir().join(format!(
            "localllm-srvtest-tools-post-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::env::set_var("LOCALLLM_SETTINGS", &settings_file);

        let app = crate::router_for_test_with(
            std::sync::Arc::new(crate::test_support::TaggedGen("test")),
            crate::route::Profile::default().policy(),
            1000,
        );

        // POST with valid surface → 200 and persists.
        let body = serde_json::json!({"surface": "openai", "disabled": ["Foo"]});
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/tools")
            .header("content-type", "application/json")
            .header("x-admin-token", "test-token")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let resp: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
        assert_eq!(status, 200, "expected 200 for valid surface: {resp}");
        assert_eq!(resp["saved"], serde_json::json!(true));
        // Verify the filter was actually persisted.
        assert_eq!(
            crate::settings::load_tool_filter("openai"),
            vec!["Foo".to_string()]
        );

        // POST with bogus surface → 400.
        let body2 = serde_json::json!({"surface": "bogus", "disabled": []});
        let request2 = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/tools")
            .header("content-type", "application/json")
            .header("x-admin-token", "test-token")
            .body(Body::from(body2.to_string()))
            .unwrap();
        let response2 = app.clone().oneshot(request2).await.unwrap();
        let status2 = response2.status().as_u16();
        let bytes2 = response2.into_body().collect().await.unwrap().to_bytes();
        let resp2: serde_json::Value = serde_json::from_slice(&bytes2).unwrap_or_default();
        assert_eq!(status2, 400, "expected 400 for unknown surface: {resp2}");
        assert!(
            resp2["error"]
                .as_str()
                .map(|s| s.contains("unknown"))
                .unwrap_or(false),
            "error should mention 'unknown': {resp2}"
        );

        let _ = std::fs::remove_file(&settings_file);
        std::env::remove_var("LOCALLLM_SETTINGS");
    }

    #[cfg(unix)]
    #[test]
    fn token_file_is_created_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("tok-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("admin-token");
        super::write_token_to(&path, "sekret");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "sekret");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// POST then GET /admin/budget round-trips the config.
    #[tokio::test]
    async fn budget_get_post_round_trip() {
        use axum::body::Body;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let _guard = crate::settings::SETTINGS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let settings_file =
            std::env::temp_dir().join(format!("localllm-bud-{}.json", uuid::Uuid::new_v4()));
        std::env::set_var("LOCALLLM_SETTINGS", &settings_file);

        let app = crate::router_for_test_with(
            std::sync::Arc::new(crate::test_support::TaggedGen("test")),
            crate::route::Profile::default().policy(),
            1000,
        );

        let post = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/budget")
            .header("content-type", "application/json")
            .header("x-admin-token", "test-token")
            .body(Body::from(r#"{"enabled":true,"daily_usd":5.0}"#))
            .unwrap();
        let resp = app.clone().oneshot(post).await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        let get = axum::http::Request::builder()
            .method("GET")
            .uri("/admin/budget")
            .header("x-admin-token", "test-token")
            .body(Body::empty())
            .unwrap();
        let resp2 = app.oneshot(get).await.unwrap();
        let bytes = resp2.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["enabled"], true);
        assert_eq!(body["daily_usd"], 5.0);

        let _ = std::fs::remove_file(&settings_file);
        std::env::remove_var("LOCALLLM_SETTINGS");
    }

    /// POST /admin/model/profile — happy path: kv_type and history_turns persist.
    #[tokio::test]
    async fn set_model_profile_persists_kv_and_history() {
        use crate::config::KvType;
        use axum::body::Body;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        // Isolate settings storage so this test never touches the real file.
        let _guard = crate::settings::SETTINGS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let settings_file =
            std::env::temp_dir().join(format!("localllm-srvtest-{}.json", uuid::Uuid::new_v4()));
        std::env::set_var("LOCALLLM_SETTINGS", &settings_file);

        // Use the first CATALOG entry — guaranteed to exist.
        let entry = &crate::catalog::CATALOG[0];
        let body = serde_json::json!({
            "repo": entry.repo,
            "file": entry.file,
            "kv_type": "q4",
            "history_turns": 3
        });

        // Build the test router. admin_token is hardcoded as "test-token".
        let app = crate::router_for_test_with(
            std::sync::Arc::new(crate::test_support::TaggedGen("test")),
            crate::route::Profile::default().policy(),
            1000,
        );

        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/model/profile")
            .header("content-type", "application/json")
            .header("x-admin-token", "test-token")
            .body(Body::from(body.to_string()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let resp_body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // Expect 200 {"saved": true} (current model is "test"/"test", not this entry).
        assert_eq!(status, 200, "expected 200, got {status}: {resp_body}");
        assert_eq!(resp_body.get("saved"), Some(&serde_json::Value::Bool(true)));

        // Assert the profile was actually persisted.
        let key = crate::settings::model_ctx_key(entry.repo, entry.file);
        let prof = crate::settings::load_model_profile(&key);
        assert_eq!(prof.kv_type, Some(KvType::Q4), "kv_type should be Q4");
        assert_eq!(prof.history_turns, Some(3), "history_turns should be 3");

        // Cleanup.
        let _ = std::fs::remove_file(&settings_file);
        std::env::remove_var("LOCALLLM_SETTINGS");
    }

    /// POST /admin/model/profile — quant field is persisted when valid, 400 when unknown.
    #[tokio::test]
    async fn set_model_profile_persists_and_validates_quant() {
        use axum::body::Body;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        // Isolate settings storage so this test never touches the real file.
        let _guard = crate::settings::SETTINGS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let settings_file =
            std::env::temp_dir().join(format!("localllm-srvtest-{}.json", uuid::Uuid::new_v4()));
        std::env::set_var("LOCALLLM_SETTINGS", &settings_file);

        // Use the first CATALOG entry — guaranteed to exist and Q4_K_M is always valid.
        let entry = &crate::catalog::CATALOG[0];
        let key = crate::settings::model_ctx_key(entry.repo, entry.file);

        let app = crate::router_for_test_with(
            std::sync::Arc::new(crate::test_support::TaggedGen("test")),
            crate::route::Profile::default().policy(),
            1000,
        );

        // --- Happy path: valid quant Q4_K_M ---
        let body = serde_json::json!({
            "repo": entry.repo,
            "file": entry.file,
            "quant": "Q4_K_M"
        });
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/model/profile")
            .header("content-type", "application/json")
            .header("x-admin-token", "test-token")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let resp_body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 200, "expected 200, got {status}: {resp_body}");
        assert_eq!(resp_body.get("saved"), Some(&serde_json::Value::Bool(true)));

        // Assert quant was persisted.
        let prof = crate::settings::load_model_profile(&key);
        assert_eq!(
            prof.quant,
            Some("Q4_K_M".to_string()),
            "quant should be Q4_K_M"
        );

        // --- Sad path: unknown quant Q9_NOPE → 400 ---
        let app2 = crate::router_for_test_with(
            std::sync::Arc::new(crate::test_support::TaggedGen("test")),
            crate::route::Profile::default().policy(),
            1000,
        );
        let body2 = serde_json::json!({
            "repo": entry.repo,
            "file": entry.file,
            "quant": "Q9_NOPE"
        });
        let request2 = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/model/profile")
            .header("content-type", "application/json")
            .header("x-admin-token", "test-token")
            .body(Body::from(body2.to_string()))
            .unwrap();
        let response2 = app2.oneshot(request2).await.unwrap();
        let status2 = response2.status().as_u16();
        let bytes2 = response2.into_body().collect().await.unwrap().to_bytes();
        let resp_body2: serde_json::Value = serde_json::from_slice(&bytes2).unwrap();
        assert_eq!(
            status2, 400,
            "expected 400 for unknown quant, got {status2}: {resp_body2}"
        );
        assert!(
            resp_body2
                .get("error")
                .and_then(|v| v.as_str())
                .map(|s| s.contains("Q9_NOPE"))
                .unwrap_or(false),
            "error should mention Q9_NOPE: {resp_body2}"
        );

        // Cleanup.
        let _ = std::fs::remove_file(&settings_file);
        std::env::remove_var("LOCALLLM_SETTINGS");
    }

    /// Minimal AppState for wiring tests: TaggedGen engine, Closed-by-default
    /// breaker (caller may pre-trip). No network, no files touched by the paths
    /// under test (record_cloud_success and handle_degrade are pure w.r.t. disk).
    #[cfg(test)]
    fn wiring_state(breaker: std::sync::Arc<crate::breaker::CircuitBreaker>) -> super::AppState {
        use crate::model_manager::{EngineBuilder, ModelManager, ModelSpec};
        let builder: EngineBuilder = Box::new(|_spec| {
            Box::pin(async {
                Ok(
                    std::sync::Arc::new(crate::test_support::TaggedGen("switched"))
                        as std::sync::Arc<dyn super::Generator>,
                )
            })
        });
        let manager = ModelManager::new(
            std::sync::Arc::new(crate::test_support::TaggedGen("test"))
                as std::sync::Arc<dyn super::Generator>,
            ModelSpec {
                repo: "test".into(),
                file: "test".into(),
                quant: None,
            },
            builder,
        );
        super::AppState {
            manager,
            model_id: "test-model".to_string(),
            policy: std::sync::Arc::new(std::sync::RwLock::new(
                crate::route::Profile::default().policy(),
            )),
            local_ctx_window: 1000,
            usage: std::sync::Arc::new(crate::usage::Usage::new()),
            cloud_token_alert: 200_000,
            admin_token: std::sync::Arc::from("test-token"),
            total_ram_mb: 16384,
            requested_ctx_ceiling: 32768,
            kv_kind: crate::fit::KvKind::Q8,
            tool_registry: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            tool_descs: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            port: 31415,
            budget: std::sync::Arc::new(crate::budget::Budget::new()),
            breaker,
            recent_prompts: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
        }
    }

    #[test]
    fn handle_degrade_trips_breaker() {
        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        let state = wiring_state(breaker.clone());
        // Non-overflow reason → handle_degrade returns None (local can serve) and
        // the breaker opens.
        let out = super::handle_degrade(
            &state,
            crate::usage::DegradeReason::Quota,
            crate::route::RouteReason::Difficulty,
        );
        assert!(out.is_none());
        assert_eq!(
            breaker.snapshot(crate::route_log::now_secs() as u64).state,
            "open"
        );
    }

    #[test]
    fn record_cloud_success_closes_breaker() {
        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        breaker.on_failure(0, crate::usage::DegradeReason::ServerError);
        let state = wiring_state(breaker.clone());
        super::record_cloud_success(&state, 10);
        assert_eq!(
            breaker.snapshot(crate::route_log::now_secs() as u64).state,
            "closed"
        );
    }

    #[tokio::test]
    async fn breaker_get_requires_token() {
        let app = super::make_seeded_test_router(std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeMap::new(),
        )));
        let code = crate::axum_test_get_status(app, "/admin/breaker").await;
        assert_eq!(code, 401);
    }

    #[tokio::test]
    async fn breaker_get_reports_closed_by_default() {
        let app = super::make_seeded_test_router(std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeMap::new(),
        )));
        let body =
            crate::axum_test_get_with_header(app, "/admin/breaker", "x-admin-token", "test-token")
                .await;
        assert_eq!(body["state"], "closed");
    }

    #[test]
    fn record_outcome_emits_truncated_for_local_length() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-tr-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);

        use crate::api::common::FinishReason;
        super::record_outcome(
            "t1",
            "local",
            None,
            10,
            Some(10),
            None,
            None,
            Some(FinishReason::Length),
        );
        super::record_outcome(
            "t2",
            "local",
            None,
            10,
            Some(10),
            None,
            None,
            Some(FinishReason::Stop),
        );
        super::record_outcome(
            "t3",
            "cloud",
            None,
            10,
            Some(10),
            None,
            None,
            Some(FinishReason::Length),
        );

        let lines = crate::route_log::read_all();
        let has_trunc = |rid: &str| {
            lines.iter().any(|l| {
                matches!(l,
            crate::route_log::LogLine::Feedback(f) if f.rid == rid && f.signal == "truncated")
            })
        };
        assert!(has_trunc("t1"), "local Length must emit truncated");
        assert!(!has_trunc("t2"), "local Stop must not");
        assert!(
            !has_trunc("t3"),
            "cloud Length must not (cloud is not judged here)"
        );

        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn cascade_escalation_emits_cascade_feedback() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-cf-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);

        // Weak local result (finish=Length) + cascade wanted → escalates; the
        // unreachable upstream degrades, but the cascade signal must be logged.
        let weak = Ok(crate::api::common::ChatResult {
            content: vec![crate::api::common::ContentPart::Text("x".into())],
            finish_reason: crate::api::common::FinishReason::Length,
            prompt_tokens: 1,
            completion_tokens: 1,
        });
        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        let state = wiring_state(breaker);
        let _ = super::cascade_or_result(
            true,
            weak,
            crate::cloud::Provider::OpenAI,
            "/v1/chat/completions",
            &axum::http::HeaderMap::new(),
            axum::body::Bytes::from("{}"),
            &state,
            1,
            "rc1",
            "openai",
        )
        .await;
        let lines = crate::route_log::read_all();
        assert!(lines.iter().any(|l| matches!(l,
            crate::route_log::LogLine::Feedback(f) if f.rid == "rc1" && f.signal == "cascade")));

        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tokset(s: &str) -> std::collections::BTreeSet<String> {
        crate::history_select::tokenize(s)
            .into_iter()
            .filter(|t| t.len() >= 2)
            .collect()
    }

    #[test]
    fn detect_reask_matches_similar_recent_prompt() {
        use std::collections::VecDeque;
        let mut buf: VecDeque<super::RecentPrompt> = VecDeque::new();
        buf.push_front(super::RecentPrompt {
            rid: "old1".into(),
            ts: 1000,
            tokens: tokset("como faço deploy do serviço no kubernetes"),
        });
        // Near-identical re-ask 30s later → match.
        let t = tokset("como faço deploy do serviço no kubernetes agora");
        assert_eq!(
            super::detect_reask(&buf, &t, 1030),
            Some("old1".to_string())
        );
        // Different topic → no match.
        let t2 = tokset("escreva um poema sobre gatos persas");
        assert_eq!(super::detect_reask(&buf, &t2, 1030), None);
        // Same prompt but 3 minutes later → outside window.
        assert_eq!(super::detect_reask(&buf, &t, 1000 + 181), None);
        // Empty token set never matches.
        let empty = std::collections::BTreeSet::new();
        assert_eq!(super::detect_reask(&buf, &empty, 1030), None);
    }

    #[test]
    fn detect_reask_first_match_wins() {
        use std::collections::VecDeque;
        let mut buf: VecDeque<super::RecentPrompt> = VecDeque::new();
        // Newest first: both similar; the front (newest) must win.
        buf.push_front(super::RecentPrompt {
            rid: "older".into(),
            ts: 990,
            tokens: tokset("erro de compilação no módulo de rede"),
        });
        buf.push_front(super::RecentPrompt {
            rid: "newer".into(),
            ts: 1000,
            tokens: tokset("erro de compilação no módulo de rede"),
        });
        let t = tokset("erro de compilação no módulo de rede ainda");
        assert_eq!(
            super::detect_reask(&buf, &t, 1010),
            Some("newer".to_string())
        );
    }

    #[test]
    fn reask_buffer_only_keeps_local_decisions_and_caps() {
        use std::collections::VecDeque;
        let mut buf: VecDeque<super::RecentPrompt> = VecDeque::new();
        for i in 0..12 {
            buf.push_front(super::RecentPrompt {
                rid: format!("r{i}"),
                ts: 1000 + i,
                tokens: tokset(&format!(
                    "prompt número {i} totalmente diferente dos outros assunto{i}"
                )),
            });
            buf.truncate(super::REASK_BUFFER_CAP);
        }
        assert_eq!(buf.len(), super::REASK_BUFFER_CAP);
        assert_eq!(buf.front().unwrap().rid, "r11"); // newest kept
    }

    #[tokio::test]
    async fn route_decision_records_active_local_model() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-lm-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let logp = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &logp);

        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        let state = wiring_state(breaker);
        let req = crate::api::common::ChatRequest {
            messages: vec![crate::api::common::ChatMessage {
                role: crate::api::common::Role::User,
                text: Some("oi".into()),
                tool_calls: vec![],
                tool_result: None,
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
            model: String::new(),
        };
        let headers = axum::http::HeaderMap::new();
        let _ = super::route_decision(&state, &req, &headers, "rlm1", "openai");

        let lines = crate::route_log::read_all();
        let dec = lines
            .iter()
            .find_map(|l| match l {
                crate::route_log::LogLine::Decision(d) if d.rid == "rlm1" => Some(d.clone()),
                _ => None,
            })
            .expect("decision line");
        assert_eq!(dec.local_model.as_deref(), Some("test/test"));

        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
