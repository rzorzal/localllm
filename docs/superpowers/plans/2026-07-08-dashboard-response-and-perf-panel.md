# Dashboard Response Text + Performance Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store the full model response (local and cloud) in the route log and show it next to the prompt in the recent-decisions row; replace the latency/quality cards with one two-column Desempenho panel.

**Architecture:** Add `output_text` to `OutcomeEntry` (route-log JSONL). Local generation sites extract text from the structured `ChatResult` (result paths) or accumulate stream deltas (streaming paths). The cloud reverse-proxy accumulates relayed bytes and parses provider SSE/JSON to recover the assistant text. A line-count cap keeps the route log bounded. The dashboard surfaces `output_text` on `RecentRow`; the frontend renders a scrollable "Resposta" box and a two-column performance panel.

**Tech Stack:** Rust (axum, serde, futures streams), vanilla JS (`src/manager_ui/app.js`), CSS.

## Global Constraints

- Route-log path: `~/Library/Application Support/localllm/routing-log.jsonl` (override `LOCALLLM_ROUTE_LOG`). One JSON object per line. Never conflate with the app-log `/tmp/localllm.log`.
- All route-log writes are best-effort — they must NEVER fail or block a request.
- `output_text` is stored in full (no length cap) per product decision; the route log is bounded instead by a line-count cap (Task 5).
- Frontend renders stored model text via `textContent` only — never `innerHTML` (injection safety).
- Frontend is embedded via `include_str!`; the running app is the `.app` bundle. Changes require `bash scripts/build-app.sh --fast` + relaunch to observe (Task 9).
- Follow existing patterns: `el(tag, cls, html)` DOM helper, `dash-card` styling, serde `#[serde(default, skip_serializing_if = "Option::is_none")]` for optional fields.

---

### Task 1: `output_text` field + content-text helper + dashboard join

**Files:**
- Modify: `src/api/common.rs` (add `content_text` helper near `ChatResult`, ~line 68)
- Modify: `src/route_log.rs:60-72` (`OutcomeEntry`), `src/route_log.rs:325-339` (`RecentRow`), `src/route_log.rs:484-492` (join)
- Test: inline `#[cfg(test)]` in `src/route_log.rs` and `src/api/common.rs`

**Interfaces:**
- Produces: `crate::api::common::content_text(parts: &[ContentPart]) -> String`
- Produces: `OutcomeEntry.output_text: Option<String>`
- Produces: `RecentRow.output_text: Option<String>` (serialized field `output_text`)

- [ ] **Step 1: Write the failing test for `content_text`**

In `src/api/common.rs` `#[cfg(test)]` module:

```rust
#[test]
fn content_text_concatenates_text_parts_only() {
    let parts = vec![
        ContentPart::Text("hello ".into()),
        ContentPart::Call(ToolCall::default()),
        ContentPart::Text("world".into()),
    ];
    assert_eq!(content_text(&parts), "hello world");
}
```

(If `ToolCall` has no `Default`, construct it with its real fields — check the struct near the top of `common.rs`.)

- [ ] **Step 2: Run test, verify it fails**

Run: `cargo test -p localllm content_text_concatenates -- --nocapture`
Expected: FAIL — `cannot find function content_text`.

- [ ] **Step 3: Implement `content_text`**

Add near `ChatResult` in `src/api/common.rs`:

```rust
/// Concatenate the text parts of a content vector, ignoring tool calls.
pub fn content_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text(t) => Some(t.as_str()),
            ContentPart::Call(_) => None,
        })
        .collect::<Vec<_>>()
        .join("")
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cargo test -p localllm content_text_concatenates`
Expected: PASS.

- [ ] **Step 5: Add `output_text` to `OutcomeEntry`**

In `src/route_log.rs`, inside `OutcomeEntry` (after `cost_saved_usd`, line ~71):

```rust
    /// Full generated response text (local or cloud). None on legacy lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,
```

- [ ] **Step 6: Add `output_text` to `RecentRow` and the join**

In `RecentRow` (after `feedback`, line ~338):

```rust
    /// Full response text joined from the outcome (may be absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,
```

In the recent-row join (`src/route_log.rs:485-492`), add the last field:

```rust
            RecentRow {
                entry: (*d).clone(),
                completion_tok: o.and_then(|o| o.completion_tok),
                ttft_ms: o.and_then(|o| o.ttft_ms),
                gen_ms: o.and_then(|o| o.gen_ms),
                cost_saved_usd: o.map(|o| o.cost_saved_usd).unwrap_or(0.0),
                feedback: feedback.get(d.rid.as_str()).cloned().unwrap_or_default(),
                output_text: o.and_then(|o| o.output_text.clone()),
            }
```

- [ ] **Step 7: Write the failing join test**

In `src/route_log.rs` `#[cfg(test)]` module:

```rust
#[test]
fn dashboard_surfaces_output_text_on_recent_row() {
    let now = 1_000_000;
    let entries = vec![
        LogLine::Decision(RouteEntry { ts: now, rid: "x".into(), dest: "local".into(), ..Default::default() }),
        LogLine::Outcome(OutcomeEntry { rid: "x".into(), ts: now, output_text: Some("resposta".into()), ..Default::default() }),
    ];
    let d = build_dashboard(&entries, now, 50, false);
    let row = d.recent.iter().find(|r| r.entry.rid == "x").unwrap();
    assert_eq!(row.output_text.as_deref(), Some("resposta"));
}
```

- [ ] **Step 8: Run tests, verify pass**

Run: `cargo test -p localllm dashboard_surfaces_output_text`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/api/common.rs src/route_log.rs
git commit -m "feat(route-log): output_text on OutcomeEntry + RecentRow join"
```

---

### Task 2: `record_outcome` accepts `output_text`; wire the 5 result-based local sites

**Files:**
- Modify: `src/server.rs:374-396` (`record_outcome` signature + body)
- Modify: `src/server.rs` record_outcome calls at 1978, 2086, 2224, 2380, 2488
- Modify: `src/server.rs` existing tests at 2608, 2618, 3174, 3184, 3194 (add the new arg)

**Interfaces:**
- Consumes: `crate::api::common::content_text` (Task 1), `OutcomeEntry.output_text` (Task 1)
- Produces: `record_outcome(rid, dest, model, prompt_tok, completion_tok, ttft_ms, gen_ms, finish, output_text: Option<String>)`

- [ ] **Step 1: Add the parameter to `record_outcome`**

In `src/server.rs:374`, add a final parameter and pass it through:

```rust
fn record_outcome(
    rid: &str,
    dest: &str,
    model: Option<&str>,
    prompt_tok: u64,
    completion_tok: Option<u64>,
    ttft_ms: Option<u64>,
    gen_ms: Option<u64>,
    finish: Option<crate::api::common::FinishReason>,
    output_text: Option<String>,
) {
```

In the `append_outcome` call inside it (line ~389), add:

```rust
    crate::route_log::append_outcome(&crate::route_log::OutcomeEntry {
        rid: rid.to_string(),
        ts: crate::route_log::now_secs(),
        completion_tok,
        ttft_ms,
        gen_ms,
        cost_saved_usd,
        output_text,
    });
```

- [ ] **Step 2: Update the 5 result-based call sites**

At each of lines 1978, 2086, 2224, 2380, 2488, the local `result: ChatResult` is in scope. Add the final argument to each call:

```rust
            Some(crate::api::common::content_text(&result.content)),
```

Insert it as the last argument (after the `finish_reason` argument) in all five `record_outcome(...)` calls.

- [ ] **Step 3: Update existing tests to pass `None`**

At `src/server.rs` test call sites 2608, 2618, 3174, 3184, 3194, add a trailing `None` argument to each `super::record_outcome(...)` call (these tests don't assert on output).

- [ ] **Step 4: Compile**

Run: `cargo build -p localllm`
Expected: builds clean (the stream sites 2035 and 2428 still fail to compile because they call `record_outcome` with the old arity — fixed in Tasks 3 and 4). If only those two remain, proceed; otherwise fix the missing arg.

Note: because Tasks 3 and 4 also change `record_outcome` callers, it is acceptable for this task to leave the build red ONLY at lines 2035/2428. To keep each task independently green, do Tasks 2–4 as one commit group: implement Step 2 here, then Tasks 3 and 4, then run the full build/tests once before committing. Commit instruction is at the end of Task 4.

---

### Task 3: OpenAI streaming site — accumulate text, pass `output_text`

**Files:**
- Modify: `src/server.rs:2020-2039` (openai stream scan state + `done` branch)

**Interfaces:**
- Consumes: `record_outcome(..., output_text)` (Task 2)

- [ ] **Step 1: Extend the scan state with a text accumulator**

Change the scan initializer at `src/server.rs:2022` from:

```rust
            .scan((false, 0u64, None::<Instant>), move |st, result| {
                let (started, ctok, first_at) = st;
```

to:

```rust
            .scan((false, 0u64, None::<Instant>, String::new()), move |st, result| {
                let (started, ctok, first_at, acc) = st;
```

- [ ] **Step 2: Accumulate delta text**

In the `Ok(delta)` branch, where `delta.text` is handled (line ~2028), append to `acc`:

```rust
                        if let Some(t) = &delta.text {
                            *ctok += crate::route::estimate_text_tokens(t) as u64;
                            acc.push_str(t);
                        }
```

- [ ] **Step 3: Pass `output_text` at `done`**

Update the `record_outcome` call in the `done` branch (line ~2035):

```rust
                            record_outcome(&rid_stream, "local", Some(&model_clone),
                                est_prompt_tokens as u64, Some(*ctok), ttft,
                                Some(started_at.elapsed().as_millis() as u64),
                                delta.finish_reason.clone(),
                                Some(std::mem::take(acc)));
```

---

### Task 4: Anthropic streaming site — add missing `record_outcome` with text

**Files:**
- Modify: `src/server.rs:2412-2458` (anthropic stream scan — currently records nothing)

**Interfaces:**
- Consumes: `record_outcome(..., output_text)` (Task 2)

**Context:** This path (used by Claude Code) currently only threads a `bool` `started` and never calls `record_outcome`, so anthropic streaming requests have no outcome, latency, or response recorded. This task adds it, mirroring the OpenAI stream site.

- [ ] **Step 1: Capture the model + rid before the stream**

Just above the `let delta_stream = ...` at line 2413, add clones (check whether `model` is in scope in this handler; if the variable is named differently, use that name):

```rust
        let model_clone = model.clone();
        let est_prompt_tokens = est_prompt_tokens;
```

- [ ] **Step 2: Extend the scan state**

Change line 2428 from:

```rust
            .scan(false, move |started, result| {
                let was_started = *started;
```

to:

```rust
            .scan((false, 0u64, None::<Instant>, String::new()), move |st, result| {
                let (started, ctok, first_at, acc) = st;
                let was_started = *started;
```

- [ ] **Step 3: Accumulate text + TTFT and record on done**

Replace the `if delta.done { ... }` block (lines 2432-2435) with:

```rust
                        if first_at.is_none() { *first_at = Some(Instant::now()); }
                        if let Some(t) = &delta.text {
                            *ctok += crate::route::estimate_text_tokens(t) as u64;
                            acc.push_str(t);
                        }
                        if delta.done {
                            let secs = started_at.elapsed().as_secs_f64();
                            tracing::info!(target: "localllm::req", "{rid_stream} [anthropic] done (stream): {secs:.1}s");
                            let ttft = first_at.map(|f| (f - started_at).as_millis() as u64);
                            record_outcome(&rid_stream, "local", Some(&model_clone),
                                est_prompt_tokens as u64, Some(*ctok), ttft,
                                Some(started_at.elapsed().as_millis() as u64),
                                delta.finish_reason.clone(),
                                Some(std::mem::take(acc)));
                        }
```

- [ ] **Step 4: Fix `*started` assignment**

The original code sets `*started = true;` after building events. It still works because `started` is now the first tuple element (a `&mut bool`). Confirm the line `*started = true;` (line ~2437) is unchanged and compiles.

- [ ] **Step 5: Build + full test suite (covers Tasks 2–4)**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean build, all tests pass.

- [ ] **Step 6: Commit Tasks 2–4**

```bash
git add src/server.rs
git commit -m "feat(server): record output_text for local result + streaming paths"
```

---

### Task 5: Route-log line-count cap (bounded growth)

**Files:**
- Modify: `src/route_log.rs` (add `cap_lines_file`, near `prune_file` ~line 190)
- Modify: `src/lib.rs:162` (call the cap on boot)
- Test: inline `#[cfg(test)]` in `src/route_log.rs`

**Interfaces:**
- Produces: `crate::route_log::cap_lines_file(max_lines: usize)`

- [ ] **Step 1: Write the failing test**

The pure logic is line trimming; test it through a temp file using the existing `LOCALLLM_ROUTE_LOG` override and `ROUTE_LOG_ENV_LOCK` (see existing tests around line 811):

```rust
#[test]
fn cap_lines_file_keeps_newest() {
    let _guard = ROUTE_LOG_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!("caplines-{}", now_secs()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("routing-log.jsonl");
    std::env::set_var("LOCALLLM_ROUTE_LOG", &path);
    let body: String = (0..10).map(|i| format!("line{i}\n")).collect();
    std::fs::write(&path, body).unwrap();
    cap_lines_file(4);
    let out = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines, vec!["line6", "line7", "line8", "line9"]);
    std::env::remove_var("LOCALLLM_ROUTE_LOG");
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cargo test -p localllm cap_lines_file_keeps_newest`
Expected: FAIL — `cannot find function cap_lines_file`.

- [ ] **Step 3: Implement `cap_lines_file`**

In `src/route_log.rs`, near `prune_file`:

```rust
/// Best-effort: cap the route log at `max_lines`, keeping the newest lines.
/// Guards against unbounded growth when full response bodies are stored.
pub fn cap_lines_file(max_lines: usize) {
    let Some(path) = log_path() else { return };
    let Ok(text) = std::fs::read_to_string(&path) else { return };
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return;
    }
    let tail = lines[lines.len() - max_lines..].join("\n");
    let _ = crate::integrations::atomic_write(&path, format!("{tail}\n").as_bytes());
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cargo test -p localllm cap_lines_file_keeps_newest`
Expected: PASS.

- [ ] **Step 5: Wire into boot**

In `src/lib.rs`, right after the existing `prune_file` call at line 162:

```rust
    crate::route_log::prune_file(crate::route_log::now_secs(), 30 * 24 * 3600);
    crate::route_log::cap_lines_file(20_000);
    crate::route_log::rotate_app_log(crate::route_log::now_secs(), 7 * 24 * 3600);
```

- [ ] **Step 6: Commit**

```bash
git add src/route_log.rs src/lib.rs
git commit -m "feat(route-log): cap line count so stored responses don't grow unbounded"
```

---

### Task 6: Cloud response text extraction in the reverse-proxy

**Files:**
- Modify: `src/cloud.rs:27-82` (`MeteredStream` fields + `poll_next` accumulation/parse)
- Modify: `src/cloud.rs:147-217` (`forward` — pass provider + stream flag + build buffer)
- Test: `src/cloud.rs` `#[cfg(test)]` — `extract_cloud_text`

**Interfaces:**
- Produces: `fn extract_cloud_text(buf: &[u8], provider: Provider, is_stream: bool) -> Option<String>`

- [ ] **Step 1: Write failing tests for `extract_cloud_text`**

In `src/cloud.rs` tests module:

```rust
#[test]
fn extract_cloud_text_anthropic_sse() {
    let sse = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n";
    assert_eq!(extract_cloud_text(sse.as_bytes(), Provider::Anthropic, true).as_deref(), Some("Hello"));
}

#[test]
fn extract_cloud_text_openai_sse() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\ndata: [DONE]\n\n";
    assert_eq!(extract_cloud_text(sse.as_bytes(), Provider::OpenAI, true).as_deref(), Some("Hi there"));
}

#[test]
fn extract_cloud_text_anthropic_json() {
    let body = "{\"content\":[{\"type\":\"text\",\"text\":\"full reply\"}]}";
    assert_eq!(extract_cloud_text(body.as_bytes(), Provider::Anthropic, false).as_deref(), Some("full reply"));
}

#[test]
fn extract_cloud_text_openai_json() {
    let body = "{\"choices\":[{\"message\":{\"content\":\"full reply\"}}]}";
    assert_eq!(extract_cloud_text(body.as_bytes(), Provider::OpenAI, false).as_deref(), Some("full reply"));
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test -p localllm extract_cloud_text`
Expected: FAIL — `cannot find function extract_cloud_text`.

- [ ] **Step 3: Implement `extract_cloud_text`**

Add to `src/cloud.rs` (uses `serde_json`):

```rust
/// Recover the assistant text from a relayed cloud response buffer. Streaming
/// bodies are provider SSE; non-streaming are the JSON response body. Best-effort:
/// returns None if nothing parses.
fn extract_cloud_text(buf: &[u8], provider: Provider, is_stream: bool) -> Option<String> {
    let s = std::str::from_utf8(buf).ok()?;
    if is_stream {
        let mut out = String::new();
        for line in s.lines() {
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data.trim() == "[DONE]" { continue; }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else { continue };
            match provider {
                Provider::Anthropic => {
                    if v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                        if let Some(t) = v.pointer("/delta/text").and_then(|t| t.as_str()) {
                            out.push_str(t);
                        }
                    }
                }
                Provider::OpenAI => {
                    if let Some(t) = v.pointer("/choices/0/delta/content").and_then(|t| t.as_str()) {
                        out.push_str(t);
                    }
                }
            }
        }
        (!out.is_empty()).then_some(out)
    } else {
        let v = serde_json::from_str::<serde_json::Value>(s).ok()?;
        match provider {
            Provider::Anthropic => {
                let arr = v.get("content")?.as_array()?;
                let out: String = arr.iter()
                    .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect();
                (!out.is_empty()).then_some(out)
            }
            Provider::OpenAI => {
                v.pointer("/choices/0/message/content").and_then(|t| t.as_str()).map(|s| s.to_string())
            }
        }
    }
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cargo test -p localllm extract_cloud_text`
Expected: PASS (all four).

- [ ] **Step 5: Add buffer + provider + stream flag to `MeteredStream`**

In `src/cloud.rs:27-34`:

```rust
struct MeteredStream<S> {
    inner: S,
    rid: String,
    provider: Provider,
    is_stream: bool,
    started: std::time::Instant,
    first_chunk_ms: Option<u64>,
    bytes: u64,
    buf: Vec<u8>,
    done: bool,
}
```

- [ ] **Step 6: Accumulate bytes (capped) and set `output_text` on done**

In `poll_next`, the `Ready(Some(Ok(chunk)))` arm (line ~47), after `this.bytes += ...`:

```rust
                this.bytes += chunk.len() as u64;
                const CLOUD_BUF_CAP: usize = 256 * 1024;
                if this.buf.len() < CLOUD_BUF_CAP {
                    let room = CLOUD_BUF_CAP - this.buf.len();
                    this.buf.extend_from_slice(&chunk[..chunk.len().min(room)]);
                }
                std::task::Poll::Ready(Some(Ok(chunk)))
```

In the `Ready(None)` arm, update the `append_outcome` to include `output_text` (line ~61):

```rust
                    crate::route_log::append_outcome(&crate::route_log::OutcomeEntry {
                        rid: this.rid.clone(),
                        ts: crate::route_log::now_secs(),
                        completion_tok: Some(est_tok),
                        ttft_ms: this.first_chunk_ms,
                        gen_ms: Some(this.started.elapsed().as_millis() as u64),
                        cost_saved_usd: 0.0,
                        output_text: extract_cloud_text(&this.buf, this.provider, this.is_stream),
                    });
```

- [ ] **Step 7: Populate the new fields at construction**

In `forward` (line ~198), detect streaming from the upstream `content-type` before consuming headers, then set the fields:

```rust
    let is_stream = upstream
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream"))
        .unwrap_or(false);
```

(Place this before `let mut resp_headers = ...` so `upstream.headers()` is still available.) Then in the `Some(m)` arm:

```rust
            let metered = MeteredStream {
                inner: stream,
                rid: m.rid,
                provider,
                is_stream,
                started: std::time::Instant::now(),
                first_chunk_ms: None,
                bytes: 0,
                buf: Vec::new(),
                done: false,
            };
```

- [ ] **Step 8: Build + test**

Run: `cargo build -p localllm && cargo test -p localllm cloud`
Expected: clean build, cloud tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/cloud.rs
git commit -m "feat(cloud): capture relayed response text into output_text"
```

---

### Task 7: Frontend — "Resposta" box in recent decisions

**Files:**
- Modify: `src/manager_ui/app.js:1117-1126` (expandable detail row in `renderDecisionsTable`)
- Modify: `src/manager_ui/style.css` (reuse `.prompt-box`; add `.resp-label` mirroring `.prompt-label` if needed)

**Interfaces:**
- Consumes: `e.output_text` on each recent row (Task 1 / Tasks 2–6 populate it)

- [ ] **Step 1: Render the response box after the prompt box**

In `renderDecisionsTable`, inside the `detail` row builder (after the prompt `if/else` at lines 1120-1125, before `detail.append(dcell)`):

```javascript
    if (e.output_text) {
      dcell.append(el("div", "prompt-label", "Resposta"));
      const respBox = el("div", "prompt-box");
      respBox.textContent = e.output_text;
      dcell.append(respBox);
    } else {
      dcell.append(el("div", "muted", "sem resposta registrada para esta linha"));
    }
```

(Use `textContent`, never `innerHTML` — the response is untrusted model output.)

- [ ] **Step 2: Verify JS parses**

Run: `node --check src/manager_ui/app.js`
Expected: prints nothing (exit 0).

- [ ] **Step 3: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): show response text next to prompt in recent decisions"
```

---

### Task 8: Frontend — Desempenho panel (latency + quality, two columns)

**Files:**
- Modify: `src/manager_ui/app.js:856-881` (replace the two cards in `renderDashboard`)
- Modify: `src/manager_ui/style.css` (add `.perf-panel`, `.perf-col`, `.perf-group`, `.perf-row`)

**Interfaces:**
- Consumes: `d.local_latency`, `d.cloud_latency`, `d.feedback` (unchanged shapes)

- [ ] **Step 1: Replace the latency + accuracy card blocks**

Delete the block from `// Latency by route ...` (line 856) through `wrap.append(acc);` (line 881) and replace with:

```javascript
  // Performance panel — latency (left) and quality (right), side by side.
  const L = d.local_latency || { avg_ttft_ms: 0, avg_tok_s: 0, n: 0 };
  const C = d.cloud_latency || { avg_ttft_ms: 0, avg_tok_s: 0, n: 0 };
  const fb = d.feedback || { local_total: 0, local_flagged: 0, cloud_total: 0, cloud_trivial: 0 };
  const flaggedPct = fb.local_total > 0 ? Math.round((fb.local_flagged / fb.local_total) * 100) + "%" : "—";
  const trivialPct = fb.cloud_total > 0 ? Math.round((fb.cloud_trivial / fb.cloud_total) * 100) + "%" : "—";

  const perf = el("div", "perf-panel");
  perf.append(el("div", "dash-card-head", "DESEMPENHO"));
  const perfCols = el("div", "perf-cols");

  const metricRow = (label, value) => {
    const r = el("div", "perf-row");
    r.append(el("span", "perf-k", label));
    const v = el("span", "perf-v"); v.textContent = value; r.append(v);
    return r;
  };
  const group = (title, rows) => {
    const g = el("div", "perf-group");
    g.append(el("div", "perf-group-head", title));
    rows.forEach(r => g.append(r));
    return g;
  };

  const latCol = el("div", "perf-col");
  latCol.append(el("div", "perf-col-head", "LATÊNCIA"));
  latCol.append(group("Local", [
    metricRow("TTFT", `${L.avg_ttft_ms} ms`),
    metricRow("velocidade", `${(L.avg_tok_s || 0).toFixed(0)} tok/s`),
    metricRow("amostras", `${L.n}`),
  ]));
  latCol.append(group("Cloud", [
    metricRow("TTFT", `${C.avg_ttft_ms} ms`),
    metricRow("velocidade", `${(C.avg_tok_s || 0).toFixed(0)} tok/s`),
    metricRow("amostras", `${C.n}`),
  ]));

  const qCol = el("div", "perf-col");
  qCol.append(el("div", "perf-col-head", "QUALIDADE"));
  qCol.append(group("Local", [
    metricRow("problemas", flaggedPct),
    metricRow("contagem", `${fb.local_flagged}/${fb.local_total}`),
  ]));
  qCol.append(group("Cloud", [
    metricRow("trivial", trivialPct),
    metricRow("contagem", `${fb.cloud_trivial}/${fb.cloud_total}`),
  ]));

  perfCols.append(latCol, qCol);
  perf.append(perfCols);
  wrap.append(perf);
```

- [ ] **Step 2: Add the CSS**

Append to `src/manager_ui/style.css`:

```css
.perf-panel { margin-top: 12px; }
.perf-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 640px) { .perf-cols { grid-template-columns: 1fr; } }
.perf-col-head { font-size: 11px; letter-spacing: .06em; color: var(--muted); margin-bottom: 6px; }
.perf-group { margin-bottom: 10px; }
.perf-group-head { font-size: 12px; font-weight: 600; margin-bottom: 2px; }
.perf-row { display: flex; justify-content: space-between; font-size: 13px; padding: 1px 0; }
.perf-k { color: var(--muted); }
.perf-v { font-variant-numeric: tabular-nums; }
```

(If `--muted` / `--bg` variables differ, match the names already used in `style.css`.)

- [ ] **Step 3: Verify JS parses**

Run: `node --check src/manager_ui/app.js`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): two-column Desempenho panel replacing latency/quality cards"
```

---

### Task 9: Rebuild bundle + manual verification

**Files:** none (build + verify)

- [ ] **Step 1: Full build + tests**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean, all pass.

- [ ] **Step 2: Rebuild the .app bundle**

Run: `bash scripts/build-app.sh --fast`
Expected: `==> SUCCESS: .../target/localllm.app`.

- [ ] **Step 3: Confirm embedded JS is current**

Run: `strings -a target/localllm.app/Contents/MacOS/localllm | grep -c "perf-panel"`
Expected: `1` or more.

- [ ] **Step 4: Restart the running app**

Run: `kill "$(pgrep -f 'localllm.app')" 2>/dev/null; sleep 1; open target/localllm.app`

- [ ] **Step 5: Manual checks**

- Send a request through a client (or use existing history), open Dashboard.
- Expand a recent decision row → a **Resposta** box appears below the prompt and scrolls; rows without a recorded response show the muted empty state.
- The **DESEMPENHO** panel shows Latência and Qualidade side by side; narrowing the window collapses it to one column.

- [ ] **Step 6: Final commit if any tweaks were needed**

```bash
git add -A && git commit -m "chore: dashboard response + perf panel verification tweaks"
```

---

## Self-Review

**Spec coverage:**
- Response text stored (full, no cap): Tasks 1–6. ✓
- Shown next to prompt in recent row: Task 7. ✓
- Route-log rotation / bounded growth: Task 5. ✓
- Two-column Desempenho panel: Task 8. ✓
- Local + cloud capture (user upgraded scope): Tasks 2–4 (local), Task 6 (cloud). ✓
- `textContent` injection safety: Task 7 Step 1, Task 8 metricRow. ✓
- Storage-model separation (jsonl vs app-log): Global Constraints. ✓
- Rebuild/relaunch rollout: Task 9. ✓
- Bonus gap found: anthropic streaming had no `record_outcome` — added in Task 4. ✓

**Placeholder scan:** none — every code step shows concrete code.

**Type consistency:** `content_text(&[ContentPart]) -> String`, `output_text: Option<String>` used identically across `OutcomeEntry`, `RecentRow`, `record_outcome`, cloud `append_outcome`, and frontend `e.output_text`. `extract_cloud_text(&[u8], Provider, bool) -> Option<String>` consistent. Frontend consumes `e.output_text`, `d.local_latency`, `d.cloud_latency`, `d.feedback`.
