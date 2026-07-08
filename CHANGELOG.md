# Changelog

All notable changes to localllm.
## [0.2.0] - 2026-07-08

### Bug Fixes

- OpenAI stream final-chunk delta omits content; tighten stream tests
- Address final review — streaming tool-call clarity, role in first chunk, /v1/models id, cleanup
- Accept Anthropic 'system' field as string or block array
- Wire ctx_len into device-map max_seq_len so GPU context isn't capped at 512
- Size PagedAttention KV cache to ctx_len for full GPU context window
- Persistent single llama.cpp context via worker thread (fixes Metal multi-context redefinition); incremental streaming; --backend select
- Review hardening — clear_kv_cache_seq Ok(false) fallback, .kvstate provenance validation, worker panic supervision, UTF-8-safe logs, drop redundant unsafe impl, accurate tscg doc
- Reject prompt larger than context with clean error instead of SIGABRT
- Append tool block to first system message only (was duplicated → -43% prompt); logs to stderr; add --dump-prompt debug
- Strip volatile x-anthropic-billing-header from system prompt (its cc_version changes per call, sat at prompt front and broke ALL prefix-cache reuse)
- Make the binary the bundle's direct CFBundleExecutable + auto-enable tray on bundle launch (__CFBundleIdentifier); a shell-wrapper executable breaks NSStatusItem registration so the menu-bar icon never appeared
- Assert credential forwarding, shared client, relay upstream headers
- Raise body limit for large cloud-routed prompts; generic 502 body
- Use tokio::sync::Mutex for env-var test lock (no poison/hang across await)
- Accept tool_result array content + unknown block types
- Accept array content parts + developer role (Codex)
- Remove now-dead mb() helper (pristine build)
- Create admin-token file atomically at 0600 (no world-readable window)
- Spec tests (503-while-switching, drain) + stream inflight guard + errored 503 msg + token-write warn
- Reject path-traversal in model file + 400 tests + sysinfo-0 warn
- JSON-encode injected admin token + correct headless-mode comment
- Live model + status lines; auto-refresh manager window
- Final-review fixes for /v1/responses
- Close disable TOCTOU + cover non-object/absent-file paths
- Codex writes standard [model_providers.localllm] table; O(1) empty check
- Preserve sibling Codex providers; surface revert failures; atomic settings write
- Capture llama/ggml logs + rebuild context after decode OOM
- Ctx reload stays on detail pane with its own toast
- Share one process-global llama backend across model switches
- Score difficulty from the latest turn, not client overhead
- Guard truncate_history against 0 turns; clearer profile-save error toast
- Match quant tag as delimited suffix (exclude _L/_S variants)
- Row size/est/fit reflect the selected quant variant
- Download+load the selected quant variant on switch
- Restore apply_history_window doc; assert pre-filter seen recording
- Add port to test AppState literal
- Repair mojibake from perl double-encoding
- Dashboard clear works in webview (inline confirm); add smart-history toggle
- Model delete uses inline confirm (webview-safe)
- Render breaker label via textContent
- Group CloudDown windows; keep Open schedule stable
- Feedback CSV row emits all 13 columns
- Harden feedback widgets (textContent, empty-state, config link)
- Capability card border only between rows
- Isolate settings in tool-request http tests + share capability slope
- Survive server-thread failure instead of killing the app
- Linux GUI backend features + pin CUDA_COMPUTE_CAP
- Cuda kernel build — MSVC on PATH (win) + gcc-11 via ubuntu-22.04 (linux)
- Cuda round 3 — compute-cap 80 (Ampere) + pin MSVC 14.39
- Un-regress windows + limit cuda arch (post-mistralrs)
- Cuda link OOM + windows VS2026 (round 6)
- Print detected variant before API call; distinguish 404
- Ps1 — print variant before API, handle 404, harden parse
- Install git-cliff binary instead of broken Docker action

### Build & CI

- Add release-fast profile + build-app.sh --fast (non-LTO, ~30s link)
- Feature-flag inference backends (metal/cuda/cpu)
- Compile_error guard for backend-less non-macOS builds
- Build-app.sh requests --features metal explicitly
- GitHub Actions build+test matrix + lint gate
- Derive .app plist version from Cargo.toml
- Add git-cliff config for release changelog
- Release workflow (bump + changelog + 5-artifact publish)

### Documentation

- Cite mistralrs source confirming device-map max_seq_len is estimator-only
- Rewrite README for 3B default, GPU/Metal active, model/context switching guide
- Design spec for local/cloud model router (load balancer)
- Append review-fix section to task-3-report
- Design spec for hot-swappable model backend (sub-project 1/3)
- Design spec for model catalog + recommendation + status (sub-project 2/3)
- Design spec for Model Manager window + cross-platform tray (sub-project 3/3)
- Design spec for capability-adjusted routing threshold
- Design spec for /v1/responses OpenAI Responses API endpoint
- Tray client-toggle design (sub-project A)
- Tray client-toggle implementation plan (sub-project A)
- Context-fit core + load-time ctx auto-clamp (sub-project 1)
- Context-fit core + load-time ctx auto-clamp (sub-project 1)
- Fix llama-log target name in doc comment (final-review minor)
- Per-model ctx + KV-aware catalog (sub-project 2)
- Per-model ctx + KV-aware catalog (sub-project 2)
- Model Manager per-model ctx control (sub-project 3)
- Model Manager per-model ctx control (sub-project 3)
- Per-model execution profile (sub-1)
- Per-model execution profile implementation plan (sub-1)
- Quant tier (sub-2)
- Quant tier implementation plan (sub-2)
- Tool filter (sub-3)
- Tool filter implementation plan (sub-3)
- Config nav hub + dashboard + blur-hide + quit unwire
- Config nav + dashboard implementation plan
- Smart history filter — methods, alternatives, design (deferred)
- Full rewrite covering routing, tray/Config, dashboard, tools, smart history
- Add UI screenshots (config, dashboard, models, tools)
- Fase A — observability, cost & daily budget design
- Fase A implementation plan — observability, cost & budget
- Circuit breaker design spec
- Circuit breaker implementation plan
- Routing feedback loop design spec
- Feedback loop implementation plan
- Per-model effective capability design spec
- Effective capability implementation plan
- Record D3 multi-local-tiering as deferred
- Document semantic routing as DEFERRED
- Backend portability design (metal/cuda/cpu feature wiring)
- Implementation plan (3 tasks: cargo wiring, guard, build-app)
- Sync spec cpu wiring to shipped (bare, no openmp)
- CI design (5-config matrix + fmt/clippy lint gate)
- Fmt blocking + one-time reformat baseline; clippy non-blocking
- Implementation plan (rustfmt baseline + ci.yml)
- Release automation design (workflow_dispatch bump + git-cliff + 5 artifacts)
- Implementation plan (build-app version, cliff.toml, release.yml)
- Install-helper design (auto-detect scripts + README selector)
- Install-helper plan (install.sh, install.ps1, README section)
- Add Install section (one-liners + download selector table)
- Drop stale mistralrs references (backend removed)

### Features

- Internal chat request/response types
- OpenAI chat completions translation layer
- Anthropic messages translation layer
- Mistralrs engine with optimizations and tool parsing
- HTTP router and non-streaming handlers for both APIs
- CLI config and real server boot
- SSE streaming for OpenAI and Anthropic endpoints
- Per-request logging, buffered streaming tool-calls, GPU by default
- Default ctx_len 32768 to fit agentic clients (Claude Code ~26k prompts)
- Default to lightweight Qwen2.5-3B (single-file, 8k ctx) for coexisting local use
- In-process HuggingFace model download
- Llama.cpp engine with chat-template prompt + tool parsing
- Longest-common-prefix KV reuse + TTFT measurement (warm prefix = fast turns)
- Tscg compact tool serialization
- Use Tscg-compressed tool schemas in the prompt
- Quantized KV cache (--kv-type, default q8)
- Persist prefix KV to disk + warm-start on restart
- Tray menu shows URL, model, context, KV cache, backend + Quit (matches by id)
- Nicer gradient chat-bubble icon (tray + .app .icns), Accessory activation policy (fixes missing menu-bar item + Dock icon), file logging to /tmp/localllm.log, 'Open Logs' menu item
- Quiet default port 31415 (avoid dev/LLM-tool port clashes); ctx-len default 32768 (Claude Code's ~20k prompts work out of the box); tray status goes green '🟢 Running' only once the server is actually serving; click URL copies it to clipboard
- Routing profiles and policy knobs
- Context-gate decision fn + token estimation
- Byte-faithful reverse-proxy to provider upstream
- Thread routing policy + ctx window through AppState
- Route over-window requests to cloud reverse-proxy
- Difficulty score + full decide (threshold + cascade decision)
- Cascade weak local results to cloud (buffered/non-stream paths)
- Persist routing profile + serde/clap derives on Profile
- --profile flag + resolve profile (cli>saved>default) into shared policy
- Routing profile submenu — live switch + persist
- Session cloud counters + one-shot gates + gated notifier
- Forward returns ForwardOutcome (relay vs classified degrade)
- Degrade cloud failures to local + session usage alerts
- Cross-platform notifications via notify-rust (fixes icon)
- Ensure_model_with_progress (per-chunk progress callback)
- Swappable Generator + status (no switch logic yet)
- Start_switch state machine (drain/drop/build/swap/restore)
- Admin token (--admin-token / random) + constant-time check + 0600 file
- ModelManager backend + token-guarded /admin/model switch endpoints
- Curated catalog + pure catalog_view (status/RAM/fit/recommend)
- Delete_cached (remove a cached model file)
- /admin/models catalog + delete endpoints + sysinfo RAM
- /manager SPA route + admin-token init script
- Model Manager window (tao+wry) with in-memory token injection
- Active_params_b capability resolver (catalog→name-parse→neutral)
- Capability-adjusted threshold (active model param size shifts routing)
- Expand model list across sizes and newer generations
- Responses request types + to_internal
- Responses from_internal (non-streaming object)
- Responses buffered SSE event renderer
- POST /v1/responses handler + route + HTTP tests
- Module scaffold — ClientInjector trait, ClientPrior, atomic_write
- Claude Code JSON injector (surgical enable/disable)
- Codex TOML injector (format-preserving enable/disable)
- Persist client-integration toggle state, preserve profile on save
- Toggle wiring — enable/disable all clients + tray CheckMenuItem
- Log the local-vs-cloud decision and its inputs per request
- Pure context-fit math (KV sizing, max_ctx_fit, ctx_bounds)
- Clamp ctx to the device memory budget at load (fixes Phi-4 OOM)
- Pass total RAM to load, use fitted ctx as the router window
- Persist per-model ctx override
- KV-aware fit + per-model ctx bounds in the catalog view
- POST /admin/model/ctx + load-path per-model ctx override
- Per-model context-window control on the detail pane
- Per-model ExecProfile with model_ctx migration
- Per-model KV in fit view + recommendation fields
- Pure saved->catalog->global resolver
- Thread gpu_layers into llama model load
- Resolve per-model profile on load and switch
- Turn-based history truncation helper
- Truncate history per active model's window
- POST /admin/model/profile sets exec profile
- Per-model KV, history, and gpu_layers controls
- QuantVariant type + variants_for/files_for_quant with fallback
- Harvest_quants dev binary + generated variant table
- Per-model quant selection field + resolution
- ModelSpec carries optional quant
- Per-variant fit/status + selected quant in ModelView
- Resolve quant variant file list on model switch
- Quant in profile/switch/delete endpoints
- Per-model quant variant selector
- Per-surface tool blocklist persistence
- Filter_tools drops blocklisted tools
- Per-surface tool discovery + request-time filtering
- GET/POST /admin/tools discovery + blocklist endpoints
- Per-client tool filter view
- Persist + resolve last active model
- Persist active model on switch, restore at boot
- GET/POST /admin/integrations wiring state + toggle
- JSONL routing history with retention prune
- Dashboard rollups (hour/day/month + recent)
- Record routing decision to route_log
- GET /admin/dashboard rollups endpoint
- Prune routing log + rotate app log on startup
- Config landing + hash router + integrations toggle
- Config submenu, read-only wired line, unwire on quit, hide on blur
- Firestore-style model columns, polished config landing, rich dashboard
- Persist discovered tool set; refresh + persist on change
- Dashboard clear, routing screen in config, tray config-home + glyphs
- Blocked tools stay visibly blocked even when not re-sent
- Render tool descriptions; seen set is cumulative (never deleted)
- Smart history filter (BM25+MMR), global Config toggle
- Dashboard refresh button
- Log score breakdown + latest-turn prompt snippet
- Tools filters + collapsible descriptions; dashboard table filters, score popover, prompt expand
- Tools desc persistence, dashboard prompt/popover, responsive, Balanced threshold config, tray hard-exit
- Two-line log model (decision+outcome) with rid + retrocompat
- Static cloud model price table + price_for
- Dashboard join with cost, latency and fallback windows
- Record post-generation outcome (tokens, latency, cost saved)
- Log provider-degrade fallback with reason for dashboard windows
- Daily cloud-spend tracker with log seed
- Persist budget enabled + daily USD cap
- Daily budget cap — enforce, endpoints, cloud-cost accrual
- Budget config screen + nav card
- Dashboard cost, latency and fallback windows
- GET /admin/export csv|jsonl + dashboard button
- Optional Prometheus /metrics endpoint
- Cloud circuit breaker state machine
- Gate route_decision on the circuit breaker
- Trip on degrade, close + notify on recovery
- GET /admin/breaker + POST /admin/breaker/reset
- Share circuit breaker + "Retry cloud now" menu item
- Dashboard circuit-breaker status widget
- LogLine::Feedback line kind joined into recent rows
- Accuracy stats + threshold suggestion in dashboard
- Meter cloud relays; emit cloud_trivial + real cloud outcomes
- Emit cascade + truncated signals
- In-memory re-ask detector
- Dashboard feedback signals + accuracy card + suggestion banner
- Params_b_for_key resolves nominal size from compact key
- Record active local model per decision
- Per-model effective-capability estimate
- Per-model effective-capability dashboard card
- Install.sh — auto-detect + install localllm (macOS/Linux)
- Install.ps1 — auto-detect + install localllm (Windows)

### Performance

- Assemble prompt stable-first (all system blocks) then user turns last, so only the question varies at the tail → near-total prefix-cache reuse on warm turns
- N_ubatch=2048 (prefill is bandwidth-bound here, no measurable gain but harmless); add PROMPT_FORMAT_VERSION to .kvstate provenance so stale-format states are never warm-loaded

### Refactor

- Cross-platform tray (de-gate + platform abstraction) + share admin token
- Forward() takes explicit upstream path (drop Provider::path)
- Bind now once in route_decision
- Remove unused mistralrs backend (llama-only)

### Tuning

- Balanced threshold 0.6 -> 0.45 (route more to cloud)

### Merge

- Auto-route clients through localllm (responses API + tray toggle)
- Ctx-aware model fit — Phi OOM fix, per-model ctx, KV-aware catalog + UI
- Fix model-switch backend re-init (share one global llama backend)
- Route difficulty from latest turn (keep simple agentic asks local)

### Plan

- Phase 1 (embedded llama.cpp engine + prewarm), 5 tasks
- Phase 2 (Tscg tool-schema compression), 2 tasks
- Phase 4 (KV quantization + persistent prefix KV across restart); Phase 3 deferred
- Phase A (routing core + cloud reverse-proxy), 5 tasks
- Phase B (difficulty score + cascade fallback), 2 tasks
- Phase C (tray profile selector + persistence), 3 tasks
- Phase D (usage tracking + graceful degrade), 3 tasks
- Hot-swappable model backend (sub-project 1/3), 5 tasks
- Model catalog + recommendation + status (sub-project 2/3), 3 tasks
- Model Manager window + cross-platform tray (sub-project 3/3), 3 tasks
- Capability-adjusted routing threshold, 2 tasks
- /v1/responses OpenAI Responses API endpoint, 5 tasks

### Polish

- Step-256 client validation + disable Usar padrão on default

### Spec

- Plug-and-play single-binary constraint; embed llama.cpp (llama-cpp-2), reorder phases

