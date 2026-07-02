# Tool filter (sub-project 3)

**Date:** 2026-07-02
**Status:** Approved design, ready for planning
**Scope:** Sub-project 3 of 3. Per-client tool filtering.
**Depends on:** Sub-1 (request-shaping choke point `apply_history_window`) and its
branch lineage — branch from `feat/quant-tier` (the latest sub-branch), not `main`.

## Context

Sub-1 exposed `history_turns` to trim conversation history sent to the local
model. This sub-project addresses the other half of the same complaint: large
agentic clients (Claude Code, Codex) send many tool schemas every turn, which
inflates the prompt and degrades tool selection. This lets the user see which
tools a client is sending and mute the noisy ones. Filtering happens at request
time in the same normalized `{messages, tools}` choke point as history
truncation, so no model rebuild is needed — the KV prefix cache re-diffs when the
tool set changes.

## Goals

- Discover, per client, the tool set arriving on requests.
- Let the user disable specific tools per client (checkboxes), persisted across
  restarts.
- Filter the disabled tools out of the request before local inference.
- No model reload / no explicit rebuild: filtering is request-time.

## Non-goals

- Distinguishing two different clients that use the same API surface (see
  "Client key" — attribution is by API surface, not a precise client identity).
- Filtering the CLOUD path: the cloud reverse-proxy forwards raw bytes untouched;
  tool filtering is local-only.
- Editing tool schemas (only whole-tool drop by name).

## Client key (attribution)

A request's client is keyed by its **API surface**, known at the handler:

| Surface key | Endpoint | Typical client |
|---|---|---|
| `anthropic` | `POST /v1/messages` | Claude Code |
| `openai` | `POST /v1/chat/completions` | Codex |
| `openai-responses` | `POST /v1/responses` | OpenAI Responses clients |

Two clients on the same surface share one filter. This is the deliberate
simplicity trade decided during design.

## Filter semantics

The persisted config is a **blocklist** (disabled tool names) per surface. Any
tool NOT in the blocklist passes. A newly-appearing tool (e.g. after a client
update) is enabled by default — nothing breaks silently; the user mutes noise
explicitly.

## Architecture

### 1. Discovery registry (in-memory) — `AppState`

`AppState` gains a shared registry of the most-recent tool names seen per
surface:

```rust
pub tool_registry: Arc<Mutex<BTreeMap<String, Vec<String>>>>, // surface -> sorted unique tool names
```

On each request, the shaping step records the request's tool names into the
registry for that surface (replace with the latest sorted-unique set — the newest
request reflects the client's current tools). This is the live "what's being used
now" source. It is not persisted (a restart re-discovers on the first request).

### 2. Persistence — `settings.rs`

`Settings` gains:

```rust
#[serde(default)]
tool_filters: std::collections::BTreeMap<String, Vec<String>>, // surface -> disabled tool names
```

with helpers `load_tool_filter(surface) -> Vec<String>` and
`save_tool_filter(surface, &[String])` (preserving the rest of settings, matching
the existing per-key save pattern). Only the blocklist persists.

### 3. Request-time filtering — `api/common.rs`

A pure function:

```rust
/// Drop tools whose name appears in `disabled` (case-sensitive exact match).
pub fn filter_tools(tools: Vec<ToolSpec>, disabled: &[String]) -> Vec<ToolSpec>;
```

### 4. Wiring — `server.rs`

Extend the existing request-shaping performed alongside `apply_history_window`.
For each of the three handlers, the surface key is a constant. A single helper:

```rust
fn shape_tools(state: &AppState, surface: &str, req: &mut ChatRequest) {
    // record seen
    {
        let mut reg = state.tool_registry.lock().unwrap();
        let mut names: Vec<String> = req.tools.iter().map(|t| t.name.clone()).collect();
        names.sort(); names.dedup();
        reg.insert(surface.to_string(), names);
    }
    // filter disabled
    let disabled = crate::settings::load_tool_filter(surface);
    if !disabled.is_empty() {
        let tools = std::mem::take(&mut req.tools);
        req.tools = crate::api::common::filter_tools(tools, &disabled);
    }
}
```

Called in each handler right after `apply_history_window(&state, &mut internal)`,
passing the handler's surface constant (`"anthropic"` / `"openai"` /
`"openai-responses"`). Applied before `route_decision` so routing sees the
filtered tool count too. The cloud path is untouched.

### 5. Endpoints — `server.rs`

- `GET /admin/tools` (admin-guarded): returns, per surface, the merged view:
  ```json
  { "anthropic": { "seen": ["Bash","Read",…], "disabled": ["Read"] }, … }
  ```
  `seen` from the registry, `disabled` from settings. A surface with neither is
  omitted; a surface present in only one source still appears.
- `POST /admin/tools` (admin-guarded): body `{ "surface": "...", "disabled": [".."] }`.
  Validates `surface` is one of the three known keys (400 otherwise). Persists the
  blocklist for that surface (empty list clears it). No reload.

### 6. UI — `manager_ui`

A new "Tools" view in the manager (a top-level section alongside the model
picker; reached from the crumbs/nav). For each surface that has `seen` or
`disabled` tools: the surface label and its tools as checkboxes (checked =
enabled = not in the blocklist). A "Salvar" button posts the resulting disabled
list for that surface to `POST /admin/tools`. A hint states changes apply on the
next request (no rebuild) and a warning notes that disabling a tool the client
relies on removes that capability. If a surface has no `seen` tools yet (fresh
restart), show "envie um request deste cliente para descobrir as tools."

## Data flow

```
request → adapter → internal{messages,tools}
        → apply_history_window (sub-1)
        → shape_tools(surface): record seen → registry; drop disabled from tools
        → route_decision → local generate            [local only; cloud untouched]

UI Tools view → GET /admin/tools (registry ∪ settings)
             → POST /admin/tools {surface,disabled} → settings.tool_filters
```

## Error handling

- Unknown surface in `POST /admin/tools` → 400.
- Bad/absent settings → empty blocklist (no filtering); never blocks startup.
- Registry lock poisoning → treat as empty seen (do not panic the request path;
  use `lock().unwrap_or_else(|e| e.into_inner())` or equivalent recovery).
- A blocklist naming a tool the client no longer sends → harmless no-op.

## Testing

- `filter_tools` (pure): drops only named tools, preserves order of the rest,
  empty disabled = passthrough, name not present = no-op.
- settings: `tool_filters` round-trip; save/clear preserves the rest of settings.
- registry: `shape_tools` records sorted-unique seen names; filtering removes the
  disabled set for the surface.
- endpoint: `GET /admin/tools` merges registry + settings; `POST` persists and
  rejects an unknown surface.
- request path (integration): a request whose surface has a disabled tool reaches
  the generator with that tool removed; other surfaces unaffected; cloud path not
  filtered.

## Open questions for planning

- Whether the "Tools" UI is a new top-level view or a panel within an existing
  screen — decide during planning based on the current `manager_ui` nav structure.
- Whether to also record a per-surface "last seen" timestamp for the UI — likely
  YAGNI; omit unless the nav needs it.
