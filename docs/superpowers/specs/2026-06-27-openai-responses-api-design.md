# OpenAI Responses API endpoint (`/v1/responses`) — design

**Date:** 2026-06-27
**Status:** approved (design)
**Sub-project:** B of 2. Part of the "auto-route clients through localllm" tray
toggle feature. This sub-project is the server-side dependency that makes the
**Codex CLI** usable against localllm. Sub-project A (the tray toggle + client
config injection) is specced separately and builds **after** this one.

## Why

Codex (2026) speaks **only** the OpenAI **Responses API** (`wire_api =
"responses"`); OpenAI dropped Chat Completions support in Codex. localllm today
serves only `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic), so
pointing Codex at localllm fails. To let the tray toggle redirect Codex
transparently, the server must answer the Responses API.

When Codex is configured with:

```toml
[model_providers.localllm]
name = "localllm"
base_url = "http://127.0.0.1:31415/v1"
env_key = "LOCALLLM_KEY"
wire_api = "responses"
```

it POSTs to `{base_url}/responses` → **`/v1/responses`**, with the provider key
in the auth header.

## Scope

Add a third HTTP wire that speaks the OpenAI Responses API and routes
local/cloud **exactly like** the existing `/v1/messages` and
`/v1/chat/completions` paths. It reuses the internal `ChatRequest` /
`ChatResult` / `StreamDelta` types (`src/api/common.rs`) and the engine — **no
new model logic, no new routing logic.**

### In scope
- `POST /v1/responses` — non-streaming and streaming (SSE).
- Request parse for the shapes Codex actually sends: `input` as a plain string
  **or** as an array of items (input messages + `function_call_output` items),
  `instructions`, flat `tools`, `stream`, `max_output_tokens`, `temperature`.
- Response render: `output` array with `message` (carrying `output_text`)
  and/or `function_call` items, plus `usage`.
- Streaming event sequence (see below).
- Cloud reverse-proxy of `/v1/responses` to `https://api.openai.com/v1/responses`,
  byte-faithful, carrying the client's own key.

### Out of scope (YAGNI)
- `reasoning` items / reasoning summaries.
- Image / non-text input parts (dropped, same as the Chat path does).
- Built-in tools (`web_search`, `file_search`, `code_interpreter`).
- Stateful conversations: `previous_response_id` / `store=true`. **Stateless
  only** — every request carries full `input`.
- `background` mode, `include`, structured-output `text.format` schemas.

## Architecture

New module `src/api/openai_responses.rs`, mirroring the structure of
`src/api/openai.rs` (wire types → `to_internal` → engine → `from_internal` /
streaming renderer). Registered in `src/api/mod.rs`.

```
Codex ──POST /v1/responses──▶ handle_oai_responses (server.rs)
                                   │ parse RespRequest
                                   │ to_internal() ──▶ ChatRequest
                                   ▼
                              route_decision()
                          ┌────────┴─────────┐
                       Cloud                Local
                  cloud::forward      engine.generate / stream
                  (→ api.openai.com    │
                   /v1/responses)      ▼
                                  from_internal() / SSE events
```

### Wire types (`RespRequest`)

```jsonc
{
  "model": "…",
  "instructions": "system text",        // optional → System message (prepended)
  "input": "hi"                         // string form
        // OR array form:
        // [ {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},
        //   {"type":"function_call_output","call_id":"call_1","output":"sunny"} ],
  "tools": [ {"type":"function","name":"get_weather","parameters":{…}} ],
  "stream": false,
  "max_output_tokens": 1024,
  "temperature": 0.7
}
```

Use `#[serde(untagged)]` for the `input` string-or-array enum, and a
`#[serde(tag="type")]` enum for input items (`message`, `function_call_output`,
and an `Other` catch-all that is ignored). Content parts accept `input_text` /
`output_text` text and ignore the rest (mirror `OaiContentPart::Other`).

Note the Responses `tools` shape is **flat** (`name`/`parameters` at the top
level), unlike Chat Completions' nested `function` object.

### `to_internal(RespRequest) -> Result<ChatRequest, String>`
- `instructions` (if present) → a leading `ChatMessage { role: System }`.
- `input` string → one `User` message.
- `input` array:
  - `message` items → `ChatMessage` by `role` (text flattened from content
    parts; `developer`/`system` → `Role::System`).
  - `function_call_output { call_id, output }` → `Role::Tool` `ChatMessage`
    with `ToolResult { tool_call_id: call_id, content: output }`.
- `tools` → `ToolSpec` (flat name/description/parameters).
- `max_output_tokens` → `max_tokens`; `temperature`, `stream`, `model` mapped
  through. Unknown item type → `Err` → 400.

### `from_internal(ChatResult, model) -> RespResponse` (non-streaming)
Build the Responses object:

```jsonc
{
  "id": "resp_<uuid>",
  "object": "response",
  "created_at": 1719500000,
  "model": "…",
  "status": "completed",
  "output": [
    // when text present:
    {"type":"message","id":"msg_<uuid>","role":"assistant","status":"completed",
     "content":[{"type":"output_text","text":"…","annotations":[]}]},
    // for each tool call:
    {"type":"function_call","id":"fc_<uuid>","call_id":"call_…",
     "name":"get_weather","arguments":"{…}","status":"completed"}
  ],
  "usage": {"input_tokens":N,"output_tokens":M,"total_tokens":N+M}
}
```

`finish_reason` is not a top-level Responses field; map internal
`FinishReason::Length` → `status:"incomplete"` with
`incomplete_details.reason:"max_output_tokens"`, otherwise `status:"completed"`.

### Streaming (SSE)

Each event is two SSE lines: `event: <type>\n` + `data: <json>\n\n`, each
carrying an incrementing `sequence_number`. Mirror the buffered-streaming
decision already used for tools in `openai.rs`:
`stream_chunks_from_result`-style replay guarantees correct framing.

**Text path** (incremental, reusing `StreamDelta`):
1. `response.created` — `{response:{id,status:"in_progress",…}}`
2. `response.output_item.added` — message item, `output_index:0`
3. `response.content_part.added` — `{part:{type:"output_text",text:""}}`
4. `response.output_text.delta` — `{item_id,output_index:0,content_index:0,delta:"…"}` (repeated)
5. `response.output_text.done` — full text
6. `response.content_part.done`, `response.output_item.done`
7. `response.completed` — `{response:{…,status:"completed",usage}}`

**Tool-call path** (buffered, generated non-streaming then replayed):
- `response.created`
- per call: `response.output_item.added` (`function_call` item) →
  `response.function_call_arguments.delta` (full args once) →
  `response.function_call_arguments.done` → `response.output_item.done`
- `response.completed`

A minimal-but-valid subset Codex accepts is `response.created` →
`response.output_text.delta`* → `response.completed` (text) and the
`function_call` add/delta/done → `response.completed` (tools). Emit the
intermediate `output_item.added` / `content_part.added` wrappers too — Codex's
event reader expects item/content framing for tool calls.

### Cloud reverse-proxy change

`cloud::forward` currently derives the upstream path from
`Provider::path()` (hardcoded `/v1/chat/completions` for OpenAI). The Responses
handler must forward to `/v1/responses` on the same OpenAI base.

**Chosen approach:** generalize `forward` to take the upstream path explicitly —
`forward(provider, upstream_path, headers, body)` — and have each handler pass
its own path (`/v1/messages`, `/v1/chat/completions`, `/v1/responses`). This
keeps `Provider` purely about base-URL/credential selection and avoids a
combinatorial enum. `Provider::path()` is removed; the two existing handlers
pass their literal paths. (Alternative considered: a `Provider::OpenAIResponses`
variant — rejected because provider identity and endpoint path are orthogonal.)

The `LOCALLLM_OPENAI_BASE` env override continues to work for tests (point at a
wiremock server asserting the `/v1/responses` path).

### Handler (`server.rs`)
`handle_oai_responses`, registered `.route("/v1/responses",
post(handle_oai_responses))`. Structure copied from `handle_oai_chat`:
parse → `route_decision(&state, &internal, &headers)` → `Decision::Cloud` →
`cloud::forward(Provider::OpenAI, "/v1/responses", …)`; `Local*` → engine
generate/stream rendered through the new renderer. Reuse the same
degrade-on-upstream-failure and `est_prompt_tokens` accounting as the other
wires.

## Error handling
- Malformed/unknown `input` item type → `400` with the `to_internal` error text
  (same pattern as the Chat handler).
- Cloud upstream non-2xx → existing degrade path (warn + relay status), no new
  behavior.
- Streaming: pipe events straight through; never buffer the cloud SSE body
  (matches existing streaming guidance).

## Testing
Pure unit tests in `openai_responses.rs` (no network), mirroring `openai.rs`:
- Parse Codex-shaped `input` **array** with a `message` + a
  `function_call_output` item → correct internal `Role::Tool` `ToolResult`.
- Parse `input` **string** form → single user message.
- `instructions` → leading `System` message.
- Flat `tools` → `ToolSpec`.
- `from_internal` renders an `output_text` message; renders a `function_call`
  item with `call_id`/`name`/`arguments`.
- `FinishReason::Length` → `status:"incomplete"`.
- Streaming renderer: text path emits
  `response.created` … `response.output_text.delta` … `response.completed` in
  order with monotonic `sequence_number`; tool path emits the
  `function_call` add → arguments.delta → completed sequence.
- Unknown input item type → `Err`.

HTTP-level test in `tests/http.rs`: `POST /v1/responses` (local route, stubbed
engine) returns a well-formed `response` object; a streamed request returns the
event sequence. Cloud route asserted against a wiremock upstream verifying the
`/v1/responses` path and forwarded auth header.

## Files touched
- `src/api/openai_responses.rs` — **new** (wire types, conversions, renderers,
  unit tests).
- `src/api/mod.rs` — `pub mod openai_responses;`.
- `src/server.rs` — `handle_oai_responses` + route registration; update the two
  existing cloud-forward call sites to pass their upstream path.
- `src/cloud.rs` — `forward` takes an explicit upstream path; remove
  `Provider::path()`.
- `tests/http.rs` — endpoint + cloud-forward tests.

## Follow-up (sub-project A, separate spec)
Tray toggle that injects `base_url`/`wire_api` into `~/.codex/config.toml` and
`ANTHROPIC_BASE_URL` into `~/.claude/settings.json`, with safe read-modify-write
+ backup/restore. Depends on this endpoint for Codex to function.
