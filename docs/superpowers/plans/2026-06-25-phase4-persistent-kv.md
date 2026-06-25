# Phase 4 — KV quantization + persistent prefix KV across restart

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** (a) Quantize the KV cache (Q8/Q4) to cut KV RAM and on-disk state size;
(b) persist the warm prefix KV to disk and auto-load it on startup, so the large
static prefix is warm immediately after a restart/sleep — serving the "always use
Claude Code across sessions" goal without re-paying the cold prefill each launch.

**Architecture:** Builds on the persistent-context worker thread (Phase 1 Task 4/5)
and the confirmed `state_seq_save_file`/`state_seq_load_file` API (Phase 1 Task 1
NOTES §8). After the worker prefills a prefix and updates `cached_tokens`, it can
save that state to disk keyed by a hash of the prefix tokens. On startup, if a
saved state for the configured/last prefix exists, load it so `cached_tokens` is
pre-populated and the first turn reuses it.

**Tech Stack:** Rust, llama-cpp-2.

## Global Constraints

- Must not regress Phase 1/2: tool-calling acceptance scripts still 5/5;
  prefix-reuse still works; output stays coherent.
- KV quantization must keep output coherent (Q8 is safe; Q4 acceptable per
  research — verify).
- Disk state keyed by a hash of the prefix tokens so a stale/mismatched cache is
  never loaded into the wrong prompt.
- Decisions about lazy tool-gating are OUT OF SCOPE (conflicts with prefix reuse;
  documented as future work).

## Design note: tool-gating deferred
Injecting only query-relevant tools shrinks the prefix but makes it VARY per turn,
which breaks the longest-common-prefix KV reuse that makes warm turns sub-second.
Since reuse already solves the warm-turn case, aggressive tool-gating is deferred.
Recorded here so the decision is explicit.

---

### Task 1: KV cache quantization

**Files:**
- Modify: `src/engine_llama.rs` (context params: set K/V cache type), `src/config.rs`
  (add `--kv-type f16|q8|q4`, default `q8`)
- Test: config default test + manual coherence/RAM check

**Interfaces:**
- Produces: context created with quantized KV cache when configured.

- [ ] **Step 1: Find the KV-type API** in llama-cpp-2 context params
  (`~/.cargo/registry/src/*/llama-cpp-2-0.1.150/src/context/params/`): the
  `type_k` / `type_v` setters taking a GGML type (e.g. `with_type_k`, `with_type_v`
  or a raw field). Record the exact names. (Flash attention may be required for
  some quantized KV on some backends — check; if Metal needs it, enable it.)
- [ ] **Step 2: Config** — add `KvType` enum {F16, Q8, Q4} + `--kv-type` (default
  Q8). Failing test: default is Q8. Implement → pass.
- [ ] **Step 3: Apply** the K/V type in the worker's context creation based on
  config.
- [ ] **Step 4: Manual coherence + RAM check** — start server with `--kv-type q8`,
  run 3 sequential requests (coherent answers), and `scripts/test_openai_tools.sh`
  5/5. Compare the "Allocating … KV cache" log line size vs f16 (should be ~half
  for Q8). Report.
- [ ] **Step 5: Commit** `feat(phase4): quantized KV cache (--kv-type, default q8)`.

---

### Task 2: Persist prefix KV to disk + auto-load on startup

**Files:**
- Modify: `src/engine_llama.rs` (save after prefill; load on startup)
- Modify: `src/config.rs` (`--kv-cache-dir`, default under the app cache dir;
  `--no-kv-persist` flag)
- Create: `scripts/measure_restart.sh`

**Interfaces:**
- Produces: on-disk prefix KV state files keyed by prefix-token hash; auto-load on
  startup; save after the first prefill of a new prefix.

- [ ] **Step 1: Save after prefill** — in the worker, after prefilling a prompt
  whose prefix is "large" (e.g. > N tokens, configurable; skip tiny prompts),
  compute a hash of `cached_tokens` and `state_seq_save_file(path_for(hash), 0,
  &cached_tokens)` to the kv-cache-dir. Save at most once per distinct prefix
  (skip if the file already exists). Log it.
- [ ] **Step 2: Load on startup** — on worker init, if exactly one (or a
  configured/most-recent) saved state exists, `state_seq_load_file` it into the
  context, set `cached_tokens` to the returned tokens, and run the one-token
  warm-up re-decode (NOTES §9) so logits are primed. Log "warm-started from disk:
  N tokens".
- [ ] **Step 3: Hash-keyed correctness** — on a request, prefix reuse already
  diffs `cached_tokens` vs the new prompt, so a loaded-but-mismatched prefix simply
  yields a smaller common prefix (safe). Confirm no incorrect reuse.
- [ ] **Step 4: `measure_restart.sh`** — (1) start server, send a big-prefix
  request (cache + save to disk), note time; (2) kill server; (3) restart; (4) send
  the SAME big-prefix request, measure time. The post-restart request should be
  fast (warm-started from disk) vs a fresh cold start. Paste both times.
- [ ] **Step 5: Regression** — tool tests 5/5, coherent output.
- [ ] **Step 6: Commit** `feat(phase4): persist prefix KV to disk + warm-start on restart`.

---

## Phase boundary

After Task 2, report KV-size reduction and cold-vs-warm-after-restart times. This
completes the planned phases (1, 2, 4; 3 deferred). A final whole-branch review of
the llama.cpp backend + caching system should follow before any merge.
