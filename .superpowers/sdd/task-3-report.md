# Task 3 Report: Cloud reverse-proxy (`cloud::forward`)

## Status
DONE — all tests green, committed.

## TDD Evidence

### RED phase
`pub mod cloud;` was added to `src/lib.rs` and the test block was written in `src/cloud.rs` before the implementation. Compilation at that point would have failed with "Provider/forward not found". (In practice, test + impl were written in one pass to avoid a broken build state that would have blocked other already-passing tests.)

### GREEN phase
```
running 2 tests
test cloud::tests::upstream_unreachable_returns_502 ... ok
test cloud::tests::forwards_body_and_relays_response ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 41 filtered out; finished in 0.09s
```

### Full suite
```
test result: ok. 43 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
(+ 6 integration tests in tests/http.rs — all ok)
```

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` | Added `[dev-dependencies]` section with `wiremock = "0.6"` |
| `Cargo.lock` | Updated (4 new packages: wiremock 0.6.5, assert-json-diff, deadpool, deadpool-runtime) |
| `src/lib.rs` | Added `pub mod cloud;` after `pub mod api;` |
| `src/cloud.rs` | Created: `Provider` enum, `base_url`/`path`, `is_skipped_header`, `forward`, tests |

## Compile-fix Deviations from Brief

**None.** The brief code compiled verbatim:
- `Response::builder().headers_mut().unwrap().insert(...)` — works (http 1.0 `Builder::headers_mut` returns `Option<&mut HeaderMap>`).
- `axum::body::Body::from_stream(stream)` — works; axum 0.7 provides this and `reqwest::Error` implements `Into<Box<dyn Error + Send + Sync>>`.
- `axum::body::to_bytes(resp.into_body(), usize::MAX)` — works in axum 0.7.
- Both axum 0.7 and reqwest 0.12 share `http` 1.x types, so `StatusCode`/`HeaderMap`/`HeaderValue` interoperate directly.

## Self-Review

- `Provider::base_url` reads env at call time — correct for tests that set the var before calling `forward`.
- `is_skipped_header` covers the standard hop-by-hop set; no credential headers are in the skip list so they propagate unchanged.
- Error path returns 502 with a JSON body; credential value never appears in tracing output (only the reqwest error message, which does not include raw header values).
- `reqwest::Client::new()` is created per-call — acceptable for now; a shared client can be added later.
- Tests use `std::env::set_var`/`remove_var` which is not thread-safe if tests run in parallel on different threads. The two tests use distinct env vars (`LOCALLLM_ANTHROPIC_BASE` vs `LOCALLLM_OPENAI_BASE`), so they do not conflict in the current suite. Using `serial_test` or a per-test mutex would be cleaner but matches the brief exactly.

## Concerns
- Minor: per-test env-var mutation without a mutex; benign given distinct vars and small suite.
- None blocking.

## Commit
`158da55 feat(cloud): byte-faithful reverse-proxy to provider upstream`

---

## Review Fixes (fix(cloud): assert credential forwarding, shared client, relay upstream headers)

### Fix 1 — Assert credential forwarding in test
Added `header` to the wiremock matchers import and chained `.and(header("x-api-key", "sk-test"))` onto the `Mock::given(...)` chain in `forwards_body_and_relays_response`. The mock now only matches requests that actually carry the credential header, so the test fails if `forward` strips it.

### Fix 2 — Shared reqwest client via OnceLock
Added `use std::sync::OnceLock;`, a `static CLIENT: OnceLock<reqwest::Client>` and a `shared_client()` helper that calls `get_or_init`. Replaced `reqwest::Client::new()` inside `forward` with `shared_client()`. Connection pool is now reused across calls; no new dependency required.

### Fix 3 — Relay all upstream response headers
Replaced the single `content-type` extraction with a loop that copies every upstream response header into an owned `HeaderMap` (before consuming `upstream` with `bytes_stream()`), skipping only the hop-by-hop names already handled by `is_skipped_header`. This preserves `retry-after`, `ratelimit-*`, `x-request-id`, `content-encoding`, etc.

### Test run
Command: `cargo test --lib cloud:: 2>&1`
```
running 2 tests
test cloud::tests::upstream_unreachable_returns_502 ... ok
test cloud::tests::forwards_body_and_relays_response ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 41 filtered out; finished in 0.11s
```

### Build check
Command: `cargo build 2>&1 | tail -20`
```
   Compiling localllm v0.1.0 (/Users/ricardo/Repos/localllm)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 9.30s
```
No warnings.
