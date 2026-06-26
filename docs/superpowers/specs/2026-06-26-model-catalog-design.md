# Model catalog + recommendation + status (design)

Date: 2026-06-26
Status: Proposed

## Goal

Provide the data the model-picker window needs: a curated catalog of local models
in a **family → model (params/quant)** drilldown, each annotated with **status**
(in use / downloaded / needs-download), an **estimated RAM cost**, a **fit
verdict** for this machine, and a single **recommended** model. Plus the ability
to **delete** a downloaded model to free disk.

This is **sub-project 2 of 3** of the model-picker feature:
- **1 — hot-swap backend** (done): runtime model switch via `ModelManager` +
  token-guarded `/admin/model` endpoints.
- **2 — this spec:** catalog + recommendation + status + delete.
- **3 — the window:** a `wry` webview, **built with the `frontend-design` skill,
  Firestore-style drilldown**, consuming sub-projects 1 + 2.

## Constraints / context

- Single self-contained binary, plug-and-play, **offline**. The catalog is
  curated by us and **embedded** (compiled in); no network to list models.
- Builds on sub-project 1 (branch `feat/model-hotswap`): `ModelManager.status()`
  returns `SwitchStatus { current: ModelSpec, … }` (the active model);
  `/admin/*` endpoints are token-guarded via `check_admin`; `AppState` is the
  shared handler state; `download::cache_path(repo,file)` gives the cache path
  (existence = downloaded).
- New dep: `sysinfo` (cross-platform total-RAM detection).

## Decisions (from brainstorming)

- **Catalog source:** embedded curated static list (Approach A). Rejected: remote
  JSON (needs network/hosting), live HuggingFace query (uncurated, heavy).
- **Recommendation:** *largest model that fits a RAM budget*. Budget = 65% of
  total RAM (leave ~35% for OS + apps). Show per-model est. RAM + a fit verdict.
- **RAM estimate:** curated GGUF `size_mb × 1.2` (disk + ~20% runtime/KV at the
  default context).
- **Delete:** a downloaded model can be deleted to free disk; the **in-use**
  model cannot (409 — switch away first).
- **Exposure:** token-guarded `GET /admin/models` and `DELETE /admin/models`,
  same `/admin/*` pattern as sub-project 1.

## Architecture

New module **`src/catalog.rs`** — pure data + pure annotation logic. It owns no
HTTP and no `sysinfo`; the handler injects `total_ram_mb`, the active `ModelSpec`,
and an `is_downloaded` closure, keeping the logic unit-testable with no I/O.

```
GET  /admin/models  ─► catalog::catalog_view(CATALOG, total_ram_mb,
                                Some(active), |r,f| cache_path(r,f).exists())
                       → Vec<FamilyView>   (family → models, annotated)
DELETE /admin/models ─► guard: not in-use → download::delete_cached(repo,file)
```

Touched: `src/lib.rs` (`pub mod catalog;`, read total RAM once at startup),
`src/server.rs` (two handlers + `AppState.total_ram_mb` + `router` param),
`src/download.rs` (`delete_cached`), `Cargo.toml` (`sysinfo`).

## Data model (`src/catalog.rs`)

```rust
pub struct CatalogEntry {
    pub family: &'static str,        // "Qwen2.5"
    pub display_name: &'static str,  // "Qwen2.5 7B Instruct"
    pub params: &'static str,        // "7B"  (display)
    pub params_b: f32,               // 7.0  (ordering "largest that fits")
    pub quant: &'static str,         // "Q4_K_M"
    pub repo: &'static str,
    pub file: &'static str,
    pub size_mb: u32,                // curated known download size
}

pub const CATALOG: &[CatalogEntry] = &[ /* curated, smallest→largest per family */ ];

#[derive(serde::Serialize)] #[serde(rename_all="snake_case")]
pub enum ModelStatus { InUse, Downloaded, NeedsDownload }

#[derive(serde::Serialize)] #[serde(rename_all="snake_case")]
pub enum FitVerdict { Fits, Tight, WontFit }

#[derive(serde::Serialize)]
pub struct ModelView {
    pub display_name: String, pub params: String, pub quant: String,
    pub repo: String, pub file: String,
    pub size_mb: u32, pub est_ram_mb: u32,
    pub status: ModelStatus, pub fit: FitVerdict, pub recommended: bool,
}
#[derive(serde::Serialize)]
pub struct FamilyView { pub family: String, pub models: Vec<ModelView> }
```

Seed catalog (editable): Qwen2.5 3B/7B/14B/32B (Q4_K_M), Llama 3.1 8B, plus a
small Phi/Gemma — curated `repo`/`file`/`size_mb` per entry. (Exact list filled
in implementation; the default model `Qwen/Qwen2.5-3B-Instruct-GGUF` must be one
of the entries so the active model is always in the catalog.)

## Annotation logic (pure)

`catalog_view(entries, total_ram_mb, active: Option<&ModelSpec>, is_downloaded: impl Fn(&str,&str)->bool) -> Vec<FamilyView>`:

- `est_ram_mb = (size_mb as f32 * 1.2).round() as u32`.
- `budget_mb = total_ram_mb * 65 / 100`.
- **fit:** `est ≤ budget → Fits`; `budget < est ≤ total*0.85 → Tight`;
  `est > total*0.85 → WontFit`.
- **status:** `active==Some({repo,file}) → InUse`; else `is_downloaded → Downloaded`;
  else `NeedsDownload`.
- **recommended (one global, or none):** among `Fits`, the largest `params_b`
  (tie-break larger `size_mb`); if none fit, the smallest `est_ram_mb`; empty
  catalog → none.
- **order:** families in first-seen `CATALOG` order; entries keep `CATALOG`
  order. Deterministic arithmetic over injected inputs — no clock, no I/O.

## Endpoints (`src/server.rs`)

- `GET /admin/models` (token-guarded): read cached `total_ram_mb`, active from
  `manager.status().current`, return `Json(catalog_view(...))`. Infallible.
- `DELETE /admin/models` (token-guarded): body `{"repo","file"}` (parse via the
  `Bytes` pattern). If `{repo,file}==manager.status().current` → **409**
  `{"error":"cannot delete the model in use"}`. Else
  `download::delete_cached(repo,file)` → `200 {"deleted":bool}`. Bad body → 400,
  bad token → 401, FS error → 500 (clean message).

`download::delete_cached(repo,file) -> std::io::Result<bool>`: remove
`cache_path(repo,file)` (+ any stray `.part`); `Ok(true)` if a file was removed,
`Ok(false)` if nothing was there.

## RAM detection

Add `sysinfo`. In `run_server_with_ready_and_policy`, read total RAM **once** at
startup (it doesn't change), store `total_ram_mb: u64` in `AppState`. `router(...)`
gains a `total_ram_mb` param. `router_for_test*` pass a fixed value (16384) so
catalog tests are deterministic. A `sysinfo` failure → treat total RAM as 0
(graceful: nothing fits → smallest recommended; logged once).

## Error handling

| Case | Handling |
|---|---|
| `GET /admin/models` | infallible; `is_downloaded` FS error → `false` (NeedsDownload) |
| total RAM 0 (sysinfo fail) | budget 0 → smallest model recommended; UI still renders |
| DELETE in-use | 409 |
| DELETE not cached | `200 {"deleted":false}` (idempotent) |
| DELETE FS error | 500, clean message |
| bad body / bad token | 400 / 401 |

## Testing (headless)

- `catalog_view` unit tests (bulk): pinned `total_ram_mb` + fake `is_downloaded`
  + `active` → assert exact status (InUse/Downloaded/NeedsDownload), fit
  (Fits/Tight/WontFit at chosen sizes), exactly one global `recommended`
  (largest-that-fits; smallest on tiny RAM), and family/entry ordering.
- `download::delete_cached`: temp cached file → `true` + gone; second call → `false`.
- Integration (`tests/http.rs`, fixed injected RAM + token): `GET /admin/models`
  401 without token / 200 non-empty with a recommended entry; `DELETE` 401
  without token / 409 for the in-use model.
- `sysinfo` value not asserted (machine-dependent; injected in tests).

## Non-goals

- The window UI (sub-project 3).
- Remote/auto-updating catalog (embedded only).
- Multi-file (split) GGUF entries (curated single-file models for v1).
- Routing using model capability tier (separate enhancement; see the
  routing-consider-local-model-capability note — sub-project 2's catalog tier
  can feed it later).

## Risks

- Curated `size_mb` drift vs the real GGUF → est. RAM slightly off; low impact
  (fit verdict has a Tight band). Keep sizes roughly accurate.
- `sysinfo` portability — it's cross-platform; the graceful 0-RAM fallback covers
  any detection failure.
