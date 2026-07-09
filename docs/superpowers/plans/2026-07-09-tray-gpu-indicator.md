# Tray GPU Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a menu-bar tray line showing whether the active local model runs on GPU (Metal) or CPU, and how many layers, updating on model hot-swap.

**Architecture:** A pure `gpu_menu_label(Option<u32>) -> String` maps the resolved per-model `gpu_layers` to an at-a-glance label. A thin `active_gpu_layers(repo, file)` resolves that value the same way the engine loader does (`profile::resolve(saved_profile, catalog).gpu_layers`). A new informational `MenuItem` after `Backend:` shows it, refreshed inside the existing status poll's model-changed branch.

**Tech Stack:** Rust, `tray_icon` menu items, existing `settings`/`profile`/`catalog` modules.

## Global Constraints

- Config-derived only: label reflects `profile::resolve(&saved, catalog, _, _).gpu_layers` (= `saved.gpu_layers.or(catalog.rec_gpu_layers)`); no runtime offload measurement.
- Label mapping is EXACT: `None → "🟢 GPU: all layers"`, `Some(0) → "⚪ CPU only"`, `Some(n>0) → "🟢 GPU: {n} layers"`.
- Tray only (no dashboard/Config surface). Do not change `gpu_layers` resolution or the engine.
- Live-update on hot-swap: refresh in the same poll branch that updates the `Model:` line (`if last_model.as_ref() != Some(&st.current)`), guarded by a `last_gpu` dedupe.
- macOS-only assumption is acceptable (Metal always compiled; OOM errors at load, no silent CPU fallback).

---

### Task 1: `gpu_menu_label` + `active_gpu_layers` helpers

**Files:**
- Modify: `src/tray.rs` (add two module-scope fns near the other label helpers like `model_menu_label` ~line 331; add a test to the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Produces: `fn gpu_menu_label(gpu_layers: Option<u32>) -> String` and `fn active_gpu_layers(repo: &str, file: &str) -> Option<u32>`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `src/tray.rs`:

```rust
    #[test]
    fn gpu_menu_label_maps_layer_counts() {
        assert_eq!(super::gpu_menu_label(None), "🟢 GPU: all layers");
        assert_eq!(super::gpu_menu_label(Some(0)), "⚪ CPU only");
        assert_eq!(super::gpu_menu_label(Some(24)), "🟢 GPU: 24 layers");
    }
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cargo test -p localllm gpu_menu_label`
Expected: FAIL — `cannot find function gpu_menu_label`.

- [ ] **Step 3: Implement both helpers**

Add near the other tray label helpers (e.g. right after `model_menu_label` at ~line 331) in `src/tray.rs`:

```rust
/// At-a-glance GPU/CPU label for the tray info block. `None` means the engine
/// offloads all layers (`with_n_gpu_layers(u32::MAX)`); `Some(0)` is CPU-only.
fn gpu_menu_label(gpu_layers: Option<u32>) -> String {
    match gpu_layers {
        None => "🟢 GPU: all layers".to_string(),
        Some(0) => "⚪ CPU only".to_string(),
        Some(n) => format!("🟢 GPU: {n} layers"),
    }
}

/// Config-derived GPU-layer count the active model's engine was built with: the
/// saved per-model exec profile's `gpu_layers`, falling back to the catalog
/// recommendation (`None` → full offload). `gpu_layers` is independent of the
/// ctx/kv globals, so inert placeholders are passed for those `resolve` args.
fn active_gpu_layers(repo: &str, file: &str) -> Option<u32> {
    let key = crate::settings::model_ctx_key(repo, file);
    let saved = crate::settings::load_model_profile(&key);
    let catalog = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == repo && e.file == file);
    crate::profile::resolve(&saved, catalog, 0, crate::config::KvType::Q8).gpu_layers
}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `cargo test -p localllm gpu_menu_label`
Expected: PASS.

- [ ] **Step 5: Build (confirms `active_gpu_layers` compiles against the real signatures)**

Run: `cargo build -p localllm`
Expected: clean build. (`active_gpu_layers` is not yet called — a `dead_code` warning on it is acceptable here; Task 2 wires it. Do NOT add `#[allow(dead_code)]`.)

- [ ] **Step 6: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): gpu_menu_label + active_gpu_layers helpers"
```

---

### Task 2: Wire the GPU line into the tray menu + live update + build

**Files:**
- Modify: `src/tray.rs` (menu build after `backend_line`; handle + dedupe declarations near `model_handle`/`last_model`; poll update inside the model-changed branch)

**Interfaces:**
- Consumes: `gpu_menu_label`, `active_gpu_layers` (Task 1). `st.current` is a `ModelSpec { repo: String, file: String }`.

- [ ] **Step 1: Declare the handle + dedupe state**

In `src/tray.rs`, right after the `model_handle` / `last_model` declarations (~lines 523–524, the block commented "Kept so we can live-update the \"Model:\" line after a hot-swap"), add:

```rust
    // Kept so we can live-update the "GPU:" line after a hot-swap; last_gpu
    // avoids redundant set_text on every poll tick.
    let mut gpu_handle: Option<MenuItem> = None;
    let mut last_gpu: Option<String> = None;
```

- [ ] **Step 2: Add the GPU line to the menu build**

In the `StartCause::Init` menu build, immediately after the `backend_line` definition (~line 578), add:

```rust
                // GPU/CPU offload indicator. Initial text is a neutral
                // placeholder; the first status-poll tick fills the real
                // label (the model-changed branch runs on tick 1 because
                // last_gpu/last_model start None).
                let gpu_line = MenuItem::new("GPU: …", false, None);
                gpu_handle = Some(gpu_line.clone());
```

- [ ] **Step 3: Insert the line into the menu, after `backend_line`**

The menu is built with sequential `menu.append(...)` calls (~lines 607–626). Find the backend append:

```rust
                menu.append(&backend_line).expect("append backend");
```

and add the GPU line directly after it:

```rust
                menu.append(&backend_line).expect("append backend");
                menu.append(&gpu_line).expect("append gpu");
```

Leave every other `append` call and its order unchanged (the next line is the `separator()` before `routing_line`).

- [ ] **Step 4: Refresh the GPU line in the poll's model-changed branch**

In the status poll, extend the existing model-changed branch. It currently reads (~lines 661–666):

```rust
                        if last_model.as_ref() != Some(&st.current) {
                            if let Some(m) = &model_handle {
                                m.set_text(model_menu_label(&st.current));
                            }
                            last_model = Some(st.current);
                        }
```

Replace it with (compute the GPU label from the current model BEFORE `st.current` is moved into `last_model`):

```rust
                        if last_model.as_ref() != Some(&st.current) {
                            if let Some(m) = &model_handle {
                                m.set_text(model_menu_label(&st.current));
                            }
                            let gpu = gpu_menu_label(active_gpu_layers(
                                &st.current.repo,
                                &st.current.file,
                            ));
                            if last_gpu.as_deref() != Some(gpu.as_str()) {
                                if let Some(g) = &gpu_handle {
                                    g.set_text(&gpu);
                                }
                                last_gpu = Some(gpu);
                            }
                            last_model = Some(st.current);
                        }
```

- [ ] **Step 5: Build + full test suite**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean build (the earlier `active_gpu_layers` dead_code warning is gone now that it is called); all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): live GPU/CPU indicator line after Backend"
```

- [ ] **Step 7: Rebuild the app bundle**

Run: `bash scripts/build-app.sh --fast`
Expected: `==> SUCCESS`.

- [ ] **Step 8: Commit any tweaks**

```bash
git add -A && git commit -m "chore: tray GPU indicator verification tweaks" --allow-empty
```

---

### Task 3: Manual verification (controller / user)

**Files:** none (manual).

- [ ] **Step 1: Restart the app bundle** (`target/localllm.app`).

- [ ] **Step 2: Open the tray menu.** Expected: an info line `🟢 GPU: all layers` directly under `Backend:` (default profile → full offload), replacing the brief `GPU: …` placeholder within one poll tick.

- [ ] **Step 3: Force a CPU/partial profile.** In Config → the active model's exec profile, set `gpu_layers = 0`, then hot-swap to that model (or set `gpu_layers = 20`). Expected: the tray line flips to `⚪ CPU only` (or `🟢 GPU: 20 layers`) WITHOUT restarting the app.

---

## Self-Review

**Spec coverage:**
- Pure `gpu_menu_label` with exact mapping → Task 1 Step 3 + test. ✓
- Config-derived `active_gpu_layers` via `profile::resolve(...).gpu_layers` → Task 1 Step 3. ✓
- New tray line after `Backend:` → Task 2 Steps 2–3. ✓
- Live-update on hot-swap in the model-changed branch with `last_gpu` dedupe → Task 2 Steps 1, 4. ✓
- Tray only; no engine/resolution change → nothing else touched. ✓
- Unit tests (3 cases) + manual hot-swap check → Task 1 test + Task 3. ✓
- Rebuild bundle → Task 2 Step 7 + Task 3. ✓

**Placeholder scan:** all code steps carry concrete code; the `GPU: …` string is an intentional initial UI placeholder (documented), not a plan placeholder; Task 3 is a real reproduction. No TBD/TODO.

**Type consistency:** `gpu_menu_label(Option<u32>) -> String` and `active_gpu_layers(&str, &str) -> Option<u32>` defined in Task 1, called in Task 2 Step 4. `profile::resolve(&ExecProfile, Option<&CatalogEntry>, u32, KvType) -> Resolved` and `settings::model_ctx_key(&str,&str) -> String` match the real signatures. `st.current.repo`/`.file` are `String` fields of `ModelSpec`. `MenuItem`, `last_gpu` dedupe mirror the existing `model_handle`/`last_model` pattern.
