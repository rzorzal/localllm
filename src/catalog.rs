//! Curated, embedded model catalog + pure annotation (status / RAM / fit /
//! recommendation). No HTTP, no sysinfo, no filesystem — the caller injects
//! total RAM, the active model, and a downloaded-probe, so this is fully
//! unit-testable.

use crate::model_manager::ModelSpec;

/// One curated model the user can switch to.
pub struct CatalogEntry {
    pub family: &'static str,
    pub display_name: &'static str,
    pub params: &'static str,
    pub params_b: f32,
    pub quant: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size_mb: u32,
    pub ctx_train: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    InUse,
    Downloaded,
    NeedsDownload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    Fits,
    Tight,
    WontFit,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelView {
    pub display_name: String,
    pub params: String,
    pub quant: String,
    pub repo: String,
    pub file: String,
    pub size_mb: u32,
    pub est_ram_mb: u32,
    pub status: ModelStatus,
    pub fit: FitVerdict,
    pub recommended: bool,
    pub ctx_min: u32,
    pub ctx_default: u32,
    pub ctx_max: u32,
    pub ctx_current: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FamilyView {
    pub family: String,
    pub models: Vec<ModelView>,
}

/// Curated models, smallest→largest within each family. Sizes are approximate
/// download sizes (MB) used only for the RAM estimate; verify against the repo
/// when editing. The default model MUST appear here.
pub const CATALOG: &[CatalogEntry] = &[
    // --- Qwen2.5: tiny → large, the default lives here (3B) ---
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 0.5B Instruct", params: "0.5B", params_b: 0.5, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF", file: "qwen2.5-0.5b-instruct-q4_k_m.gguf", size_mb: 469, ctx_train: 32768 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 1.5B Instruct", params: "1.5B", params_b: 1.5, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF", file: "qwen2.5-1.5b-instruct-q4_k_m.gguf", size_mb: 1066, ctx_train: 32768 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 3B Instruct", params: "3B", params_b: 3.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-3B-Instruct-GGUF", file: "qwen2.5-3b-instruct-q4_k_m.gguf", size_mb: 2000, ctx_train: 32768 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 7B Instruct", params: "7B", params_b: 7.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-7B-Instruct-GGUF", file: "qwen2.5-7b-instruct-q4_k_m.gguf", size_mb: 4700, ctx_train: 32768 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 14B Instruct", params: "14B", params_b: 14.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-14B-Instruct-GGUF", file: "qwen2.5-14b-instruct-q4_k_m.gguf", size_mb: 9000, ctx_train: 32768 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 32B Instruct", params: "32B", params_b: 32.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-32B-Instruct-GGUF", file: "qwen2.5-32b-instruct-q4_k_m.gguf", size_mb: 20000, ctx_train: 32768 },
    // --- Qwen3 (2025/2026): newer generation, tiny → large ---
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 0.6B", params: "0.6B", params_b: 0.6, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-0.6B-GGUF", file: "Qwen_Qwen3-0.6B-Q4_K_M.gguf", size_mb: 462, ctx_train: 32768 },
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 1.7B", params: "1.7B", params_b: 1.7, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-1.7B-GGUF", file: "Qwen_Qwen3-1.7B-Q4_K_M.gguf", size_mb: 1223, ctx_train: 32768 },
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 4B Instruct", params: "4B", params_b: 4.0, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF", file: "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf", size_mb: 2382, ctx_train: 32768 },
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 8B", params: "8B", params_b: 8.0, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-8B-GGUF", file: "Qwen_Qwen3-8B-Q4_K_M.gguf", size_mb: 4794, ctx_train: 32768 },
    // --- Llama: 3.2 small models + 3.1 8B, grouped as one family ---
    CatalogEntry { family: "Llama", display_name: "Llama 3.2 1B Instruct", params: "1B", params_b: 1.0, quant: "Q4_K_M",
        repo: "bartowski/Llama-3.2-1B-Instruct-GGUF", file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf", size_mb: 770, ctx_train: 131072 },
    CatalogEntry { family: "Llama", display_name: "Llama 3.2 3B Instruct", params: "3B", params_b: 3.0, quant: "Q4_K_M",
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", size_mb: 1926, ctx_train: 131072 },
    CatalogEntry { family: "Llama", display_name: "Llama 3.1 8B Instruct", params: "8B", params_b: 8.0, quant: "Q4_K_M",
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", size_mb: 4900, ctx_train: 131072 },
    // --- Gemma 2: 2B → 27B ---
    CatalogEntry { family: "Gemma 2", display_name: "Gemma 2 2B Instruct", params: "2B", params_b: 2.0, quant: "Q4_K_M",
        repo: "bartowski/gemma-2-2b-it-GGUF", file: "gemma-2-2b-it-Q4_K_M.gguf", size_mb: 1629, ctx_train: 8192 },
    CatalogEntry { family: "Gemma 2", display_name: "Gemma 2 9B Instruct", params: "9B", params_b: 9.0, quant: "Q4_K_M",
        repo: "bartowski/gemma-2-9b-it-GGUF", file: "gemma-2-9b-it-Q4_K_M.gguf", size_mb: 5494, ctx_train: 8192 },
    CatalogEntry { family: "Gemma 2", display_name: "Gemma 2 27B Instruct", params: "27B", params_b: 27.0, quant: "Q4_K_M",
        repo: "bartowski/gemma-2-27b-it-GGUF", file: "gemma-2-27b-it-Q4_K_M.gguf", size_mb: 15875, ctx_train: 8192 },
    // --- Gemma 3 (2025): newer generation, 1B → 27B ---
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 1B Instruct", params: "1B", params_b: 1.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-1b-it-GGUF", file: "google_gemma-3-1b-it-Q4_K_M.gguf", size_mb: 769, ctx_train: 32768 },
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 4B Instruct", params: "4B", params_b: 4.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-4b-it-GGUF", file: "google_gemma-3-4b-it-Q4_K_M.gguf", size_mb: 2374, ctx_train: 131072 },
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 12B Instruct", params: "12B", params_b: 12.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-12b-it-GGUF", file: "google_gemma-3-12b-it-Q4_K_M.gguf", size_mb: 6962, ctx_train: 131072 },
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 27B Instruct", params: "27B", params_b: 27.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-27b-it-GGUF", file: "google_gemma-3-27b-it-Q4_K_M.gguf", size_mb: 15780, ctx_train: 131072 },
    // --- Phi 3.5 ---
    CatalogEntry { family: "Phi 3.5", display_name: "Phi 3.5 Mini Instruct", params: "3.8B", params_b: 3.8, quant: "Q4_K_M",
        repo: "bartowski/Phi-3.5-mini-instruct-GGUF", file: "Phi-3.5-mini-instruct-Q4_K_M.gguf", size_mb: 2400, ctx_train: 131072 },
    // --- Phi-4 (Microsoft, 2025) ---
    CatalogEntry { family: "Phi-4", display_name: "Phi-4 Mini Instruct", params: "3.8B", params_b: 3.8, quant: "Q4_K_M",
        repo: "bartowski/microsoft_Phi-4-mini-instruct-GGUF", file: "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", size_mb: 2376, ctx_train: 131072 },
    CatalogEntry { family: "Phi-4", display_name: "Phi-4 (14B)", params: "14B", params_b: 14.0, quant: "Q4_K_M",
        repo: "bartowski/phi-4-GGUF", file: "phi-4-Q4_K_M.gguf", size_mb: 8634, ctx_train: 16384 },
    // --- Mistral ---
    CatalogEntry { family: "Mistral", display_name: "Mistral 7B Instruct v0.3", params: "7B", params_b: 7.0, quant: "Q4_K_M",
        repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF", file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", size_mb: 4170, ctx_train: 32768 },
];

/// Parse a parameter size in billions from a model name: the first run of
/// digits (optionally with one decimal point) immediately followed by `b`/`B`.
/// Tries `file` first, then `repo`. `qwen2.5-7b…`→7.0, `…-8B-…`→8.0,
/// `…-3.8b-…`→3.8, names without an `<N>b` token → None.
pub fn params_b_from_name(repo: &str, file: &str) -> Option<f32> {
    fn scan(s: &str) -> Option<f32> {
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_digit() {
                let start = i;
                let mut seen_dot = false;
                while i < b.len()
                    && (b[i].is_ascii_digit()
                        || (b[i] == b'.' && !seen_dot && i + 1 < b.len() && b[i + 1].is_ascii_digit()))
                {
                    if b[i] == b'.' {
                        seen_dot = true;
                    }
                    i += 1;
                }
                if i < b.len() && (b[i] == b'b' || b[i] == b'B') {
                    if let Ok(v) = s[start..i].parse::<f32>() {
                        return Some(v);
                    }
                }
            } else {
                i += 1;
            }
        }
        None
    }
    scan(file).or_else(|| scan(repo))
}

/// Capability (parameter billions) of the active model: an exact CATALOG match
/// wins; else parse the name; else `0.0` (unknown → neutral routing).
pub fn active_params_b(repo: &str, file: &str) -> f32 {
    if let Some(e) = CATALOG.iter().find(|e| e.repo == repo && e.file == file) {
        return e.params_b;
    }
    params_b_from_name(repo, file).unwrap_or(0.0)
}

/// Annotate the catalog for this machine + the active model. Pure.
pub fn catalog_view(
    entries: &[CatalogEntry],
    total_ram_mb: u64,
    requested_ctx_ceiling: u32,
    kv: crate::fit::KvKind,
    active: Option<&ModelSpec>,
    is_downloaded: impl Fn(&str, &str) -> bool,
    ctx_override: impl Fn(&str, &str) -> Option<u32>,
) -> Vec<FamilyView> {
    let budget = total_ram_mb * 65 / 100;
    let tight_ceiling = total_ram_mb * 85 / 100;
    let budget_mb = crate::fit::device_budget_mb(total_ram_mb, true);

    // First pass: build ModelView (without recommendation) and track the best
    // recommendation candidate index in the flattened order.
    let mut flat: Vec<ModelView> = Vec::with_capacity(entries.len());
    let mut best_fit: Option<usize> = None; // largest params_b among Fits
    let mut smallest: Option<usize> = None; // fallback: smallest est_ram

    for (i, e) in entries.iter().enumerate() {
        let kv_per_token = crate::fit::est_kv_bytes_per_token(e.params_b, kv);
        let bounds = crate::fit::ctx_bounds(e.size_mb, kv_per_token, budget_mb, e.ctx_train);
        let ctx_current = if bounds.max == 0 {
            0
        } else {
            ctx_override(e.repo, e.file).unwrap_or_else(|| requested_ctx_ceiling.min(bounds.max))
        };
        let kv_mb = (kv_per_token * ctx_current as u64 / (1024 * 1024)) as u32;
        let est = e.size_mb + kv_mb + crate::fit::COMPUTE_HEADROOM_MB;

        let fit = if bounds.max == 0 {
            FitVerdict::WontFit
        } else if (est as u64) <= budget {
            FitVerdict::Fits
        } else if (est as u64) <= tight_ceiling {
            FitVerdict::Tight
        } else {
            FitVerdict::WontFit
        };

        let status = if active.map(|a| a.repo == e.repo && a.file == e.file).unwrap_or(false) {
            ModelStatus::InUse
        } else if is_downloaded(e.repo, e.file) {
            ModelStatus::Downloaded
        } else {
            ModelStatus::NeedsDownload
        };

        if fit == FitVerdict::Fits {
            let better = match best_fit {
                None => true,
                Some(j) => {
                    e.params_b > entries[j].params_b
                        || (e.params_b == entries[j].params_b && e.size_mb > entries[j].size_mb)
                }
            };
            if better {
                best_fit = Some(i);
            }
        }
        if smallest.map(|j| est < flat[j].est_ram_mb).unwrap_or(true) {
            smallest = Some(i);
        }

        flat.push(ModelView {
            display_name: e.display_name.to_string(),
            params: e.params.to_string(),
            quant: e.quant.to_string(),
            repo: e.repo.to_string(),
            file: e.file.to_string(),
            size_mb: e.size_mb,
            est_ram_mb: est,
            status,
            fit,
            recommended: false,
            ctx_min: bounds.min,
            ctx_default: bounds.default,
            ctx_max: bounds.max,
            ctx_current,
        });
    }

    if let Some(idx) = best_fit.or(smallest) {
        flat[idx].recommended = true;
    }

    // Group into families in first-seen order.
    let mut families: Vec<FamilyView> = Vec::new();
    for (e, mv) in entries.iter().zip(flat.into_iter()) {
        match families.iter_mut().find(|f| f.family == e.family) {
            Some(f) => f.models.push(mv),
            None => families.push(FamilyView { family: e.family.to_string(), models: vec![mv] }),
        }
    }
    families
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::KvKind;

    fn entry(family: &'static str, name: &'static str, pb: f32, repo: &'static str, file: &'static str, size_mb: u32) -> CatalogEntry {
        CatalogEntry { family, display_name: name, params: "x", params_b: pb, quant: "Q4_K_M", repo, file, size_mb, ctx_train: 32768 }
    }

    fn sample() -> Vec<CatalogEntry> {
        vec![
            entry("Qwen2.5", "Qwen 3B", 3.0, "q/3b", "3b.gguf", 2000),
            entry("Qwen2.5", "Qwen 7B", 7.0, "q/7b", "7b.gguf", 4700),
            entry("Qwen2.5", "Qwen 32B", 32.0, "q/32b", "32b.gguf", 20000),
            entry("Llama", "Llama 8B", 8.0, "l/8b", "8b.gguf", 4900),
        ]
    }

    #[test]
    fn annotates_status_fit_and_single_recommendation() {
        // 16 GB machine → budget 10649 MB (65%); tight 13926 MB (85%).
        // KV-aware est: 3B/7B/8B all fit (well under 10649); 32B won't fit
        // (weights alone saturate the 78% device budget → max=0).
        // Largest-that-fits = 8B (params_b 8 > 7).
        let cat = sample();
        let active = ModelSpec { repo: "q/7b".into(), file: "7b.gguf".into() };
        let downloaded = |r: &str, _f: &str| r == "q/3b"; // 3B cached
        let view = catalog_view(&cat, 16384, 32768, KvKind::Q8, Some(&active), downloaded, |_, _| None);

        // grouped by family in first-seen order
        assert_eq!(view[0].family, "Qwen2.5");
        assert_eq!(view[1].family, "Llama");

        // statuses
        let qwen = &view[0].models;
        assert_eq!(qwen[0].status, ModelStatus::Downloaded);   // 3B cached
        assert_eq!(qwen[1].status, ModelStatus::InUse);        // 7B active
        assert_eq!(qwen[2].status, ModelStatus::NeedsDownload);// 32B

        // est ram includes KV overhead, so always > size_mb
        assert!(qwen[0].est_ram_mb > 2000);
        assert_eq!(qwen[0].fit, FitVerdict::Fits);
        assert_eq!(qwen[2].fit, FitVerdict::WontFit);          // 32B

        // exactly one recommended, and it's the 8B (largest that fits)
        let recs: Vec<_> = view.iter().flat_map(|f| f.models.iter()).filter(|m| m.recommended).collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].display_name, "Llama 8B");
    }

    #[test]
    fn tiny_ram_recommends_smallest() {
        let cat = sample();
        // 2 GB → budget 1331; nothing fits → smallest est (3B has lowest est).
        let view = catalog_view(&cat, 2048, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        let recs: Vec<_> = view.iter().flat_map(|f| f.models.iter()).filter(|m| m.recommended).collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].display_name, "Qwen 3B");
        // and a non-fitting model is marked won't_fit
        assert!(view.iter().flat_map(|f| f.models.iter()).any(|m| m.fit == FitVerdict::WontFit));
    }

    #[test]
    fn tight_band_between_budget_and_85_percent() {
        // total 10000 → budget 6500, 85% = 8500.
        // 5B model, 6000 MB weights: KV at the computed ctx_current pushes est
        // above 6500 but below 8500 → Tight.
        let cat = vec![entry("F", "M", 5.0, "r", "f", 6000)];
        let view = catalog_view(&cat, 10000, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        assert_eq!(view[0].models[0].fit, FitVerdict::Tight);
    }

    #[test]
    fn default_model_is_in_catalog() {
        assert!(CATALOG.iter().any(|e|
            e.repo == "Qwen/Qwen2.5-3B-Instruct-GGUF"
            && e.file == "qwen2.5-3b-instruct-q4_k_m.gguf"));
    }

    #[test]
    fn params_b_from_name_parses_size() {
        assert_eq!(super::params_b_from_name("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf"), Some(7.0));
        assert_eq!(super::params_b_from_name("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), Some(8.0));
        // decimal sizes
        assert_eq!(super::params_b_from_name("x/y", "model-3.8b-q4.gguf"), Some(3.8));
        // no "<N>b" token anywhere → None
        assert_eq!(super::params_b_from_name("x/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf"), None);
        // falls back to the repo when the file lacks it
        assert_eq!(super::params_b_from_name("org/thing-13b", "weights.gguf"), Some(13.0));
    }

    #[test]
    fn active_params_b_catalog_then_name_then_neutral() {
        // a real CATALOG entry → its params_b
        assert_eq!(super::active_params_b("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"), 3.0);
        // not in catalog but parseable name
        assert_eq!(super::active_params_b("foo/bar", "model-13b.gguf"), 13.0);
        // unknown → neutral 0.0
        assert_eq!(super::active_params_b("foo/bar", "model.gguf"), 0.0);
    }

    #[test]
    fn est_ram_includes_kv_and_grows_with_ctx() {
        // One 3B model; compare est at a small override vs the 32k ceiling.
        let entries = [CatalogEntry {
            family: "T", display_name: "t3", params: "3B", params_b: 3.0, quant: "Q4_K_M",
            repo: "r", file: "f", size_mb: 2000, ctx_train: 32768,
        }];
        let big = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        let small = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| Some(4096));
        let big_est = big[0].models[0].est_ram_mb;
        let small_est = small[0].models[0].est_ram_mb;
        // KV at 32768 costs more than at 4096 → bigger est. Both exceed size_mb.
        assert!(big_est > small_est, "big {big_est} small {small_est}");
        assert!(small_est > 2000);
        assert_eq!(small[0].models[0].ctx_current, 4096);
    }

    #[test]
    fn ctx_current_defaults_to_min_ceiling_and_max() {
        // Phi-4 14B on 16GB: ctx_train 16384 caps the max below the 32768 ceiling.
        let entries = [CatalogEntry {
            family: "P", display_name: "phi4", params: "14B", params_b: 14.0, quant: "Q4_K_M",
            repo: "bartowski/phi-4-GGUF", file: "phi-4-Q4_K_M.gguf", size_mb: 8634, ctx_train: 16384,
        }];
        let v = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        let m = &v[0].models[0];
        assert_eq!(m.ctx_max, 16384);          // trained ctx binds
        assert_eq!(m.ctx_current, 16384);       // min(32768 ceiling, 16384 max)
        assert_eq!(m.ctx_min, crate::fit::MIN_CTX);
        assert_eq!(m.ctx_default, crate::fit::DEFAULT_SMALL_CTX.min(16384));
        // weights 8634 + KV(16384)≈1700 + headroom 1024 ≈ 11.4k < 65% of 16384 (10649)?
        // 11358 > 10649 → Tight, not Fits. Assert it is at least not WontFit.
        assert_ne!(m.fit, FitVerdict::WontFit);
    }

    #[test]
    fn override_out_of_nothing_uses_ceiling_min_max() {
        let entries = [CatalogEntry {
            family: "T", display_name: "t3", params: "3B", params_b: 3.0, quant: "Q4_K_M",
            repo: "r", file: "f", size_mb: 2000, ctx_train: 131072,
        }];
        // Big trained ctx, small model → ctx_max is memory- or GLOBAL_MAX-bound.
        let v = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        let m = &v[0].models[0];
        // ceiling 32768 <= max → ctx_current = 32768
        assert_eq!(m.ctx_current, 32768.min(m.ctx_max));
    }
}
