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
    /// Recommended per-model overrides. `None` = use the global default.
    pub rec_kv: Option<crate::config::KvType>,
    pub rec_gpu_layers: Option<u32>,
    pub rec_history_turns: Option<u32>,
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
pub struct VariantView {
    pub quant: String,
    pub size_mb: u32,
    pub est_ram_mb: u32,
    pub fit: FitVerdict,
    pub status: ModelStatus,
    pub selected: bool,
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
    pub kv_current: String,
    pub kv_default: String,
    pub gpu_layers_current: Option<u32>,
    pub history_turns_current: Option<u32>,
    pub history_turns_default: Option<u32>,
    pub variants: Vec<VariantView>,
    pub quant_selected: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FamilyView {
    pub family: String,
    pub models: Vec<ModelView>,
}

/// A quant variant as stored in the generated table (`catalog_variants.rs`).
#[derive(Debug, Clone, Copy)]
pub struct QuantVariant {
    pub quant: &'static str,
    pub files: &'static [&'static str], // >1 = split model
    pub size_mb: u32,
}

/// An owned quant variant returned to callers. Owned so the fallback can be
/// synthesized from a `CatalogEntry` at runtime (which cannot produce a
/// `&'static` slice).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ResolvedVariant {
    pub quant: String,
    pub files: Vec<String>,
    pub size_mb: u32,
}

/// All quant variants available for a model: the generated table entry for its
/// repo, or a single-element fallback synthesized from the model's default file.
pub fn variants_for(entry: &CatalogEntry) -> Vec<ResolvedVariant> {
    if let Some((_, vs)) = crate::catalog_variants::QUANT_VARIANTS
        .iter()
        .find(|(repo, _)| *repo == entry.repo)
    {
        return vs
            .iter()
            .map(|v| ResolvedVariant {
                quant: v.quant.to_string(),
                files: v.files.iter().map(|f| f.to_string()).collect(),
                size_mb: v.size_mb,
            })
            .collect();
    }
    vec![ResolvedVariant {
        quant: entry.quant.to_string(),
        files: vec![entry.file.to_string()],
        size_mb: entry.size_mb,
    }]
}

/// The file list for a specific quant of a model, or `None` if not available.
pub fn files_for_quant(entry: &CatalogEntry, quant: &str) -> Option<Vec<String>> {
    variants_for(entry)
        .into_iter()
        .find(|v| v.quant.eq_ignore_ascii_case(quant))
        .map(|v| v.files)
}

/// Curated models, smallest→largest within each family. Sizes are approximate
/// download sizes (MB) used only for the RAM estimate; verify against the repo
/// when editing. The default model MUST appear here.
pub const CATALOG: &[CatalogEntry] = &[
    // --- Qwen2.5: tiny → large, the default lives here (3B) ---
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 0.5B Instruct", params: "0.5B", params_b: 0.5, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF", file: "qwen2.5-0.5b-instruct-q4_k_m.gguf", size_mb: 469, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 1.5B Instruct", params: "1.5B", params_b: 1.5, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF", file: "qwen2.5-1.5b-instruct-q4_k_m.gguf", size_mb: 1066, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 3B Instruct", params: "3B", params_b: 3.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-3B-Instruct-GGUF", file: "qwen2.5-3b-instruct-q4_k_m.gguf", size_mb: 2000, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 7B Instruct", params: "7B", params_b: 7.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-7B-Instruct-GGUF", file: "qwen2.5-7b-instruct-q4_k_m.gguf", size_mb: 4700, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 14B Instruct", params: "14B", params_b: 14.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-14B-Instruct-GGUF", file: "qwen2.5-14b-instruct-q4_k_m.gguf", size_mb: 9000, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 32B Instruct", params: "32B", params_b: 32.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-32B-Instruct-GGUF", file: "qwen2.5-32b-instruct-q4_k_m.gguf", size_mb: 20000, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    // --- Qwen3 (2025/2026): newer generation, tiny → large ---
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 0.6B", params: "0.6B", params_b: 0.6, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-0.6B-GGUF", file: "Qwen_Qwen3-0.6B-Q4_K_M.gguf", size_mb: 462, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 1.7B", params: "1.7B", params_b: 1.7, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-1.7B-GGUF", file: "Qwen_Qwen3-1.7B-Q4_K_M.gguf", size_mb: 1223, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 4B Instruct", params: "4B", params_b: 4.0, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF", file: "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf", size_mb: 2382, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Qwen3", display_name: "Qwen3 8B", params: "8B", params_b: 8.0, quant: "Q4_K_M",
        repo: "bartowski/Qwen_Qwen3-8B-GGUF", file: "Qwen_Qwen3-8B-Q4_K_M.gguf", size_mb: 4794, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    // --- Llama: 3.2 small models + 3.1 8B, grouped as one family ---
    CatalogEntry { family: "Llama", display_name: "Llama 3.2 1B Instruct", params: "1B", params_b: 1.0, quant: "Q4_K_M",
        repo: "bartowski/Llama-3.2-1B-Instruct-GGUF", file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf", size_mb: 770, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Llama", display_name: "Llama 3.2 3B Instruct", params: "3B", params_b: 3.0, quant: "Q4_K_M",
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", size_mb: 1926, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Llama", display_name: "Llama 3.1 8B Instruct", params: "8B", params_b: 8.0, quant: "Q4_K_M",
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", size_mb: 4900, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    // --- Gemma 2: 2B → 27B ---
    CatalogEntry { family: "Gemma 2", display_name: "Gemma 2 2B Instruct", params: "2B", params_b: 2.0, quant: "Q4_K_M",
        repo: "bartowski/gemma-2-2b-it-GGUF", file: "gemma-2-2b-it-Q4_K_M.gguf", size_mb: 1629, ctx_train: 8192,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Gemma 2", display_name: "Gemma 2 9B Instruct", params: "9B", params_b: 9.0, quant: "Q4_K_M",
        repo: "bartowski/gemma-2-9b-it-GGUF", file: "gemma-2-9b-it-Q4_K_M.gguf", size_mb: 5494, ctx_train: 8192,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Gemma 2", display_name: "Gemma 2 27B Instruct", params: "27B", params_b: 27.0, quant: "Q4_K_M",
        repo: "bartowski/gemma-2-27b-it-GGUF", file: "gemma-2-27b-it-Q4_K_M.gguf", size_mb: 15875, ctx_train: 8192,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    // --- Gemma 3 (2025): newer generation, 1B → 27B ---
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 1B Instruct", params: "1B", params_b: 1.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-1b-it-GGUF", file: "google_gemma-3-1b-it-Q4_K_M.gguf", size_mb: 769, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 4B Instruct", params: "4B", params_b: 4.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-4b-it-GGUF", file: "google_gemma-3-4b-it-Q4_K_M.gguf", size_mb: 2374, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 12B Instruct", params: "12B", params_b: 12.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-12b-it-GGUF", file: "google_gemma-3-12b-it-Q4_K_M.gguf", size_mb: 6962, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Gemma 3", display_name: "Gemma 3 27B Instruct", params: "27B", params_b: 27.0, quant: "Q4_K_M",
        repo: "bartowski/google_gemma-3-27b-it-GGUF", file: "google_gemma-3-27b-it-Q4_K_M.gguf", size_mb: 15780, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    // --- Phi 3.5 ---
    CatalogEntry { family: "Phi 3.5", display_name: "Phi 3.5 Mini Instruct", params: "3.8B", params_b: 3.8, quant: "Q4_K_M",
        repo: "bartowski/Phi-3.5-mini-instruct-GGUF", file: "Phi-3.5-mini-instruct-Q4_K_M.gguf", size_mb: 2400, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    // --- Phi-4 (Microsoft, 2025) ---
    CatalogEntry { family: "Phi-4", display_name: "Phi-4 Mini Instruct", params: "3.8B", params_b: 3.8, quant: "Q4_K_M",
        repo: "bartowski/microsoft_Phi-4-mini-instruct-GGUF", file: "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", size_mb: 2376, ctx_train: 131072,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    CatalogEntry { family: "Phi-4", display_name: "Phi-4 (14B)", params: "14B", params_b: 14.0, quant: "Q4_K_M",
        repo: "bartowski/phi-4-GGUF", file: "phi-4-Q4_K_M.gguf", size_mb: 8634, ctx_train: 16384,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
    // --- Mistral ---
    CatalogEntry { family: "Mistral", display_name: "Mistral 7B Instruct v0.3", params: "7B", params_b: 7.0, quant: "Q4_K_M",
        repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF", file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", size_mb: 4170, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
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

/// Nominal parameter size (billions) for a compact model key
/// (`model_ctx_key(repo,file)` = `"{repo}/{file}"`). The key is not reliably
/// splittable back to (repo,file) because `repo` itself contains `/`, so this
/// resolves without splitting: a CATALOG entry whose `"{repo}/{file}"` equals
/// the key wins; else scan the whole key for a `NNb` size token; else 0.0.
pub fn params_b_for_key(key: &str) -> f32 {
    if let Some(e) = CATALOG
        .iter()
        .find(|e| format!("{}/{}", e.repo, e.file) == key)
    {
        return e.params_b;
    }
    // Reuse the same digit+"b" scan the name parser uses by passing the whole
    // key as the "repo" arg and an empty file.
    params_b_from_name(key, "").unwrap_or(0.0)
}

/// Map a `KvType` to the fit-math `KvKind` and a lowercase tag.
fn kv_to_kind(t: crate::config::KvType) -> (crate::fit::KvKind, &'static str) {
    match t {
        crate::config::KvType::Q8 => (crate::fit::KvKind::Q8, "q8"),
        crate::config::KvType::Q4 => (crate::fit::KvKind::Q4, "q4"),
        crate::config::KvType::F16 => (crate::fit::KvKind::F16, "f16"),
    }
}

/// Lowercase tag for the global fallback `KvKind`.
fn kv_kind_tag(k: crate::fit::KvKind) -> &'static str {
    match k {
        crate::fit::KvKind::Q8 => "q8",
        crate::fit::KvKind::Q4 => "q4",
        crate::fit::KvKind::F16 => "f16",
    }
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
    profile_override: impl Fn(&str, &str) -> crate::settings::ExecProfile,
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
        let prof = profile_override(e.repo, e.file);
        let (eff_kv_kind, kv_tag) = match prof.kv_type.or(e.rec_kv.clone()) {
            Some(t) => kv_to_kind(t),
            None => (kv, kv_kind_tag(kv)),
        };
        let kv_default_tag = match e.rec_kv.clone() {
            Some(t) => kv_to_kind(t).1,
            None => kv_kind_tag(kv),
        };
        // Resolve the selected quant and the size it implies BEFORE row-level
        // bounds so that size_mb/est_ram_mb/fit on the row match the selection.
        let quant_selected = prof.quant.clone().unwrap_or_else(|| e.quant.to_string());
        let raw_variants = crate::catalog::variants_for(e);
        let row_size_mb = raw_variants
            .iter()
            .find(|v| v.quant.eq_ignore_ascii_case(&quant_selected))
            .map(|v| v.size_mb)
            .unwrap_or(e.size_mb);

        // kv_per_token is constant per entry — hoist out of the variants closure.
        let kv_per_token = crate::fit::est_kv_bytes_per_token(e.params_b, eff_kv_kind);
        let bounds = crate::fit::ctx_bounds(row_size_mb, kv_per_token, budget_mb, e.ctx_train);
        let ctx_current = if bounds.max == 0 {
            0
        } else {
            ctx_override(e.repo, e.file).unwrap_or_else(|| requested_ctx_ceiling.min(bounds.max))
        };
        let kv_mb = (kv_per_token * ctx_current as u64 / (1024 * 1024)) as u32;
        let est = row_size_mb + kv_mb + crate::fit::COMPUTE_HEADROOM_MB;

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

        let variants: Vec<VariantView> = raw_variants
            .into_iter()
            .map(|rv| {
                // kv_per_token hoisted above (constant per entry, same params + eff_kv_kind).
                let vbounds = crate::fit::ctx_bounds(rv.size_mb, kv_per_token, budget_mb, e.ctx_train);
                let vkv_mb = (kv_per_token * ctx_current as u64 / (1024 * 1024)) as u32;
                let vest = rv.size_mb + vkv_mb + crate::fit::COMPUTE_HEADROOM_MB;
                let vfit = if vbounds.max == 0 {
                    FitVerdict::WontFit
                } else if (vest as u64) <= budget {
                    FitVerdict::Fits
                } else if (vest as u64) <= tight_ceiling {
                    FitVerdict::Tight
                } else {
                    FitVerdict::WontFit
                };
                let vstatus = if rv.files.iter().all(|f| is_downloaded(e.repo, f)) {
                    ModelStatus::Downloaded
                } else {
                    ModelStatus::NeedsDownload
                };
                let selected = rv.quant.eq_ignore_ascii_case(&quant_selected);
                VariantView { quant: rv.quant, size_mb: rv.size_mb, est_ram_mb: vest, fit: vfit, status: vstatus, selected }
            })
            .collect();

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
            size_mb: row_size_mb,
            est_ram_mb: est,
            status,
            fit,
            recommended: false,
            ctx_min: bounds.min,
            ctx_default: bounds.default,
            ctx_max: bounds.max,
            ctx_current,
            kv_current: kv_tag.to_string(),
            kv_default: kv_default_tag.to_string(),
            gpu_layers_current: prof.gpu_layers.or(e.rec_gpu_layers),
            history_turns_current: prof.history_turns.or(e.rec_history_turns),
            history_turns_default: e.rec_history_turns,
            variants,
            quant_selected,
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
        CatalogEntry { family, display_name: name, params: "x", params_b: pb, quant: "Q4_K_M",
            repo, file, size_mb, ctx_train: 32768,
            rec_kv: None, rec_gpu_layers: None, rec_history_turns: None }
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
        let active = ModelSpec { repo: "q/7b".into(), file: "7b.gguf".into(), quant: None };
        let downloaded = |r: &str, _f: &str| r == "q/3b"; // 3B cached
        let view = catalog_view(&cat, 16384, 32768, KvKind::Q8, Some(&active), downloaded, |_, _| None, |_, _| crate::settings::ExecProfile::default());

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
        let view = catalog_view(&cat, 2048, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, |_, _| crate::settings::ExecProfile::default());
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
        let view = catalog_view(&cat, 10000, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, |_, _| crate::settings::ExecProfile::default());
        let m = &view[0].models[0];
        assert_eq!(m.fit, FitVerdict::Tight);
        // Pin the estimate to the Tight band so a shift in the fit constants
        // can't silently reclassify this to Fits/WontFit while still "passing".
        assert!(m.est_ram_mb > 6500 && m.est_ram_mb <= 8500, "est {}", m.est_ram_mb);
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
            rec_kv: None, rec_gpu_layers: None, rec_history_turns: None,
        }];
        let big = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, |_, _| crate::settings::ExecProfile::default());
        let small = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| Some(4096), |_, _| crate::settings::ExecProfile::default());
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
            rec_kv: None, rec_gpu_layers: None, rec_history_turns: None,
        }];
        let v = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, |_, _| crate::settings::ExecProfile::default());
        let m = &v[0].models[0];
        assert_eq!(m.ctx_max, 16384);          // trained ctx binds
        assert_eq!(m.ctx_current, 16384);       // min(32768 ceiling, 16384 max)
        assert_eq!(m.ctx_min, crate::fit::MIN_CTX);
        assert_eq!(m.ctx_default, crate::fit::DEFAULT_SMALL_CTX.min(16384));
        // weights 8634 + KV(16384) + headroom 1024 lands above 65% of 16384
        // (10649) but under 85% (13926) → Tight. Pin both the verdict and the
        // estimate so a KV-heuristic shift can't silently reclassify it.
        assert_eq!(m.fit, FitVerdict::Tight);
        assert!(m.est_ram_mb > 10649, "est {}", m.est_ram_mb);
    }

    #[test]
    fn override_out_of_nothing_uses_ceiling_min_max() {
        let entries = [CatalogEntry {
            family: "T", display_name: "t3", params: "3B", params_b: 3.0, quant: "Q4_K_M",
            repo: "r", file: "f", size_mb: 2000, ctx_train: 131072,
            rec_kv: None, rec_gpu_layers: None, rec_history_turns: None,
        }];
        // Big trained ctx, small model → ctx_max is memory- or GLOBAL_MAX-bound.
        let v = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, |_, _| crate::settings::ExecProfile::default());
        let m = &v[0].models[0];
        // ceiling 32768 <= max → ctx_current = 32768
        assert_eq!(m.ctx_current, 32768.min(m.ctx_max));
    }

    #[test]
    fn catalog_view_uses_per_model_kv_from_profile() {
        use crate::settings::ExecProfile;
        let cat = vec![entry("Qwen2.5", "Qwen 7B", 7.0, "q/7b", "7b.gguf", 4700)];
        // Global kv = Q8, but this model's saved profile forces F16 → larger KV est.
        let f16_profile = |_r: &str, _f: &str| ExecProfile {
            kv_type: Some(crate::config::KvType::F16), ..Default::default()
        };
        let none = |_r: &str, _f: &str| ExecProfile::default();
        let with_f16 = catalog_view(&cat, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, f16_profile);
        let with_q8  = catalog_view(&cat, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, none);
        // F16 KV is heavier per token → smaller max ctx for the same budget.
        assert!(with_f16[0].models[0].ctx_max <= with_q8[0].models[0].ctx_max);
        assert_eq!(with_f16[0].models[0].kv_current, "f16");
        assert_eq!(with_q8[0].models[0].kv_current, "q8");
    }

    #[test]
    fn variants_for_falls_back_to_default_when_repo_absent() {
        // A repo not present in QUANT_VARIANTS → single synthesized variant from the entry.
        let e = entry("Qwen2.5", "Qwen 3B", 3.0, "q/absent", "model-q4_k_m.gguf", 2000);
        let vs = variants_for(&e);
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0].quant, "Q4_K_M");
        assert_eq!(vs[0].files, vec!["model-q4_k_m.gguf".to_string()]);
        assert_eq!(vs[0].size_mb, 2000);
    }

    #[test]
    fn files_for_quant_returns_default_variant_files_or_none() {
        let e = entry("Qwen2.5", "Qwen 3B", 3.0, "q/absent", "model-q4_k_m.gguf", 2000);
        assert_eq!(files_for_quant(&e, "q4_k_m"), Some(vec!["model-q4_k_m.gguf".to_string()]));
        assert_eq!(files_for_quant(&e, "Q8_0"), None); // not available for this (fallback) model
    }

    #[test]
    fn catalog_view_exposes_variants_and_selection() {
        use crate::settings::ExecProfile;
        let cat = vec![entry("Qwen2.5", "Qwen 7B", 7.0, "q/7b", "7b-q4_k_m.gguf", 4700)];
        // no saved profile quant → selected is the default (entry) quant
        let none = |_r: &str, _f: &str| ExecProfile::default();
        let v = catalog_view(&cat, 16384, 32768, KvKind::Q8, None, |_, _| true, |_, _| None, none);
        let m = &v[0].models[0];
        assert_eq!(m.quant_selected, "Q4_K_M");
        assert_eq!(m.variants.len(), 1);              // fallback single variant
        assert_eq!(m.variants[0].quant, "Q4_K_M");
        assert!(m.variants[0].selected);
        assert_eq!(m.variants[0].status, ModelStatus::Downloaded); // is_downloaded → true
        assert!(m.variants[0].est_ram_mb > m.variants[0].size_mb); // KV-aware est
    }

    #[test]
    fn row_size_reflects_selected_variant() {
        use crate::settings::ExecProfile;
        let cat = vec![entry("Qwen2.5", "Qwen 7B", 7.0, "q/7b", "7b-q4_k_m.gguf", 4700)];
        let none = |_r: &str, _f: &str| ExecProfile::default();
        let v = catalog_view(&cat, 16384, 32768, KvKind::Q8, None, |_, _| true, |_, _| None, none);
        let m = &v[0].models[0];
        let sel = m.variants.iter().find(|x| x.selected).unwrap();
        // Row-level fields must derive from the selected variant (not hardcoded e.size_mb).
        assert_eq!(m.size_mb, sel.size_mb);
        assert_eq!(m.est_ram_mb, sel.est_ram_mb);
        assert_eq!(m.fit, sel.fit);
    }

    #[test]
    fn params_b_for_key_catalog_then_scan_then_zero() {
        // A real CATALOG entry: key = "{repo}/{file}".
        let key = format!(
            "{}/{}",
            "Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"
        );
        assert_eq!(super::params_b_for_key(&key), 3.0);
        // Not in catalog but the key string carries a size token.
        assert_eq!(super::params_b_for_key("foo/bar-13b.gguf"), 13.0);
        // Unknown → 0.0.
        assert_eq!(super::params_b_for_key("foo/bar-model.gguf"), 0.0);
    }
}
