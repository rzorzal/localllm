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
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 3B Instruct", params: "3B", params_b: 3.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-3B-Instruct-GGUF", file: "qwen2.5-3b-instruct-q4_k_m.gguf", size_mb: 2000 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 7B Instruct", params: "7B", params_b: 7.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-7B-Instruct-GGUF", file: "qwen2.5-7b-instruct-q4_k_m.gguf", size_mb: 4700 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 14B Instruct", params: "14B", params_b: 14.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-14B-Instruct-GGUF", file: "qwen2.5-14b-instruct-q4_k_m.gguf", size_mb: 9000 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 32B Instruct", params: "32B", params_b: 32.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-32B-Instruct-GGUF", file: "qwen2.5-32b-instruct-q4_k_m.gguf", size_mb: 20000 },
    CatalogEntry { family: "Llama 3.1", display_name: "Llama 3.1 8B Instruct", params: "8B", params_b: 8.0, quant: "Q4_K_M",
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", size_mb: 4900 },
    CatalogEntry { family: "Phi 3.5", display_name: "Phi 3.5 Mini Instruct", params: "3.8B", params_b: 3.8, quant: "Q4_K_M",
        repo: "bartowski/Phi-3.5-mini-instruct-GGUF", file: "Phi-3.5-mini-instruct-Q4_K_M.gguf", size_mb: 2400 },
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
    active: Option<&ModelSpec>,
    is_downloaded: impl Fn(&str, &str) -> bool,
) -> Vec<FamilyView> {
    let budget = total_ram_mb * 65 / 100;
    let tight_ceiling = total_ram_mb * 85 / 100;

    // First pass: build ModelView (without recommendation) and track the best
    // recommendation candidate index in the flattened order.
    let mut flat: Vec<ModelView> = Vec::with_capacity(entries.len());
    let mut best_fit: Option<usize> = None; // largest params_b among Fits
    let mut smallest: Option<usize> = None; // fallback: smallest est_ram

    for (i, e) in entries.iter().enumerate() {
        let est = (e.size_mb as f32 * 1.2).round() as u32;
        let fit = if (est as u64) <= budget {
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
        if smallest.map(|j| est < (entries[j].size_mb as f32 * 1.2).round() as u32).unwrap_or(true) {
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

    fn entry(family: &'static str, name: &'static str, pb: f32, repo: &'static str, file: &'static str, size_mb: u32) -> CatalogEntry {
        CatalogEntry { family, display_name: name, params: "x", params_b: pb, quant: "Q4_K_M", repo, file, size_mb }
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
        // 16 GB machine → budget 10649 MB; total*0.85 = 13926 MB.
        // est = size*1.2: 3B→2400 fits, 7B→5640 fits, 8B→5880 fits,
        // 32B→24000 wont_fit. Largest-that-fits = 8B (params_b 8 > 7).
        let cat = sample();
        let active = ModelSpec { repo: "q/7b".into(), file: "7b.gguf".into() };
        let downloaded = |r: &str, _f: &str| r == "q/3b"; // 3B cached
        let view = catalog_view(&cat, 16384, Some(&active), downloaded);

        // grouped by family in first-seen order
        assert_eq!(view[0].family, "Qwen2.5");
        assert_eq!(view[1].family, "Llama");

        // statuses
        let qwen = &view[0].models;
        assert_eq!(qwen[0].status, ModelStatus::Downloaded);   // 3B cached
        assert_eq!(qwen[1].status, ModelStatus::InUse);        // 7B active
        assert_eq!(qwen[2].status, ModelStatus::NeedsDownload);// 32B

        // est ram + fit
        assert_eq!(qwen[0].est_ram_mb, 2400);
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
        // 2 GB → budget 1331; nothing fits → smallest est_ram (3B → 2400).
        let view = catalog_view(&cat, 2048, None, |_, _| false);
        let recs: Vec<_> = view.iter().flat_map(|f| f.models.iter()).filter(|m| m.recommended).collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].display_name, "Qwen 3B");
        // and a non-fitting model is marked won't_fit
        assert!(view.iter().flat_map(|f| f.models.iter()).any(|m| m.fit == FitVerdict::WontFit));
    }

    #[test]
    fn tight_band_between_budget_and_85_percent() {
        // total 10000 → budget 6500, 85% = 8500. Pick est in (6500, 8500].
        // size 6000 → est 7200 → Tight.
        let cat = vec![entry("F", "M", 5.0, "r", "f", 6000)];
        let view = catalog_view(&cat, 10000, None, |_, _| false);
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
}
