//! Pure resolution of a model's effective execution parameters from three
//! layers of precedence: saved per-model profile → catalog recommendation →
//! global CLI default. No I/O.

use crate::catalog::CatalogEntry;
use crate::config::KvType;
use crate::settings::ExecProfile;

/// Fully resolved execution parameters for a model load.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub ctx: u32,
    pub kv_type: KvType,
    pub gpu_layers: Option<u32>,
    pub history_turns: Option<u32>,
}

/// Resolve each field: saved profile wins, else catalog recommendation, else
/// the global CLI default. `ctx` has no catalog recommendation here (catalog ctx
/// defaults are memory-derived elsewhere), so it is `saved.ctx` or `global_ctx`.
pub fn resolve(
    saved: &ExecProfile,
    catalog: Option<&CatalogEntry>,
    global_ctx: u32,
    global_kv: KvType,
) -> Resolved {
    let rec_kv = catalog.and_then(|c| c.rec_kv.clone());
    let rec_gpu = catalog.and_then(|c| c.rec_gpu_layers);
    let rec_hist = catalog.and_then(|c| c.rec_history_turns);
    Resolved {
        ctx: saved.ctx.unwrap_or(global_ctx),
        kv_type: saved.kv_type.clone().or(rec_kv).unwrap_or(global_kv),
        gpu_layers: saved.gpu_layers.or(rec_gpu),
        history_turns: saved.history_turns.or(rec_hist),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat(rec_kv: Option<KvType>, rec_gpu: Option<u32>, rec_hist: Option<u32>) -> CatalogEntry {
        CatalogEntry {
            family: "F", display_name: "M", params: "7B", params_b: 7.0, quant: "Q4_K_M",
            repo: "r", file: "f", size_mb: 4700, ctx_train: 32768,
            rec_kv, rec_gpu_layers: rec_gpu, rec_history_turns: rec_hist,
        }
    }

    #[test]
    fn saved_wins_over_catalog_and_global() {
        let saved = ExecProfile {
            ctx: Some(8192), kv_type: Some(KvType::Q4),
            gpu_layers: Some(10), history_turns: Some(2),
        };
        let c = cat(Some(KvType::F16), Some(99), Some(9));
        let r = resolve(&saved, Some(&c), 32768, KvType::Q8);
        assert_eq!(r, Resolved { ctx: 8192, kv_type: KvType::Q4, gpu_layers: Some(10), history_turns: Some(2) });
    }

    #[test]
    fn catalog_fills_gaps_when_saved_empty() {
        let saved = ExecProfile::default();
        let c = cat(Some(KvType::F16), Some(20), Some(4));
        let r = resolve(&saved, Some(&c), 32768, KvType::Q8);
        assert_eq!(r, Resolved { ctx: 32768, kv_type: KvType::F16, gpu_layers: Some(20), history_turns: Some(4) });
    }

    #[test]
    fn global_fallback_when_nothing_set() {
        let r = resolve(&ExecProfile::default(), None, 32768, KvType::Q8);
        assert_eq!(r, Resolved { ctx: 32768, kv_type: KvType::Q8, gpu_layers: None, history_turns: None });
    }
}
