//! Pure context-fit math: how much memory a model's KV cache costs at a given
//! context length, and the largest context that fits a memory budget. No I/O,
//! no llama/ggml calls — just numbers in, numbers out, so it is fully testable.

/// KV-cache element storage, decoupled from `llama_cpp_2::KvCacheType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvKind {
    F16,
    Q8,
    Q4,
}

/// Bytes per stored KV element for each quantization (llama.cpp sizes).
fn bytes_per_elem(kv: KvKind) -> f64 {
    match kv {
        KvKind::F16 => 2.0,
        KvKind::Q8 => 1.0625, // q8_0: 34 bytes / 32 elems
        KvKind::Q4 => 0.5625, // q4_0: 18 bytes / 32 elems
    }
}

pub const MIN_CTX: u32 = 2048;
pub const DEFAULT_SMALL_CTX: u32 = 8192;
pub const GLOBAL_MAX_CTX: u32 = 32768;
pub const COMPUTE_HEADROOM_MB: u32 = 1024;
pub const METAL_BUDGET_PCT: u64 = 78;
pub const CPU_BUDGET_PCT: u64 = 65;

/// KV-cache bytes per token, exact from the model's dimensions.
/// `2` covers both the K and V caches.
pub fn kv_bytes_per_token(n_layer: u32, n_head_kv: u32, head_dim: u32, kv: KvKind) -> u64 {
    let elems = 2.0 * n_layer as f64 * n_head_kv as f64 * head_dim as f64;
    (elems * bytes_per_elem(kv)).round() as u64
}

/// Coarse KV bytes/token from parameter count alone, for catalog entries whose
/// GGUF metadata is not yet available (pre-download). Tuned so a typical dense
/// transformer lands within ~2x of the exact value. `ELEMS_PER_TOKEN_PER_B`
/// (~6700) is the empirical `2*n_layer*n_head_kv*head_dim / params_b` for common
/// models (Qwen 3B ≈ 6144, Phi-4 14B ≈ 7314).
pub fn est_kv_bytes_per_token(params_b: f32, kv: KvKind) -> u64 {
    const ELEMS_PER_TOKEN_PER_B: f64 = 6700.0;
    let elems = ELEMS_PER_TOKEN_PER_B * params_b as f64;
    (elems * bytes_per_elem(kv)).round() as u64
}

/// Largest context (multiple of 256, capped at `n_ctx_train` and `GLOBAL_MAX_CTX`)
/// whose `weights + KV(ctx) + COMPUTE_HEADROOM` fits in `budget_mb`. Returns 0 if
/// even `MIN_CTX` does not fit.
pub fn max_ctx_fit(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> u32 {
    let overhead = weights_mb.saturating_add(COMPUTE_HEADROOM_MB);
    let avail_mb = budget_mb.saturating_sub(overhead);
    if avail_mb == 0 || kv_per_token_bytes == 0 {
        return 0;
    }
    let avail_bytes = avail_mb as u64 * 1024 * 1024;
    let by_mem = avail_bytes / kv_per_token_bytes;
    let cap = by_mem.min(n_ctx_train as u64).min(GLOBAL_MAX_CTX as u64);
    let rounded = (cap / 256) * 256; // floor to a multiple of 256
    if rounded < MIN_CTX as u64 {
        0
    } else {
        rounded as u32
    }
}

/// The three ctx values surfaced for a model. `max == 0` means "won't fit even
/// at MIN_CTX" — the caller refuses the load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtxBounds {
    pub min: u32,
    pub default: u32,
    pub max: u32,
}

/// `max = max_ctx_fit(...)`; `min = MIN_CTX`; `default = min(DEFAULT_SMALL_CTX, max)`.
/// When the model won't fit, `default` and `max` are both 0.
pub fn ctx_bounds(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> CtxBounds {
    let max = max_ctx_fit(weights_mb, kv_per_token_bytes, budget_mb, n_ctx_train);
    if max == 0 {
        return CtxBounds { min: MIN_CTX, default: 0, max: 0 };
    }
    CtxBounds { min: MIN_CTX, default: DEFAULT_SMALL_CTX.min(max), max }
}

/// Usable memory budget in MB: `METAL_BUDGET_PCT` of RAM when a GPU is present
/// (Apple unified memory ≈ the Metal working-set cap), else `CPU_BUDGET_PCT`.
pub fn device_budget_mb(total_ram_mb: u64, gpu_present: bool) -> u32 {
    let pct = if gpu_present { METAL_BUDGET_PCT } else { CPU_BUDGET_PCT };
    (total_ram_mb * pct / 100) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_bytes_per_token_phi4_q8() {
        // Phi-4: n_layer 40, n_head_kv 10, head_dim 128, Q8.
        // 2*40*10*128 = 102400 elems * 1.0625 = 108800 bytes.
        assert_eq!(kv_bytes_per_token(40, 10, 128, KvKind::Q8), 108800);
    }

    #[test]
    fn kv_bytes_scales_with_kv_kind() {
        let f16 = kv_bytes_per_token(40, 10, 128, KvKind::F16);
        let q8 = kv_bytes_per_token(40, 10, 128, KvKind::Q8);
        let q4 = kv_bytes_per_token(40, 10, 128, KvKind::Q4);
        assert!(f16 > q8 && q8 > q4);
        assert_eq!(f16, 204800); // 102400 * 2.0
    }

    #[test]
    fn est_kv_within_2x_of_exact() {
        // Phi-4 exact 108800; heuristic from 14B should be within 2x.
        let est = est_kv_bytes_per_token(14.0, KvKind::Q8) as f64;
        let exact = 108800.0;
        assert!(est > exact / 2.0 && est < exact * 2.0, "est={est} exact={exact}");
    }

    #[test]
    fn max_ctx_fit_caps_at_global_max_for_small_model() {
        // 3B (~2000 MB weights), tiny KV, big budget → capped at GLOBAL_MAX_CTX.
        let kv = kv_bytes_per_token(36, 2, 128, KvKind::Q8); // Qwen 3B ~19584 B/tok
        let m = max_ctx_fit(2000, kv, 12780, 131072);
        assert_eq!(m, GLOBAL_MAX_CTX);
    }

    #[test]
    fn max_ctx_fit_phi4_16gb_caps_at_ctx_train() {
        // Phi-4 on 16GB Metal: weights ~8634, budget 12780, trained ctx 16384.
        // Memory would allow ~30k, but n_ctx_train (16384) binds first — and
        // that is well under the requested 32768, so the load will clamp.
        let kv = kv_bytes_per_token(40, 10, 128, KvKind::Q8); // 108800 B/tok
        let m = max_ctx_fit(8634, kv, 12780, 16384);
        assert_eq!(m, 16384);
    }

    #[test]
    fn max_ctx_fit_memory_bound_below_ctx_train() {
        // Same big model but a large trained ctx → MEMORY binds, not n_ctx_train.
        // avail = 12780 - (8634+1024) = 3122 MB → 3122MiB/108800B ≈ 30088 tokens
        // → floored to a multiple of 256 (29952), and below GLOBAL_MAX_CTX.
        let kv = kv_bytes_per_token(40, 10, 128, KvKind::Q8);
        let m = max_ctx_fit(8634, kv, 12780, 131072);
        assert!(m < GLOBAL_MAX_CTX && m >= 29000, "got {m}");
        assert_eq!(m % 256, 0);
    }

    #[test]
    fn max_ctx_fit_zero_when_weights_exceed_budget() {
        assert_eq!(max_ctx_fit(13000, 108800, 12780, 16384), 0);
    }

    #[test]
    fn ctx_bounds_default_is_small_and_within_max() {
        let kv = kv_bytes_per_token(36, 2, 128, KvKind::Q8);
        let b = ctx_bounds(2000, kv, 12780, 131072);
        assert_eq!(b.min, MIN_CTX);
        assert_eq!(b.max, GLOBAL_MAX_CTX);
        assert_eq!(b.default, DEFAULT_SMALL_CTX); // 8192 < 32768
    }

    #[test]
    fn ctx_bounds_wont_fit_reports_zero() {
        let b = ctx_bounds(13000, 108800, 12780, 16384);
        assert_eq!(b.max, 0);
        assert_eq!(b.default, 0);
        assert_eq!(b.min, MIN_CTX);
    }

    #[test]
    fn device_budget_gpu_vs_cpu() {
        assert_eq!(device_budget_mb(16384, true), 12779);  // 78%
        assert_eq!(device_budget_mb(16384, false), 10649); // 65%
    }
}
