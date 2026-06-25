//! CLI configuration parsed with clap.
//!
//! `Config` is the single source of truth for all runtime parameters.
//! `force_cpu` defaults to `false`: the Metal toolchain is installed and
//! shaders compile, so the model runs on the Apple GPU. Pass
//! `--force-cpu true` to force CPU (e.g. if the Metal toolchain breaks again).

/// Which inference backend to use.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    /// llama.cpp via the llama-cpp-2 crate (default). Works on Metal without
    /// pre-compiling shaders. Supports incremental token streaming.
    Llama,
    /// mistral.rs GgufModelBuilder. Feature-rich but requires
    /// MISTRALRS_METAL_PRECOMPILE=0 on this machine.
    Mistralrs,
}

/// KV-cache quantization type. Controls the precision of the K and V tensors
/// in the llama.cpp KV cache. Lower precision = less RAM, slightly lower quality.
///
/// Q8 is the default: ~50% the RAM of F16 with negligible quality loss.
/// Q4 saves ~75% RAM but may reduce output quality on some models.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq, Default)]
pub enum KvType {
    /// 8-bit quantized KV cache. ~50% the size of F16. Default.
    #[default]
    Q8,
    /// Full 16-bit float KV cache. Maximum quality, most RAM.
    F16,
    /// 4-bit quantized KV cache. ~25% the size of F16. May reduce quality.
    Q4,
}

/// All runtime configuration, parsed from command-line arguments.
#[derive(clap::Parser, Debug)]
pub struct Config {
    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// HuggingFace model ID to load. Default is the lightweight Qwen2.5-3B
    /// (~2 GB Q4), chosen so the model coexists with the user's other apps on
    /// a 16 GB machine for small local tasks. Pass --model-id + --gguf-file to
    /// run a larger model (e.g. Qwen2.5-7B) when more RAM is free.
    #[arg(long, default_value = "Qwen/Qwen2.5-3B-Instruct-GGUF")]
    pub model_id: String,

    /// GGUF filename(s). Repeat the flag for split models.
    #[arg(long = "gguf-file", default_values_t = [
        "qwen2.5-3b-instruct-q4_k_m.gguf".to_string(),
    ])]
    pub gguf_files: Vec<String>,

    /// Context window in tokens. On GPU (PagedAttention) this sizes the KV
    /// cache and is the usable context length (~55 KB/token; 8192 ≈ 0.4 GB).
    /// Default 8192: ample for small tasks while staying light on RAM so the
    /// model coexists with other apps. Raise it for longer documents.
    #[arg(long, default_value_t = 8192)]
    pub ctx_len: usize,

    /// Disable paged attention.
    #[arg(long, default_value_t = false)]
    pub no_paged_attn: bool,

    /// Force CPU execution. Defaults to false (use the Apple GPU via Metal).
    /// Set to true to fall back to CPU.
    #[arg(long, default_value_t = false)]
    pub force_cpu: bool,

    /// Inference backend. Default `llama` (embedded llama.cpp, single binary,
    /// no MISTRALRS_METAL_PRECOMPILE needed, supports KV persistence).
    #[arg(long, value_enum, default_value_t = Backend::Llama)]
    pub backend: Backend,

    /// KV-cache quantization type. Only applies to the llama backend.
    /// Q8 (default) cuts KV-cache RAM ~50% vs F16 with negligible quality loss.
    /// Q4 cuts ~75% but may reduce output quality.
    #[arg(long, value_enum, default_value_t = KvType::Q8)]
    pub kv_type: KvType,
}

impl Config {
    /// Map CLI config into the `EngineConfig` expected by `Engine::load`.
    pub fn engine_config(&self) -> crate::engine::EngineConfig {
        crate::engine::EngineConfig {
            model_id: self.model_id.clone(),
            gguf_files: self.gguf_files.clone(),
            ctx_len: self.ctx_len,
            paged_attn: !self.no_paged_attn,
            force_cpu: self.force_cpu,
        }
    }

    /// Map CLI `kv_type` to the llama-cpp-2 `KvCacheType`.
    pub fn llama_kv_cache_type(&self) -> llama_cpp_2::context::params::KvCacheType {
        use llama_cpp_2::context::params::KvCacheType;
        match self.kv_type {
            KvType::Q8 => KvCacheType::Q8_0,
            KvType::Q4 => KvCacheType::Q4_0,
            KvType::F16 => KvCacheType::F16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn default_backend_is_llama() {
        let c = Config::parse_from(["localllm"]);
        assert_eq!(c.backend, Backend::Llama);
    }

    #[test]
    fn defaults_are_localhost_8080_qwen() {
        let c = Config::parse_from(["localllm"]);
        assert_eq!(c.port, 8080);
        assert_eq!(c.ctx_len, 8192);
        assert!(c.model_id.contains("Qwen2.5-3B-Instruct"));
    }

    #[test]
    fn engine_config_maps_fields_correctly() {
        let c = Config::parse_from(["localllm"]);
        let ec = c.engine_config();
        // no_paged_attn defaults to false → paged_attn should be true
        assert!(ec.paged_attn, "paged_attn should be true when no_paged_attn=false");
        // default is the single-file 3B gguf
        assert_eq!(ec.gguf_files.len(), 1);
        assert!(ec.gguf_files[0].contains("qwen2.5-3b-instruct-q4_k_m"));
    }

    #[test]
    fn default_kv_type_is_q8() {
        let c = Config::parse_from(["localllm"]);
        assert_eq!(c.kv_type, KvType::Q8, "default kv_type must be Q8");
    }

    #[test]
    fn kv_type_flag_parses_correctly() {
        let c = Config::parse_from(["localllm", "--kv-type", "f16"]);
        assert_eq!(c.kv_type, KvType::F16);

        let c = Config::parse_from(["localllm", "--kv-type", "q4"]);
        assert_eq!(c.kv_type, KvType::Q4);

        let c = Config::parse_from(["localllm", "--kv-type", "q8"]);
        assert_eq!(c.kv_type, KvType::Q8);
    }
}
