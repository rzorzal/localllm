//! CLI configuration parsed with clap.
//!
//! `Config` is the single source of truth for all runtime parameters.
//! `force_cpu` defaults to `false`: the Metal toolchain is installed and
//! shaders compile, so the model runs on the Apple GPU. Pass
//! `--force-cpu true` to force CPU (e.g. if the Metal toolchain breaks again).

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

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
}
