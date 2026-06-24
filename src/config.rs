//! CLI configuration parsed with clap.
//!
//! `Config` is the single source of truth for all runtime parameters.
//! `force_cpu` defaults to `true` because the Metal shader toolchain is
//! broken on this machine and GPU load fails; CPU is the only working path.
//! A user with a working Metal toolchain can pass `--force-cpu false`.

/// All runtime configuration, parsed from command-line arguments.
#[derive(clap::Parser, Debug)]
pub struct Config {
    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// HuggingFace model ID to load.
    #[arg(long, default_value = "Qwen/Qwen2.5-7B-Instruct-GGUF")]
    pub model_id: String,

    /// GGUF shard filename(s). Repeat the flag for split models.
    #[arg(long = "gguf-file", default_values_t = [
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf".to_string(),
        "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf".to_string(),
    ])]
    pub gguf_files: Vec<String>,

    /// KV-cache context length.
    #[arg(long, default_value_t = 16384)]
    pub ctx_len: usize,

    /// Disable paged attention.
    #[arg(long, default_value_t = false)]
    pub no_paged_attn: bool,

    /// Force CPU execution (sidesteps broken Metal toolchain on this machine).
    /// Defaults to true because GPU load fails on this hardware.
    #[arg(long, default_value_t = true)]
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
        assert_eq!(c.ctx_len, 16384);
        assert!(c.model_id.contains("Qwen2.5-7B-Instruct"));
    }

    #[test]
    fn engine_config_maps_fields_correctly() {
        let c = Config::parse_from(["localllm"]);
        let ec = c.engine_config();
        // no_paged_attn defaults to false → paged_attn should be true
        assert!(ec.paged_attn, "paged_attn should be true when no_paged_attn=false");
        // both default gguf shards should be present
        assert_eq!(ec.gguf_files.len(), 2);
        assert!(ec.gguf_files[0].contains("00001-of-00002"));
        assert!(ec.gguf_files[1].contains("00002-of-00002"));
    }
}
