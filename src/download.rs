//! In-process model downloader.
//!
//! Downloads GGUF (or any) files from HuggingFace into a local cache dir,
//! skipping files that are already present and non-empty.

use std::path::PathBuf;

use anyhow::Context;
use futures_util::StreamExt;
use tracing::info;

// ---------------------------------------------------------------------------
// URL builder
// ---------------------------------------------------------------------------

/// Return the canonical HuggingFace resolve URL for a repo file.
pub fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

// ---------------------------------------------------------------------------
// Cache path builder
// ---------------------------------------------------------------------------

/// Return the local cache path for a repo file.
///
/// Layout: `<sys_cache>/localllm/<sanitized_repo>/<file>`
/// where `sanitized_repo` replaces `/` with `--` (mirrors the HF hub layout).
pub fn cache_path(repo: &str, file: &str) -> PathBuf {
    let base = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"));
    let sanitized = repo.replace('/', "--");
    base.join("localllm").join(sanitized).join(file)
}

// ---------------------------------------------------------------------------
// ensure_model
// ---------------------------------------------------------------------------

/// For each file in `files`, if it already exists in cache and is non-empty,
/// skip it; otherwise download it from HuggingFace, streaming to a `.part`
/// temp file and renaming on success.  Returns the local paths in order.
pub async fn ensure_model(repo: &str, files: &[String]) -> anyhow::Result<Vec<PathBuf>> {
    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .build()
        .context("building reqwest client")?;

    let mut paths = Vec::with_capacity(files.len());

    for file in files {
        let dest = cache_path(repo, file);

        // Skip if already present and non-empty.
        if dest.exists() && dest.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            info!("cache hit: {}", dest.display());
            paths.push(dest);
            continue;
        }

        // Ensure parent directory exists.
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating cache dir {}", parent.display()))?;
        }

        let url = hf_url(repo, file);
        info!("downloading {} → {}", url, dest.display());

        let part_path = dest.with_extension("part");

        let response = client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?
            .error_for_status()
            .with_context(|| format!("HTTP error for {url}"))?;

        let total = response.content_length();
        let mut stream = response.bytes_stream();

        let mut part_file = tokio::fs::File::create(&part_path)
            .await
            .with_context(|| format!("creating {}", part_path.display()))?;

        let mut downloaded: u64 = 0;
        let mut last_logged: u64 = 0;
        const LOG_EVERY: u64 = 50 * 1024 * 1024; // 50 MB

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("reading stream for {url}"))?;
            tokio::io::AsyncWriteExt::write_all(&mut part_file, &chunk)
                .await
                .with_context(|| format!("writing to {}", part_path.display()))?;
            downloaded += chunk.len() as u64;
            if downloaded - last_logged >= LOG_EVERY {
                last_logged = downloaded;
                match total {
                    Some(t) => info!("  {file}: {:.1} / {:.1} MB", mb(downloaded), mb(t)),
                    None    => info!("  {file}: {:.1} MB downloaded", mb(downloaded)),
                }
            }
        }

        // Flush + rename to final path.
        tokio::io::AsyncWriteExt::flush(&mut part_file).await?;
        drop(part_file);
        std::fs::rename(&part_path, &dest)
            .with_context(|| format!("renaming {} → {}", part_path.display(), dest.display()))?;

        info!("download complete: {} ({:.1} MB)", dest.display(), mb(downloaded));
        paths.push(dest);
    }

    Ok(paths)
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

// ---------------------------------------------------------------------------
// Tests — RED step: written BEFORE any implementation existed
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_url_returns_correct_resolve_url() {
        let repo = "Qwen/Qwen2.5-3B-Instruct-GGUF";
        let file = "qwen2.5-3b-instruct-q4_k_m.gguf";
        let expected = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf";
        assert_eq!(hf_url(repo, file), expected);
    }

    #[test]
    fn cache_path_ends_with_filename() {
        let repo = "Qwen/Qwen2.5-3B-Instruct-GGUF";
        let file = "qwen2.5-3b-instruct-q4_k_m.gguf";
        let path = cache_path(repo, file);
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), file);
    }

    #[test]
    fn cache_path_contains_localllm_component() {
        let path = cache_path("SomeOrg/SomeModel-GGUF", "model.gguf");
        let components: Vec<_> = path.components().collect();
        let has_localllm = components
            .iter()
            .any(|c| c.as_os_str() == "localllm");
        assert!(has_localllm, "cache path should contain 'localllm' dir: {path:?}");
    }

    #[test]
    fn cache_path_sanitizes_repo_slash() {
        let path = cache_path("Org/Repo", "file.gguf");
        // The directory containing the file must NOT be named "Repo" alone
        // (the slash was replaced) — it should be "Org--Repo".
        let parent_name = path.parent().unwrap().file_name().unwrap().to_str().unwrap();
        assert_eq!(parent_name, "Org--Repo");
    }

    /// Skip-path integration test: create a dummy file at the cache location
    /// and confirm `ensure_model` returns it without any network call.
    #[tokio::test]
    async fn ensure_model_skips_existing_nonempty_file() {
        let repo = "test-org/test-model";
        let file = "tiny.gguf".to_string();

        let dest = cache_path(repo, &file);
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&dest, b"dummy content").unwrap();

        let result = ensure_model(repo, &[file]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], dest);

        // Cleanup
        let _ = std::fs::remove_file(&dest);
    }
}
