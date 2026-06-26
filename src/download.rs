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

/// Download a single URL to `dest` (via a `.part` temp), invoking `on_progress`
/// with (bytes_done, total_from_Content-Length) throttled to ~1% changes.
async fn download_to_with_progress(
    url: &str,
    dest: &std::path::Path,
    on_progress: impl Fn(u64, Option<u64>),
) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().use_rustls_tls().build()
        .context("building reqwest client")?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating cache dir {}", parent.display()))?;
    }
    let part_path = dest.with_extension("part");
    let response = client.get(url).send().await
        .with_context(|| format!("GET {url}"))?
        .error_for_status().with_context(|| format!("HTTP error for {url}"))?;
    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut part_file = tokio::fs::File::create(&part_path).await
        .with_context(|| format!("creating {}", part_path.display()))?;
    let mut downloaded: u64 = 0;
    let mut last_pct: i64 = -1;
    on_progress(0, total);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("reading stream for {url}"))?;
        tokio::io::AsyncWriteExt::write_all(&mut part_file, &chunk).await
            .with_context(|| format!("writing to {}", part_path.display()))?;
        downloaded += chunk.len() as u64;
        if let Some(t) = total {
            if t > 0 {
                let pct = (downloaded * 100 / t) as i64;
                if pct != last_pct {
                    last_pct = pct;
                    on_progress(downloaded, total);
                }
            }
        } else {
            on_progress(downloaded, None);
        }
    }
    tokio::io::AsyncWriteExt::flush(&mut part_file).await?;
    drop(part_file);
    std::fs::rename(&part_path, dest)
        .with_context(|| format!("renaming {} → {}", part_path.display(), dest.display()))?;
    on_progress(downloaded, total.or(Some(downloaded)));
    Ok(())
}

/// Like [`ensure_model`] but reports download progress per chunk. `on_progress`
/// receives (bytes_done, total) for the file currently downloading; cache-hit
/// files report 100% immediately (done==total==file len).
pub async fn ensure_model_with_progress(
    repo: &str,
    files: &[String],
    on_progress: impl Fn(u64, Option<u64>),
) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(files.len());
    for file in files {
        let dest = cache_path(repo, file);
        if dest.exists() && dest.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            info!("cache hit: {}", dest.display());
            let len = dest.metadata().map(|m| m.len()).unwrap_or(0);
            on_progress(len, Some(len)); // 100%
            paths.push(dest);
            continue;
        }
        let url = hf_url(repo, file);
        info!("downloading {} → {}", url, dest.display());
        download_to_with_progress(&url, &dest, &on_progress).await?;
        info!("download complete: {}", dest.display());
        paths.push(dest);
    }
    Ok(paths)
}

/// Delete a cached model file (and any leftover `.part`). Returns whether the
/// main file existed and was removed. Best-effort on the `.part`.
pub fn delete_cached(repo: &str, file: &str) -> std::io::Result<bool> {
    let path = cache_path(repo, file);
    let _ = std::fs::remove_file(path.with_extension("part"));
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// For each file in `files`, if it already exists in cache and is non-empty,
/// skip it; otherwise download it from HuggingFace, streaming to a `.part`
/// temp file and renaming on success.  Returns the local paths in order.
pub async fn ensure_model(repo: &str, files: &[String]) -> anyhow::Result<Vec<PathBuf>> {
    ensure_model_with_progress(repo, files, |_, _| {}).await
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

    #[tokio::test]
    async fn progress_reports_total_and_reaches_full() {
        use std::sync::{Arc, Mutex};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = vec![b'x'; 1000];
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        // Point hf_url at the mock by using a repo/file whose resolve URL is the mock.
        // ensure_model_with_progress builds the URL via hf_url(repo,file); override the
        // base by setting the file to an absolute path is not possible, so we test the
        // callback wiring through a direct download against the mock server URL instead.
        type ProgressLog = Vec<(u64, Option<u64>)>;
        let seen: Arc<Mutex<ProgressLog>> = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();

        // Use a temp cache by pointing HOME/XDG cache via the repo/file path under a temp dir
        // is overkill here; instead assert the callback contract via a unit on the streaming
        // helper. We verify: final callback has done == total == 1000.
        let dir = std::env::temp_dir().join(format!("dl-prog-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("model.bin");
        download_to_with_progress(&server.uri(), &dest, |done, total| {
            seen2.lock().unwrap().push((done, total));
        })
        .await
        .unwrap();

        let s = seen.lock().unwrap();
        let (last_done, last_total) = *s.last().unwrap();
        assert_eq!(last_done, 1000);
        assert_eq!(last_total, Some(1000));
        assert!(dest.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_cached_removes_file_then_reports_absent() {
        // Create a real cached file at the canonical path for a throwaway repo.
        let repo = format!("test--del-{}", uuid::Uuid::new_v4());
        let file = "m.gguf";
        let path = cache_path(&repo, file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"data").unwrap();

        assert!(delete_cached(&repo, file).unwrap()); // removed
        assert!(!path.exists());
        assert!(!delete_cached(&repo, file).unwrap()); // already gone

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
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
