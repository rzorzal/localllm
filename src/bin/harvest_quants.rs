//! Dev-time harvester: queries the HuggingFace tree API for each CATALOG repo,
//! detects target-quant GGUF files (grouping split shards), and regenerates
//! `src/catalog_variants.rs`. NOT part of the server runtime.
//!
//! Run: `cargo run --bin harvest_quants`

use std::collections::BTreeMap;

/// Target quant tags, longest-first so `Q4_K_M` matches before a hypothetical `Q4`.
const TARGET_QUANTS: &[&str] = &["Q4_K_M", "Q5_K_M", "Q3_K_M", "Q6_K", "Q8_0", "Q2_K"];

/// The canonical target tag contained in a filename (case-insensitive), or None.
fn quant_tag_of(filename: &str) -> Option<&'static str> {
    let lower = filename.to_ascii_lowercase();
    if !lower.ends_with(".gguf") {
        return None;
    }
    TARGET_QUANTS
        .iter()
        .copied()
        .find(|tag| lower.contains(&tag.to_ascii_lowercase()))
}

/// Strip a `-NNNNN-of-NNNNN` shard suffix and the `.gguf` extension so shards of
/// one variant share a key.
fn variant_key(filename: &str) -> String {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    // Remove a trailing "-\d{5}-of-\d{5}" if present.
    if let Some(of_pos) = stem.rfind("-of-") {
        // shard suffix begins at the '-' before the first shard number
        // e.g. "...-Q6_K-00001-of-00002"
        if let Some(dash_before) = stem[..of_pos].rfind('-') {
            let tail = &stem[dash_before + 1..of_pos];
            if tail.len() == 5 && tail.chars().all(|c| c.is_ascii_digit()) {
                return stem[..dash_before].to_string();
            }
        }
    }
    stem.to_string()
}

/// Group HF files (name, size_bytes) into variants: (quant_tag, sorted files, size_mb).
fn group_variants(files: Vec<(String, u64)>) -> Vec<(String, Vec<String>, u32)> {
    // key = variant_key; value = (quant_tag, files, total_bytes)
    let mut map: BTreeMap<String, (&'static str, Vec<String>, u64)> = BTreeMap::new();
    for (name, size) in files {
        let Some(tag) = quant_tag_of(&name) else { continue };
        let key = variant_key(&name);
        let e = map.entry(key).or_insert((tag, Vec::new(), 0));
        e.1.push(name);
        e.2 += size;
    }
    map.into_values()
        .map(|(tag, mut fs, bytes)| {
            fs.sort();
            (tag.to_string(), fs, (bytes / 1_000_000) as u32)
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use std::collections::BTreeSet;
    // Unique repos from the catalog.
    let repos: BTreeSet<&str> = localllm::catalog::CATALOG.iter().map(|e| e.repo).collect();
    let client = reqwest::Client::builder().use_rustls_tls().build()?;

    // Start from existing generated data so a failed fetch keeps prior data (merge).
    let mut table: BTreeMap<String, Vec<(String, Vec<String>, u32)>> = BTreeMap::new();
    for (repo, vs) in localllm::catalog_variants::QUANT_VARIANTS {
        table.insert(
            repo.to_string(),
            vs.iter()
                .map(|v| (v.quant.to_string(), v.files.iter().map(|f| f.to_string()).collect(), v.size_mb))
                .collect(),
        );
    }

    let mut skipped: Vec<String> = Vec::new();
    let mut missing_default: Vec<String> = Vec::new();
    for repo in &repos {
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=false");
        match fetch_files(&client, &url).await {
            Ok(files) => {
                let mut variants = group_variants(files);
                variants.sort_by(|a, b| a.0.cmp(&b.0));
                if !variants.iter().any(|v| v.0 == "Q4_K_M") {
                    missing_default.push(repo.to_string());
                }
                table.insert(repo.to_string(), variants);
            }
            Err(e) => {
                eprintln!("WARN {repo}: fetch failed ({e}) — keeping prior data");
                skipped.push(repo.to_string());
            }
        }
    }

    write_generated(&table)?;
    eprintln!(
        "harvested {} repos ({} skipped). repos missing Q4_K_M: {:?}",
        repos.len() - skipped.len(),
        skipped.len(),
        missing_default
    );
    Ok(())
}

/// Fetch the tree listing → Vec<(path, size)> for .gguf files.
async fn fetch_files(client: &reqwest::Client, url: &str) -> anyhow::Result<Vec<(String, u64)>> {
    #[derive(serde::Deserialize)]
    struct Node {
        path: String,
        #[serde(default)]
        size: u64,
        #[serde(rename = "type")]
        kind: String,
    }
    let text = client.get(url).send().await?.error_for_status()?.text().await?;
    let nodes: Vec<Node> = serde_json::from_str(&text)?;
    Ok(nodes
        .into_iter()
        .filter(|n| n.kind == "file" && n.path.ends_with(".gguf"))
        .map(|n| (n.path, n.size))
        .collect())
}

/// Write `src/catalog_variants.rs` from the merged table.
fn write_generated(table: &BTreeMap<String, Vec<(String, Vec<String>, u32)>>) -> anyhow::Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    out.push_str("// @generated by `cargo run --bin harvest_quants` — do not edit by hand.\n");
    out.push_str("//! Per-repo GGUF quant variants. Regenerate with `cargo run --bin harvest_quants`.\n");
    out.push_str("use crate::catalog::QuantVariant;\n\n");
    out.push_str("pub static QUANT_VARIANTS: &[(&str, &[QuantVariant])] = &[\n");
    for (repo, variants) in table {
        if variants.is_empty() { continue; }
        writeln!(out, "    ({repo:?}, &[")?;
        for (quant, files, size_mb) in variants {
            let files_lit = files.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>().join(", ");
            writeln!(out, "        QuantVariant {{ quant: {quant:?}, files: &[{files_lit}], size_mb: {size_mb} }},")?;
        }
        out.push_str("    ]),\n");
    }
    out.push_str("];\n");
    std::fs::write("src/catalog_variants.rs", out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_tag_matches_target_set_case_insensitive() {
        assert_eq!(quant_tag_of("Qwen_Qwen3-8B-Q4_K_M.gguf"), Some("Q4_K_M"));
        assert_eq!(quant_tag_of("model-q8_0.gguf"), Some("Q8_0"));
        assert_eq!(quant_tag_of("model-q2_k.gguf"), Some("Q2_K"));
        assert_eq!(quant_tag_of("model-iq4_xs.gguf"), None); // IQ not in target set
        assert_eq!(quant_tag_of("readme.md"), None);
    }

    #[test]
    fn variant_key_strips_shard_suffix() {
        assert_eq!(variant_key("gemma-2-27b-it-Q6_K-00001-of-00002.gguf"),
                   "gemma-2-27b-it-Q6_K");
        assert_eq!(variant_key("model-q4_k_m.gguf"), "model-q4_k_m");
    }

    #[test]
    fn group_variants_sums_split_sizes_and_sorts_files() {
        let files = vec![
            ("m-Q6_K-00002-of-00002.gguf".to_string(), 2_000_000u64),
            ("m-Q6_K-00001-of-00002.gguf".to_string(), 3_000_000u64),
            ("m-q4_k_m.gguf".to_string(), 2_097_152u64),
            ("notes.txt".to_string(), 10u64),
        ];
        let mut got = group_variants(files);
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, vec![
            ("Q4_K_M".to_string(), vec!["m-q4_k_m.gguf".to_string()], 2u32),
            ("Q6_K".to_string(),
             vec!["m-Q6_K-00001-of-00002.gguf".to_string(), "m-Q6_K-00002-of-00002.gguf".to_string()],
             5u32),
        ]);
    }
}
