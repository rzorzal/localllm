# Fase A — Observabilidade, Custo & Budget — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrumentar o roteamento local↔cloud com custo em $, budget diário (tela nova), latência (TTFT/tok/s) e visualização dos motivos de fallback, mais export do log.

**Architecture:** O `route_log` passa a gravar 2 linhas por request (decisão + desfecho) correlacionadas por `rid`. Um módulo `pricing` converte tokens→$. Um módulo `budget` mantém o gasto cloud do dia e força local quando estoura. O dashboard agrega tudo no backend; o frontend só pinta. Uma tela nova `/budget` configura o teto.

**Tech Stack:** Rust (axum 0.7, serde, futures), frontend vanilla JS embutido (`src/manager_ui/*`).

## Global Constraints

- Testes: `cargo test --lib <modulo>::`; build: `cargo build`.
- Sem novas dependências (não há `chrono`/`time`): o "dia" do budget é **dia-UTC** = `ts.div_euclid(86400)`. Documentar; refinar pra tz local em fase futura.
- Escrita no log é best-effort: nunca falha um request (segue o padrão de `route_log::append`).
- Retrocompat: linhas antigas do log (sem `kind`) parseiam como decisão.
- Preços em `pricing.rs` são estimativas mantidas no código; usuário não edita.
- Endpoints admin são token-guarded (`check_admin`), como os existentes.
- Motivo do fallback-por-budget é gravado como `RouteEntry.reason = "BudgetExceeded"` com `dest="local"` (NÃO um novo `RouteReason`, que é cloud-only); fallback-por-provider usa `RouteEntry.degrade_reason`.

---

### Task 1: Modelo de log — `LogLine`, campos novos, `OutcomeEntry`

**Files:**
- Modify: `src/route_log.rs`

**Interfaces:**
- Produces:
  - `RouteEntry` ganha `rid: String`, `model: Option<String>`, `degrade_reason: Option<String>`.
  - `pub struct OutcomeEntry { rid: String, ts: i64, completion_tok: Option<u64>, ttft_ms: Option<u64>, gen_ms: Option<u64>, cost_saved_usd: f64 }`
  - `pub enum LogLine { Decision(RouteEntry), Outcome(OutcomeEntry) }` com `fn ts(&self) -> i64`.
  - `pub fn read_all() -> Vec<LogLine>` (retrocompat).
  - `pub fn append_outcome(entry: &OutcomeEntry)`.
  - `append(&RouteEntry)` inalterada na assinatura; grava `{"kind":"d",...}`.

- [ ] **Step 1: Escreve o teste de retrocompat + append_outcome**

Adicionar em `#[cfg(test)] mod tests` de `src/route_log.rs`:

```rust
#[test]
fn read_all_parses_new_and_legacy_lines() {
    let dir = std::env::temp_dir().join(format!("localllm-ll-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("routing-log.jsonl");
    std::env::set_var("LOCALLLM_ROUTE_LOG", &path);
    // legacy decision line (no "kind")
    let legacy = r#"{"ts":10,"surface":"openai","dest":"local","score":0.1,"prompt_tok":5}"#;
    std::fs::write(&path, format!("{legacy}\n")).unwrap();
    // new decision + outcome via the API
    append(&RouteEntry { ts: 20, rid: "req-1".into(), surface: "openai".into(),
        dest: "local".into(), prompt_tok: 7, ..Default::default() });
    append_outcome(&OutcomeEntry { rid: "req-1".into(), ts: 21, completion_tok: Some(9),
        ttft_ms: Some(30), gen_ms: Some(100), cost_saved_usd: 0.5 });
    let lines = read_all();
    assert_eq!(lines.len(), 3);
    assert!(matches!(lines[0], LogLine::Decision(ref d) if d.prompt_tok == 5));
    assert!(matches!(lines[2], LogLine::Outcome(ref o) if o.completion_tok == Some(9)));
    std::env::remove_var("LOCALLLM_ROUTE_LOG");
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Roda e verifica que falha**

Run: `cargo test --lib route_log::tests::read_all_parses_new_and_legacy_lines`
Expected: FAIL — `OutcomeEntry`/`append_outcome`/`LogLine` não existem.

- [ ] **Step 3: Implementa**

Em `src/route_log.rs`, adicionar aos campos de `RouteEntry` (logo após `prompt_snippet`):

```rust
    /// Correlation id shared with the request's OutcomeEntry (e.g. "req-1a2b3c4d").
    #[serde(default)]
    pub rid: String,
    /// Cloud model the client asked for (the one that WOULD have served). Used
    /// to price $ saved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// When this request fell back to local due to a provider degrade, the reason
    /// (Quota/Auth/ServerError/Offline). None otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,
```

Depois da definição de `RouteEntry`, adicionar:

```rust
/// Post-generation outcome for a request, correlated to its RouteEntry by `rid`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OutcomeEntry {
    pub rid: String,
    pub ts: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tok: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_ms: Option<u64>,
    /// What this LOCAL request would have cost on cloud (0.0 for cloud requests).
    #[serde(default)]
    pub cost_saved_usd: f64,
}

/// A parsed log line: a routing decision or a post-generation outcome.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind")]
pub enum LogLine {
    #[serde(rename = "d")]
    Decision(RouteEntry),
    #[serde(rename = "o")]
    Outcome(OutcomeEntry),
}

impl LogLine {
    pub fn ts(&self) -> i64 {
        match self {
            LogLine::Decision(d) => d.ts,
            LogLine::Outcome(o) => o.ts,
        }
    }
}
```

Trocar `append` e `read_all`, e adicionar `append_outcome`:

```rust
pub fn append(entry: &RouteEntry) {
    append_line(&LogLine::Decision(entry.clone()));
}

pub fn append_outcome(entry: &OutcomeEntry) {
    append_line(&LogLine::Outcome(entry.clone()));
}

fn append_line(line: &LogLine) {
    let Some(path) = log_path() else { return };
    let Ok(text) = serde_json::to_string(line) else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{text}");
    }
}

/// Read every line: new tagged lines, or legacy untagged decision lines.
pub fn read_all() -> Vec<LogLine> {
    let Some(path) = log_path() else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    text.lines()
        .filter_map(|l| {
            serde_json::from_str::<LogLine>(l).ok().or_else(|| {
                serde_json::from_str::<RouteEntry>(l).ok().map(LogLine::Decision)
            })
        })
        .collect()
}
```

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib route_log::tests::read_all_parses_new_and_legacy_lines`
Expected: PASS. (Outros testes de `route_log` vão quebrar por causa da mudança de tipo de `read_all`/`prune`/`build_dashboard` — corrigidos na Task 3. Se o build falhar aqui, seguir pra Step 5 mesmo assim e só then rodar o teste específico com `--no-fail-fast` não é possível; então: adaptar `prune`/`prune_file` já neste passo conforme abaixo.)

Ajustar `prune` e `prune_file` pra `LogLine`:

```rust
pub fn prune(entries: &[LogLine], now: i64, max_age_secs: i64) -> Vec<LogLine> {
    let cutoff = now - max_age_secs;
    entries.iter().filter(|e| e.ts() >= cutoff).cloned().collect()
}

pub fn prune_file(now: i64, max_age_secs: i64) {
    let Some(path) = log_path() else { return };
    let kept = prune(&read_all(), now, max_age_secs);
    let mut buf = String::new();
    for e in &kept {
        if let Ok(l) = serde_json::to_string(e) {
            buf.push_str(&l);
            buf.push('\n');
        }
    }
    let _ = crate::integrations::atomic_write(&path, buf.as_bytes());
}
```

Atualizar o teste `prune_drops_entries_older_than_max_age` e `prune_keeps_everything_when_all_fresh`: o helper `entry(ts,dest)` retorna `RouteEntry`; envolver em `LogLine::Decision(entry(...))` nas chamadas de `prune`. Ex.:

```rust
let kept = prune(&[LogLine::Decision(entry(now - month - 1, "local")),
                   LogLine::Decision(entry(now - 10, "cloud"))], now, month);
assert_eq!(kept.len(), 1);
assert!(matches!(&kept[0], LogLine::Decision(d) if d.dest == "cloud"));
```

- [ ] **Step 5: Commit**

```bash
git add src/route_log.rs
git commit -m "feat(route_log): two-line log model (decision+outcome) with rid + retrocompat"
```

---

### Task 2: Módulo `pricing`

**Files:**
- Create: `src/pricing.rs`
- Modify: `src/lib.rs` (declarar `pub mod pricing;`)

**Interfaces:**
- Produces:
  - `pub struct Price { pub in_per_1m: f64, pub out_per_1m: f64 }` com `pub fn cost(&self, prompt_tok: u64, completion_tok: u64) -> f64`.
  - `pub fn price_for(model_id: &str) -> Price`.

- [ ] **Step 1: Cria o teste**

Criar `src/pricing.rs` com:

```rust
//! Static, code-maintained cloud model pricing ($/1M tokens). No user editing.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    pub in_per_1m: f64,
    pub out_per_1m: f64,
}

impl Price {
    /// USD cost for a call of the given token counts.
    pub fn cost(&self, prompt_tok: u64, completion_tok: u64) -> f64 {
        (prompt_tok as f64 * self.in_per_1m + completion_tok as f64 * self.out_per_1m) / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_match_and_unknown_falls_back() {
        assert_eq!(price_for("claude-opus-4-8"), Price { in_per_1m: 15.0, out_per_1m: 75.0 });
        assert_eq!(price_for("claude-sonnet-4-6"), Price { in_per_1m: 3.0, out_per_1m: 15.0 });
        assert_eq!(price_for("gpt-5"), Price { in_per_1m: 1.25, out_per_1m: 10.0 });
        // unknown → fallback
        assert_eq!(price_for("some-random-model"), FALLBACK);
    }

    #[test]
    fn cost_arithmetic() {
        let p = Price { in_per_1m: 3.0, out_per_1m: 15.0 };
        // 1M prompt @3 + 1M completion @15 = 18
        assert!((p.cost(1_000_000, 1_000_000) - 18.0).abs() < 1e-9);
    }
}
```

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib pricing::`
Expected: FAIL — `price_for`/`FALLBACK` não definidos.

- [ ] **Step 3: Implementa a tabela**

Adicionar em `src/pricing.rs` (antes do bloco de testes):

```rust
/// Estimated price for an unrecognized model (mid-tier). Documented estimate.
pub const FALLBACK: Price = Price { in_per_1m: 3.0, out_per_1m: 15.0 };

/// (substring of the model id, $/1M input, $/1M output). First match wins, so
/// order more-specific patterns before broader ones. Estimates as of 2026-07;
/// update as providers change pricing.
const MODEL_PRICES: &[(&str, f64, f64)] = &[
    ("claude-opus-4", 15.0, 75.0),
    ("claude-sonnet-4", 3.0, 15.0),
    ("claude-haiku-4", 1.0, 5.0),
    ("claude-3-5-haiku", 0.8, 4.0),
    ("claude-3-opus", 15.0, 75.0),
    ("gpt-5", 1.25, 10.0),
    ("gpt-4o-mini", 0.15, 0.6),
    ("gpt-4o", 2.5, 10.0),
    ("gpt-4.1", 2.0, 8.0),
    ("o3", 2.0, 8.0),
    ("o1", 15.0, 60.0),
];

/// Price for a model id, matched case-insensitively by substring; `FALLBACK`
/// when nothing matches.
pub fn price_for(model_id: &str) -> Price {
    let m = model_id.to_lowercase();
    for (pat, in_p, out_p) in MODEL_PRICES {
        if m.contains(pat) {
            return Price { in_per_1m: *in_p, out_per_1m: *out_p };
        }
    }
    FALLBACK
}
```

Em `src/lib.rs`, adicionar junto às declarações de módulo (perto de `pub mod route_log;`):

```rust
pub mod pricing;
```

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib pricing::`
Expected: PASS (2 testes).

- [ ] **Step 5: Commit**

```bash
git add src/pricing.rs src/lib.rs
git commit -m "feat(pricing): static cloud model price table + price_for"
```

---

### Task 3: `build_dashboard` — join, custo, latência, janelas

**Files:**
- Modify: `src/route_log.rs`
- Modify: `src/server.rs:775-776` (`handle_dashboard` — chamador)

**Interfaces:**
- Consumes: `LogLine`, `OutcomeEntry` (Task 1); `pricing` não é usado aqui (o custo já vem calculado no `OutcomeEntry.cost_saved_usd`, gravado na Task 4).
- Produces:
  - `Bucket` ganha `cost_saved_usd: f64`.
  - `pub struct RouteLatency { pub avg_ttft_ms: u64, pub avg_tok_s: f64, pub n: u64 }`
  - `pub struct FallbackWindow { pub kind: String, pub reason: String, pub start_ts: i64, pub end_ts: i64, pub count: u64 }`
  - `pub struct RecentRow { #[serde(flatten)] entry: RouteEntry, completion_tok: Option<u64>, ttft_ms: Option<u64>, gen_ms: Option<u64>, cost_saved_usd: f64 }`
  - `Dashboard` ganha `local_latency: RouteLatency`, `cloud_latency: RouteLatency`, `windows: Vec<FallbackWindow>`; `recent` passa a `Vec<RecentRow>`.
  - `pub fn build_dashboard(entries: &[LogLine], now: i64, recent_n: usize) -> Dashboard` (assinatura muda o tipo de `entries`).

- [ ] **Step 1: Atualiza os testes existentes + novo teste de custo/latência**

No `mod tests` de `src/route_log.rs`, os helpers `e(...)`/`entry(...)` retornam `RouteEntry`. Trocar `dashboard_buckets_and_token_math` e `dashboard_recent_is_capped` pra construir `Vec<LogLine>` e casar os novos tipos. Substituir `dashboard_buckets_and_token_math` por:

```rust
#[test]
fn dashboard_buckets_cost_and_join() {
    let now = 10_000_000i64;
    let mut lines = vec![
        LogLine::Decision(RouteEntry { ts: now - 10, rid: "a".into(), surface: "openai".into(),
            dest: "local".into(), prompt_tok: 100, ..Default::default() }),
        LogLine::Decision(RouteEntry { ts: now - 20, rid: "b".into(), surface: "openai".into(),
            dest: "cloud".into(), prompt_tok: 200, ..Default::default() }),
    ];
    lines.push(LogLine::Outcome(OutcomeEntry { rid: "a".into(), ts: now - 9,
        completion_tok: Some(20), ttft_ms: Some(30), gen_ms: Some(100), cost_saved_usd: 0.42 }));
    lines.push(LogLine::Outcome(OutcomeEntry { rid: "b".into(), ts: now - 19,
        completion_tok: Some(50), ttft_ms: Some(80), gen_ms: Some(500), cost_saved_usd: 0.0 }));
    let d = build_dashboard(&lines, now, 10);
    // hour bucket: local 120 saved, all-cloud 120+250
    assert_eq!(d.hour.tokens_saved, 120);
    assert_eq!(d.hour.tokens_if_all_cloud, 370);
    assert!((d.hour.cost_saved_usd - 0.42).abs() < 1e-9);
    // local latency: 20 tok / 0.1s = 200 tok/s, ttft 30
    assert_eq!(d.local_latency.n, 1);
    assert_eq!(d.local_latency.avg_ttft_ms, 30);
    assert!((d.local_latency.avg_tok_s - 200.0).abs() < 1.0);
    // recent carries the joined completion_tok
    assert_eq!(d.recent.len(), 2);
    let row_a = d.recent.iter().find(|r| r.entry.rid == "a").unwrap();
    assert_eq!(row_a.completion_tok, Some(20));
}

#[test]
fn dashboard_groups_fallback_windows() {
    let now = 1_000_000i64;
    let lines = vec![
        LogLine::Decision(RouteEntry { ts: now - 300, rid: "1".into(), surface: "openai".into(),
            dest: "local".into(), degrade_reason: Some("Quota".into()), ..Default::default() }),
        LogLine::Decision(RouteEntry { ts: now - 290, rid: "2".into(), surface: "openai".into(),
            dest: "local".into(), degrade_reason: Some("Quota".into()), ..Default::default() }),
        LogLine::Decision(RouteEntry { ts: now - 100, rid: "3".into(), surface: "openai".into(),
            dest: "local".into(), reason: Some("BudgetExceeded".into()), ..Default::default() }),
    ];
    let d = build_dashboard(&lines, now, 10);
    assert_eq!(d.windows.len(), 2);
    let prov = d.windows.iter().find(|w| w.kind == "provider").unwrap();
    assert_eq!(prov.count, 2);
    assert_eq!(prov.reason, "Quota");
    let bud = d.windows.iter().find(|w| w.kind == "budget").unwrap();
    assert_eq!(bud.count, 1);
}
```

Em `dashboard_recent_is_capped`, trocar a construção de entries pra `LogLine::Decision(...)` e `d.recent.len()`.

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib route_log::tests::dashboard`
Expected: FAIL — campos/tipos novos ausentes.

- [ ] **Step 3: Implementa**

Adicionar `cost_saved_usd` ao `Bucket`:

```rust
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Bucket {
    pub local_count: u64,
    pub cloud_count: u64,
    pub tokens_saved: u64,
    pub tokens_if_all_cloud: u64,
    /// Σ cost_saved_usd of local requests in this window.
    pub cost_saved_usd: f64,
}
```

Novos tipos + `Dashboard` atualizado:

```rust
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RouteLatency {
    pub avg_ttft_ms: u64,
    pub avg_tok_s: f64,
    pub n: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FallbackWindow {
    /// "budget" | "provider"
    pub kind: String,
    /// human reason (e.g. "Quota", "BudgetExceeded")
    pub reason: String,
    pub start_ts: i64,
    pub end_ts: i64,
    pub count: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecentRow {
    #[serde(flatten)]
    pub entry: RouteEntry,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tok: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ms: Option<u64>,
    pub cost_saved_usd: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dashboard {
    pub hour: Bucket,
    pub day: Bucket,
    pub month: Bucket,
    pub local_latency: RouteLatency,
    pub cloud_latency: RouteLatency,
    pub windows: Vec<FallbackWindow>,
    pub recent: Vec<RecentRow>,
}
```

Trocar `accumulate` pra receber o outcome já juntado:

```rust
fn accumulate(bucket: &mut Bucket, d: &RouteEntry, o: Option<&OutcomeEntry>) {
    let completion = o.and_then(|o| o.completion_tok).unwrap_or(0);
    let toks = d.prompt_tok + completion;
    bucket.tokens_if_all_cloud += toks;
    if d.dest == "local" {
        bucket.local_count += 1;
        bucket.tokens_saved += toks;
        bucket.cost_saved_usd += o.map(|o| o.cost_saved_usd).unwrap_or(0.0);
    } else {
        bucket.cloud_count += 1;
    }
}
```

Reescrever `build_dashboard`:

```rust
pub fn build_dashboard(entries: &[LogLine], now: i64, recent_n: usize) -> Dashboard {
    use std::collections::HashMap;
    // Split decisions and index outcomes by rid.
    let mut decisions: Vec<&RouteEntry> = Vec::new();
    let mut outcomes: HashMap<&str, &OutcomeEntry> = HashMap::new();
    for l in entries {
        match l {
            LogLine::Decision(d) => decisions.push(d),
            LogLine::Outcome(o) => { outcomes.insert(o.rid.as_str(), o); }
        }
    }
    let (mut hour, mut day, mut month) = (Bucket::default(), Bucket::default(), Bucket::default());
    // Latency accumulators (sum, then average).
    let (mut l_ttft, mut l_tps, mut l_n) = (0u64, 0f64, 0u64);
    let (mut c_ttft, mut c_tps, mut c_n) = (0u64, 0f64, 0u64);
    for d in &decisions {
        let o = outcomes.get(d.rid.as_str()).copied();
        let age = now - d.ts;
        if age <= MONTH { accumulate(&mut month, d, o); }
        if age <= DAY { accumulate(&mut day, d, o); }
        if age <= HOUR { accumulate(&mut hour, d, o); }
        if let Some(o) = o {
            let tps = match (o.completion_tok, o.gen_ms) {
                (Some(ct), Some(ms)) if ms > 0 => ct as f64 / (ms as f64 / 1000.0),
                _ => 0.0,
            };
            if d.dest == "local" {
                if let Some(t) = o.ttft_ms { l_ttft += t; }
                l_tps += tps; l_n += 1;
            } else {
                if let Some(t) = o.ttft_ms { c_ttft += t; }
                c_tps += tps; c_n += 1;
            }
        }
    }
    let latency = |ttft: u64, tps: f64, n: u64| RouteLatency {
        avg_ttft_ms: if n > 0 { ttft / n } else { 0 },
        avg_tok_s: if n > 0 { tps / n as f64 } else { 0.0 },
        n,
    };
    // Fallback windows: group contiguous same-reason local decisions.
    let mut sorted: Vec<&RouteEntry> = decisions.iter().filter(|d| {
        d.degrade_reason.is_some() || d.reason.as_deref() == Some("BudgetExceeded")
    }).copied().collect();
    sorted.sort_by_key(|d| d.ts);
    let mut windows: Vec<FallbackWindow> = Vec::new();
    const GAP: i64 = 120; // seconds; same-reason events within this gap merge
    for d in sorted {
        let (kind, reason) = if let Some(r) = &d.degrade_reason {
            ("provider", r.clone())
        } else {
            ("budget", "BudgetExceeded".to_string())
        };
        match windows.last_mut() {
            Some(w) if w.kind == kind && w.reason == reason && d.ts - w.end_ts <= GAP => {
                w.end_ts = d.ts; w.count += 1;
            }
            _ => windows.push(FallbackWindow {
                kind: kind.to_string(), reason, start_ts: d.ts, end_ts: d.ts, count: 1,
            }),
        }
    }
    // Recent rows: newest-first decisions joined to their outcome.
    let mut recent: Vec<RecentRow> = decisions.iter().map(|d| {
        let o = outcomes.get(d.rid.as_str()).copied();
        RecentRow {
            entry: (*d).clone(),
            completion_tok: o.and_then(|o| o.completion_tok),
            ttft_ms: o.and_then(|o| o.ttft_ms),
            gen_ms: o.and_then(|o| o.gen_ms),
            cost_saved_usd: o.map(|o| o.cost_saved_usd).unwrap_or(0.0),
        }
    }).collect();
    recent.sort_by(|a, b| b.entry.ts.cmp(&a.entry.ts));
    recent.truncate(recent_n);
    Dashboard {
        hour, day, month,
        local_latency: latency(l_ttft, l_tps, l_n),
        cloud_latency: latency(c_ttft, c_tps, c_n),
        windows,
        recent,
    }
}
```

Em `src/server.rs`, `handle_dashboard` já chama `read_all()` + `build_dashboard(&entries, ...)`; como `read_all` agora retorna `Vec<LogLine>`, o código compila sem mudança de chamada. Confirmar que compila.

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib route_log::`
Expected: PASS (todos, incluindo os 2 novos). Rodar também `cargo build` pra garantir `server.rs` compila.

- [ ] **Step 5: Commit**

```bash
git add src/route_log.rs src/server.rs
git commit -m "feat(route_log): dashboard join with cost, latency and fallback windows"
```

---

### Task 4: Captura de métricas pós-geração

**Files:**
- Modify: `src/server.rs` (helper `record_outcome` + call-sites)

**Interfaces:**
- Consumes: `route_log::{OutcomeEntry, append_outcome}`, `pricing::price_for`.
- Produces: `fn record_outcome(rid, dest, model, prompt_tok, completion_tok, ttft_ms, gen_ms)` — calcula `cost_saved_usd` (só local) e grava o `OutcomeEntry`.

- [ ] **Step 1: Teste do helper (cálculo de custo)**

Em `src/server.rs`, no `#[cfg(test)] mod tests`, adicionar (o helper é pura decisão de custo; testamos a aritmética via `pricing` já coberto e aqui a gravação):

```rust
#[test]
fn record_outcome_saves_cost_only_for_local() {
    let dir = std::env::temp_dir().join(format!("localllm-ro-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("routing-log.jsonl");
    std::env::set_var("LOCALLLM_ROUTE_LOG", &path);
    record_outcome("r1", "local", Some("claude-sonnet-4-6"), 1_000_000, Some(1_000_000), Some(10), Some(1000));
    record_outcome("r2", "cloud", Some("claude-sonnet-4-6"), 1_000_000, Some(1_000_000), Some(10), Some(1000));
    let lines = crate::route_log::read_all();
    let o1 = lines.iter().find_map(|l| match l {
        crate::route_log::LogLine::Outcome(o) if o.rid == "r1" => Some(o.clone()), _ => None }).unwrap();
    let o2 = lines.iter().find_map(|l| match l {
        crate::route_log::LogLine::Outcome(o) if o.rid == "r2" => Some(o.clone()), _ => None }).unwrap();
    assert!((o1.cost_saved_usd - 18.0).abs() < 1e-6); // sonnet: 3+15
    assert_eq!(o2.cost_saved_usd, 0.0); // cloud saves nothing
    std::env::remove_var("LOCALLLM_ROUTE_LOG");
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib server::tests::record_outcome_saves_cost_only_for_local`
Expected: FAIL — `record_outcome` não existe.

- [ ] **Step 3: Implementa o helper + wiring**

Adicionar em `src/server.rs` (perto de `record_cloud_success`):

```rust
/// Record a request's post-generation outcome. `cost_saved_usd` is what a LOCAL
/// request would have cost on cloud (0 for cloud requests). Best-effort.
fn record_outcome(
    rid: &str,
    dest: &str,
    model: Option<&str>,
    prompt_tok: u64,
    completion_tok: Option<u64>,
    ttft_ms: Option<u64>,
    gen_ms: Option<u64>,
) {
    let cost_saved_usd = if dest == "local" {
        let price = crate::pricing::price_for(model.unwrap_or(""));
        price.cost(prompt_tok, completion_tok.unwrap_or(0))
    } else {
        0.0
    };
    crate::route_log::append_outcome(&crate::route_log::OutcomeEntry {
        rid: rid.to_string(),
        ts: crate::route_log::now_secs(),
        completion_tok,
        ttft_ms,
        gen_ms,
        cost_saved_usd,
    });
}
```

Wiring nos call-sites (o `dest` é `"local"` quando `want_cascade`/local serviu, `"cloud"` quando repassou). Nos caminhos **não-stream e buffered-stream** de cada surface, onde já existe `result.completion_tokens` e `started_at`, adicionar após o `tracing::info! done`:

```rust
record_outcome(&rid, "local", Some(&model), est_prompt_tokens as u64,
    Some(result.completion_tokens as u64), None, Some(started_at.elapsed().as_millis() as u64));
```

Sites exatos (todos usam `result`, `started_at`, `est_prompt_tokens`, `model`, `rid`):
- `handle_oai_chat`: buffered-stream (após `src/server.rs:1307`) e non-stream (bloco `else` após `1368`).
- `handle_anth_messages`: bloco equivalente non-stream/buffered.
- `handle_responses` (`handle_oai_responses`): bloco non-stream após `src/server.rs:1482`.

(Nesses caminhos `ttft_ms = None` porque a resposta é gerada inteira antes de emitir.)

No caminho **streaming incremental** de `handle_oai_chat` (bloco `if stream_flag` em `1319`), medir TTFT no 1º delta e completion por contagem de deltas. Trocar o `.scan(false, ...)` por `.scan((false, 0u64, None::<Instant>), ...)` acumulando, e gravar no `delta.done`:

```rust
// state tuple: (started, completion_tok_est, first_delta_at)
let sse_stream = delta_stream
    .scan((false, 0u64, None::<Instant>), move |st, result| {
        let (ref mut started, ref mut ctok, ref mut first_at) = *st;
        let was_started = *started;
        let event: Result<Event, Infallible> = match result {
            Ok(delta) => {
                if first_at.is_none() { *first_at = Some(Instant::now()); }
                if let Some(t) = &delta.text {
                    *ctok += crate::route::estimate_tokens(t) as u64;
                }
                if delta.done {
                    let ttft = first_at.map(|f| (f - started_at).as_millis() as u64);
                    record_outcome(&rid_stream, "local", Some(&model_clone),
                        est_prompt_tokens as u64, Some(*ctok), ttft,
                        Some(started_at.elapsed().as_millis() as u64));
                }
                let line = crate::api::openai::stream_chunk(&delta, &id_clone, &model_clone, was_started);
                *started = true;
                let data = line.strip_prefix("data: ").unwrap_or(&line);
                Ok(Event::default().data(data.to_string()))
            }
            Err(e) => Ok(Event::default().data(format!("[ERROR] {e}"))),
        };
        futures::future::ready(Some(event))
    })
    .chain(futures::stream::once(async { Ok::<Event, Infallible>(Event::default().data("[DONE]")) }));
```

Precisa mover `est_prompt_tokens` e `model` pra dentro do closure: adicionar `let est_prompt_tokens = est_prompt_tokens;` antes e `move` já captura. Garantir `rid_stream`/`model_clone` (já existem no bloco).

Se `crate::route::estimate_tokens` não existir como pública, usar `crate::route::estimate_prompt_tokens`-equivalente; verificar em `src/route/mod.rs` e expor `pub fn estimate_tokens(text: &str) -> usize` (wrapper simples: `text.split_whitespace().count() * 4 / 3` ou o que o módulo já usa). Se já houver função de estimativa por string, reusar.

(Streaming das outras surfaces: mesmo padrão, opcional nesta task — anotar como follow-up se o tempo apertar; os caminhos non-stream/buffered já cobrem a maioria.)

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib server::tests::record_outcome_saves_cost_only_for_local` e `cargo build`
Expected: PASS + build ok.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): record post-generation outcome (tokens, latency, cost saved)"
```

---

### Task 5: Logar o fallback do provider com o motivo

**Files:**
- Modify: `src/server.rs` (`handle_degrade` + call-sites que caem pra local)

**Interfaces:**
- Consumes: `route_log::{RouteEntry, append}`.
- Produces: quando um request cloud degrada pra local, grava uma `RouteEntry` (dest="local") com `degrade_reason = <DegradeReason>` e `rid`/`model`/`prompt_tok`.

- [ ] **Step 1: Teste**

Em `src/server.rs` tests, testar a função pura de mapear `DegradeReason`→string, que a Task usa:

```rust
#[test]
fn degrade_reason_label_is_stable() {
    assert_eq!(degrade_reason_label(crate::usage::DegradeReason::Quota), "Quota");
    assert_eq!(degrade_reason_label(crate::usage::DegradeReason::Auth), "Auth");
    assert_eq!(degrade_reason_label(crate::usage::DegradeReason::ServerError), "ServerError");
    assert_eq!(degrade_reason_label(crate::usage::DegradeReason::Offline), "Offline");
}
```

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib server::tests::degrade_reason_label_is_stable`
Expected: FAIL — `degrade_reason_label` não existe.

- [ ] **Step 3: Implementa**

Adicionar em `src/server.rs`:

```rust
/// Stable short label for a degrade reason, logged so the dashboard can group
/// provider-fallback windows.
fn degrade_reason_label(r: crate::usage::DegradeReason) -> &'static str {
    match r {
        crate::usage::DegradeReason::Auth => "Auth",
        crate::usage::DegradeReason::Quota => "Quota",
        crate::usage::DegradeReason::ServerError => "ServerError",
        crate::usage::DegradeReason::Offline => "Offline",
    }
}

/// Log that a cloud request fell back to local with the given provider reason.
fn log_degrade_fallback(rid: &str, surface: &str, model: Option<&str>, prompt_tok: u64,
    reason: crate::usage::DegradeReason) {
    crate::route_log::append(&crate::route_log::RouteEntry {
        ts: crate::route_log::now_secs(),
        rid: rid.to_string(),
        surface: surface.to_string(),
        dest: "local".to_string(),
        prompt_tok,
        model: model.map(|m| m.to_string()),
        degrade_reason: Some(degrade_reason_label(reason).to_string()),
        ..Default::default()
    });
}
```

Nos call-sites onde o `Degrade` cai pra local (ex. `handle_oai_chat` `src/server.rs:1268` "cloud degraded → serving local", e equivalentes nas 3 surfaces + no `cascade_or_result`/`handle_degrade` que retorna `None`), chamar antes de servir local:

```rust
log_degrade_fallback(&rid, "openai", Some(&model), est_prompt_tokens as u64, d);
```

(`d` é o `DegradeReason` do `ForwardOutcome::Degrade(d)`.) Fazer o mesmo em `handle_anth_messages` (`"anthropic"`) e `handle_oai_responses` (`"openai-responses"`).

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib server::tests::degrade_reason_label_is_stable` + `cargo build`
Expected: PASS + build ok.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): log provider-degrade fallback with reason for dashboard windows"
```

---

### Task 6: Módulo `budget`

**Files:**
- Create: `src/budget.rs`
- Modify: `src/lib.rs` (`pub mod budget;`)

**Interfaces:**
- Consumes: `route_log::{LogLine}`.
- Produces:
  - `pub struct Budget` com:
    - `pub fn new() -> Self`
    - `pub fn note_cloud_cost(&self, usd: f64)`
    - `pub fn spent_today(&self, now: i64) -> f64`
    - `pub fn is_over(&self, now: i64, limit: f64) -> bool`
    - `pub fn seed_from_log(&self, entries: &[LogLine], now: i64)`
  - `pub fn day_index(ts: i64) -> i64` (dia-UTC).

- [ ] **Step 1: Teste**

Criar `src/budget.rs`:

```rust
//! Daily cloud-spend tracker for the budget cap. Day = UTC day (no tz dep).

use std::sync::Mutex;

/// UTC day bucket for a unix-seconds timestamp.
pub fn day_index(ts: i64) -> i64 {
    ts.div_euclid(86_400)
}

#[derive(Debug, Default)]
struct DaySpend {
    day: i64,
    spent: f64,
}

#[derive(Debug, Default)]
pub struct Budget {
    inner: Mutex<DaySpend>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_log::{LogLine, RouteEntry, OutcomeEntry};

    #[test]
    fn accumulates_and_resets_on_new_day() {
        let b = Budget::new();
        let t0 = 100_000i64; // some day
        b.note_cloud_cost_at(1.5, t0);
        b.note_cloud_cost_at(1.0, t0 + 10);
        assert!((b.spent_today(t0 + 20) - 2.5).abs() < 1e-9);
        // next day → resets
        let t1 = t0 + 86_400;
        assert!((b.spent_today(t1) - 0.0).abs() < 1e-9);
        assert!(b.is_over(t0 + 20, 2.0));
        assert!(!b.is_over(t0 + 20, 3.0));
        assert!(!b.is_over(t0 + 20, 0.0)); // 0 limit = disabled
    }

    #[test]
    fn seeds_today_cloud_cost_from_log() {
        let b = Budget::new();
        let now = 200_000i64;
        // one cloud decision today with a cost; one local (ignored); one old cloud
        let lines = vec![
            LogLine::Decision(RouteEntry { ts: now - 10, rid: "c1".into(),
                dest: "cloud".into(), prompt_tok: 1_000_000, model: Some("claude-sonnet-4-6".into()),
                ..Default::default() }),
            LogLine::Outcome(OutcomeEntry { rid: "c1".into(), ts: now - 9,
                completion_tok: Some(0), ..Default::default() }),
            LogLine::Decision(RouteEntry { ts: now - 86_400 - 10, rid: "old".into(),
                dest: "cloud".into(), prompt_tok: 1_000_000, model: Some("claude-sonnet-4-6".into()),
                ..Default::default() }),
        ];
        b.seed_from_log(&lines, now);
        // c1: 1M prompt @ sonnet $3 = $3.0 (completion 0)
        assert!((b.spent_today(now) - 3.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib budget::`
Expected: FAIL — métodos não existem.

- [ ] **Step 3: Implementa**

Adicionar em `src/budget.rs` (antes dos testes):

```rust
impl Budget {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add cost to today's total, resetting first if the UTC day rolled over.
    pub fn note_cloud_cost_at(&self, usd: f64, now: i64) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let today = day_index(now);
        if g.day != today {
            g.day = today;
            g.spent = 0.0;
        }
        g.spent += usd;
    }

    /// Convenience: cost at the current wall clock.
    pub fn note_cloud_cost(&self, usd: f64) {
        self.note_cloud_cost_at(usd, crate::route_log::now_secs());
    }

    /// Today's spend (0 if the stored day is stale).
    pub fn spent_today(&self, now: i64) -> f64 {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.day == day_index(now) { g.spent } else { 0.0 }
    }

    /// Whether today's spend has reached `limit`. `limit <= 0` means disabled.
    pub fn is_over(&self, now: i64, limit: f64) -> bool {
        limit > 0.0 && self.spent_today(now) >= limit
    }

    /// Seed today's spend by summing today's cloud costs from the log. Cloud
    /// cost = price_for(model).cost(prompt, completion) using the joined outcome.
    pub fn seed_from_log(&self, entries: &[crate::route_log::LogLine], now: i64) {
        use std::collections::HashMap;
        use crate::route_log::LogLine;
        let mut outcomes: HashMap<&str, u64> = HashMap::new();
        for l in entries {
            if let LogLine::Outcome(o) = l {
                outcomes.insert(o.rid.as_str(), o.completion_tok.unwrap_or(0));
            }
        }
        let today = day_index(now);
        let mut total = 0.0;
        for l in entries {
            if let LogLine::Decision(d) = l {
                if d.dest == "cloud" && day_index(d.ts) == today {
                    let completion = outcomes.get(d.rid.as_str()).copied().unwrap_or(0);
                    let price = crate::pricing::price_for(d.model.as_deref().unwrap_or(""));
                    total += price.cost(d.prompt_tok, completion);
                }
            }
        }
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.day = today;
        g.spent = total;
    }
}
```

Em `src/lib.rs`: `pub mod budget;`.

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib budget::`
Expected: PASS (2 testes).

- [ ] **Step 5: Commit**

```bash
git add src/budget.rs src/lib.rs
git commit -m "feat(budget): daily cloud-spend tracker with log seed"
```

---

### Task 7: Settings do budget

**Files:**
- Modify: `src/settings.rs`

**Interfaces:**
- Produces: `pub fn load_budget() -> (bool, f64)` e `pub fn save_budget(enabled: bool, daily_usd: f64) -> anyhow::Result<()>`.

- [ ] **Step 1: Teste**

Em `settings.rs` tests (dentro do `mod tests`, usando `with_temp_settings`):

```rust
#[test]
fn budget_round_trips_and_preserves_profile() {
    with_temp_settings(|| {
        save_profile(Profile::Balanced).unwrap();
        save_budget(true, 5.0).unwrap();
        assert_eq!(load_budget(), (true, 5.0));
        assert_eq!(load_profile(), Profile::Balanced);
        save_budget(false, 0.0).unwrap();
        assert_eq!(load_budget(), (false, 0.0));
    });
}
```

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib settings::tests::budget_round_trips_and_preserves_profile`
Expected: FAIL.

- [ ] **Step 3: Implementa**

Nos campos do `Settings` struct (perto de `balanced_threshold`):

```rust
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    budget_enabled: bool,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    budget_daily_usd: f64,
```

Adicionar helper de skip (perto do topo do arquivo, junto de outros helpers):

```rust
fn is_zero_f64(v: &f64) -> bool { *v == 0.0 }
```

Funções:

```rust
/// Load (enabled, daily_usd) for the budget cap.
pub fn load_budget() -> (bool, f64) {
    let s = load_settings();
    (s.budget_enabled, s.budget_daily_usd)
}

/// Persist the budget cap config, preserving the rest.
pub fn save_budget(enabled: bool, daily_usd: f64) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.budget_enabled = enabled;
    s.budget_daily_usd = daily_usd.max(0.0);
    save_settings(&s)
}
```

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib settings::tests::budget_round_trips_and_preserves_profile`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/settings.rs
git commit -m "feat(settings): persist budget enabled + daily USD cap"
```

---

### Task 8: Enforce do budget + endpoints + wiring no AppState

**Files:**
- Modify: `src/server.rs` (AppState, router, handlers, enforce, note cost), `src/lib.rs`/`src/tray.rs` (construção do AppState se necessário)

**Interfaces:**
- Consumes: `budget::Budget`, `settings::{load_budget}`, `pricing::price_for`.
- Produces: rotas `GET/POST /admin/budget`; enforce que força local logando `reason="BudgetExceeded"`; `state.budget: Arc<Budget>`.

- [ ] **Step 1: Teste do endpoint GET**

Em `server.rs` tests (seguindo o padrão dos testes de handler existentes que montam o router; reusar o helper de app dos testes atuais). Teste mínimo do POST→GET round-trip:

```rust
#[tokio::test]
async fn budget_get_post_round_trip() {
    let dir = std::env::temp_dir().join(format!("localllm-bud-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("LOCALLLM_SETTINGS", dir.join("settings.json"));
    let app = test_app(); // helper existente que monta o Router com token conhecido
    // POST enable + $5
    let resp = post_json(&app, "/admin/budget", r#"{"enabled":true,"daily_usd":5.0}"#).await;
    assert_eq!(resp.status(), StatusCode::OK);
    // GET reflects it
    let body = get_json(&app, "/admin/budget").await;
    assert_eq!(body["enabled"], true);
    assert_eq!(body["daily_usd"], 5.0);
    std::env::remove_var("LOCALLLM_SETTINGS");
    let _ = std::fs::remove_dir_all(&dir);
}
```

(Se não houver `test_app`/`post_json`/`get_json`, seguir o padrão do teste `set_model_profile_persists_kv_and_history` em `server.rs:1859` pra montar `Router` e enviar `Request` via `tower::ServiceExt::oneshot`. Adaptar nomes.)

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib server::tests::budget_get_post_round_trip`
Expected: FAIL — rota `/admin/budget` não existe.

- [ ] **Step 3: Implementa**

No `AppState` (campo novo):

```rust
    /// Daily cloud-spend tracker for the budget cap.
    pub budget: std::sync::Arc<crate::budget::Budget>,
```

Na construção do `AppState` (no builder do router, onde `usage` é criado): criar `Budget`, semear do log e inserir:

```rust
let budget = std::sync::Arc::new(crate::budget::Budget::new());
budget.seed_from_log(&crate::route_log::read_all(), crate::route_log::now_secs());
```

…e passar `budget` no struct. Se `AppState` é construído em mais de um lugar (server + testes), ajustar todos.

Registrar rotas (perto de `/admin/threshold`):

```rust
        .route("/admin/budget", get(handle_budget_get).post(handle_budget_set))
```

Handlers:

```rust
async fn handle_budget_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) { return resp; }
    let (enabled, daily_usd) = crate::settings::load_budget();
    let spent = state.budget.spent_today(crate::route_log::now_secs());
    let remaining = (daily_usd - spent).max(0.0);
    let over = enabled && daily_usd > 0.0 && spent >= daily_usd;
    Json(json!({ "enabled": enabled, "daily_usd": daily_usd,
        "spent_today": spent, "remaining": remaining, "over": over })).into_response()
}

#[derive(serde::Deserialize)]
struct BudgetSetBody { enabled: Option<bool>, daily_usd: Option<f64> }

async fn handle_budget_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) { return resp; }
    let body: BudgetSetBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    };
    let (cur_en, cur_usd) = crate::settings::load_budget();
    let enabled = body.enabled.unwrap_or(cur_en);
    let daily_usd = body.daily_usd.unwrap_or(cur_usd).max(0.0);
    if let Err(e) = crate::settings::save_budget(enabled, daily_usd) {
        tracing::warn!("failed to persist budget: {e}");
    }
    handle_budget_get(State(state), headers).await
}
```

**Enforce:** em `route_decision` (ou logo após, no início de cada handler antes de repassar cloud), interceptar: se `Decision::Cloud(_)` e budget ligado+estourado → virar local e logar. Implementar dentro de `route_decision`, após `let decision = crate::route::decide(...)` e antes de logar:

```rust
    let (budget_enabled, budget_daily) = crate::settings::load_budget();
    let decision = if matches!(decision, crate::route::Decision::Cloud(_))
        && state.budget.is_over(crate::route_log::now_secs(), budget_daily)
        && budget_enabled
    {
        // Log a forced-local decision line with the budget reason.
        crate::route_log::append(&crate::route_log::RouteEntry {
            ts: crate::route_log::now_secs(), rid: rid.to_string(),
            surface: surface.to_string(), dest: "local".to_string(),
            prompt_tok: prompt_tokens as u64,
            reason: Some("BudgetExceeded".to_string()),
            model: None, ..Default::default()
        });
        crate::route::Decision::Local
    } else {
        decision
    };
```

(Como `route_decision` já grava um `RouteEntry` de decisão logo abaixo, evitar linha duplicada: preferir **setar** `reason="BudgetExceeded"` e `dest="local"` no `RouteEntry` que já é gravado, em vez de gravar uma linha extra. Ajustar a montagem existente do `RouteEntry` em `route_decision` pra usar o `dest`/`reason` derivados da decisão final, e não gravar a linha extra acima. Ver o bloco de log atual em `src/server.rs:217-231`.)

**Note cost no sucesso cloud:** em `record_cloud_success`, adicionar o custo ao budget. Como completion cloud pode faltar, usar prompt-only:

```rust
fn record_cloud_success(state: &AppState, est_prompt_tokens: usize) {
    state.usage.note_success();
    // Charge the budget with a prompt-only estimate (completion often absent on
    // the passthrough cloud stream).
    let (_, _daily) = crate::settings::load_budget();
    // model unknown here → fallback price; acceptable estimate for the cap.
    let cost = crate::pricing::price_for("").cost(est_prompt_tokens as u64, 0);
    state.budget.note_cloud_cost(cost);
    if state.usage.record_cloud_call(est_prompt_tokens, state.cloud_token_alert) {
        crate::usage::notify("localllm — high cloud usage",
            "High cloud token use this session — consider the Save tokens profile.");
    }
}
```

(Nota: `record_cloud_success` não conhece o model id; o custo do budget usa `FALLBACK`. Aceitável — é um teto de segurança, não fatura exata. Se quiser mais preciso, passar `model` como argumento; fora do escopo mínimo desta task.)

- [ ] **Step 4: Roda e verifica passa**

Run: `cargo test --lib server::tests::budget_get_post_round_trip` + `cargo build`
Expected: PASS + build ok.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): daily budget cap — enforce, endpoints, cloud-cost accrual"
```

---

### Task 9: Tela `/budget` + card no nav

**Files:**
- Modify: `src/manager_ui/app.js` (NAV_CARDS + `renderBudget` + rota)
- Modify: `src/manager_ui/style.css` (estilos da tela)

**Interfaces:**
- Consumes: `GET/POST /admin/budget`.

- [ ] **Step 1: Card no nav + rota**

Em `app.js`, adicionar ao `NAV_CARDS`:

```javascript
  { route: "/budget", icon: "$", title: "Budget", desc: "Teto de gasto cloud por dia" },
```

No hash-router (função que mapeia rota→render, perto de `renderTools`/`renderDashboard`; ver `app.js` ~969-986), adicionar o caso `"/budget": renderBudget`.

- [ ] **Step 2: `renderBudget`**

Adicionar em `app.js`:

```javascript
async function renderBudget() {
  currentView = renderBudget;
  setCrumbs([{ label: "Config", onClick: renderConfig }, { label: "Budget" }]);
  const wrap = el("div", "detail");
  wrap.append(el("h2", null, "Budget diário"));
  wrap.append(el("div", "ctxhint", "Teto de gasto cloud por dia (estimado pela tabela de preços). Ao estourar, roteia tudo local até o dia virar (UTC). Zera sozinho no dia seguinte."));

  const panel = el("div", "intpanel");
  const head = el("div", "intpanel-head");
  const txt = el("div", "intpanel-txt");
  txt.append(el("div", "intpanel-title", "Ativar budget"));
  txt.append(el("div", "intpanel-sub", "Quando ligado e o gasto do dia atinge o teto, força local."));
  head.append(txt);
  const sw = el("button", "switch"); sw.setAttribute("role", "switch");
  sw.append(el("span", "switch-knob"));
  head.append(sw);
  panel.append(head);

  const row = el("div", "budget-row");
  const label = el("span", "dt-lbl", "Teto $/dia");
  const input = el("input", "ctxinput"); input.type = "number"; input.min = 0; input.step = 0.5;
  const saveBtn = el("button", "btn primary", "Salvar");
  row.append(label, input, saveBtn);
  panel.append(row);

  const readout = el("div", "budget-readout");
  panel.append(readout);
  wrap.append(panel);
  view.innerHTML = ""; view.append(wrap);

  let enabled = false;
  const paint = (b) => {
    enabled = !!b.enabled;
    sw.classList.toggle("on", enabled);
    sw.setAttribute("aria-checked", enabled ? "true" : "false");
    input.value = b.daily_usd;
    const over = b.over;
    readout.className = "budget-readout" + (over ? " over" : "");
    readout.textContent = `gasto hoje $${(b.spent_today||0).toFixed(2)} / $${(b.daily_usd||0).toFixed(2)} · resta $${(b.remaining||0).toFixed(2)}` + (over ? " · ESTOUROU (local)" : "");
  };
  try { paint(await api("GET", "/admin/budget")); } catch (e) { readout.textContent = e.message; return; }

  sw.onclick = async () => {
    try { paint(await api("POST", "/admin/budget", { enabled: !enabled })); toast("Budget atualizado"); }
    catch (e) { toast(e.message, true); }
  };
  saveBtn.onclick = async () => {
    const v = Number(input.value);
    try { paint(await api("POST", "/admin/budget", { daily_usd: isNaN(v) ? 0 : v })); toast("Teto salvo"); }
    catch (e) { toast(e.message, true); }
  };
}
```

- [ ] **Step 3: CSS**

Em `style.css`, adicionar:

```css
.budget-row { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-top: 16px; }
.budget-readout { margin-top: 14px; font-size: 14px; color: var(--muted); font-variant-numeric: tabular-nums; }
.budget-readout.over { color: var(--red); font-weight: 600; }
```

- [ ] **Step 4: Verificação manual**

Run: `cargo build` (embute os assets). Relançar localllm, abrir Config → Budget, ligar/salvar, confirmar leitura "gasto hoje $X / $Y". (Sem teste automatizado de UI neste repo.)

- [ ] **Step 5: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): budget config screen + nav card"
```

---

### Task 10: Dashboard — $, latência e janelas de fallback

**Files:**
- Modify: `src/manager_ui/app.js` (render do dashboard)
- Modify: `src/manager_ui/style.css`

**Interfaces:**
- Consumes: `Dashboard` novo (`cost_saved_usd`, `local_latency`, `cloud_latency`, `windows`, `recent[].completion_tok/ttft_ms/gen_ms`).

- [ ] **Step 1: $ nos cards + latência**

No render dos cards do dashboard (função que monta `.dash-cards`), adicionar por bucket a linha de $:

```javascript
// dentro do card de economia, após tokens:
card.append(el("div", "dash-card-split", `$ ${(b.cost_saved_usd||0).toFixed(2)} economizado`));
```

Adicionar um card/linha de latência usando `d.local_latency`/`d.cloud_latency`:

```javascript
const lat = el("div", "dash-card");
lat.append(el("div", "dash-card-head", "LATÊNCIA"));
const L = d.local_latency, C = d.cloud_latency;
lat.append(el("div", "dash-card-alt",
  `local: TTFT ${L.avg_ttft_ms}ms · ${L.avg_tok_s.toFixed(0)} tok/s (${L.n})`));
lat.append(el("div", "dash-card-alt",
  `cloud: TTFT ${C.avg_ttft_ms}ms · ${C.avg_tok_s.toFixed(0)} tok/s (${C.n})`));
panel.append(lat);
```

- [ ] **Step 2: Janelas de fallback**

Antes da tabela de decisões, renderizar `d.windows`:

```javascript
if (d.windows && d.windows.length) {
  const box = el("div", "fallback-box");
  box.append(el("div", "fallback-title", "Períodos em local"));
  d.windows.slice().reverse().forEach(w => {
    const when = w.start_ts === w.end_ts
      ? fmtDateTime(w.start_ts)
      : `${fmtDateTime(w.start_ts)} – ${fmtDateTime(w.end_ts)}`;
    const why = w.kind === "budget"
      ? "budget diário estourado"
      : `cloud indisponível (${w.reason})`;
    const line = el("div", "fallback-line" + (w.kind === "budget" ? " budget" : " provider"));
    line.textContent = `${when} · ${why} · ${w.count} pedido(s) atendido(s) local · você seguiu trabalhando`;
    box.append(line);
  });
  panel.append(box);
}
```

- [ ] **Step 3: CSS**

```css
.fallback-box { margin: 16px 0; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; background: var(--panel); }
.fallback-title { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.fallback-line { font-size: 12.5px; line-height: 1.6; padding-left: 12px; border-left: 3px solid var(--line); margin-bottom: 6px; }
.fallback-line.budget { border-left-color: var(--amber); }
.fallback-line.provider { border-left-color: var(--indigo); }
```

- [ ] **Step 4: Verificação manual**

Run: `cargo build`, relançar, abrir Dashboard. Confirmar cards de $, linha de latência e (se houver histórico) as janelas. Gerar um budget-estouro baixo ($0.01) pra ver a janela budget.

- [ ] **Step 5: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): dashboard cost, latency and fallback windows"
```

---

### Task 11: Export do log

**Files:**
- Modify: `src/server.rs` (rota + handler)
- Modify: `src/manager_ui/app.js` (botão)

**Interfaces:**
- Produces: `GET /admin/export?format=jsonl|csv` → arquivo (attachment).

- [ ] **Step 1: Teste do CSV**

Em `server.rs` tests, testar a função pura que serializa CSV:

```rust
#[test]
fn export_csv_has_header_and_rows() {
    let lines = vec![
        crate::route_log::LogLine::Decision(crate::route_log::RouteEntry {
            ts: 10, rid: "r1".into(), surface: "openai".into(), dest: "local".into(),
            prompt_tok: 5, ..Default::default() }),
    ];
    let csv = export_csv(&lines);
    assert!(csv.starts_with("kind,rid,ts,surface,dest,reason,degrade_reason,model,prompt_tok"));
    assert!(csv.contains("d,r1,10,openai,local"));
}
```

- [ ] **Step 2: Roda e verifica falha**

Run: `cargo test --lib server::tests::export_csv_has_header_and_rows`
Expected: FAIL — `export_csv` não existe.

- [ ] **Step 3: Implementa**

```rust
/// Flatten the log to CSV (decisions + outcomes, one row each).
fn export_csv(lines: &[crate::route_log::LogLine]) -> String {
    use crate::route_log::LogLine;
    let mut out = String::from(
        "kind,rid,ts,surface,dest,reason,degrade_reason,model,prompt_tok,completion_tok,ttft_ms,gen_ms,cost_saved_usd\n");
    let esc = |s: &str| if s.contains(',') { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() };
    for l in lines {
        match l {
            LogLine::Decision(d) => out.push_str(&format!(
                "d,{},{},{},{},{},{},{},{},,,,\n",
                esc(&d.rid), d.ts, esc(&d.surface), esc(&d.dest),
                esc(d.reason.as_deref().unwrap_or("")),
                esc(d.degrade_reason.as_deref().unwrap_or("")),
                esc(d.model.as_deref().unwrap_or("")), d.prompt_tok)),
            LogLine::Outcome(o) => out.push_str(&format!(
                "o,{},{},,,,,,,{},{},{},{}\n",
                esc(&o.rid), o.ts,
                o.completion_tok.map(|v| v.to_string()).unwrap_or_default(),
                o.ttft_ms.map(|v| v.to_string()).unwrap_or_default(),
                o.gen_ms.map(|v| v.to_string()).unwrap_or_default(),
                o.cost_saved_usd)),
        }
    }
    out
}

#[derive(serde::Deserialize)]
struct ExportQuery { format: Option<String> }

async fn handle_export(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Query(q): axum::extract::Query<ExportQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) { return resp; }
    let lines = crate::route_log::read_all();
    let (body, ct, fname) = if q.format.as_deref() == Some("csv") {
        (export_csv(&lines), "text/csv", "routing-log.csv")
    } else {
        let jsonl = lines.iter().filter_map(|l| serde_json::to_string(l).ok())
            .collect::<Vec<_>>().join("\n");
        (jsonl, "application/x-ndjson", "routing-log.jsonl")
    };
    (
        [(axum::http::header::CONTENT_TYPE, ct),
         (axum::http::header::CONTENT_DISPOSITION,
          Box::leak(format!("attachment; filename=\"{fname}\"").into_boxed_str()) as &str)],
        body,
    ).into_response()
}
```

Registrar rota: `.route("/admin/export", get(handle_export))`.

- [ ] **Step 4: Botão no dashboard**

Em `app.js`, no cabeçalho do dashboard (`.dash-bar-actions`), adicionar:

```javascript
const exp = el("button", "btn", "Exportar CSV");
exp.onclick = () => { window.open(`/admin/export?format=csv&token=${encodeURIComponent(adminToken())}`, "_blank"); };
actions.append(exp);
```

(Verificar como o token é lido no front — se `api()` usa `Authorization: Bearer`, o download via `window.open` precisa do token na query. Confirmar que `check_admin` aceita `?token=`; se não, estender `check_admin` pra também ler `token` da query — mudança pequena. Se já aceitar, usar `adminToken()` helper existente.)

- [ ] **Step 5: Commit**

```bash
git add src/server.rs src/manager_ui/app.js
git commit -m "feat(export): GET /admin/export csv|jsonl + dashboard button"
```

---

### Task 12 (opcional/stretch): `/metrics` Prometheus

**Files:**
- Modify: `src/server.rs`

**Interfaces:**
- Produces: `GET /metrics` (texto Prometheus). Só implementar se sobrar tempo; não bloqueia a fase.

- [ ] **Step 1: Teste**

```rust
#[test]
fn metrics_text_has_counters() {
    let now = 1_000i64;
    let lines = vec![
        crate::route_log::LogLine::Decision(crate::route_log::RouteEntry {
            ts: now, rid: "r".into(), dest: "local".into(), prompt_tok: 10, ..Default::default() }),
        crate::route_log::LogLine::Outcome(crate::route_log::OutcomeEntry {
            rid: "r".into(), ts: now, cost_saved_usd: 1.5, ..Default::default() }),
    ];
    let text = metrics_text(&lines, now);
    assert!(text.contains("localllm_requests_total{dest=\"local\"} 1"));
    assert!(text.contains("localllm_cost_saved_usd_total"));
}
```

- [ ] **Step 2: Roda e verifica falha** — `cargo test --lib server::tests::metrics_text_has_counters` → FAIL.

- [ ] **Step 3: Implementa** — `fn metrics_text(lines, now) -> String` construindo contadores a partir de `build_dashboard(lines, now, 0)` (month bucket) + handler `GET /metrics` (sem token, ou token-guarded conforme preferência; padrão: token-guarded pra não vazar dados).

```rust
fn metrics_text(lines: &[crate::route_log::LogLine], now: i64) -> String {
    let d = crate::route_log::build_dashboard(lines, now, 0);
    format!(
        "# HELP localllm_requests_total Requests by destination (30d)\n\
         # TYPE localllm_requests_total counter\n\
         localllm_requests_total{{dest=\"local\"}} {}\n\
         localllm_requests_total{{dest=\"cloud\"}} {}\n\
         # HELP localllm_cost_saved_usd_total USD saved by local routing (30d)\n\
         # TYPE localllm_cost_saved_usd_total counter\n\
         localllm_cost_saved_usd_total {:.6}\n\
         # HELP localllm_ttft_ms Average TTFT by route\n\
         localllm_ttft_ms{{dest=\"local\"}} {}\n\
         localllm_ttft_ms{{dest=\"cloud\"}} {}\n",
        d.month.local_count, d.month.cloud_count, d.month.cost_saved_usd,
        d.local_latency.avg_ttft_ms, d.cloud_latency.avg_ttft_ms)
}
```

- [ ] **Step 4: Roda e verifica passa** — PASS + `cargo build`.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs
git commit -m "feat(metrics): optional Prometheus /metrics endpoint"
```

---

## Self-Review

**Cobertura do spec:**
- Seção 1 (dados/captura) → Tasks 1, 3, 4. ✓
- Seção 2 (preços/$) → Tasks 2, 3, 4. ✓
- Seção 3 (budget) → Tasks 6, 7, 8, 9. ✓
- Seção 4 (dashboard viz) → Tasks 3, 5, 10. ✓
- Seção 5 (export + /metrics opcional) → Tasks 11, 12. ✓

**Consistência de tipos:** `LogLine`/`OutcomeEntry`/`RouteEntry`/`Bucket.cost_saved_usd`/`RouteLatency`/`FallbackWindow`/`RecentRow` definidos na Task 1/3 e consumidos coerentemente em 4/6/8/11/12. `record_outcome`, `price_for`, `Budget::{note_cloud_cost,is_over,spent_today,seed_from_log}`, `load_budget/save_budget` batem entre tasks.

**Notas de risco reafirmadas:**
- Budget usa **dia-UTC** (sem dep de tz).
- Custo cloud no budget usa `FALLBACK` (model id desconhecido em `record_cloud_success`) — teto de segurança, não fatura exata.
- Streaming de completion tokens é **estimado** (`estimate_tokens`); TTFT exato só no caminho stream.
- `check_admin` via `?token=` pro download pode exigir extensão pequena (Task 11 Step 4).
