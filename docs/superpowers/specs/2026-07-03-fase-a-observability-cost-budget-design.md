# Fase A — Observabilidade, Custo & Budget Diário

**Data:** 2026-07-03
**Status:** Aprovado (brainstorming) — pronto pra plano de implementação

## Contexto

`localllm` roteia requests de clientes (Claude Code / Codex) entre modelo local e
cloud. Hoje o `route_log` grava um `RouteEntry` **no momento da decisão**, antes da
geração — então `completion_tok`, latência e custo **não são capturados**. O path
cloud repassa bytes crus. A infra de *degrade* já existe (`cloud::forward` →
`ForwardOutcome::Degrade` → `handle_degrade`: tenta cloud, no limite do provider
cai pra local e notifica uma vez via gate `Usage::note_degrade`), mas esses eventos
não são logados nem visualizados.

Esta é a **Fase A** de um roadmap de 5 fases (A→E). Escopo: observabilidade + custo
+ budget diário. Não toca no núcleo de decisão de dificuldade (Fase D) além de um
novo motivo de rota (`BudgetExceeded`).

## Objetivos

1. **$ economizado** — quanto se poupou roteando local, via tabela de preços
   embutida e automática (sem edição manual).
2. **Budget diário** — teto de gasto cloud em $/dia, com toggle liga/desliga, em
   **tela nova** de Config. Ao estourar, força local; reseta e volta cloud no dia
   seguinte.
3. **Latência** — TTFT e tokens/s por rota (local vs cloud) no dashboard.
4. **Explicar o "porquê local"** no dashboard — dois motivos distintos de fallback:
   (a) budget diário estourado; (b) cloud sem tokens (limite do provider).
5. **Export** — baixar o `route_log` (CSV/JSONL). Prometheus `/metrics` opcional.

## Fora de escopo (Fase A)

- Circuit breaker avançado — backoff, probe pra re-tentar cloud antes do provider
  re-liberar. Fica na **Fase C (#10)**. O básico (tenta cloud → fallback local →
  notifica uma vez) já roda.
- Preço editável pelo usuário. A tabela é mantida no código; usuário não edita.
- Custo cloud **gasto** exato exige parsear `usage` do SSE cloud — é secundário. O
  foco é **$ economizado**, calculado com completion tokens **locais**.

---

## Arquitetura

### Seção 1 — Modelo de dados + captura de métricas

**Abordagem escolhida: duas linhas por request.**

- `RouteEntry` (linha de **decisão**, gravada na hora como hoje) ganha campos:
  - `rid: String` — id de correlação (já existe nos logs, ex. `req-1a2b3c4d`;
    passa a ser persistido).
  - `model: Option<String>` — id que o cliente pediu (o modelo cloud que *teria*
    atendido). Barato, conhecido na decisão.
  - `degrade_reason: Option<String>` — preenchido quando este request virou local
    por degrade do provider (Quota/Auth/ServerError/Offline). Ver Seção 4.
- Novo `OutcomeEntry` (linha de **desfecho**, gravada quando a geração termina):
  ```
  { rid, completion_tok, ttft_ms, gen_ms, cost_saved_usd }
  ```
  Serializado no mesmo JSONL, distinguido por um campo `kind` ou pela presença de
  `rid` + ausência dos campos de decisão. **Decisão de formato:** adicionar
  `#[serde(tag = "kind")]`-style não; em vez disso, o `route_log` passa a ler um
  enum `LogLine { Decision(RouteEntry), Outcome(OutcomeEntry) }` com `kind`
  explícito (`"d"` | `"o"`) pra parse robusto. Linhas antigas sem `kind` são
  tratadas como `Decision` (retrocompat).

**Captura:**
- **Local:** o handler de `generate`/`generate_stream` já tem o `ChatResult`
  (`completion_tokens`) ou conta deltas no stream. Cronometra 1º delta (TTFT) e fim
  (gen_ms). Grava `OutcomeEntry` ao concluir.
- **Cloud:** envolve o stream repassado pra timestampar 1º byte (TTFT) e ler
  `usage` do chunk final quando presente (completion tokens; senão fica `None`).
  `$ economizado` **não** depende disso — só de requests locais.
- Robustez: a linha de decisão sempre é gravada (como hoje). Se o request morre no
  meio, fica sem `OutcomeEntry` — o dashboard trata outcome ausente como desconhecido.

### Seção 2 — Preços + $ economizado

- Novo módulo **`src/pricing.rs`**:
  - `MODEL_PRICES: &[(&str, f64, f64)]` — (padrão de model id, $/1M input, $/1M
    output) pros modelos comuns Anthropic/OpenAI.
  - `pub fn price_for(model_id: &str) -> Price` — casa por prefixo/substring; se
    nada casa, retorna um **fallback estimado** (constante documentada, ex. média
    de mid-tier).
  - `pub struct Price { in_per_1m: f64, out_per_1m: f64 }` com
    `fn cost(&self, prompt_tok, completion_tok) -> f64`.
- **$ economizado** de um request local =
  `price_for(model).cost(prompt_tok, completion_tok)` — o que aquele request
  *teria* custado na cloud. Somado no `OutcomeEntry.cost_saved_usd`.
- `Bucket` (rollups hora/dia/mês) ganha `cost_saved_usd: f64`, acumulado a partir
  dos outcomes juntados por `rid` à decisão (só `dest == "local"`).

### Seção 3 — Budget diário

- **Settings** (`settings.rs`): `budget_enabled: bool`, `budget_daily_usd: f64`
  (ambos `#[serde(default)]`). Load/save preservando o resto.
- **Componente `Budget`** (novo módulo `src/budget.rs`, estilo `Usage`):
  - Gasto cloud de **hoje** (chave de dia local `YYYY-MM-DD`) + total, atômico/lock
    leve.
  - `note_cloud_cost(usd)` — soma ao gasto de hoje; se o dia virou, zera antes.
  - `spent_today() -> f64`, `is_over(limit) -> bool`.
  - **Seed no boot:** soma o custo cloud de hoje a partir do `route_log`
    (decisão+outcome de `dest == cloud` com `ts` de hoje) pra sobreviver a restart.
- **Enforce (roteamento):** antes de repassar pra cloud, se
  `budget_enabled && budget.is_over(budget_daily_usd)` → força local com novo
  `RouteReason::BudgetExceeded`. A linha de decisão grava esse reason.
- **Acúmulo:** a cada completion cloud bem-sucedida, `budget.note_cloud_cost(custo)`
  (path pós-geração; custo = preço do model × tokens do request cloud, prompt +
  completion se disponível, senão só prompt).
- **Endpoints:** `GET/POST /admin/budget` →
  `{ enabled, daily_usd, spent_today, remaining, over }`.
- **Tela nova `/budget`:**
  - Novo card no `NAV_CARDS` (`app.js`): `{ route:"/budget", icon:"$", title:"Budget",
    desc:"Teto de gasto cloud por dia" }`.
  - Toggle liga/desliga (reusa `.switch`).
  - Input `$/dia` (reusa estilo alinhado dos inputs de profile).
  - Leitura ao vivo: *"gasto hoje $X / $Y · resta $Z"*, badge vermelho quando `over`.

### Seção 4 — Dashboard: 2 motivos + $ + latência

- **Cards de $** (hora/dia/mês): `$ economizado` ao lado dos tokens. Com budget
  ligado, mostra *"gasto hoje $X / $Y"*.
- **Latência** por rota: TTFT médio + tok/s (local vs cloud), agregados dos outcomes.
- **Janelas de fallback** — banner/timeline, dois motivos derivados do log:
  - **Budget:** entradas `RouteReason::BudgetExceeded` →
    *"budget diário estourado às HH:MM · N pedidos atendidos local · você seguiu
    trabalhando"*.
  - **Provider:** requer **logar o degrade** — hoje o fallback-to-local do
    `handle_degrade` não grava o motivo. Passa a gravar `RouteEntry` (dest=local)
    com `degrade_reason` = `DegradeReason` (Quota/Auth/ServerError/Offline). Pinta
    *"cloud indisponível HH:MM–HH:MM (quota) · N local"*.
  - Janelas contíguas do mesmo motivo são agrupadas.
- **Agregação no backend** (`build_dashboard`): números de latência (TTFT/tok/s) e
  as janelas de fallback são computados no backend e adicionados ao `Dashboard`
  (`Bucket` + uma lista de janelas). O frontend só pinta.

### Seção 5 — Export

- **`GET /admin/export?format=jsonl|csv`** (token-guarded) — baixa o `route_log`
  com decisão+outcome juntados por `rid`. `Content-Disposition: attachment`. Botão
  "Exportar" no Dashboard.
- **Opcional/stretch:** `GET /metrics` formato Prometheus (contadores local/cloud,
  `$ economizado`, TTFT médio). Só se sobrar tempo; não bloqueia a fase.

---

## Endpoints novos

| Método | Rota | Corpo/Query | Retorno |
|--------|------|-------------|---------|
| GET | `/admin/budget` | — | `{enabled, daily_usd, spent_today, remaining, over}` |
| POST | `/admin/budget` | `{enabled?, daily_usd?}` | idem, atualizado |
| GET | `/admin/export` | `?format=jsonl\|csv` | arquivo (attachment) |
| GET | `/metrics` (opcional) | — | texto Prometheus |

## Settings novos

`budget_enabled: bool`, `budget_daily_usd: f64` — ambos default off/0, skip
serialize quando no default.

## Testes

- `pricing`: `price_for` casa modelos conhecidos + fallback; `cost` aritmética.
- `route_log`: parse retrocompat (linha antiga sem `kind` = Decision); join
  decisão+outcome por `rid`; `cost_saved_usd` acumulado só em local.
- `budget`: `note_cloud_cost` soma e zera na virada do dia; `is_over`; seed do log.
- `settings`: round-trip `budget_enabled`/`budget_daily_usd` preservando o resto.
- Dashboard: agregação de latência e agrupamento de janelas de fallback.
- Enforce: budget ligado + over → decisão vira Local(`BudgetExceeded`).

## Sequência de implementação (dentro da Fase A)

1. **Base de dados/métricas** (Seção 1): `rid`/`model`/`degrade_reason` em
   `RouteEntry`, `OutcomeEntry`, `LogLine` enum, captura local+cloud. — desbloqueia
   o resto.
2. **Preços + $** (Seção 2): `pricing.rs`, `cost_saved_usd` nos buckets.
3. **Budget** (Seção 3): settings, componente, enforce, endpoints, tela nova.
4. **Dashboard viz** (Seção 4): cards $, latência, janelas de fallback + log de
   degrade.
5. **Export** (Seção 5): `/admin/export`; `/metrics` se sobrar.

## Riscos / notas

- **Retrocompat do log:** linhas antigas sem `kind` devem parsear como `Decision`.
- **Custo de leitura no boot** pro seed do budget: uma leitura do log — aceitável.
- **Completion tokens cloud** podem faltar (provider não manda `usage`): tratar
  `None`; budget cai pra custo de prompt-only nesse caso.
- **Fuso do "dia":** budget reseta em dia **local** da máquina.
