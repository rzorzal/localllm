# Fase D3 — Multi-Local-Model Tiering (DEFERRED / NOT BUILT)

**Date:** 2026-07-07
**Status:** DEFERRED by user decision — not implemented. Documented for future reference.

## Decision

During the Fase D roadmap (D1→D4), after shipping D1 (feedback loop) and D2
(per-model effective capability), the user chose to **skip D3** ("acho que não
precisamos fazer o d3") and move to a polish/cleanup pass instead. This document
records what D3 would have been so it can be revisited later without
re-deriving it.

## What D3 was (original feature #6)

**Multi-local-model tiering:** run more than one local model and route between
them by difficulty, instead of a single active local model plus cloud. E.g. a
tiny fast model for trivial prompts, a mid model for moderate ones, and cloud
only for the hardest — a local "cascade" with more than one rung.

### Sketch of the design space (not decided)

- **Model set:** a small ordered set of local models (by `params_b`), e.g.
  0.5B → 3B → 7B, plus cloud as the top tier. Today the router is binary
  (single local vs cloud); D3 makes local itself multi-tier.
- **Selection:** the existing `difficulty_score` maps to a tier via thresholds
  (one threshold per rung) rather than a single local/cloud cutoff. D1's
  feedback signals and D2's effective-capability estimates would inform where
  each rung's threshold sits.
- **Serving cost:** the hard part. Holding multiple GGUF models resident is RAM-
  expensive; the alternative is hot-swapping (the existing `ModelManager`
  hot-swap path) per request, which is far too slow to do per-request. A
  realistic D3 would likely require either (a) small models that fit resident
  alongside the main one, or (b) a separate lightweight "triage" model that only
  decides the tier, not serves. Both are substantial.
- **Interaction with existing cascade:** Fase B's `LocalThenCascade` already
  does local→cloud escalation on a weak result. D3 generalizes this to
  local(small)→local(big)→cloud, reusing `is_weak_result` at each rung.

## Why deferred

- **Cost/complexity:** multi-model residency or per-request swap is a large
  systems change (RAM, load latency, KV-cache management) relative to the
  routing-quality gain, which D1/D2 already improved cheaply.
- **YAGNI for now:** D1 (measurement) + D2 (per-model capability insight) give
  the user visibility into whether the single-local+cloud split is even the
  bottleneck. D3 should only be picked up if that data shows a clear win from
  an intermediate local tier.

## If revisited

Start from a fresh brainstorm. Key open questions to answer first:
1. Resident multi-model vs triage-only model vs per-request swap — pick the
   serving model before designing routing.
2. Reuse `LocalThenCascade` semantics extended to N rungs, or a new tier enum?
3. Per-rung thresholds: static, or fed by D1 feedback / D2 effective capability?

Related: [[../../../.claude/projects/-Users-ricardo-Repos-localllm/memory/fases-roadmap-status]]
(roadmap status memory). Fase D4 (semantic routing) and Fase E (cross-platform
tray) remain on the roadmap; the polish pass followed this deferral.
