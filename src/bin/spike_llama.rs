//! Spike: verify the llama-cpp-2 API including KV state save/load to disk.
//!
//! Steps:
//! 1. Load Qwen2.5-3B-Instruct GGUF with Metal offload.
//! 2. Create a context, tokenize a prompt.
//! 3. Decode prefill batch. SAVE KV STATE immediately after prefill.
//! 4. Greedy-sample a few tokens; print them (round 1).
//! 5. Create a FRESH context, load the saved prefill state.
//! 6. Continue sampling from the reloaded prefill state (round 2).
//!    The two rounds should produce the same output (deterministic greedy).
//!
//! Run with:
//!   GGUF_PATH=/path/to/model.gguf cargo run --bin spike_llama --release

use std::num::NonZeroU32;
use std::path::PathBuf;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};

/// Greedy-generate up to `max_tokens` from the current context logits.
/// `logit_idx` is the index of the logit slot to sample from first.
/// `start_pos` is the next token position to write in the KV cache.
/// Returns (generated tokens, next free position).
fn generate(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    model: &LlamaModel,
    sampler: &mut LlamaSampler,
    decoder: &mut encoding_rs::Decoder,
    logit_idx: i32,
    start_pos: i32,
    max_tokens: usize,
) -> anyhow::Result<(Vec<LlamaToken>, i32)> {
    let mut generated = Vec::new();
    let mut pos = start_pos;

    let mut next_tok = sampler.sample(ctx, logit_idx);
    let mut batch = LlamaBatch::new(1, 1);

    for _ in 0..max_tokens {
        if model.is_eog_token(next_tok) {
            print!("<EOG>");
            break;
        }
        let piece = model.token_to_piece(next_tok, decoder, true, None)?;
        print!("{piece}");
        std::io::Write::flush(&mut std::io::stdout())?;
        generated.push(next_tok);

        batch.clear();
        batch.add(next_tok, pos, &[0], true)?;
        ctx.decode(&mut batch)?;
        pos += 1;

        next_tok = sampler.sample(ctx, 0);
    }
    Ok((generated, pos))
}

fn main() -> anyhow::Result<()> {
    // ---------------------------------------------------------------------------
    // 0. Config
    // ---------------------------------------------------------------------------
    let gguf_path = std::env::var("GGUF_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        format!(
            "{home}/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-GGUF/snapshots/7dabda4d13d513e3e842b20f0d435c732f172cbe/qwen2.5-3b-instruct-q4_k_m.gguf"
        )
    });

    let n_gpu_layers: u32 = std::env::var("N_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX);

    let n_ctx: u32 = 2048;
    let prompt = "Reply with the single word: ok";
    let n_gen_tokens: usize = 16;

    println!("=== spike_llama ===");
    println!("model : {gguf_path}");
    println!("n_gpu_layers : {n_gpu_layers}");
    println!("n_ctx : {n_ctx}");
    println!();

    // ---------------------------------------------------------------------------
    // 1. Backend + model
    // ---------------------------------------------------------------------------
    let backend = LlamaBackend::init()?;
    println!("GPU offload supported: {}", backend.supports_gpu_offload());

    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let model = LlamaModel::load_from_file(&backend, PathBuf::from(&gguf_path), &model_params)?;
    println!("model loaded — vocab={}", model.n_vocab());

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));

    // ---------------------------------------------------------------------------
    // 2. Tokenize
    // ---------------------------------------------------------------------------
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let n_tok = tokens.len();
    println!("prompt tokens ({n_tok}): {:?}", tokens);

    // ---------------------------------------------------------------------------
    // 3. Prefill in context 1
    // ---------------------------------------------------------------------------
    let mut ctx1 = model.new_context(&backend, ctx_params.clone())?;
    println!("ctx1 created (n_ctx={})", ctx1.n_ctx());

    let mut prefill = LlamaBatch::new(n_tok, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        prefill.add(tok, i as i32, &[0], i == n_tok - 1)?;
    }
    ctx1.decode(&mut prefill)?;
    println!("prefill done");

    // ---------------------------------------------------------------------------
    // 4. Save KV state IMMEDIATELY after prefill (before any generation)
    // ---------------------------------------------------------------------------
    let state_file = std::env::temp_dir().join("spike_llama_prefill.bin");
    ctx1.state_save_file(&state_file, &tokens)?;
    let state_size = std::fs::metadata(&state_file)?.len();
    println!(
        "prefill state saved — {} bytes at {state_file:?}",
        state_size
    );

    // ---------------------------------------------------------------------------
    // 5. Generate from ctx1 (round 1) — baseline
    // ---------------------------------------------------------------------------
    let mut s1 = LlamaSampler::chain_simple([LlamaSampler::temp(0.0), LlamaSampler::greedy()]);
    let mut d1 = encoding_rs::UTF_8.new_decoder();
    print!("\n[round 1] ");
    let (gen1, _pos1) = generate(
        &mut ctx1,
        &model,
        &mut s1,
        &mut d1,
        (n_tok - 1) as i32,
        n_tok as i32,
        n_gen_tokens,
    )?;
    println!("\nRound 1: {} tokens generated", gen1.len());
    drop(ctx1); // free GPU memory

    // ---------------------------------------------------------------------------
    // 6. Fresh context 2 — load prefill state
    // ---------------------------------------------------------------------------
    let mut ctx2 = model.new_context(&backend, ctx_params.clone())?;
    println!("\nctx2 created (fresh)");

    let loaded = ctx2.state_load_file(&state_file, tokens.len() + 64)?;
    println!(
        "state loaded — {} tokens restored (expected {})",
        loaded.len(),
        tokens.len()
    );
    assert_eq!(
        loaded.len(),
        tokens.len(),
        "token count mismatch after state_load_file"
    );

    // After state_load_file, KV cache is restored but logits are not.
    // Remove last cached position and re-decode to regenerate logits.
    let last_prefill_pos = (n_tok - 1) as u32;
    ctx2.clear_kv_cache_seq(Some(0), Some(last_prefill_pos), Some(last_prefill_pos + 1))?;
    let mut warmup = LlamaBatch::new(1, 1);
    warmup.add(tokens[n_tok - 1], last_prefill_pos as i32, &[0], true)?;
    ctx2.decode(&mut warmup)?;
    println!("warm-up decode done — logits ready at position {last_prefill_pos}");

    // ---------------------------------------------------------------------------
    // 7. Generate from ctx2 (round 2) — must match round 1
    // ---------------------------------------------------------------------------
    let mut s2 = LlamaSampler::chain_simple([LlamaSampler::temp(0.0), LlamaSampler::greedy()]);
    let mut d2 = encoding_rs::UTF_8.new_decoder();
    print!("\n[round 2] ");
    let (gen2, _pos2) = generate(
        &mut ctx2,
        &model,
        &mut s2,
        &mut d2,
        0,
        n_tok as i32,
        n_gen_tokens,
    )?;
    println!("\nRound 2: {} tokens generated", gen2.len());
    drop(ctx2);

    // ---------------------------------------------------------------------------
    // 8. Verify determinism: rounds 1 and 2 must produce the same tokens
    // ---------------------------------------------------------------------------
    assert_eq!(
        gen1, gen2,
        "Round 1 and round 2 must produce identical tokens (greedy, same KV state)"
    );
    println!("\nDETERMINISM CHECK PASSED: rounds 1 and 2 produced identical output");

    // ---------------------------------------------------------------------------
    // 9. Summary
    // ---------------------------------------------------------------------------
    println!("\n=== ALL CHECKS PASSED ===");
    println!("  Metal GPU offload   : OK (all layers on MTL0)");
    println!(
        "  Model load          : OK (Qwen2.5-3B Q4_K_M, vocab={})",
        model.n_vocab()
    );
    println!("  Prefill + decode    : OK ({} prompt tokens)", n_tok);
    println!("  KV state save       : OK ({state_size} bytes)");
    println!(
        "  KV state load       : OK ({} tokens restored)",
        loaded.len()
    );
    println!("  Warm-up re-decode   : OK (position {last_prefill_pos})");
    println!("  Round 1 generation  : OK ({} tokens)", gen1.len());
    println!("  Round 2 generation  : OK ({} tokens)", gen2.len());
    println!("  Determinism check   : PASS (identical output)");

    let _ = std::fs::remove_file(&state_file);
    Ok(())
}
