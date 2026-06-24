use mistralrs::{GgufModelBuilder, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // GGUF files are already quantized — no ISQ needed.
    // The qwen2.5-7b-instruct-q4_k_m.gguf file is already Q4_K_M quantized.
    let model = GgufModelBuilder::new(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        vec!["qwen2.5-7b-instruct-q4_k_m.gguf".to_string()],
    )
    .with_logging()
    .build()
    .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Reply with the single word: ok");
    let resp = model.send_chat_request(messages).await?;
    println!("{}", resp.choices[0].message.content.clone().unwrap_or_default());
    Ok(())
}
