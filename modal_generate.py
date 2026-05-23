import modal

app = modal.App("inference-engine")

image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "accelerate>=0.24.0",
)

@app.function(gpu="A10G", image=image, timeout=300)
def generate(prompt: str, max_new_tokens: int = 200):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    past_key_values = None
    generated_tokens = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values
        next_token = logits.argmax(dim=-1).item()

        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)
        input_ids = torch.tensor([[next_token]], device="cuda")

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


@app.local_entrypoint()
def main():
    prompt = "Explain how a transformer model works in simple terms:"
    print(f"Prompt: {prompt}\n")
    result = generate.remote(prompt)
    print(f"Response: {result}")
