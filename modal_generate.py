import modal

app = modal.App("inference-engine")

image = (
    modal.Image.debian_slim()
    .pip_install("torch>=2.0.0", "transformers>=4.35.0", "accelerate>=0.24.0")
    .add_local_dir("engine", remote_path="/root/engine")
)

@app.function(gpu="A10G", image=image, timeout=600, secrets=[modal.Secret.from_name("huggingface")])
def generate(prompt: str, max_new_tokens: int = 200):

    from engine.model.loader import load_model
    from engine.runner import Runner, GenerationConfig

    model, tokenizer = load_model(device="cuda")
    runner = Runner(model, tokenizer, device="cuda")
    config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.9)

    tokens = list(runner.generate(prompt, config))
    return "".join(tokens)


@app.local_entrypoint()
def main():
    prompt = "Explain how a transformer model works in simple terms:"
    print(f"Prompt: {prompt}\n\nResponse: ", end="", flush=True)
    result = generate.remote(prompt)
    print(result)
