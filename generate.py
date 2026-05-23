from engine import load_model, Runner, GenerationConfig

model, tokenizer = load_model()
runner = Runner(model, tokenizer)
config = GenerationConfig(max_new_tokens=200, temperature=0.8, top_p=0.9)

prompt = "Explain how a transformer model works in simple terms:"
print(f"Prompt: {prompt}\n\nResponse: ", end="", flush=True)

for token in runner.generate(prompt, config):
    print(token, end="", flush=True)

print()
