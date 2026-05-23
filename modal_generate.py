import time
import modal

app = modal.App("inference-engine")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.0.0", "transformers>=4.35.0", "accelerate>=0.24.0", "ninja")
    .add_local_dir("engine", remote_path="/root/engine")
    .add_local_dir("kernels", remote_path="/root/kernels")
    .add_local_dir("tests", remote_path="/root/tests")
)

PROMPTS = [
    "Explain how a transformer model works in simple terms:",
    "What is the difference between supervised and unsupervised learning?",
    "Describe how attention mechanisms work in neural networks:",
    "What are the main challenges in training large language models?",
]


@app.function(gpu="A10G", image=image, timeout=300)
def run_kernel_test():
    import sys
    sys.path.insert(0, "/root")
    from tests.test_kernels import test_paged_attention_decode
    test_paged_attention_decode()
    return "PASSED"


@app.function(gpu="A10G", image=image, timeout=600)
def run_scheduler(max_new_tokens: int = 100):
    import sys
    sys.path.insert(0, "/root")

    from engine.model.loader import load_model
    from engine.block_manager import BlockManager
    from engine.scheduler import Scheduler
    from engine.runner import GenerationConfig
    from engine.model.transformer import TINYLLAMA_CONFIG

    model, tokenizer = load_model(device="cuda")

    # Compile CUDA kernel before timing starts — JIT takes ~60s on first run
    from engine.model.attention import _load_cuda_kernel
    _load_cuda_kernel()

    cfg = TINYLLAMA_CONFIG
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_blocks=512,
        device="cuda",
    )
    scheduler = Scheduler(model, tokenizer, block_manager, device="cuda")
    config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.9)

    # Submit all requests simultaneously
    seq_ids = [scheduler.add_request(p, config) for p in PROMPTS]
    outputs = {sid: "" for sid in seq_ids}

    start = time.perf_counter()
    total_tokens = 0

    while scheduler.has_work():
        step_outputs = scheduler.step()
        for sid, token in step_outputs.items():
            if token:
                outputs[sid] += token
                total_tokens += 1

    elapsed = time.perf_counter() - start
    throughput = total_tokens / elapsed

    _chat_tokens = ["<|assistant|>", "<|user|>", "<|system|>", "<|im_start|>", "<|im_end|>"]
    def _clean(text):
        for t in _chat_tokens:
            text = text.replace(t, "")
        return text.strip()

    return {
        "outputs": {PROMPTS[i]: _clean(outputs[sid]) for i, sid in enumerate(seq_ids)},
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "throughput_tok_per_sec": round(throughput, 1),
    }


@app.local_entrypoint()
def main():
    print("\n--- Phase 4: kernel correctness test ---")
    status = run_kernel_test.remote()
    print(f"Kernel test: {status}\n")

    result = run_scheduler.remote(max_new_tokens=100)

    print(f"\n{'='*60}")
    print(f"Throughput: {result['throughput_tok_per_sec']} tokens/sec")
    print(f"Total tokens: {result['total_tokens']} in {result['elapsed_sec']}s")
    print(f"Concurrent requests: {len(PROMPTS)}")
    print(f"{'='*60}\n")

    for prompt, output in result["outputs"].items():
        print(f"Q: {prompt}")
        print(f"A: {output}\n")
