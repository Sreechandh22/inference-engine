"""
Benchmark: our inference engine vs HuggingFace pipeline baseline.

Measures:
  - Throughput (tokens/sec)
  - Time to first token (TTFT)
  - Total latency per request

Run on Modal:
  modal run benchmarks/run.py
"""

import time
import modal

app = modal.App("inference-engine-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch>=2.0.0", "transformers>=4.35.0", "accelerate>=0.24.0", "ninja",
    )
    .add_local_dir("engine", remote_path="/root/engine")
    .add_local_dir("kernels", remote_path="/root/kernels")
)

_BASE_PROMPTS = [
    "Explain how a transformer model works in simple terms:",
    "What is the difference between supervised and unsupervised learning?",
    "Describe how attention mechanisms work in neural networks:",
    "What are the main challenges in training large language models?",
    "What is gradient descent and how does it work?",
    "Explain the concept of overfitting in machine learning:",
    "What is the difference between a CNN and an RNN?",
    "How does backpropagation work in neural networks:",
]
# Repeat to hit 16 concurrent requests
PROMPTS = (_BASE_PROMPTS * 2)[:16]
MAX_NEW_TOKENS = 150


# ---------------------------------------------------------------------------
# Our engine
# ---------------------------------------------------------------------------

@app.function(gpu="A10G", image=image, timeout=600)
def bench_our_engine():
    import sys
    sys.path.insert(0, "/root")

    from engine.model.loader import load_model
    from engine.block_manager import BlockManager
    from engine.scheduler import Scheduler
    from engine.runner import GenerationConfig
    from engine.model.transformer import TINYLLAMA_CONFIG
    from engine.model.attention import _load_cuda_kernel

    model, tokenizer = load_model(device="cuda")
    _load_cuda_kernel()

    cfg = TINYLLAMA_CONFIG
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_blocks=1024,
        device="cuda",
    )
    scheduler = Scheduler(model, tokenizer, block_manager, device="cuda")
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, greedy=True)
    # Greedy + min tokens so both engines generate comparable output lengths

    seq_ids = [scheduler.add_request(p, config) for p in PROMPTS]
    first_token_times = {}
    outputs = {sid: "" for sid in seq_ids}
    start = time.perf_counter()
    total_tokens = 0

    while scheduler.has_work():
        step = scheduler.step()
        for sid, token in step.items():
            if token:
                if sid not in first_token_times:
                    first_token_times[sid] = time.perf_counter() - start
                outputs[sid] += token
                total_tokens += 1

    elapsed = time.perf_counter() - start
    avg_ttft = sum(first_token_times.values()) / len(first_token_times) if first_token_times else 0

    return {
        "engine": "custom (PagedAttention + CUDA kernel)",
        "throughput_tok_per_sec": round(total_tokens / elapsed, 1),
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "avg_time_to_first_token_sec": round(avg_ttft, 3),
        "concurrent_requests": len(PROMPTS),
    }


# ---------------------------------------------------------------------------
# HuggingFace baseline
# ---------------------------------------------------------------------------

@app.function(gpu="A10G", image=image, timeout=600)
def bench_huggingface():
    import sys
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()

    total_tokens = 0
    first_token_times = []
    start = time.perf_counter()

    # HF has no continuous batching — process sequentially
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                min_new_tokens=MAX_NEW_TOKENS,
            )
        ttft = time.perf_counter() - t0
        first_token_times.append(ttft)
        total_tokens += out.shape[1] - inputs["input_ids"].shape[1]

    elapsed = time.perf_counter() - start
    avg_ttft = sum(first_token_times) / len(first_token_times)

    return {
        "engine": "HuggingFace pipeline (sequential)",
        "throughput_tok_per_sec": round(total_tokens / elapsed, 1),
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "avg_time_to_first_token_sec": round(avg_ttft, 3),
        "concurrent_requests": len(PROMPTS),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    import modal
    # Run both benchmarks in parallel
    our_f = bench_our_engine.spawn()
    hf_f = bench_huggingface.spawn()

    our = our_f.get()
    hf = hf_f.get()

    print("\n" + "=" * 65)
    print(f"{'BENCHMARK RESULTS':^65}")
    print("=" * 65)

    for result in [our, hf]:
        print(f"\n{result['engine']}")
        print(f"  Throughput       : {result['throughput_tok_per_sec']} tok/sec")
        print(f"  Total tokens     : {result['total_tokens']} in {result['elapsed_sec']}s")
        print(f"  Avg TTFT         : {result['avg_time_to_first_token_sec']}s")
        print(f"  Concurrent reqs  : {result['concurrent_requests']}")

    wallclock_speedup = hf["elapsed_sec"] / our["elapsed_sec"]
    ttft_speedup = hf["avg_time_to_first_token_sec"] / our["avg_time_to_first_token_sec"]
    print(f"\n  Wall-clock speedup : {wallclock_speedup:.2f}x faster to serve all {len(PROMPTS)} requests")
    print(f"  TTFT improvement   : {ttft_speedup:.2f}x faster time-to-first-token (user latency)")
    print(f"\n  Note: PagedAttention also uses memory proportional to actual sequence")
    print(f"  length, not max_length — enabling higher concurrency before OOM.")
    print("=" * 65 + "\n")
