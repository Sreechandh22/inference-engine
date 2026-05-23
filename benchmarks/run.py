"""
Concurrency sweep benchmark: our engine vs HuggingFace sequential.
Measures throughput, TTFT, and peak GPU memory at 1/4/16/32/64 concurrent requests.

Run:
  modal run benchmarks/run.py

Plots saved to benchmarks/plots/concurrency_sweep.png (requires matplotlib locally).
"""

import os
import time
import modal

app = modal.App("inference-engine-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.0.0", "transformers>=4.35.0", "accelerate>=0.24.0", "ninja")
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
MAX_NEW_TOKENS = 150
CONCURRENCIES = [1, 4, 16, 32, 64]


def _prompts(n: int):
    return (_BASE_PROMPTS * ((n // len(_BASE_PROMPTS)) + 1))[:n]


@app.function(gpu="A10G", image=image, timeout=1200)
def bench_our_engine(num_requests: int):
    import sys
    import torch
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
        num_blocks=2048,
        device="cuda",
    )
    scheduler = Scheduler(model, tokenizer, block_manager, device="cuda")
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, greedy=True)

    prompts = _prompts(num_requests)
    torch.cuda.reset_peak_memory_stats()

    seq_ids = [scheduler.add_request(p, config) for p in prompts]
    first_token_times = {}
    start = time.perf_counter()
    total_tokens = 0

    while scheduler.has_work():
        step = scheduler.step()
        for sid, token in step.items():
            if token:
                if sid not in first_token_times:
                    first_token_times[sid] = time.perf_counter() - start
                total_tokens += 1

    elapsed = time.perf_counter() - start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    avg_ttft = sum(first_token_times.values()) / len(first_token_times) if first_token_times else 0

    return {
        "engine": "ours",
        "num_requests": num_requests,
        "throughput": round(total_tokens / elapsed, 1),
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "avg_ttft": round(avg_ttft, 3),
        "peak_mem_gb": round(peak_mem_gb, 2),
        "oom": False,
    }


@app.function(gpu="A10G", image=image, timeout=1200)
def bench_huggingface(num_requests: int):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda().eval()

    prompts = _prompts(num_requests)
    torch.cuda.reset_peak_memory_stats()

    total_tokens = 0
    first_token_times = []
    start = time.perf_counter()

    try:
        for prompt in prompts:
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
            first_token_times.append(time.perf_counter() - t0)
            total_tokens += out.shape[1] - inputs["input_ids"].shape[1]
    except torch.cuda.OutOfMemoryError:
        return {"engine": "huggingface", "num_requests": num_requests, "oom": True}

    elapsed = time.perf_counter() - start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    avg_ttft = sum(first_token_times) / len(first_token_times)

    return {
        "engine": "huggingface",
        "num_requests": num_requests,
        "throughput": round(total_tokens / elapsed, 1),
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "avg_ttft": round(avg_ttft, 3),
        "peak_mem_gb": round(peak_mem_gb, 2),
        "oom": False,
    }


@app.local_entrypoint()
def main():
    # Spawn all runs in parallel — 10 GPU functions, Modal queues them
    our_futures = {c: bench_our_engine.spawn(c) for c in CONCURRENCIES}
    hf_futures = {c: bench_huggingface.spawn(c) for c in CONCURRENCIES}

    our = {c: f.get() for c, f in our_futures.items()}
    hf = {c: f.get() for c, f in hf_futures.items()}

    # Print table
    print("\n" + "=" * 82)
    print(f"{'CONCURRENCY SWEEP  —  ' + str(MAX_NEW_TOKENS) + ' tokens/req  —  A10G  —  TinyLlama-1.1B':^82}")
    print("=" * 82)
    print(f"{'Concurrency':>12} | {'Engine':>12} | {'Throughput':>12} | {'Avg TTFT':>10} | {'Peak Mem':>10}")
    print("-" * 82)
    for c in CONCURRENCIES:
        for label, r in [("ours", our[c]), ("huggingface", hf[c])]:
            if r["oom"]:
                print(f"{c:>12} | {label:>12} | {'OOM':>12} | {'OOM':>10} | {'OOM':>10}")
            else:
                print(
                    f"{c:>12} | {label:>12} | "
                    f"{r['throughput']:>10.1f}/s | "
                    f"{r['avg_ttft']:>9.3f}s | "
                    f"{r['peak_mem_gb']:>8.2f} GB"
                )
    print("=" * 82 + "\n")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Inference Engine vs HuggingFace — Concurrency Sweep (A10G, TinyLlama-1.1B, 150 tok/req)",
            fontsize=12,
        )

        our_x   = [c for c in CONCURRENCIES if not our[c]["oom"]]
        hf_x    = [c for c in CONCURRENCIES if not hf[c]["oom"]]
        hf_oom  = [c for c in CONCURRENCIES if hf[c]["oom"]]

        def _vals(results, key, xs):
            return [results[c][key] for c in xs]

        # Throughput
        ax = axes[0]
        ax.plot(our_x, _vals(our, "throughput", our_x), "o-", label="Ours (PagedAttn + batched decode)", color="steelblue")
        ax.plot(hf_x,  _vals(hf,  "throughput", hf_x),  "s--", label="HuggingFace (sequential)", color="coral")
        for c in hf_oom:
            ax.axvline(x=c, color="coral", linestyle=":", alpha=0.4)
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel("Throughput (tok/sec)")
        ax.set_title("Throughput")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # TTFT
        ax = axes[1]
        ax.plot(our_x, _vals(our, "avg_ttft", our_x), "o-", label="Ours", color="steelblue")
        ax.plot(hf_x,  _vals(hf,  "avg_ttft", hf_x),  "s--", label="HuggingFace", color="coral")
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel("Avg TTFT (sec)")
        ax.set_title("Time to First Token")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Memory
        ax = axes[2]
        ax.plot(our_x, _vals(our, "peak_mem_gb", our_x), "o-", label="Ours", color="steelblue")
        ax.plot(hf_x,  _vals(hf,  "peak_mem_gb", hf_x),  "s--", label="HuggingFace", color="coral")
        for c in hf_oom:
            ax.axvline(x=c, color="coral", linestyle=":", alpha=0.4)
            ax.text(c, ax.get_ylim()[1] * 0.9, f"HF OOM\n@ {c}", color="coral", fontsize=7, ha="center")
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel("Peak GPU memory (GB)")
        ax.set_title("GPU Memory Usage")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("benchmarks/plots", exist_ok=True)
        out = "benchmarks/plots/concurrency_sweep.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out}")

    except ImportError:
        print("matplotlib not installed — skipping plot.  pip install matplotlib")
