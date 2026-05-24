# inference-engine

**21.2x faster serving than HuggingFace at 64 concurrent requests. Memory flat at 3 GB regardless of load.**

Built a production-style LLM inference engine from scratch — the same ideas behind vLLM, implemented without using it.

What I wrote from scratch:
- **PagedAttention block manager** — KV cache in fixed-size pages, block table per sequence, ~4% memory waste vs ~60% with naive allocation
- **Continuous batching scheduler** — waiting queue + running set, one GPU forward pass per token step over all active sequences
- **Custom CUDA kernel** — `paged_attention_decode_batched`, grid `dim3(num_heads, num_seqs)`, gathers KV from non-contiguous physical blocks, GQA-aware
- **TinyLlama transformer** — RoPE, SwiGLU, GQA, RMSNorm, loads pretrained weights
- **OpenAI-compatible API** — `/v1/chat/completions` (plus legacy `/v1/completions`) with streaming SSE, drop-in for any OpenAI client

## Demo

**[sreechandh22.github.io/inference-engine](https://sreechandh22.github.io/inference-engine/)** — live inference, tokens stream back in real time. Runs on an A10G via Modal (scales to zero when idle — first request takes ~60s cold start, subsequent requests are fast).

Or hit the API directly:

```bash
curl https://sreechandh22--inference-engine-server-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain how attention mechanisms work:"}], "max_tokens": 100}'
```

First request takes ~60s (cold start — model load + kernel JIT compile). After that it's fast.

## Results

Benchmarked on A10G (Modal), TinyLlama-1.1B, 150 tokens/request. HuggingFace baseline runs requests sequentially (no continuous batching).

| Concurrency | This engine | HuggingFace | Speedup |
|---|---|---|---|
| 4 | 178.0 tok/sec | 57.3 tok/sec | 3.1x |
| 16 | 672.8 tok/sec | 49.3 tok/sec | 13.6x |
| 32 | 1187.8 tok/sec | 64.7 tok/sec | 18.4x |
| 64 | **1340.1 tok/sec** | 63.1 tok/sec | **21.2x** |

**Peak GPU memory: 2.96 → 3.02 GB across 1–64 concurrent sequences.** PagedAttention allocates blocks per actual token length — no padding, no max-length reservation.

HuggingFace stays flat at ~60 tok/sec regardless of load (sequential, one request at a time). This engine scales with concurrency because all running sequences are batched into one GPU forward pass per token step.

![Concurrency sweep](benchmarks/plots/concurrency_sweep.png)

## What's inside

```
engine/
  block_manager.py   — PagedAttention: KV cache in fixed-size blocks, ~4% waste vs ~60% naive
  scheduler.py       — continuous batching: one forward pass per step over all running sequences
  model/             — TinyLlama transformer, RoPE, GQA, RMSNorm
  server.py          — FastAPI: /generate, /stream (SSE), /health
kernels/
  paged_attention.cu — custom CUDA kernel: gathers KV from non-contiguous blocks
benchmarks/
  run.py             — parallel benchmark vs HuggingFace
tests/
  test_kernels.py    — single + batched kernel correctness vs PyTorch reference
```

## How it works

**PagedAttention:** KV cache split into 16-token blocks. Each sequence gets a block table (logical→physical). No padding, no max-length reservation. Memory scales with actual sequence length.

**Continuous batching:** The scheduler runs a waiting queue + running set. Each token step: prefill any new sequences, then batch all decode-ready sequences into one forward pass. When a sequence finishes, the next waiting request takes its slot immediately.

**Custom CUDA kernel:** `paged_attention_decode_batched` — grid `dim3(num_heads, num_seqs)`, each block gathers KV from non-contiguous physical blocks via the block table, computes numerically-stable softmax, returns weighted V sum. Correctness tested against PyTorch reference (max diff < 0.05 in float16).

## API

OpenAI-compatible endpoint — any OpenAI client works as-is:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain attention mechanisms:"}], "max_tokens": 100}'
```

```json
{
  "id": "chatcmpl-3f8a1c2d9e4b",
  "object": "chat.completion",
  "model": "tinyllama",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Attention mechanisms allow..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105}
}
```

Streaming (`"stream": true`) returns SSE `chat.completion.chunk` deltas as tokens are generated. The legacy `/v1/completions` (text-completion format) is still available.

## Run

```bash
# kernel correctness + generation benchmark on Modal (A10G)
modal run modal_generate.py

# throughput benchmark vs HuggingFace
modal run benchmarks/run.py
```

## Stack

Python · PyTorch · CUDA C++ · pybind11 · FastAPI · Modal
