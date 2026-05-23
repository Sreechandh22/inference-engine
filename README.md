# inference-engine

A from-scratch LLM inference engine. Custom PagedAttention memory manager, continuous batching scheduler, and CUDA kernels — no vLLM, no HuggingFace inference backend.

## Results

Benchmarked on A10G (Modal), TinyLlama-1.1B, 16 concurrent requests, 150 max tokens:

| | This engine | HuggingFace pipeline |
|---|---|---|
| Throughput | **574.7 tok/sec** | 62.6 tok/sec |
| Avg TTFT | **0.814s** | 2.396s |
| Wall-clock (16 reqs) | **3.89s** | 38.35s |

**9.86x faster wall-clock. 2.94x lower time-to-first-token.**

HF pipeline runs 16 requests sequentially. This engine runs them concurrently with one GPU forward pass per token step — which is the point.

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

## Run

```bash
# kernel correctness + generation benchmark on Modal (A10G)
modal run modal_generate.py

# throughput benchmark vs HuggingFace
modal run benchmarks/run.py
```

## Stack

Python · PyTorch · CUDA C++ · pybind11 · FastAPI · Modal
