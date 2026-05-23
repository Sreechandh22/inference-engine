# inference-engine

A from-scratch LLM inference engine built for throughput and memory efficiency.

**Not a wrapper.** Custom PagedAttention memory manager, continuous batching scheduler, and CUDA kernels — built ground up.

## What's inside

- `engine/` — block manager (PagedAttention), scheduler (continuous batching), model runner
- `kernels/` — custom CUDA kernels (PagedAttention, fused LayerNorm)
- `server/` — FastAPI serving layer + streaming playground
- `benchmarks/` — throughput, latency, and memory benchmarks vs naive HuggingFace

## Status

Building in public. Phases:
- [ ] Phase 1: working single-GPU serving
- [ ] Phase 2: PagedAttention + block manager
- [ ] Phase 3: continuous batching scheduler
- [ ] Phase 4: custom CUDA kernel
- [ ] Phase 5: benchmarks + playground
