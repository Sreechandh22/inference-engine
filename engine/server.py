"""
FastAPI inference server.

Endpoints:
  POST /generate        — batch, returns full text when done
  POST /stream          — SSE, streams tokens as they're generated
  GET  /health          — liveness check
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine.block_manager import BlockManager
from engine.model.loader import load_model
from engine.model.transformer import TINYLLAMA_CONFIG
from engine.runner import GenerationConfig
from engine.scheduler import Scheduler

app = FastAPI(title="inference-engine")

# Global state — initialized on startup
_scheduler: Optional[Scheduler] = None
_lock = asyncio.Lock()

_CHAT_TOKENS = ["<|assistant|>", "<|user|>", "<|system|>", "<|im_start|>", "<|im_end|>"]


def _clean(text: str) -> str:
    for t in _CHAT_TOKENS:
        text = text.replace(t, "")
    return text.strip()


@app.on_event("startup")
async def startup():
    global _scheduler
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(device=device)

    # Warm up CUDA kernel
    if device == "cuda":
        from engine.model.attention import _load_cuda_kernel
        _load_cuda_kernel()

    cfg = TINYLLAMA_CONFIG
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_blocks=512,
        device=device,
    )
    _scheduler = Scheduler(model, tokenizer, block_manager, device=device)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    greedy: bool = False


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    elapsed_sec: float
    tokens_per_sec: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "scheduler_ready": _scheduler is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    config = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        greedy=req.greedy,
    )

    async with _lock:
        seq_id = _scheduler.add_request(req.prompt, config)
        output = ""
        start = time.perf_counter()

        while _scheduler.has_work():
            step = _scheduler.step()
            token = step.get(seq_id)
            if token is None:
                break
            output += token

        elapsed = time.perf_counter() - start

    output = _clean(output)
    n = len(_scheduler.tokenizer.encode(output))
    return GenerateResponse(
        text=output,
        tokens_generated=n,
        elapsed_sec=round(elapsed, 3),
        tokens_per_sec=round(n / elapsed, 1) if elapsed > 0 else 0,
    )


@app.post("/stream")
async def stream(req: GenerateRequest):
    config = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        greedy=req.greedy,
    )

    async def token_generator() -> AsyncGenerator[str, None]:
        async with _lock:
            seq_id = _scheduler.add_request(req.prompt, config)
            while _scheduler.has_work():
                step = _scheduler.step()
                token = step.get(seq_id)
                if token is None:
                    break
                token = _clean(token)
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/completions
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model: str = "tinyllama"
    prompt: str
    max_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


@app.post("/v1/completions")
async def v1_completions(req: CompletionRequest):
    greedy = req.temperature == 0.0
    config = GenerationConfig(
        max_new_tokens=req.max_tokens,
        temperature=max(req.temperature, 1e-5),
        top_p=req.top_p,
        greedy=greedy,
    )
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        async def _stream() -> AsyncGenerator[str, None]:
            async with _lock:
                seq_id = _scheduler.add_request(req.prompt, config)
                while _scheduler.has_work():
                    step = _scheduler.step()
                    token = step.get(seq_id)
                    if token is None:
                        break
                    token = _clean(token)
                    if token:
                        chunk = {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": req.model,
                            "choices": [{"text": token, "index": 0, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # Non-streaming
    async with _lock:
        seq_id = _scheduler.add_request(req.prompt, config)
        output = ""
        while _scheduler.has_work():
            step = _scheduler.step()
            token = step.get(seq_id)
            if token is None:
                break
            output += token

    output = _clean(output)
    prompt_tokens = len(_scheduler.tokenizer.encode(req.prompt))
    completion_tokens = len(_scheduler.tokenizer.encode(output))

    return CompletionResponse(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[CompletionChoice(text=output, index=0, finish_reason="stop")],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
