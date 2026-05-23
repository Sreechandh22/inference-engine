import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # compute in float32 for numerical stability
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, freqs)  # [max_seq_len, head_dim/2]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # LLaMA split-half RoPE: [f0..fn/2, f0..fn/2] not interleaved pairs
        self.register_buffer("cos_cache", torch.cat([cos, cos], dim=-1), persistent=False)
        self.register_buffer("sin_cache", torch.cat([sin, sin], dim=-1), persistent=False)

    def forward(self, positions: torch.Tensor):
        return self.cos_cache[positions], self.sin_cache[positions]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [seq, num_heads, head_dim]
    # cos, sin: [seq, head_dim] — cast to match q dtype (float16 in practice)
    cos = cos.to(q.dtype).unsqueeze(1)  # [seq, 1, head_dim]
    sin = sin.to(q.dtype).unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class PagedAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads  # GQA expansion factor
        self.scale = 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(head_dim, max_seq_len, rope_base)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [seq_len, hidden_size]
        positions: torch.Tensor,        # [seq_len]
        seq_id: int,
        block_manager,
        is_prefill: bool,
    ) -> torch.Tensor:
        T, _ = hidden_states.shape

        Q = self.q_proj(hidden_states).view(T, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)

        cos, sin = self.rotary(positions)
        Q, K = apply_rope(Q, K, cos, sin)

        if is_prefill:
            # Write all prompt tokens to the block manager
            for t in range(T):
                block_manager.write_kv(self.layer_idx, seq_id, t, K[t], V[t])
            # K, V are already contiguous here — use them directly
            K_full = K   # [T, num_kv_heads, head_dim]
            V_full = V
        else:
            # Write new token to its allocated slot
            token_pos = block_manager.seq_lengths[seq_id] - 1
            block_manager.write_kv(self.layer_idx, seq_id, token_pos, K[0], V[0])
            # Gather full K, V from scattered blocks (the PagedAttention gather)
            K_full, V_full = block_manager.read_kv(self.layer_idx, seq_id)

        # GQA: expand kv heads to match query heads
        if self.num_groups > 1:
            K_full = K_full.repeat_interleave(self.num_groups, dim=1)
            V_full = V_full.repeat_interleave(self.num_groups, dim=1)

        # Attention: Q [T, H, D] x K [S, H, D] -> [H, T, S]
        S = K_full.shape[0]
        Q_t = Q.permute(1, 0, 2)          # [H, T, D]
        K_t = K_full.permute(1, 2, 0)     # [H, D, S]
        V_t = V_full.permute(1, 0, 2)     # [H, S, D]

        scores = torch.matmul(Q_t, K_t) * self.scale  # [H, T, S]

        if is_prefill and T > 1:
            # Causal mask for prefill
            mask = torch.triu(
                torch.full((T, S), float("-inf"), device=hidden_states.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + mask.unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_t)      # [H, T, D]
        out = out.permute(1, 0, 2).reshape(T, -1)  # [T, H*D]
        return self.o_proj(out)
