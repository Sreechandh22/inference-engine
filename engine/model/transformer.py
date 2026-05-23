from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.model.attention import PagedAttention, RMSNorm


@dataclass
class ModelConfig:
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


TINYLLAMA_CONFIG = ModelConfig()


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = PagedAttention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_base=config.rope_theta,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        seq_id: int,
        block_manager,
        is_prefill: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, seq_id, block_manager, is_prefill)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class TinyLlamaModel(nn.Module):
    def __init__(self, config: ModelConfig = TINYLLAMA_CONFIG):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, config) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,    # [seq_len]
        positions: torch.Tensor,    # [seq_len]
        seq_id: int,
        block_manager,
        is_prefill: bool,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)  # [seq_len, hidden_size]

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, seq_id, block_manager, is_prefill)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)  # [seq_len, vocab_size]
        return logits[-1]  # return only last token's logits
