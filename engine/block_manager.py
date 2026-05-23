import torch
from typing import Dict, List, Tuple

TOKENS_PER_BLOCK = 16


class BlockManager:
    """
    Manages the KV cache as a pool of fixed-size blocks (PagedAttention).
    Sequences map logical token positions to physical blocks via a block table.
    Near-zero memory waste vs naive max-length reservation.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int = 512,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.tokens_per_block = tokens_per_block
        self.device = device

        # Pre-allocate entire KV cache up front
        # Shape: [num_layers, num_blocks, tokens_per_block, num_kv_heads, head_dim]
        self.key_cache = torch.zeros(
            num_layers, num_blocks, tokens_per_block, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.value_cache = torch.zeros(
            num_layers, num_blocks, tokens_per_block, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[int, List[int]] = {}  # seq_id -> [block_id, ...]
        self.seq_lengths: Dict[int, int] = {}          # seq_id -> num tokens written

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("KV cache out of blocks — too many concurrent sequences")
        return self.free_blocks.pop()

    def can_allocate(self, num_tokens: int) -> bool:
        needed = (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block
        return len(self.free_blocks) >= needed

    def allocate(self, seq_id: int, num_tokens: int):
        """Allocate blocks for a new sequence (called before prefill)."""
        needed = (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block
        self.block_tables[seq_id] = [self._alloc_block() for _ in range(needed)]
        self.seq_lengths[seq_id] = num_tokens

    def append_slot(self, seq_id: int) -> Tuple[int, int]:
        """
        Reserve the next token's slot (called before each decode step).
        Allocates a new block if the current one is full.
        Returns (block_id, token_offset_within_block).
        """
        seq_len = self.seq_lengths[seq_id]
        block_idx = seq_len // self.tokens_per_block
        token_offset = seq_len % self.tokens_per_block

        if block_idx >= len(self.block_tables[seq_id]):
            self.block_tables[seq_id].append(self._alloc_block())

        self.seq_lengths[seq_id] += 1
        return self.block_tables[seq_id][block_idx], token_offset

    def can_append(self, seq_id: int) -> bool:
        """Check if the next decode step can be served (has a free block if needed)."""
        seq_len = self.seq_lengths[seq_id]
        needs_new_block = (seq_len % self.tokens_per_block) == 0
        return (not needs_new_block) or (len(self.free_blocks) >= 1)

    def free(self, seq_id: int):
        """Release all blocks for a finished or preempted sequence."""
        for block_id in self.block_tables.pop(seq_id, []):
            self.free_blocks.append(block_id)
        self.seq_lengths.pop(seq_id, None)

    def write_kv(
        self,
        layer_idx: int,
        seq_id: int,
        token_pos: int,
        key: torch.Tensor,    # [num_kv_heads, head_dim]
        value: torch.Tensor,  # [num_kv_heads, head_dim]
    ):
        block_idx = token_pos // self.tokens_per_block
        token_offset = token_pos % self.tokens_per_block
        block_id = self.block_tables[seq_id][block_idx]
        self.key_cache[layer_idx, block_id, token_offset] = key
        self.value_cache[layer_idx, block_id, token_offset] = value

    def read_kv(
        self,
        layer_idx: int,
        seq_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather all K, V for a sequence from non-contiguous physical blocks.
        Returns (keys, values) each of shape [seq_len, num_kv_heads, head_dim].
        This is the gather operation that a custom CUDA kernel will replace in Phase 4.
        """
        seq_len = self.seq_lengths[seq_id]
        block_ids = torch.tensor(
            self.block_tables[seq_id], dtype=torch.long, device=self.device
        )
        # [num_blocks, tokens_per_block, num_kv_heads, head_dim]
        keys = self.key_cache[layer_idx][block_ids]
        values = self.value_cache[layer_idx][block_ids]

        # Flatten blocks and trim to actual length
        keys = keys.reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        values = values.reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        return keys, values
