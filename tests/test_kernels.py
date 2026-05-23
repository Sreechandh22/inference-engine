"""
Correctness test: custom CUDA paged_attention_decode kernel vs PyTorch reference.

Run on a CUDA machine:
    python tests/test_kernels.py
"""

import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytorch_reference(q, key_cache, value_cache, block_table, seq_len, scale, num_groups):
    """Pure-PyTorch equivalent of the CUDA kernel."""
    num_heads, head_dim = q.shape
    num_kv_heads = key_cache.shape[2]
    tokens_per_block = key_cache.shape[1]

    # Gather K, V from scattered blocks
    block_ids = block_table[:len(block_table)]
    K = key_cache[block_ids]    # [num_blocks, tpb, num_kv_heads, head_dim]
    V = value_cache[block_ids]
    K = K.reshape(-1, num_kv_heads, head_dim)[:seq_len].float()
    V = V.reshape(-1, num_kv_heads, head_dim)[:seq_len].float()

    # GQA expansion
    K = K.repeat_interleave(num_groups, dim=1)  # [seq_len, num_heads, head_dim]
    V = V.repeat_interleave(num_groups, dim=1)

    q_f = q.float()  # [num_heads, head_dim]
    scores = torch.einsum("hd,shd->hs", q_f, K) * scale  # [num_heads, seq_len]
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum("hs,shd->hd", attn, V)  # [num_heads, head_dim]
    return out.half()


def test_paged_attention_decode():
    assert torch.cuda.is_available(), "CUDA required"

    from torch.utils.cpp_extension import load
    kernel = load(
        name="paged_attn_cuda",
        sources=["kernels/paged_attention.cu"],
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )

    # TinyLlama-like dimensions
    num_heads        = 32
    num_kv_heads     = 4
    head_dim         = 64
    num_groups       = num_heads // num_kv_heads
    tokens_per_block = 16
    num_blocks       = 64
    seq_len          = 47   # arbitrary, non-block-aligned
    scale            = 1.0 / (head_dim ** 0.5)

    device = "cuda"
    torch.manual_seed(42)

    q           = torch.randn(num_heads, head_dim, dtype=torch.float16, device=device)
    key_cache   = torch.randn(num_blocks, tokens_per_block, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    value_cache = torch.randn(num_blocks, tokens_per_block, num_kv_heads, head_dim, dtype=torch.float16, device=device)

    # Block table: use first ceil(seq_len / tpb) blocks in order
    num_kv_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
    block_table = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)

    # Run custom kernel
    out_kernel = kernel.paged_attention_decode(
        q, key_cache, value_cache, block_table, seq_len, scale
    )

    # Run PyTorch reference
    out_ref = pytorch_reference(q, key_cache, value_cache, block_table, seq_len, scale, num_groups)

    max_diff = (out_kernel.float() - out_ref.float()).abs().max().item()
    mean_diff = (out_kernel.float() - out_ref.float()).abs().mean().item()

    print(f"max  |kernel - ref| = {max_diff:.6f}")
    print(f"mean |kernel - ref| = {mean_diff:.6f}")

    # float16 has ~1e-3 precision; allow some tolerance
    assert max_diff < 0.05, f"Kernel output too far from reference: max diff = {max_diff}"
    print("PASSED")


def test_batched_paged_attention_decode():
    assert torch.cuda.is_available(), "CUDA required"

    from torch.utils.cpp_extension import load
    kernel = load(
        name="paged_attn_cuda",
        sources=["kernels/paged_attention.cu"],
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )

    num_heads        = 32
    num_kv_heads     = 4
    head_dim         = 64
    num_groups       = num_heads // num_kv_heads
    tokens_per_block = 16
    num_blocks       = 64
    scale            = 1.0 / (head_dim ** 0.5)

    device = "cuda"
    torch.manual_seed(7)

    # Sequences of varying, non-block-aligned lengths sharing one cache pool
    seq_lens_list = [47, 16, 33, 5]
    num_seqs = len(seq_lens_list)

    # Distinct physical blocks per sequence (scattered)
    block_tables_list = []
    next_block = 0
    for sl in seq_lens_list:
        n = (sl + tokens_per_block - 1) // tokens_per_block
        block_tables_list.append(list(range(next_block, next_block + n)))
        next_block += n
    assert next_block <= num_blocks

    max_blocks = max(len(bt) for bt in block_tables_list)
    block_tables = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    for i, bt in enumerate(block_tables_list):
        block_tables[i, :len(bt)] = torch.tensor(bt, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    q           = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.float16, device=device)
    key_cache   = torch.randn(num_blocks, tokens_per_block, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    value_cache = torch.randn(num_blocks, tokens_per_block, num_kv_heads, head_dim, dtype=torch.float16, device=device)

    out_kernel = kernel.paged_attention_decode_batched(
        q.contiguous(), key_cache, value_cache, block_tables, seq_lens, scale
    )

    # Per-sequence PyTorch reference
    max_diff = 0.0
    for i in range(num_seqs):
        ref = pytorch_reference(
            q[i], key_cache, value_cache,
            torch.tensor(block_tables_list[i], dtype=torch.int32, device=device),
            seq_lens_list[i], scale, num_groups,
        )
        d = (out_kernel[i].float() - ref.float()).abs().max().item()
        max_diff = max(max_diff, d)

    print(f"batched max |kernel - ref| = {max_diff:.6f}")
    assert max_diff < 0.05, f"Batched kernel too far from reference: max diff = {max_diff}"
    print("BATCHED PASSED")


if __name__ == "__main__":
    test_paged_attention_decode()
    test_batched_paged_attention_decode()
