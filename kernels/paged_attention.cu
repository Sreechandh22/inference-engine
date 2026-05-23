/*
 * PagedAttention decode kernel.
 *
 * For a single decode step (T=1), computes attention over a KV cache stored
 * in non-contiguous physical blocks. The key innovation: instead of gathering
 * K and V into a contiguous buffer first (what the Python fallback does), this
 * kernel reads directly from scattered blocks via the block_table, computes
 * dot products, softmax, and the weighted V sum — all in one GPU pass.
 *
 * Grid:  (num_heads,)          — one thread block per attention head
 * Block: (THREADS_PER_BLOCK,)  — 128 threads
 * Smem:  Q values + scores buffer + reduction scratch
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <float.h>

#define THREADS_PER_BLOCK 128
#define WARP_SIZE 32
#define MAX_WARPS 4   // THREADS_PER_BLOCK / WARP_SIZE
#define MAX_SEQ_LEN 2048

// ---------------------------------------------------------------------------
// Warp-level primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Block-level reduction — result broadcast to all threads via smem[0].
__device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int w = 0; w < MAX_WARPS; w++) s += smem[w];
        smem[0] = s;
    }
    __syncthreads();
    return smem[0];
}

__device__ float block_reduce_max(float val, float* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int w = 0; w < MAX_WARPS; w++) m = fmaxf(m, smem[w]);
        smem[0] = m;
    }
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

__global__ void paged_attention_decode_kernel(
    const __half* __restrict__ q,           // [num_heads, head_dim]
    const __half* __restrict__ key_cache,   // [num_blocks, tokens_per_block, num_kv_heads, head_dim]
    const __half* __restrict__ value_cache, // [num_blocks, tokens_per_block, num_kv_heads, head_dim]
    const int*   __restrict__ block_table,  // [num_kv_blocks_for_seq]
    __half*      __restrict__ output,       // [num_heads, head_dim]
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int tokens_per_block,
    float scale
) {
    int head    = blockIdx.x;
    int kv_head = head / (num_heads / num_kv_heads);  // GQA head mapping
    int tid     = threadIdx.x;

    // Shared memory layout:
    //   [0 .. head_dim)             : Q values (float32)
    //   [head_dim .. head_dim+MAX_SEQ_LEN) : attention scores (float32)
    //   last MAX_WARPS floats       : reduction scratch
    extern __shared__ float smem[];
    float* q_smem       = smem;
    float* scores       = smem + head_dim;
    float* reduce_smem  = scores + MAX_SEQ_LEN;  // MAX_WARPS floats

    // --- Load Q for this head into shared memory ---
    const __half* q_head = q + head * head_dim;
    for (int d = tid; d < head_dim; d += THREADS_PER_BLOCK)
        q_smem[d] = __half2float(q_head[d]);
    __syncthreads();

    // --- Step 1: compute scores[t] = dot(Q, K_t) * scale ---
    // Each thread handles multiple token positions.
    for (int t = tid; t < seq_len; t += THREADS_PER_BLOCK) {
        int block_idx    = t / tokens_per_block;
        int token_offset = t % tokens_per_block;
        int phys_block   = block_table[block_idx];

        // Pointer to K[phys_block, token_offset, kv_head, 0]
        const __half* k = key_cache
            + (phys_block * tokens_per_block + token_offset) * num_kv_heads * head_dim
            + kv_head * head_dim;

        float score = 0.f;
        for (int d = 0; d < head_dim; d++)
            score += q_smem[d] * __half2float(k[d]);

        scores[t] = score * scale;
    }
    __syncthreads();

    // --- Step 2: numerically stable softmax ---
    // 2a. Find global max
    float local_max = -FLT_MAX;
    for (int t = tid; t < seq_len; t += THREADS_PER_BLOCK)
        local_max = fmaxf(local_max, scores[t]);
    float global_max = block_reduce_max(local_max, reduce_smem);
    __syncthreads();

    // 2b. Compute exp(score - max) and accumulate sum
    float local_sum = 0.f;
    for (int t = tid; t < seq_len; t += THREADS_PER_BLOCK) {
        scores[t] = expf(scores[t] - global_max);
        local_sum += scores[t];
    }
    float global_sum = block_reduce_sum(local_sum, reduce_smem);
    __syncthreads();

    // 2c. Normalize
    for (int t = tid; t < seq_len; t += THREADS_PER_BLOCK)
        scores[t] /= global_sum;
    __syncthreads();

    // --- Step 3: weighted sum of V ---
    // Each thread owns a contiguous slice of output dimensions.
    for (int d = tid; d < head_dim; d += THREADS_PER_BLOCK) {
        float out_d = 0.f;
        for (int t = 0; t < seq_len; t++) {
            int block_idx    = t / tokens_per_block;
            int token_offset = t % tokens_per_block;
            int phys_block   = block_table[block_idx];

            const __half* v = value_cache
                + (phys_block * tokens_per_block + token_offset) * num_kv_heads * head_dim
                + kv_head * head_dim;

            out_d += scores[t] * __half2float(v[d]);
        }
        output[head * head_dim + d] = __float2half(out_d);
    }
}

// ---------------------------------------------------------------------------
// C++ wrapper — called from Python
// ---------------------------------------------------------------------------

torch::Tensor paged_attention_decode(
    torch::Tensor q,            // [num_heads, head_dim]          float16
    torch::Tensor key_cache,    // [num_blocks, tpb, num_kv_heads, head_dim] float16
    torch::Tensor value_cache,
    torch::Tensor block_table,  // [num_kv_blocks]  int32
    int seq_len,
    float scale
) {
    TORCH_CHECK(seq_len <= MAX_SEQ_LEN,
        "seq_len ", seq_len, " exceeds kernel MAX_SEQ_LEN ", MAX_SEQ_LEN);
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be float16");
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");

    int num_heads        = q.size(0);
    int head_dim         = q.size(1);
    int num_kv_heads     = key_cache.size(2);
    int tokens_per_block = key_cache.size(1);

    auto output = torch::zeros({num_heads, head_dim}, q.options());

    // Shared memory: Q (head_dim) + scores (MAX_SEQ_LEN) + reduction (MAX_WARPS)
    int smem_bytes = (head_dim + MAX_SEQ_LEN + MAX_WARPS) * sizeof(float);

    paged_attention_decode_kernel<<<num_heads, THREADS_PER_BLOCK, smem_bytes>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        reinterpret_cast<const __half*>(key_cache.data_ptr()),
        reinterpret_cast<const __half*>(value_cache.data_ptr()),
        block_table.data_ptr<int>(),
        reinterpret_cast<__half*>(output.data_ptr()),
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        tokens_per_block,
        scale
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "paged_attention_decode",
        &paged_attention_decode,
        "PagedAttention single-query decode kernel (float16)"
    );
}
