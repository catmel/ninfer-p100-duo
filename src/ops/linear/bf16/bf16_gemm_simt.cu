// Pascal (sm_60) A16 general SIMT fallback for the bf16 linear family.
//
// CUDA 12.4 on GP100 has no tensor cores: the bf16 mma schedules (bf16_gemm_mma_kernel,
// ldmatrix + mma.m16n8k16 helpers in ops/common/mma.cuh) compile at arch 60 but __trap() at
// runtime, and the CUTLASS sm70 TU is Volta-only. This kernel carries the wide-T, TP2-shard,
// and any-shape A16 GEMM that `launch_bf16_mma` owns on Ampere+ product builds: plain fmaf
// dot products, one warp per (output row, token) pair, runtime N/K/T so no exact-geometry
// template is needed. Correctness first; the HFMA2-tiled variant is perf backlog
// (docs/p100-migration.md).
#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

inline constexpr int kTokensPerBlock = 8;

__global__ __launch_bounds__(32 * kTokensPerBlock) void bf16_gemm_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k, std::int32_t tokens,
    std::int32_t out_ld) {
    const std::int32_t row   = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t token = static_cast<std::int32_t>(blockIdx.y) * kTokensPerBlock +
                               static_cast<std::int32_t>(threadIdx.y);
    if (token >= tokens) { return; }

    const __nv_bfloat16* weight_row = weight + static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* x_col      = x + static_cast<std::int64_t>(token) * k;
    float accumulator = 0.0F;
    for (std::int32_t kk = static_cast<std::int32_t>(threadIdx.x); kk < k; kk += 32) {
        accumulator += __bfloat162float(__ldg(weight_row + kk)) *
                       __bfloat162float(__ldg(x_col + kk));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xFFFFFFFFu, accumulator, offset, 32);
    }
    if (threadIdx.x == 0) {
        out[static_cast<std::int64_t>(token) * out_ld + row] = __float2bfloat16_rn(accumulator);
    }
}

} // namespace

void bf16_gemm_simt_launch(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t rows   = weight.n;
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(rows),
                    static_cast<unsigned>((tokens + kTokensPerBlock - 1) / kTokensPerBlock), 1u);
    const dim3 block(32, kTokensPerBlock, 1u);
    bf16_gemm_simt_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), rows, k, tokens, out_ld);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
