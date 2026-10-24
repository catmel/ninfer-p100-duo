// sm_60 (Tesla P100 / GP100) wide-token GGML_K tile projection.
//
// The generic engine's GGML_K route for T>4 on Pascal re-decodes each weight row per 8-token
// chunk (gemv_hfma2<8> in ggml_k.cu). On GP100 -- which has no tensor
// cores and no WMMA path under CUDA 12.x -- the decode itself (shift + cvt + fma per value)
// dominates the instruction cost at prefill widths. This kernel amortizes one HFMA2-based
// decode over eight staged tokens, following the qualified w8/q4/q5/q6 decode tile geometry
// (ops/linear/pascal/linear_tile_pascal.cu) and the donor Q4_K/Q6_K sm_60 matvec patterns.
//
// Codec facts used here (mirrored from ggml_k_codec.cuh and the gemv routes in ggml_k.cu):
//   Q4_K block = 144 B: d@0, dmin@2 (fp16); packed scales@4..15; 128 B of nibbles@16.
//     value column c: byte = 16 + (c>>6)*32 + (c&31); nibble shift = ((c>>5)&1)*4;
//     group = c>>5; value = (d*scale)*code - (dmin*min).
//   Q6_K block = 210 B: low nibbles@0..127, 2-bit high plane@128..159, int8 scales@192..207,
//     d@208; value = (d*scale)*(code6 - 32) with code6 = low nibble | (2-bit section) << 4.
// Lanes own columns so that every half2 pair is (column c, column c + 32), which makes the Q4_K
// and Q6_K stage interleave identical: with Q = lane>>3 (64-column region) and i = lane&7, a
// lane covers c = 64*Q + 4*i + p (p = 0..3) and its +32 twin. One warp round consumes exactly
// one 256-value GGML block; per staged token the lane reads its four stage slots as one LDS.128.
// fp16 accounting: Q4_K decodes code (0..15) and accumulates code*x plus a separate x-sum so the
// merge applies scale*Σ(code*x) - min*Σx in fp32; Q6_K folds the -32 into the decode magic
// (bias 1056 = 0x6420), so its merge is just scale*Σ(code*x).
#include "ops/linear/ggml_k/ggml_k.h"

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 128;      // four rows per block, one warp each
constexpr int kTokensPerTile = 8;
constexpr int kStageSlots = 128;   // half2 pairs per token: (column c, column c + 32)

// GGML row blocks are not 4-byte aligned for Q6_K (210 B stride), and the nibble planes are
// only guaranteed 4-byte aligned inside the row; load through a byte rebuild like the gemv.
__device__ __forceinline__ std::uint32_t load_u32_unaligned(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ float bf16_value(std::uint32_t word, int half_index) {
    return __uint_as_float(((word >> (16 * half_index)) & 0xffffu) << 16);
}

// Stage eight columns (c0..c0+3 and their +32 twins) for one token as four interleaved half2:
// slot p = (v_{c0+p}, v_{c0+32+p}).
__device__ __forceinline__ uint4 stage_ggml_pair(const __nv_bfloat16* __restrict__ src) {
    const uint2 a = *reinterpret_cast<const uint2*>(src);
    const uint2 b = *reinterpret_cast<const uint2*>(src + 32);
    const float2 a0 = {bf16_value(a.x, 0), bf16_value(a.x, 1)};
    const float2 a1 = {bf16_value(a.y, 0), bf16_value(a.y, 1)};
    const float2 b0 = {bf16_value(b.x, 0), bf16_value(b.x, 1)};
    const float2 b1 = {bf16_value(b.y, 0), bf16_value(b.y, 1)};
    const half2 p0 = __floats2half2_rn(a0.x, b0.x);
    const half2 p1 = __floats2half2_rn(a0.y, b0.y);
    const half2 p2 = __floats2half2_rn(a1.x, b1.x);
    const half2 p3 = __floats2half2_rn(a1.y, b1.y);
    return make_uint4(*reinterpret_cast<const unsigned*>(&p0),
                      *reinterpret_cast<const unsigned*>(&p1),
                      *reinterpret_cast<const unsigned*>(&p2),
                      *reinterpret_cast<const unsigned*>(&p3));
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void __launch_bounds__(kThreads)
    ggml_k_tile_kernel(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ rows,
                       const std::uint64_t* __restrict__ descriptors, __nv_bfloat16* __restrict__ out,
                       int n, int k, int tokens, std::int64_t out_stride) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int t0 = blockIdx.y * kTokensPerTile;
    if (row >= n || t0 >= tokens) { return; }
    const int rem = tokens - t0 < kTokensPerTile ? tokens - t0 : kTokensPerTile;
    const int lane = threadIdx.x & 31;
    const int q_region = lane >> 3;  // 64-column region of the block this lane decodes
    const int i_group = lane & 7;    // 4-column group inside the region

    const std::uint64_t descriptor = __ldg(descriptors + row);
    const bool q6 = (descriptor & 1u) != 0u;
    const unsigned char* row_bytes = rows + (descriptor >> 1);

    // Staging view of the current block: 32 units of 8 columns x 8 tokens = 256 jobs, two per
    // thread on this 128-thread block.
    __shared__ half2 x_stage[kTokensPerTile][kStageSlots];

    const half2 bias = __float2half2_rn(q6 ? 1056.0F : 1024.0F);
    const int nib_shift = q6 ? 4 * (q_region & 1) : 0;  // Q6_K nibble half selected per section
    float acc[kTokensPerTile] = {};

    for (int block = 0; block < k / 256; ++block) {
        for (int job = threadIdx.x; job < kStageSlots * 2; job += kThreads) {
            const int unit = job >> 3;          // 0..31 == (region, quarter)
            const int stage_token = job & 7;    // 0..7
            if (stage_token < rem) {
                const int col = 64 * (unit >> 3) + 4 * (unit & 7);
                reinterpret_cast<uint4*>(&x_stage[stage_token][0])[unit] = stage_ggml_pair(
                    x + static_cast<std::int64_t>(t0 + stage_token) * k + block * 256 + col);
            }
        }
        __syncthreads();
        const unsigned char* bytes =
            row_bytes + static_cast<std::int64_t>(block) * (q6 ? 210 : 144);

        half2 scale_pair;
        half2 min_pair = __float2half2_rn(0.0F);
        std::uint32_t packed0 = 0u, packed1 = 0u, high_word = 0u;
        const unsigned char* code_base;
        if (q6) {
            const float d = __half2float(*reinterpret_cast<const __half*>(bytes + 208));
            const int g = 4 * q_region + (i_group >> 2);
            const float s0 = d * static_cast<float>(static_cast<std::int8_t>(__ldg(bytes + 192 + g)));
            const float s1 =
                d * static_cast<float>(static_cast<std::int8_t>(__ldg(bytes + 192 + g + 2)));
            scale_pair = __floats2half2_rn(s0, s1);
            code_base = bytes + (q_region >> 1) * 64 + i_group * 4;  // +32 twin row is lo1
            packed0 = load_u32_unaligned(code_base);
            packed1 = load_u32_unaligned(code_base + 32);
            high_word = load_u32_unaligned(bytes + 128 + (q_region >> 1) * 32 + i_group * 4);
        } else {
            const float d = __half2float(*reinterpret_cast<const __half*>(bytes));
            const float dmin = __half2float(*reinterpret_cast<const __half*>(bytes + 2));
            float s[2];
            float m[2];
#pragma unroll
            for (int gi = 0; gi < 2; ++gi) {
                const int group = 2 * q_region + gi;
                const int low = group & 3;
                const int raw_s = __ldg(bytes + 4 + low);
                const int raw_m = __ldg(bytes + 8 + low);
                const int raw_hi = __ldg(bytes + 12 + low);
                s[gi] = d * (group < 4 ? raw_s & 63 : (raw_hi & 15) | ((raw_s >> 6) << 4));
                m[gi] = dmin * (group < 4 ? raw_m & 63 : (raw_hi >> 4) | ((raw_m >> 6) << 4));
            }
            scale_pair = __floats2half2_rn(s[0], s[1]);
            min_pair = __floats2half2_rn(m[0], m[1]);
            code_base = bytes + 16 + q_region * 32 + i_group * 4;
            packed0 = load_u32_unaligned(code_base);
        }

        half2 sums[kTokensPerTile];
        half2 sums_v[kTokensPerTile];
#pragma unroll
        for (int t = 0; t < kTokensPerTile; ++t) {
            sums[t] = __float2half2_rn(0.0F);
            sums_v[t] = __float2half2_rn(0.0F);
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits;
            if (q6) {
                const int sh = 8 * p + nib_shift;
                const std::uint32_t a =
                    ((packed0 >> sh) & 0x0Fu) | (((high_word >> sh) & 0x3u) << 4);
                const std::uint32_t b =
                    ((packed1 >> sh) & 0x0Fu) | (((high_word >> (sh + 2)) & 0x3u) << 4);
                bits = 0x64006400u | a | (b << 16);
            } else {
                const std::uint32_t w = packed0 >> (8 * p);  // low nib = column c, high = c + 32
                bits = 0x64006400u | (w & 0x0Fu) | (((w >> 4) & 0x0Fu) << 16);
            }
            const half2 code2 = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
#pragma unroll
            for (int t = 0; t < kTokensPerTile; ++t) {
                if (t >= rem) { break; }
                const half2 xv = x_stage[t][32 * q_region + 4 * i_group + p];
                sums[t] = __hfma2(code2, xv, sums[t]);
                sums_v[t] = __hadd2(sums_v[t], xv);
            }
        }

        const float2 scale_f = __half22float2(scale_pair);
        const float2 min_f = __half22float2(min_pair);
#pragma unroll
        for (int t = 0; t < kTokensPerTile; ++t) {
            if (t >= rem) { break; }
            const float2 s = __half22float2(sums[t]);
            const float2 v = __half22float2(sums_v[t]);
            acc[t] += scale_f.x * s.x - min_f.x * v.x + scale_f.y * s.y - min_f.y * v.y;
        }
        __syncthreads();  // the next block round rewrites the stage
    }

#pragma unroll
    for (int t = 0; t < kTokensPerTile; ++t) {
        if (t >= rem) { break; }
        const float value = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[out_stride * (t0 + t) + row] = __float2bfloat16_rn(value);
        }
    }
}

} // namespace

void ggml_k_launch_tile_pascal(const Tensor& x, const Weight& weight, const Tensor& out,
                               cudaStream_t stream) {
    const int n = weight.n;
    const int k = weight.k;
    const int tokens = x.ne[1];
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* rows = static_cast<const unsigned char*>(weight.qdata);
    const auto* descriptors = static_cast<const std::uint64_t*>(weight.qhigh);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    const dim3 grid((n + 3) / 4, (tokens + kTokensPerTile - 1) / kTokensPerTile);
    ggml_k_tile_kernel<<<grid, kThreads, 0, stream>>>(input, rows, descriptors, output, n, k,
                                                      tokens, out.nb[1] / 2);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
