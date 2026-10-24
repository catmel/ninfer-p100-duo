// Pascal (sm_60) Q5 groupwise A16 decode GEMV, T=1.
//
// Companion to q4_decode_hfma2_pascal.cu for the int5 family. Storage is the Q5 row-split
// layout (Q5RowSplitStorage): 64-value groups with 32 low-nibble code bytes, one high byte per
// 8 values carrying bit 5 of each, and one fp16 scale per group. The qualified V100 GEMV
// unpacks to float before its FMA chain; this kernel keeps the products in half2 exactly like
// Q5SimtDecodeAtom::decode_eight -- magic half2 construction from the nibble, bit-5 insertion
// through the XOR-0xFF sign fold, one HSUB2 by 1040, then HFMA2 against the interleaved
// staged activation (half2 slot j of an 8-value block holds (v_j, v_{j+4})).
//
// Semantics (scale application, reduction, rounding) match the qualified gemv; the bounded
// half2 group interior is qualified against the naive FP64 oracle
// (docs/p100-migration.md section 7).
#include "ops/linear/q5/q5_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kMaxStagedK = 17408;  // largest registered Q5 k; 34 KB of smem.

__device__ __forceinline__ void q5_decode_word(std::uint32_t word, std::uint8_t high_byte,
                                               const half2* xp, half2 (&sum)[4], half2 bias) {
    // fold bit i = 1 - bit5(v_i): a set stored bit5 selects the negative half of the fold
    // ((nib | 16) ^ 16) - 16 = nib - 16, applied once by the 1040 bias subtract.
    const std::uint32_t folded = static_cast<std::uint32_t>(high_byte ^ 0xFFu);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        bits |= (((folded >> pair) & 1u) << 4) | (((folded >> (pair + 4)) & 1u) << 20);
        const half2 weights = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
        sum[pair]           = __hfma2(weights, xp[pair], sum[pair]);
    }
}

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void q5_decode_hfma2_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high_bits, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k) {
    extern __shared__ half2 x_stage[];  // [k / 2], 8-value blocks interleaved as (v_j, v_{j+4})
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int tid  = static_cast<int>(threadIdx.x);

    for (int base = tid * 8; base < k; base += RowsPerCta * 32 * 8) {
        const uint4 raw = load_vec<uint4>(x + base);
        const float2 f0 = bf16x2_bits_to_float2(raw.x);  // v0, v1
        const float2 f1 = bf16x2_bits_to_float2(raw.y);  // v2, v3
        const float2 f2 = bf16x2_bits_to_float2(raw.z);  // v4, v5
        const float2 f3 = bf16x2_bits_to_float2(raw.w);  // v6, v7
        const half2 p0 = __floats2half2_rn(f0.x, f2.x);  // v0, v4
        const half2 p1 = __floats2half2_rn(f0.y, f2.y);  // v1, v5
        const half2 p2 = __floats2half2_rn(f1.x, f3.x);  // v2, v6
        const half2 p3 = __floats2half2_rn(f1.y, f3.y);  // v3, v7
        reinterpret_cast<uint4*>(x_stage)[base / 8] = make_uint4(
            *reinterpret_cast<const unsigned*>(&p0), *reinterpret_cast<const unsigned*>(&p1),
            *reinterpret_cast<const unsigned*>(&p2), *reinterpret_cast<const unsigned*>(&p3));
    }
    __syncthreads();

    const int row = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    if (row >= rows) { return; }
    const auto* code_row  = codes + static_cast<std::int64_t>(row) * (k / 2);
    const auto* high_row  = high_bits + static_cast<std::int64_t>(row) * (k / 8);
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * ((k / 64) * 2);

    const half2 bias = __float2half2_rn(1040.0F);  // 1024 + 16
    const int   words = static_cast<int>(k) / 8;
    float acc          = 0.0F;
    // Warp round = 64 words (512 values, 8 groups). A lane owns two adjacent words, which
    // always share one 8-word group, hence one scale.
    for (int word_base = lane * 2; word_base < words; word_base += 32 * 2) {
        const int group = word_base / 8;
        const std::uint16_t scale_bits =
            *reinterpret_cast<const std::uint16_t*>(scale_row + group * 2);
        const half2 scale = __half2half2(__ushort_as_half(scale_bits));

        const uint2 cw = load_vec<uint2>(code_row + word_base * 4);  // 2 words = 8 bytes
        const std::uint16_t hi_pair = *reinterpret_cast<const std::uint16_t*>(high_row + word_base);
        const half2* xp = reinterpret_cast<const half2*>(x_stage) + word_base * 4;
        half2 sums[4]   = {__float2half2_rn(0.0F), __float2half2_rn(0.0F), __float2half2_rn(0.0F),
                           __float2half2_rn(0.0F)};
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            q5_decode_word((&cw.x)[t], static_cast<std::uint8_t>(hi_pair >> (8 * t)), xp + t * 4,
                           sums, bias);
        }
        half2 merged = __hadd2(__hadd2(sums[0], sums[1]), __hadd2(sums[2], sums[3]));
        const float2 tail = __half22float2(__hmul2(merged, scale));
        acc += tail.x + tail.y;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
}

} // namespace

void launch_q5_decode_hfma2_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t k    = w.k;
    const std::int32_t rows = w.n;
    if (x.ne[1] != 1 || k <= 0 || (k % 512) != 0 || w.padded_shape[1] != k ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15U) != 0) {
        throw std::invalid_argument("Q5 hfma2 decode requires T=1 and k divisible by 512");
    }
    const std::size_t stage_bytes = static_cast<std::size_t>(k) * sizeof(std::uint16_t);
    if (stage_bytes > static_cast<std::size_t>(kMaxStagedK) * sizeof(std::uint16_t)) {
        throw std::invalid_argument("Q5 hfma2 decode: k exceeds the staged activation bound");
    }
    constexpr int kRowsPerCta = 16;
    const unsigned blocks = static_cast<unsigned>((rows + kRowsPerCta - 1) / kRowsPerCta);
    q5_decode_hfma2_kernel<kRowsPerCta><<<blocks, kRowsPerCta * 32, stage_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
