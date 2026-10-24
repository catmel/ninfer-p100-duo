// Pascal (sm_60) Q4 groupwise A16 decode GEMV, T=1.
//
// Companion to w8_decode_hfma2_pascal.cu for the int4 family. The qualified V100 GEMV
// (q4_rowsplit_gemv.cu) already unpacks nibbles with the fp16 magic but converts back to
// float for its FMA chain: measured inner-loop cost is ~6 SASS instructions per value on
// GP100, where decode is instruction-issue bound. This kernel keeps the products in half2:
//
//   * one word (4 bytes, 8 nibbles) decodes to four half2 weight pairs (n_j, n_{j+4}) with
//     the XOR-0x88 sign fold and the 0x6400 offset magic -- 13 integer ops + 4 HSUB2, exactly
//     the pattern Q4SimtDecodeAtom::decode_eight uses minus the two conversions per pair
//     and the 8 FMUL/FFMA;
//   * the staged activation row is interleaved to match that pairing -- half2 slot j of an
//     8-value block holds (v_j, v_{j+4});
//   * the staged activation row is interleaved to match that pairing -- half2 slot j of an
//     8-value block holds (v_j, v_{j+4}); each lane owns two words (16 values) inside one
//     64-value group, so the products accumulate in four half2 lanes (two terms each for
//     normalized values), merge, take the group's fp16 scale once, and join the FP32
//     accumulator before the warp reduction and the single bf16 RNE store.
//
// Geometry comes from Q4RowSplitStorage: 64-value groups, 32 code bytes and one fp16 scale
// per group, rows contiguous. Semantics (scale application, reduction, rounding) match the
// qualified gemv; the bounded half2 group interior is qualified against the naive FP32
// oracle (docs/p100-migration.md section 7).
#include "ops/linear/q4/q4_launch.h"

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

constexpr int kMaxStagedK = 17408;  // largest registered Q4 k; 34 KB of smem.

__device__ __forceinline__ void q4_decode_word(std::uint32_t word, const half2* xp, half2 (&sum)[4],
                                               half2 bias) {
    // 0x6400 | (nibble ^ 8) encodes 1032 + signed nibble exactly in fp16; the bias subtract
    // lands it on [-8, 7]. Pair j of the word is (nibble j, nibble j+4), matching the staged
    // activation interleave.
    const std::uint32_t xored = word ^ 0x88888888u;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint32_t bits = ((xored >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        const half2 weights      = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
        sum[pair]                = __hfma2(weights, xp[pair], sum[pair]);
    }
}

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void q4_decode_hfma2_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t rows,
    std::int32_t k) {
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
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * ((k / 64) * 2);

    const half2 bias  = __float2half2_rn(1032.0F);  // 1024 + 8
    const int   words = static_cast<int>(k) / 8;     // 8 nibbles per word
    float acc         = 0.0F;
    // Each warp round covers 64 words (512 values, 8 groups). A lane owns two adjacent words
    // (word_base = round*64 + lane*2): a group holds 8 words and any two adjacent aligned
    // words fall inside one group, so each lane's uint2 shares exactly one scale.
    for (int word_base = lane * 2; word_base < words; word_base += 32 * 2) {
        const int group = word_base / 8;
        const std::uint16_t scale_bits =
            *reinterpret_cast<const std::uint16_t*>(scale_row + group * 2);
        const half2 scale = __half2half2(__ushort_as_half(scale_bits));

        const uint2 cw = load_vec<uint2>(code_row + word_base * 4);  // 2 words = 8 bytes
        const half2* xp = reinterpret_cast<const half2*>(x_stage) + word_base * 4;
        half2 sums[4]   = {__float2half2_rn(0.0F), __float2half2_rn(0.0F), __float2half2_rn(0.0F),
                           __float2half2_rn(0.0F)};
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            q4_decode_word((&cw.x)[t], xp + t * 4, sums, bias);
        }
        half2 merged = __hadd2(__hadd2(sums[0], sums[1]), __hadd2(sums[2], sums[3]));
        const float2 tail = __half22float2(__hmul2(merged, scale));
        acc += tail.x + tail.y;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
}

} // namespace

void launch_q4_decode_hfma2_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t k    = w.k;
    const std::int32_t rows = w.n;
    // k % 512 == 0: whole warp rounds (64 words = 512 values each), every code read is an
    // 8-byte vector, and each lane's two words stay inside one 64-value group.
    if (x.ne[1] != 1 || k <= 0 || (k % 512) != 0 || w.padded_shape[1] != k ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15U) != 0) {
        throw std::invalid_argument("Q4 hfma2 decode requires T=1 and k divisible by 512");
    }
    const std::size_t stage_bytes = static_cast<std::size_t>(k) * sizeof(std::uint16_t);
    if (stage_bytes > static_cast<std::size_t>(kMaxStagedK) * sizeof(std::uint16_t)) {
        throw std::invalid_argument("Q4 hfma2 decode: k exceeds the staged activation bound");
    }
    constexpr int kRowsPerCta = 16;
    const unsigned blocks = static_cast<unsigned>((rows + kRowsPerCta - 1) / kRowsPerCta);
    q4_decode_hfma2_kernel<kRowsPerCta><<<blocks, kRowsPerCta * 32, stage_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
