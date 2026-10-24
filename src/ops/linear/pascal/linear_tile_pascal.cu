// Pascal (sm_60) wide-T A16 decode tile GEMM for the row-split int families (W8/Q4/Q5/Q6).
//
// GP100 has no tensor cores, so every width above T=4 on the V100 design lands on a SIMT
// kernel that re-decodes each weight value once per token. This kernel decodes each weight
// word ONCE and multiplies it against up to 8 staged token rows with HFMA2 -- the llama.cpp
// Pascal (QR_GEMM / p100) pattern: weight bytes are the scarce resource on this card, and one
// decode amortized over the token tile turns the issue-bound per-token decode cost into a
// ~1/8 share.
//
//   * grid.y chunks the token axis by kTokensPerTile; ragged token chunks break out of the
//     stage loop at the dynamic remainder, so any token count is supported (stale stage lanes
//     for absent tokens feed only discarded accumulators);
//   * each CTA has 8 warps, one row per warp (grid.x chunks rows);
//   * activations stream through shared memory as half2 in the decode atoms' pair interleave
//     (weight slot j pairs (v_j, v_{j+4})), one K chunk of 2048 values at a time (32 KB stage),
//     so the staged footprint is independent of k;
//   * per group the lane's half2 accumulators merge, take the group's fp16 scale once, and
//     join per-token FP32 accumulators; the warp reduction and the single bf16 RNE store
//     happen once per row and token at the end of k;
//   * the decode arithmetic is the exact magic/XOR/bias pattern of the qualified T=1 GEMVs
//     (w8/q4/q5/q6_decode_hfma2_pascal.cu), so one oracle pass qualifies both widths.
//
// Geometry guards mirror the T=1 kernels: rows contiguous, k % 512 == 0, k <= 17408 (largest
// registered row), padded rows == k.
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads       = 256;            // 8 warps, one row per warp
constexpr int kRowsPerCta    = 8;
constexpr int kTokensPerTile = 8;
constexpr int kChunkVals     = 2048;           // 8 KB of half2 stage per token
constexpr int kStageHalf2    = kChunkVals / 2;
constexpr int kStageUint4    = kStageHalf2 / 4;  // 8-value blocks per token in the stage
constexpr int kMaxK          = 17408;

__device__ __forceinline__ uint4 stage_interleaved(const __nv_bfloat16* __restrict__ src) {
    const uint4 raw = load_vec<uint4>(src);
    const float2 f0 = bf16x2_bits_to_float2(raw.x);  // v0, v1
    const float2 f1 = bf16x2_bits_to_float2(raw.y);  // v2, v3
    const float2 f2 = bf16x2_bits_to_float2(raw.z);  // v4, v5
    const float2 f3 = bf16x2_bits_to_float2(raw.w);  // v6, v7
    const half2 p0 = __floats2half2_rn(f0.x, f2.x);  // v0, v4
    const half2 p1 = __floats2half2_rn(f0.y, f2.y);  // v1, v5
    const half2 p2 = __floats2half2_rn(f1.x, f3.x);  // v2, v6
    const half2 p3 = __floats2half2_rn(f1.y, f3.y);  // v3, v7
    return make_uint4(*reinterpret_cast<const unsigned*>(&p0),
                      *reinterpret_cast<const unsigned*>(&p1),
                      *reinterpret_cast<const unsigned*>(&p2),
                      *reinterpret_cast<const unsigned*>(&p3));
}

// One staged token's four half2 pairs at a word position.
__device__ __forceinline__ void ld_pairs(const half2* xp, half2 (&out)[4]) {
    const uint4 v = load_vec<uint4>(reinterpret_cast<const __nv_bfloat16*>(xp));
    out[0] = *reinterpret_cast<const half2*>(&v.x);
    out[1] = *reinterpret_cast<const half2*>(&v.y);
    out[2] = *reinterpret_cast<const half2*>(&v.z);
    out[3] = *reinterpret_cast<const half2*>(&v.w);
}

// W8 stages a different pairing than the nibble families: the byte-pair halves decode to
// (v_j, v_{j+2}) within each 4-byte chunk, so the stage slot order is (v0,v2),(v1,v3),
// (v4,v6),(v5,v7) -- the geometry of the qualified w8 T=1 kernel.
__device__ __forceinline__ uint4 stage_interleaved_w8(const __nv_bfloat16* __restrict__ src) {
    const uint4 raw = load_vec<uint4>(src);
    const float2 f0 = bf16x2_bits_to_float2(raw.x);  // v0, v1
    const float2 f1 = bf16x2_bits_to_float2(raw.y);  // v2, v3
    const float2 f2 = bf16x2_bits_to_float2(raw.z);  // v4, v5
    const float2 f3 = bf16x2_bits_to_float2(raw.w);  // v6, v7
    const half2 p0 = __floats2half2_rn(f0.x, f1.x);  // v0, v2
    const half2 p1 = __floats2half2_rn(f0.y, f1.y);  // v1, v3
    const half2 p2 = __floats2half2_rn(f2.x, f3.x);  // v4, v6
    const half2 p3 = __floats2half2_rn(f2.y, f3.y);  // v5, v7
    return make_uint4(*reinterpret_cast<const unsigned*>(&p0),
                      *reinterpret_cast<const unsigned*>(&p1),
                      *reinterpret_cast<const unsigned*>(&p2),
                      *reinterpret_cast<const unsigned*>(&p3));
}

// Kind: 4 = q4, 5 = q5, 6 = q6 (nibbles + plane). W8 has its own word shape below.
template <int Kind>
struct TileNibbleTraits;

template <>
struct TileNibbleTraits<4> {
    static __device__ __forceinline__ void
    accumulate(std::uint32_t word, std::uint32_t plane, const half2* stage, int half2_base,
               half2 (&sums)[4][8], half2 bias) {
        (void)plane;
        const std::uint32_t xored = word ^ 0x88888888u;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const std::uint32_t bits = ((xored >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            const half2 weights      = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                sums[pair][t] = __hfma2(weights, stage[t * kStageUint4 * 4 + half2_base + pair],
                                        sums[pair][t]);
            }
        }
    }
};

template <>
struct TileNibbleTraits<5> {
    static __device__ __forceinline__ void
    accumulate(std::uint32_t word, std::uint32_t plane, const half2* stage, int half2_base,
               half2 (&sums)[4][8], half2 bias) {
        const std::uint32_t folded = (plane & 0xFFu) ^ 0xFFu;  // bit i = 1 - bit5(v_i)
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((folded >> pair) & 1u) << 4) | (((folded >> (pair + 4)) & 1u) << 20);
            const half2 weights = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                sums[pair][t] = __hfma2(weights, stage[t * kStageUint4 * 4 + half2_base + pair],
                                        sums[pair][t]);
            }
        }
    }
};

template <>
struct TileNibbleTraits<6> {
    static __device__ __forceinline__ void
    accumulate(std::uint32_t word, std::uint32_t plane, const half2* stage, int half2_base,
               half2 (&sums)[4][8], half2 bias) {
        const std::uint32_t folded = (plane & 0xFFFFu) ^ 0xAAAAu;  // 2 bits per value
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((folded >> (2 * pair)) & 3u) << 4) |
                    (((folded >> (2 * pair + 8)) & 3u) << 20);
            const half2 weights = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                sums[pair][t] = __hfma2(weights, stage[t * kStageUint4 * 4 + half2_base + pair],
                                        sums[pair][t]);
            }
        }
    }
};

template <int Kind>
__device__ __forceinline__ half2 tile_bias() {
    return __float2half2_rn(Kind == 5 ? 1040.0F : (Kind == 6 ? 1056.0F : 1032.0F));
}

template <int Kind>
__global__ __launch_bounds__(kThreads, 2) void decode_tile_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high_bits, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    extern __shared__ half2 x_stage[];  // [kTokensPerTile][kStageHalf2]
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
    const int t0   = static_cast<int>(blockIdx.y) * kTokensPerTile;
    const int rem  = tokens - t0 < kTokensPerTile ? tokens - t0 : kTokensPerTile;
    if (row >= rows || t0 >= tokens) { return; }

    const auto* code_row = codes + static_cast<std::int64_t>(row) * (k / 2);
    const auto* high_row = Kind == 4 ? nullptr
                           : high_bits + static_cast<std::int64_t>(row) *
                                             (Kind == 5 ? k / 8 : k / 4);
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * ((k / 64) * 2);

    const half2 bias = tile_bias<Kind>();
    float acc[kTokensPerTile] = {};

    for (int chunk = 0; chunk < k; chunk += kChunkVals) {
        const int cv = k - chunk < kChunkVals ? k - chunk : kChunkVals;
        // Cooperative interleaved staging of every token's chunk (uint4 = 8 values per slot).
        for (int block8 = 0; block8 * kThreads * 8 < cv; ++block8) {
            const int slot = block8 * kThreads + static_cast<int>(threadIdx.x);
            if (slot * 8 >= cv) { break; }
            for (int t = 0; t < rem; ++t) {
                reinterpret_cast<uint4*>(x_stage)[t * kStageUint4 + slot] = stage_interleaved(
                    x + static_cast<std::int64_t>(t0 + t) * k + chunk + slot * 8);
            }
        }
        __syncthreads();

        // Word loop: a lane owns two adjacent words (16 values), always inside one 64-value
        // group; a warp round covers 64 words = 8 groups = 512 values.
        half2 sums[4][8];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
#pragma unroll
            for (int t = 0; t < 8; ++t) { sums[p][t] = __float2half2_rn(0.0F); }
        }
        const int chunk_words = cv / 8;
        for (int round = 0; round * 64 < chunk_words; ++round) {
            const int word_local = round * 64 + lane * 2;
            if (word_local + 1 >= chunk_words) { break; }  // chunks are multiples of 512 values
            const int word_global = chunk / 8 + word_local;
            const int group       = word_global / 8;
            const std::uint16_t scale_bits =
                *reinterpret_cast<const std::uint16_t*>(scale_row + group * 2);

            const uint2 cw = load_vec<uint2>(code_row + chunk / 2 + word_local * 4);  // 8 bytes
            std::uint32_t plane0 = 0, plane1 = 0;
            if (Kind == 5) {
                const std::uint16_t hi_pair = *reinterpret_cast<const std::uint16_t*>(
                    high_row + chunk / 8 + word_local);
                plane0 = hi_pair & 0xFFu;
                plane1 = hi_pair >> 8;
            } else if (Kind == 6) {
                plane0 = *reinterpret_cast<const std::uint32_t*>(high_row +
                                                                 (chunk / 8 + word_local) * 2);
                plane1 = 0;  // both words inside one u32 below
            }
            const half2* stage = &x_stage[0];
            TileNibbleTraits<Kind>::accumulate(cw.x, plane0, stage, word_local * 4, sums, bias);
            TileNibbleTraits<Kind>::accumulate(
                cw.y, Kind == 6 ? (plane0 >> 16) : plane1, stage, word_local * 4 + 4, sums, bias);

            // End of the round == end of this lane's slice of the group: merge per token,
            // scale once, fold into the FP32 accumulator, reset.
            const half2 scale = __half2half2(__ushort_as_half(scale_bits));
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const half2 merged =
                    __hadd2(__hadd2(sums[0][t], sums[1][t]), __hadd2(sums[2][t], sums[3][t]));
                const float2 tail = __half22float2(__hmul2(merged, scale));
                acc[t] += tail.x + tail.y;
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
#pragma unroll
                for (int t = 0; t < 8; ++t) { sums[p][t] = __float2half2_rn(0.0F); }
            }
        }
        __syncthreads();  // stage buffer reuse
    }

#pragma unroll
    for (int t = 0; t < kTokensPerTile; ++t) {
        if (t >= rem) { break; }
        const float value = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            // out is {rows, tokens} with the row axis contiguous: element (row, t) sits at
            // data + t * rows.
            out[static_cast<std::int64_t>(t0 + t) * rows + row] = __float2bfloat16_rn(value);
        }
    }
}

// W8 tile: one byte per value, 32-value groups. A lane owns a uint4 = 16 values (half a
// group; lanes 2i and 2i+1 share group i's scale), exactly the T=1 kernel's phase geometry,
// but each decoded byte-pair feeds all staged tokens.
__global__ __launch_bounds__(kThreads, 2) void decode_tile_w8_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t rows,
    std::int32_t k, std::int32_t tokens) {
    extern __shared__ half2 x_stage[];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
    const int t0   = static_cast<int>(blockIdx.y) * kTokensPerTile;
    const int rem  = tokens - t0 < kTokensPerTile ? tokens - t0 : kTokensPerTile;
    if (row >= rows || t0 >= tokens) { return; }

    const auto* code_row  = codes + static_cast<std::int64_t>(row) * k;
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * ((k / 32) * 2);
    const half2 bias      = __float2half2_rn(1152.0F);
    float acc[kTokensPerTile] = {};

    for (int chunk = 0; chunk < k; chunk += kChunkVals) {
        const int cv = k - chunk < kChunkVals ? k - chunk : kChunkVals;
        for (int block8 = 0; block8 * kThreads * 8 < cv; ++block8) {
            const int slot = block8 * kThreads + static_cast<int>(threadIdx.x);
            if (slot * 8 >= cv) { break; }
            for (int t = 0; t < rem; ++t) {
                reinterpret_cast<uint4*>(x_stage)[t * kStageUint4 + slot] =
                    stage_interleaved_w8(
                        x + static_cast<std::int64_t>(t0 + t) * k + chunk + slot * 8);
            }
        }
        __syncthreads();

        // Warp round = 256 values (8 groups of 32); a lane owns 8 contiguous values, so its
        // scale lookup serves the whole round (lanes 4i..4i+3 share group i).
        for (int base = 0; base + 256 <= cv; base += 256) {
            const int k_local = base + lane * 8;
            const int group   = (chunk + base) / 32 + (lane >> 2);
            const std::uint16_t scale_bits =
                *reinterpret_cast<const std::uint16_t*>(scale_row + group * 2);
            const uint2 cw = load_vec<uint2>(code_row + chunk + k_local);  // 8 bytes = 8 values
            half2 sums[2][8];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
#pragma unroll
                for (int t = 0; t < 8; ++t) { sums[p][t] = __float2half2_rn(0.0F); }
            }
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                const std::uint32_t xored = (&cw.x)[hw] ^ 0x80808080u;
                const std::uint32_t even  = (xored & 0x00FF00FFu) | 0x64006400u;
                const std::uint32_t odd   = ((xored >> 8) & 0x00FF00FFu) | 0x64006400u;
                const half2 we = __hsub2(*reinterpret_cast<const half2*>(&even), bias);
                const half2 wo = __hsub2(*reinterpret_cast<const half2*>(&odd), bias);
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const half2* xp = &x_stage[t * kStageHalf2 + k_local / 2 + hw * 2];
                    sums[0][t]      = __hfma2(we, xp[0], sums[0][t]);
                    sums[1][t]      = __hfma2(wo, xp[1], sums[1][t]);
                }
            }
            const half2 scale = __half2half2(__ushort_as_half(scale_bits));
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const float2 tail =
                    __half22float2(__hmul2(__hadd2(sums[0][t], sums[1][t]), scale));
                acc[t] += tail.x + tail.y;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < kTokensPerTile; ++t) {
        if (t >= rem) { break; }
        const float value = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(t0 + t) * rows + row] = __float2bfloat16_rn(value);
        }
    }
}

void launch_tile_common(const Tensor& x, const Weight& w, Tensor& out, const char* name) {
    const std::int32_t k = w.k;
    if (x.ne[1] <= 1 || k <= 0 || (k % 512) != 0 || w.padded_shape[1] != k ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15U) != 0) {
        throw std::invalid_argument(std::string(name) +
                                    " requires T>=2 and k divisible by 512");
    }
}

} // namespace

void launch_w8_decode_tile_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    launch_tile_common(x, w, out, "W8 tile decode");
    const std::int32_t k = w.k, rows = w.n, tokens = x.ne[1];
    if (k > kMaxK) { throw std::invalid_argument("W8 tile decode: k exceeds the staged bound"); }
    const std::size_t stage_bytes = sizeof(half2) * kTokensPerTile * kStageHalf2;
    dim3 grid(static_cast<unsigned>((rows + kRowsPerCta - 1) / kRowsPerCta),
              static_cast<unsigned>((tokens + kTokensPerTile - 1) / kTokensPerTile));
    decode_tile_w8_kernel<<<grid, kThreads, stage_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k,
        tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_decode_tile_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    launch_tile_common(x, w, out, "Q4 tile decode");
    const std::int32_t k = w.k, rows = w.n, tokens = x.ne[1];
    if (k > kMaxK) { throw std::invalid_argument("Q4 tile decode: k exceeds the staged bound"); }
    const std::size_t stage_bytes = sizeof(half2) * kTokensPerTile * kStageHalf2;
    dim3 grid(static_cast<unsigned>((rows + kRowsPerCta - 1) / kRowsPerCta),
              static_cast<unsigned>((tokens + kTokensPerTile - 1) / kTokensPerTile));
    decode_tile_kernel<4><<<grid, kThreads, stage_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        nullptr, static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_decode_tile_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    launch_tile_common(x, w, out, "Q5 tile decode");
    const std::int32_t k = w.k, rows = w.n, tokens = x.ne[1];
    if (k > kMaxK) { throw std::invalid_argument("Q5 tile decode: k exceeds the staged bound"); }
    const std::size_t stage_bytes = sizeof(half2) * kTokensPerTile * kStageHalf2;
    dim3 grid(static_cast<unsigned>((rows + kRowsPerCta - 1) / kRowsPerCta),
              static_cast<unsigned>((tokens + kTokensPerTile - 1) / kTokensPerTile));
    decode_tile_kernel<5><<<grid, kThreads, stage_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q6_decode_tile_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    launch_tile_common(x, w, out, "Q6 tile decode");
    const std::int32_t k = w.k, rows = w.n, tokens = x.ne[1];
    if (k > kMaxK) { throw std::invalid_argument("Q6 tile decode: k exceeds the staged bound"); }
    const std::size_t stage_bytes = sizeof(half2) * kTokensPerTile * kStageHalf2;
    dim3 grid(static_cast<unsigned>((rows + kRowsPerCta - 1) / kRowsPerCta),
              static_cast<unsigned>((tokens + kTokensPerTile - 1) / kTokensPerTile));
    decode_tile_kernel<6><<<grid, kThreads, stage_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
