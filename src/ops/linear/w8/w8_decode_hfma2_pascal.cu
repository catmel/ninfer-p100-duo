// Pascal (sm_60) W8 A16 decode GEMV, T=1.
//
// The V100-era rowsplit SIMT route dequantizes each int8 code into a float and runs an FP32 FMA
// chain -- about five SASS instructions per weight on GP100, where decode is instruction-issue
// bound (no tensor cores, no dp4a at sm_60, HBM2 at ~732 GB/s). This kernel re-expresses the same
// dot product around the full-rate HFMA2 pipe, following the fp16 matvec pattern established by
// the donor projects (docs/p100-donor-catalog.md, qr0n note 0007):
//
//   * the activation row is staged once per CTA as half2 in shared memory (bf16 -> fp16 is exact
//     for normed activations), amortizing the conversion over all rows the CTA covers;
//   * four int8 codes unpack to two half2 registers with five integer logic ops via the 1024-
//     offset magic: (0x6400 | byte) encodes 1024 + byte exactly as fp16, and subtracting the
//     1152 bias yields the signed byte. The unpacked weight pairs are (byte0, byte2) and
//     (byte1, byte3), and the staged activation is interleaved to match -- the pairing is a
//     layout choice, not a stride change; the set of products per warp lane is invariant.
//   * eight products per lane per group accumulate in two half2 lanes (four terms each:
//     4 * 127 * |x| stays far below the 65504 fp16 ceiling for normed activations), merge to
//     one half2, multiply by the group's fp16 scale, then join the FP32 accumulator and the
//     single bf16 round-to-nearest-even store of the qualified float route.
//
// Semantics match w8_rowsplit_k16384_decode_kernel (w8_rowsplit_gemm_decode.cu): per-32-value
// fp16 scales, warp-per-row ownership, warp reduction, bf16 RNE output. The only new arithmetic
// is the bounded half2 group interior, qualified against the naive FP32 oracle like every other
// route (docs/p100-migration.md section 7).
#include "ops/linear/w8/w8_launch.h"

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

constexpr int kMaxStagedK = 17408;  // largest registered W8 k (down projection); 34 KB of smem.

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_decode_hfma2_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t rows,
    std::int32_t k) {
    extern __shared__ half2 x_stage[];  // [k / 2], interleaved as (v0,v2),(v1,v3),(v4,v6),(v5,v7)
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int tid  = static_cast<int>(threadIdx.x);

    // Stage the activation row cooperatively: eight consecutive bf16 per participating thread.
    for (int base = tid * 8; base < k; base += RowsPerCta * 32 * 8) {
        const uint4 raw = load_vec<uint4>(x + base);
        const float2 f0 = bf16x2_bits_to_float2(raw.x);  // v0, v1
        const float2 f1 = bf16x2_bits_to_float2(raw.y);  // v2, v3
        const float2 f2 = bf16x2_bits_to_float2(raw.z);  // v4, v5
        const float2 f3 = bf16x2_bits_to_float2(raw.w);  // v6, v7
        const half2 p0 = __floats2half2_rn(f0.x, f1.x);  // v0, v2
        const half2 p1 = __floats2half2_rn(f0.y, f1.y);  // v1, v3
        const half2 p2 = __floats2half2_rn(f2.x, f3.x);  // v4, v6
        const half2 p3 = __floats2half2_rn(f2.y, f3.y);  // v5, v7
        reinterpret_cast<uint4*>(x_stage)[base / 8] = make_uint4(
            *reinterpret_cast<const unsigned*>(&p0), *reinterpret_cast<const unsigned*>(&p1),
            *reinterpret_cast<const unsigned*>(&p2), *reinterpret_cast<const unsigned*>(&p3));
    }
    __syncthreads();

    const int row = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    if (row >= rows) { return; }
    const auto* code_row  = codes + static_cast<std::int64_t>(row) * k;
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * ((k / 32) * 2);

    const half2 bias  = __float2half2_rn(1152.0F);  // 1024 + 128
    // Warp phase = 256 values = eight 32-value groups: lane j owns the 8 values at j*8 and
    // they all sit inside group (j >> 2), so each lane applies exactly one group scale --
    // the same ownership the qualified float decode kernel uses.
    const int phases = static_cast<int>(k) / 256;
    float acc        = 0.0F;
    for (int phase = 0; phase < phases; ++phase) {
        const std::int32_t k_base = phase * 256 + lane * 8;
        const uint2 words         = load_vec<uint2>(code_row + k_base);
        const std::uint16_t scale_bits =
            *reinterpret_cast<const std::uint16_t*>(scale_row + (phase * 8 + (lane >> 2)) * 2);
        const half2 scale = __half2half2(__ushort_as_half(scale_bits));
        const half2* xp   = reinterpret_cast<const half2*>(x_stage + k_base / 2);

        half2 sum_a = __float2half2_rn(0.0F);
        half2 sum_b = __float2half2_rn(0.0F);
#pragma unroll
        for (int half_word = 0; half_word < 2; ++half_word) {
            // The artifact stores raw two's-complement int8 bytes; flipping the sign bit of
            // every byte converts them to the unsigned offset (value + 128) that the 1024-
            // magnitude magic plus the 1152 bias decodes back to the signed value.
            const std::uint32_t word = (&words.x)[half_word] ^ 0x80808080u;
            const std::uint32_t even = (word & 0x00FF00FFu) | 0x64006400u;
            const std::uint32_t odd  = ((word >> 8) & 0x00FF00FFu) | 0x64006400u;
            const half2 w_even = __hsub2(*reinterpret_cast<const half2*>(&even), bias);
            const half2 w_odd  = __hsub2(*reinterpret_cast<const half2*>(&odd), bias);
            sum_a = __hfma2(w_even, xp[half_word * 2], sum_a);
            sum_b = __hfma2(w_odd, xp[half_word * 2 + 1], sum_b);
        }
        const float2 tail = __half22float2(__hmul2(__hadd2(sum_a, sum_b), scale));
        acc += tail.x + tail.y;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
}

} // namespace

void launch_w8_decode_hfma2_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t k    = w.k;
    const std::int32_t rows = w.n;
    // k % 256 == 0: whole 32-value groups per lane (8 values) x 32 lanes, no tail; the warp
    // stages the row in 8-value steps and every code/scale load is aligned.
    if (x.ne[1] != 1 || k <= 0 || (k % 256) != 0 || w.padded_shape[1] != k ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15U) != 0) {
        throw std::invalid_argument("W8 hfma2 decode requires T=1 and k divisible by 256");
    }
    const std::size_t stage_bytes = static_cast<std::size_t>(k) * sizeof(std::uint16_t);
    if (stage_bytes > static_cast<std::size_t>(kMaxStagedK) * sizeof(std::uint16_t)) {
        throw std::invalid_argument("W8 hfma2 decode: k exceeds the staged activation bound");
    }
    constexpr int kRowsPerCta = 16;
    const unsigned blocks = static_cast<unsigned>((rows + kRowsPerCta - 1) / kRowsPerCta);
    w8_decode_hfma2_kernel<kRowsPerCta><<<blocks, kRowsPerCta * 32, stage_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
