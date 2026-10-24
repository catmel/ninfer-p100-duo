#include "ops/linear/ggml_k/ggml_k.h"
#include "ops/linear/ggml_k/ggml_k_codec.cuh"
#include "core/device.h"
#include "core/layout.h"

#ifndef NINFER_PASCAL_BUILD
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#endif

#include <cuda_bf16.h>
#ifdef NINFER_PASCAL_BUILD
#include <cuda_fp16.h>  // HFMA2 decode gemv half2 arithmetic (Pascal section)
#endif
#ifndef NINFER_PASCAL_BUILD
#include <mma.h>  // CUDA 12 dropped the WMMA API below sm_70; the Pascal TU excludes the tile kernel.
#endif

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct Outputs {
    void* data[4] = {};
    int rows[4] = {};
    std::int64_t stride[4] = {};
    bool fp32[4] = {};
    int count = 0;
    bool add = false;

    __device__ void store(int row, int token, float value) const {
        int section = 0;
        while (row >= rows[section]) { row -= rows[section++]; }
        auto* column = static_cast<unsigned char*>(data[section]) + token * stride[section];
        if (fp32[section]) {
            auto* p = reinterpret_cast<float*>(column) + row;
            *p = add ? *p + value : value;
        } else {
            auto* p = reinterpret_cast<__nv_bfloat16*>(column) + row;
            *p = __float2bfloat16_rn(add ? __bfloat162float(*p) + value : value);
        }
    }
};

#ifndef NINFER_PASCAL_BUILD
constexpr int kCutlassRowTile = 4096;

using CutlassAccumulator = float;
using CutlassInput = cutlass::half_t;
using CutlassOutput = float;
using CutlassEpilogue = cutlass::epilogue::thread::LinearCombination<
    CutlassOutput, 128 / cutlass::sizeof_bits<CutlassOutput>::value,
    CutlassAccumulator, CutlassAccumulator>;
using CutlassGemm = cutlass::gemm::device::Gemm<
    CutlassInput, cutlass::layout::RowMajor,
    CutlassInput, cutlass::layout::ColumnMajor,
    CutlassOutput, cutlass::layout::RowMajor,
    CutlassAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>, CutlassEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using CutlassBf16Output = cutlass::bfloat16_t;
using CutlassBf16Epilogue = cutlass::epilogue::thread::LinearCombination<
    CutlassBf16Output, 128 / cutlass::sizeof_bits<CutlassBf16Output>::value,
    CutlassAccumulator, CutlassAccumulator>;
using CutlassBf16Gemm = cutlass::gemm::device::Gemm<
    CutlassInput, cutlass::layout::RowMajor,
    CutlassInput, cutlass::layout::ColumnMajor,
    CutlassBf16Output, cutlass::layout::RowMajor,
    CutlassAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>, CutlassBf16Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
#endif // NINFER_PASCAL_BUILD

// GGML_K rows are concatenated without per-row padding.  All registered Qwen3.8
// projection shapes happen to contain an even number of K256 blocks, so their
// Q4_K/Q6_K rows start on a four-byte boundary.  Keep the fast aligned path for
// those rows, but do not make the public GGML_K Op depend on that incidental
// property: a caller can legally provide K=256, where a Q6_K row is 210 bytes
// and the following row starts at offset 2 mod 4.  CUDA generally tolerates
// misaligned scalar loads, but an `unsigned int*` dereference is not a valid
// alignment contract. Every block remains two-byte aligned, including odd
// Q6_K blocks in production rows.
__device__ __forceinline__ unsigned load_u32_maybe_unaligned(const unsigned char* p) {
    if ((reinterpret_cast<std::uintptr_t>(p) & 3u) == 0u) {
        return __ldg(reinterpret_cast<const unsigned int*>(p));
    }
    return static_cast<unsigned>(__ldg(reinterpret_cast<const unsigned short*>(p))) |
           (static_cast<unsigned>(__ldg(reinterpret_cast<const unsigned short*>(p + 2))) << 16u);
}

// The generic decoder is useful as a correctness oracle, but calling it once per element is
// needlessly expensive in the decode path.  A K block carries its scales in a tiny header; keep
// that header in registers and consume the packed code bytes in the native GGML order.  One lane
// owns one byte of each 32-value half-group, therefore every lane performs eight FMAs per block
// instead of eight calls through the descriptor/type branch in ggml_k_value().
// GGUF GDN output columns are [value-within-key, key-head, dimension], while
// the recurrent output is [key-head, value-within-key, dimension]. Read that
// exact permutation in the multiply, preserving every original K256 block.
template <bool TiledGdn>
__device__ __forceinline__ int input_column(int column, int k) {
    if constexpr (!TiledGdn) { return column; }
    const int head = column / 128;
    // The registered full and TP2 shapes have 16 and 8 key heads respectively.
    // Explicit powers of two avoid runtime integer division in every load.
    const int grouped = k == 3072 ? (head & 7) * 3 + (head >> 3)
                                  : (head & 15) * 3 + (head >> 4);
    return grouped * 128 + (column & 127);
}

#ifndef NINFER_PASCAL_BUILD
__global__ void dequant_rows_fp16(const unsigned char* __restrict__ rows,
                                  const std::uint64_t* __restrict__ descriptors,
                                  cutlass::half_t* __restrict__ out, int row_begin,
                                  int row_count, int k) {
    const int row_local = blockIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_local < row_count && column < k) {
        out[static_cast<std::int64_t>(row_local) * k + column] = cutlass::half_t(
            ggml_k_value(rows, descriptors[row_begin + row_local], column));
    }
}

template <bool TiledGdn>
__global__ void cast_activation_fp16(const __nv_bfloat16* __restrict__ input,
                                     cutlass::half_t* __restrict__ out,
                                     std::int64_t count, int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const int column = static_cast<int>(index % k);
        const std::int64_t token_base = index - column;
        out[index] = cutlass::half_t(__bfloat162float(
            input[token_base + input_column<TiledGdn>(column, k)]));
    }
}

__global__ void scatter_cutlass_output(const float* __restrict__ input, Outputs out,
                                       int n, int tokens) {
    const std::int64_t count = static_cast<std::int64_t>(n) * tokens;
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const int row = static_cast<int>(index % n);
        const int token = static_cast<int>(index / n);
        out.store(row, token, input[index]);
    }
}

template <bool TiledGdn>
void cutlass_project(const __nv_bfloat16* input, const unsigned char* rows,
                     const std::uint64_t* descriptors, Outputs output,
                     int n, int k, int tokens, WorkspaceArena& workspace,
                     cudaStream_t stream) {
    auto scope = workspace.scope();
    const int row_tile = std::min(n, kCutlassRowTile);
    Tensor x_fp16 = workspace.alloc(DType::FP16, {k, tokens});
    Tensor w_fp16 = workspace.alloc(DType::FP16, {k, row_tile});
    const bool direct_bf16 = output.count == 1 && !output.fp32[0] &&
                             output.stride[0] == static_cast<std::int64_t>(n) * 2;
    Tensor projected;
    if (!direct_bf16) { projected = workspace.alloc(DType::FP32, {n, tokens}); }

    const std::int64_t activation_count = static_cast<std::int64_t>(k) * tokens;
    cast_activation_fp16<TiledGdn><<<static_cast<unsigned>((activation_count + 255) / 256),
                                      256, 0, stream>>>(
        input, static_cast<cutlass::half_t*>(x_fp16.data), activation_count, k);
    CUDA_CHECK(cudaGetLastError());

    CutlassGemm gemm;
    CutlassBf16Gemm bf16_gemm;
    for (int row_begin = 0; row_begin < n; row_begin += row_tile) {
        const int row_count = std::min(row_tile, n - row_begin);
        dequant_rows_fp16<<<dim3(static_cast<unsigned>((k + 255) / 256),
                                 static_cast<unsigned>(row_count)), 256, 0, stream>>>(
            rows, descriptors, static_cast<cutlass::half_t*>(w_fp16.data),
            row_begin, row_count, k);
        CUDA_CHECK(cudaGetLastError());

        cutlass::gemm::GemmCoord problem(tokens, row_count, k);
        if (direct_bf16) {
            auto* destination = static_cast<CutlassBf16Output*>(output.data[0]) + row_begin;
            typename CutlassBf16Gemm::Arguments arguments{
                problem,
                {static_cast<const cutlass::half_t*>(x_fp16.data), k},
                {static_cast<const cutlass::half_t*>(w_fp16.data), k},
                {destination, n}, {destination, n},
                {1.0f, output.add ? 1.0f : 0.0f}, 1};
            cutlass::Status status = bf16_gemm.can_implement(arguments);
            if (status != cutlass::Status::kSuccess ||
                bf16_gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess ||
                bf16_gemm(stream) != cutlass::Status::kSuccess) {
                throw std::runtime_error("GGML K BF16 CUTLASS GEMM failed");
            }
        } else {
            typename CutlassGemm::Arguments arguments{
                problem,
                {static_cast<const cutlass::half_t*>(x_fp16.data), k},
                {static_cast<const cutlass::half_t*>(w_fp16.data), k},
                {static_cast<float*>(projected.data) + row_begin, n},
                {static_cast<float*>(projected.data) + row_begin, n},
                {1.0f, 0.0f}, 1};
            cutlass::Status status = gemm.can_implement(arguments);
            if (status != cutlass::Status::kSuccess ||
                gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess ||
                gemm(stream) != cutlass::Status::kSuccess) {
                throw std::runtime_error("GGML K FP32 CUTLASS GEMM failed");
            }
        }
    }
    if (!direct_bf16) {
        const std::int64_t output_count = static_cast<std::int64_t>(n) * tokens;
        scatter_cutlass_output<<<static_cast<unsigned>((output_count + 255) / 256),
                                  256, 0, stream>>>(
            static_cast<const float*>(projected.data), output, n, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif // !NINFER_PASCAL_BUILD (CUTLASS staging path)

template <int Tokens, bool TiledGdn>
__device__ __forceinline__ void gemv_q4_row(const __nv_bfloat16* __restrict__ x,
                                            const unsigned char* __restrict__ rows,
                                            std::uint64_t descriptor, float (&sum)[Tokens],
                                            int k, int rem = Tokens) {
    const int lane = threadIdx.x & 31;
    const unsigned char* row = rows + (descriptor >> 1);
    for (int base = 0; base < k; base += 256) {
        const unsigned char* block = row + (base >> 8) * 144;
        const int pair = lane >> 3;
        const int inner = lane & 7;
        // Each lane needs only its two scale groups. Coalesced header loads avoid
        // broadcasting all sixteen scale/minimum values through shuffle calls.
        const float d = __half2float(*reinterpret_cast<const __half*>(block));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
        float scale_f[2], minimum_f[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int group = 2 * pair + i;
            const int low = group & 3;
            const int s = __ldg(block + 4 + low);
            const int m = __ldg(block + 8 + low);
            const int high = __ldg(block + 12 + low);
            scale_f[i] = d * (group < 4 ? s & 63 : (high & 15) | ((s >> 6) << 4));
            minimum_f[i] = dmin * (group < 4 ? m & 63 : (high >> 4) | ((m >> 6) << 4));
        }
        const unsigned char* packed_bytes = block + 16 + pair * 32 + inner * 4;
        const unsigned int packed = load_u32_maybe_unaligned(packed_bytes);
        const int c0 = base + pair * 64 + inner * 4;
        const int c1 = c0 + 32;
        float w0[4], w1[4];
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const unsigned char code = static_cast<unsigned char>(packed >> (byte * 8));
            w0[byte] = scale_f[0] * static_cast<float>(code & 15) - minimum_f[0];
            w1[byte] = scale_f[1] * static_cast<float>(code >> 4) - minimum_f[1];
        }
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (t >= rem) { break; }  // ragged final chunk in the multi-block wide-T route
            const __nv_bfloat16* xt = x + static_cast<std::int64_t>(t) * k;
            const uint2 x0 = __ldg(reinterpret_cast<const uint2*>(xt + input_column<TiledGdn>(c0, k)));
            const uint2 x1 = __ldg(reinterpret_cast<const uint2*>(xt + input_column<TiledGdn>(c1, k)));
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const unsigned a0 = byte < 2 ? x0.x : x0.y;
                const unsigned a1 = byte < 2 ? x1.x : x1.y;
                const float v0 = __uint_as_float(((a0 >> (16 * (byte & 1))) & 0xffffu) << 16);
                const float v1 = __uint_as_float(((a1 >> (16 * (byte & 1))) & 0xffffu) << 16);
                sum[t] = fmaf(w0[byte], v0, sum[t]);
                sum[t] = fmaf(w1[byte], v1, sum[t]);
            }
        }
    }
}

template <int Tokens, bool TiledGdn>
__device__ __forceinline__ void gemv_q6_row(const __nv_bfloat16* __restrict__ x,
                                            const unsigned char* __restrict__ rows,
                                            std::uint64_t descriptor, float (&sum)[Tokens],
                                            int k, int rem = Tokens) {
    const int lane = threadIdx.x & 31;
    const unsigned char* row = rows + (descriptor >> 1);
    for (int base = 0; base < k; base += 256) {
        const unsigned char* block = row + (base >> 8) * 210;
        const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
        const int half = lane >> 4;
        const int pair = (lane >> 3) & 1;
        const int inner = lane & 7;
        const unsigned lo0 = load_u32_maybe_unaligned(block + half * 64 + inner * 4);
        const unsigned lo1 = load_u32_maybe_unaligned(block + half * 64 + 32 + inner * 4);
        const unsigned hi = load_u32_maybe_unaligned(block + 128 + half * 32 + inner * 4);
        const int sb = half * 8 + pair * 4 + (inner >> 2);
        const float s0 = d * static_cast<float>(static_cast<std::int8_t>(__ldg(block + 192 + sb)));
        const float s1 = d * static_cast<float>(static_cast<std::int8_t>(__ldg(block + 194 + sb)));
        const int c0 = base + half * 128 + pair * 64 + inner * 4;
        float w0[4], w1[4];
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const unsigned l0 = lo0 >> (8 * byte), l1 = lo1 >> (8 * byte), h = hi >> (8 * byte);
            const int q0 = int(((l0 >> (4 * pair)) & 15) | (((h >> (4 * pair)) & 3) << 4)) - 32;
            const int q1 = int(((l1 >> (4 * pair)) & 15) | (((h >> (4 * pair + 2)) & 3) << 4)) - 32;
            w0[byte] = s0 * q0;
            w1[byte] = s1 * q1;
        }
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (t >= rem) { break; }  // ragged final chunk in the multi-block wide-T route
            const __nv_bfloat16* xt = x + static_cast<std::int64_t>(t) * k;
            const uint2 x0 = __ldg(reinterpret_cast<const uint2*>(xt + input_column<TiledGdn>(c0, k)));
            const uint2 x1 = __ldg(reinterpret_cast<const uint2*>(xt + input_column<TiledGdn>(c0 + 32, k)));
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const unsigned a0 = byte < 2 ? x0.x : x0.y;
                const unsigned a1 = byte < 2 ? x1.x : x1.y;
                const float v0 = __uint_as_float(((a0 >> (16 * (byte & 1))) & 0xffffu) << 16);
                const float v1 = __uint_as_float(((a1 >> (16 * (byte & 1))) & 0xffffu) << 16);
                sum[t] = fmaf(w0[byte], v0, sum[t]);
                sum[t] = fmaf(w1[byte], v1, sum[t]);
            }
        }
    }
}

template <int Tokens, bool TiledGdn>
__global__ void gemv_dynamic(const __nv_bfloat16* __restrict__ x,
                             const unsigned char* __restrict__ rows,
                             const std::uint64_t* __restrict__ descriptors,
                             Outputs out, int row_begin, int row_count, int k) {
    const int row = row_begin + blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= row_begin + row_count) { return; }
    const int lane = threadIdx.x & 31;
    const std::uint64_t descriptor = __ldg(descriptors + row);
    float sum[Tokens] = {};
    if ((descriptor & 1u) != 0) {
        gemv_q6_row<Tokens, TiledGdn>(x, rows, descriptor, sum, k);
    } else {
        gemv_q4_row<Tokens, TiledGdn>(x, rows, descriptor, sum, k);
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            sum[t] += __shfl_down_sync(0xffffffff, sum[t], delta);
        }
        if (lane == 0) { out.store(row, t, sum[t]); }
    }
}

template <int Tokens, bool TiledGdn, bool Q6>
__global__ void gemv_codec(const __nv_bfloat16* __restrict__ x,
                           const unsigned char* __restrict__ rows,
                           const std::uint64_t* __restrict__ descriptors,
                           Outputs out, int row_begin, int row_count, int k) {
    const int row = row_begin + blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= row_begin + row_count) { return; }
    const int lane = threadIdx.x & 31;
    const std::uint64_t descriptor = __ldg(descriptors + row);
    float sum[Tokens] = {};
    if constexpr (Q6) {
        gemv_q6_row<Tokens, TiledGdn>(x, rows, descriptor, sum, k);
    } else {
        gemv_q4_row<Tokens, TiledGdn>(x, rows, descriptor, sum, k);
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            sum[t] += __shfl_down_sync(0xffffffff, sum[t], delta);
        }
        if (lane == 0) { out.store(row, t, sum[t]); }
    }
}

template <int Tokens, bool TiledGdn>
void launch_gemv(const __nv_bfloat16* input, const unsigned char* rows,
                 const std::uint64_t* descriptors, Outputs output, int n, int k,
                 int first_q6, int type_change, cudaStream_t stream) {
    // Splitting a mixed matrix adds a CUDA Graph node per projection and costs
    // more end-to-end than the Q6 row speedup recovers. Specialize only a
    // uniform Q6 matrix; Q4 and mixed matrices retain their single launch.
    if (first_q6 != 1 || type_change != n) {
        gemv_dynamic<Tokens, TiledGdn><<<(n + 3) / 4, 128, 0, stream>>>(
            input, rows, descriptors, output, 0, n, k);
        return;
    }
    gemv_codec<Tokens, TiledGdn, true><<<(n + 3) / 4, 128, 0, stream>>>(
        input, rows, descriptors, output, 0, n, k);
}

#ifdef NINFER_PASCAL_BUILD
// sm_60 decode route: HFMA2 dequant gemv for T=1..4 and a chunked T=8 variant for the wide
// shapes the tile kernel cannot serve (section splits, add epilogues, TiledGdn permutation).
// The scalar fp32 gemv_q4/q6_row spent a quarter-rate I2F and a separate fp32 multiply per
// weight.  GP100 issues HFMA2 at the full fp32 lane rate, and the magic `0x64006400 | code`
// decode lands the 4/6-bit code directly in the fp16 mantissa (half value 1024+code, exponent
// field 0x19): one HSUB2 against 1024+bias realizes (code - 32) for Q6_K with no integer-to-
// float path at all.  Column ownership mirrors the qualified ggml_k_tile_pascal decode: a lane
// covers c = 64*Q + 4*i + p with Q = lane >> 3, i = lane & 7, p = 0..3, and its +32 twin, so
// one warp round consumes exactly one 256-value GGML_K block and every half2 is (c, c + 32).
// Activations stage bf16 -> half2 once per CTA round into ping-pong shared buffers, shared by
// the four row warps: staging block b+1 overlaps computing block b and a single barrier per
// round replaces the pair the single-buffer version needed (barriers otherwise also fence the
// compiler's own load hoisting across iterations).  Per K block the fp16 pair sums merge into
// fp32 with the row scale: Q6_K merges with scale alone, Q4_K keeps a separate Sigma-x
// accumulator so the merge applies scale*Sigma(code*x) - min*Sigma(x).  The Q6_K weight
// pattern (x, x+32, x+128, x+192, x+208 per warp block) is the real sm_60 ceiling: a
// loads-only probe on gate_up runs at 132 GB/s, within 12% of the full decode gemv, so the
// decode math is nearly free and the levers are fewer barriers and less L1 churn (__ldcs).
// The T=1..4 launches specialize a uniform-Q6 matrix (same first_q6/type_change rule as
// launch_gemv); mixed matrices keep the per-row branch.
__device__ __forceinline__ float hfma2_bf16_value(std::uint32_t word, int half_index) {
    return __uint_as_float(((word >> (16 * half_index)) & 0xffffu) << 16);
}

// Stage one unit: eight columns (c..c+3 and their +32 twins) of one token as four interleaved
// half2 slots (v_c, v_{c+32}), exactly the tile kernel's staging contract.
__device__ __forceinline__ uint4 hfma2_stage_pair(const __nv_bfloat16* __restrict__ src) {
    const uint2 a = *reinterpret_cast<const uint2*>(src);
    const uint2 b = *reinterpret_cast<const uint2*>(src + 32);
    const half2 p0 = __floats2half2_rn(hfma2_bf16_value(a.x, 0), hfma2_bf16_value(b.x, 0));
    const half2 p1 = __floats2half2_rn(hfma2_bf16_value(a.x, 1), hfma2_bf16_value(b.x, 1));
    const half2 p2 = __floats2half2_rn(hfma2_bf16_value(a.y, 0), hfma2_bf16_value(b.y, 0));
    const half2 p3 = __floats2half2_rn(hfma2_bf16_value(a.y, 1), hfma2_bf16_value(b.y, 1));
    return make_uint4(*reinterpret_cast<const unsigned*>(&p0),
                      *reinterpret_cast<const unsigned*>(&p1),
                      *reinterpret_cast<const unsigned*>(&p2),
                      *reinterpret_cast<const unsigned*>(&p3));
}

__device__ __forceinline__ std::uint32_t hfma2_ldcs(const unsigned char* p) {
    if ((reinterpret_cast<std::uintptr_t>(p) & 3u) == 0u) {
        return __ldcs(reinterpret_cast<const std::uint32_t*>(p));
    }
    return static_cast<std::uint32_t>(__ldcs(reinterpret_cast<const std::uint16_t*>(p))) |
           (static_cast<std::uint32_t>(__ldcs(reinterpret_cast<const std::uint16_t*>(p) + 1)) << 16);
}

template <int Tokens, bool TiledGdn, bool Q6>
__global__ void __launch_bounds__(128) gemv_hfma2(const __nv_bfloat16* __restrict__ x,
                                                  const unsigned char* __restrict__ rows,
                                                  const std::uint64_t* __restrict__ descriptors,
                                                  Outputs out, int n, int k, int tokens) {
    constexpr int kStageSlots = 128;  // half2 pairs per token: (column c, column c + 32)
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= n) { return; }
    const int lane = threadIdx.x & 31;
    const int tbase = Tokens < 8 ? 0 : blockIdx.y * 8;
    const int rem = tokens - tbase < Tokens ? tokens - tbase : Tokens;
    const std::uint64_t descriptor = __ldg(descriptors + row);
    const bool q6 = Q6 || (descriptor & 1u) != 0u;
    const unsigned char* row_bytes = rows + (descriptor >> 1);
    const int q_region = lane >> 3;  // 64-column region of the block this lane decodes
    const int i_group = lane & 7;    // 4-column group inside the region

    __shared__ half2 x_stage[2][Tokens][kStageSlots];

    const half2 bias = __float2half2_rn(q6 ? 1056.0F : 1024.0F);  // 1024 + 32 for Q6_K
    const int nib_shift = q6 ? 4 * (q_region & 1) : 0;
    const int block_bytes = q6 ? 210 : 144;
    const int slot_base = 32 * q_region + 4 * i_group;
    float acc[Tokens] = {};

    const int blocks = k >> 8;
    const auto stage = [&](int block) {
        const int buf = block & 1;
#pragma unroll
        for (int job = threadIdx.x; job < kStageSlots / 4 * Tokens; job += 128) {
            const int unit = job & 31;   // (region, quarter) -> slots unit*4 .. +3
            const int token = job >> 5;
            if (token < rem) {
                const int col = block * 256 + 64 * (unit >> 3) + 4 * (unit & 7);
                reinterpret_cast<uint4*>(&x_stage[buf][token][0])[unit] = hfma2_stage_pair(
                    x + static_cast<std::int64_t>(tbase + token) * k +
                    input_column<TiledGdn>(col, k));
            }
        }
    };
    stage(0);
    __syncthreads();
    for (int block = 0; block < blocks; ++block) {
        if (block + 1 < blocks) { stage(block + 1); }  // overlaps this block's compute below
        const unsigned char* bytes = row_bytes + block * block_bytes;

        half2 sums[Tokens], sums_v[Tokens];  // sums_v carries Sigma x for the Q4_K min term
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            sums[t] = __float2half2_rn(0.0F);
            sums_v[t] = __float2half2_rn(0.0F);
        }
        if (q6) {
            const std::uint32_t p0 = hfma2_ldcs(bytes + (q_region >> 1) * 64 + i_group * 4);
            const std::uint32_t p1 = hfma2_ldcs(bytes + (q_region >> 1) * 64 + i_group * 4 + 32);
            const std::uint32_t hi =
                hfma2_ldcs(bytes + 128 + (q_region >> 1) * 32 + i_group * 4);
            const float d = __half2float(*reinterpret_cast<const __half*>(bytes + 208));
            const int g = 4 * q_region + (i_group >> 2);
            const float s_x = d * static_cast<float>(static_cast<std::int8_t>(bytes[192 + g]));
            const float s_y = d * static_cast<float>(static_cast<std::int8_t>(bytes[192 + g + 2]));
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int sh = 8 * p + nib_shift;
                const std::uint32_t a = ((p0 >> sh) & 0x0Fu) | (((hi >> sh) & 0x3u) << 4);
                const std::uint32_t b =
                    ((p1 >> sh) & 0x0Fu) | (((hi >> (sh + 2)) & 0x3u) << 4);
                const std::uint32_t bits = 0x64006400u | a | (b << 16);
                const half2 code2 = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    if (Tokens == 8 && t >= rem) { break; }
                    sums[t] = __hfma2(code2, x_stage[block & 1][t][slot_base + p], sums[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (Tokens == 8 && t >= rem) { break; }
                const float2 s = __half22float2(sums[t]);
                acc[t] += s_x * s.x + s_y * s.y;
            }
        } else {
            float s_x, s_y, m_x, m_y;
            {
                const float d = __half2float(*reinterpret_cast<const __half*>(bytes));
                const float dmin = __half2float(*reinterpret_cast<const __half*>(bytes + 2));
                float s[2], m[2];
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
                s_x = s[0];
                s_y = s[1];
                m_x = m[0];
                m_y = m[1];
            }
            const std::uint32_t p0 = hfma2_ldcs(bytes + 16 + q_region * 32 + i_group * 4);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const std::uint32_t w =
                    p0 >> (8 * p);  // low nib = column c, high nib = column c + 32
                const std::uint32_t bits = 0x64006400u | (w & 0x0Fu) | (((w >> 4) & 0x0Fu) << 16);
                const half2 code2 = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    if (Tokens == 8 && t >= rem) { break; }
                    const half2 xv = x_stage[block & 1][t][slot_base + p];
                    sums[t] = __hfma2(code2, xv, sums[t]);
                    sums_v[t] = __hadd2(sums_v[t], xv);
                }
            }
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (Tokens == 8 && t >= rem) { break; }
                const float2 s = __half22float2(sums[t]);
                const float2 v = __half22float2(sums_v[t]);
                acc[t] += s_x * s.x + s_y * s.y - m_x * v.x - m_y * v.y;
            }
        }
        __syncthreads();  // stage(block+1) must complete before the next round reads it
    }

    Outputs shifted = out;
    if (tbase != 0) {
        for (int section = 0; section < out.count; ++section) {
            shifted.data[section] = static_cast<unsigned char*>(out.data[section]) +
                                    static_cast<std::int64_t>(tbase) * out.stride[section];
        }
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        if (Tokens == 8 && t >= rem) { break; }
        float value = acc[t];
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            value += __shfl_down_sync(0xffffffff, value, delta);
        }
        if (lane == 0) { shifted.store(row, t, value); }
    }
}

template <int Tokens, bool TiledGdn>
void launch_gemv_hfma2(const __nv_bfloat16* input, const unsigned char* rows,
                       const std::uint64_t* descriptors, Outputs output, int n, int k, int tokens,
                       bool uniform_q6, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n + 3) / 4),
                    Tokens < 8 ? 1u : static_cast<unsigned>((tokens + 7) / 8));
    if (uniform_q6) {
        gemv_hfma2<Tokens, TiledGdn, true><<<grid, 128, 0, stream>>>(input, rows, descriptors,
                                                                     output, n, k, tokens);
    } else {
        gemv_hfma2<Tokens, TiledGdn, false><<<grid, 128, 0, stream>>>(input, rows, descriptors,
                                                                      output, n, k, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool TiledGdn>
void launch_gemv_hfma2_decode(const __nv_bfloat16* input, const unsigned char* rows,
                              const std::uint64_t* descriptors, Outputs output, int n, int k,
                              int tokens, int first_q6, int type_change, cudaStream_t stream) {
    // Same specialization rule as launch_gemv: only a uniform-Q6 matrix earns the Q6 kernel;
    // the runtime branch in the mixed variant is warp-uniform (one descriptor per row).
    const bool uniform_q6 = first_q6 == 1 && type_change == n;
    switch (tokens) {
    case 1:
        launch_gemv_hfma2<1, TiledGdn>(input, rows, descriptors, output, n, k, tokens, uniform_q6,
                                       stream);
        break;
    case 2:
        launch_gemv_hfma2<2, TiledGdn>(input, rows, descriptors, output, n, k, tokens, uniform_q6,
                                       stream);
        break;
    case 3:
        launch_gemv_hfma2<3, TiledGdn>(input, rows, descriptors, output, n, k, tokens, uniform_q6,
                                       stream);
        break;
    default:
        launch_gemv_hfma2<4, TiledGdn>(input, rows, descriptors, output, n, k, tokens, uniform_q6,
                                       stream);
        break;
    }
}
#endif // NINFER_PASCAL_BUILD

// Volta's WMMA layout is provided by CUDA. Only the private multiplication operands
// are FP16; public activations/results remain BF16 and accumulators remain FP32.
#ifndef NINFER_PASCAL_BUILD  // CUDA 12.4 has no WMMA API below sm_70; Pascal uses gemv_hfma2.
template <int TokenTile, bool TiledGdn>
__global__ void gemm(const __nv_bfloat16* __restrict__ x,
                     const unsigned char* __restrict__ rows,
                     const std::uint64_t* __restrict__ descriptors,
                     Outputs out, int n, int k, int tokens) {
    static_assert(TokenTile == 16 || TokenTile == 32 || TokenTile == 64 || TokenTile == 96);
    using namespace nvcuda;
    // Eight threads own each output row and decode half a native K256 block
    // before multiplication. Padding the 128-wide tile by eight half elements
    // distributes successive rows across Volta's shared-memory banks. A and C
    // have disjoint lifetimes, so they share storage. The 96-token route uses
    // 37.5 KiB per CTA and 12 warps; two CTAs fit on a V100 SM while decoding
    // each weight tile one third fewer times than the 64-token route.
    constexpr int kTile = 128;
    constexpr int kStride = kTile + 8;
    constexpr int kABytes = 32 * kStride * sizeof(__half);
    constexpr int kCBytes = 32 * TokenTile * sizeof(float);
    constexpr int kAcBytes = kABytes > kCBytes ? kABytes : kCBytes;
    __shared__ __align__(32) unsigned char ac_storage[kAcBytes];
    __shared__ __align__(32) __half b[TokenTile * kStride];
    auto* a = reinterpret_cast<__half*>(ac_storage);
    const int row_begin = blockIdx.x * 32;
    const int token_begin = blockIdx.y * TokenTile;
    const int warp = threadIdx.x / 32;
    constexpr int token_warps = TokenTile / 16;
    const int warp_row = (warp / token_warps) * 16;
    const int warp_token = (warp % token_warps) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int base = 0; base < k; base += kTile) {
        // Exactly eight threads decode each of the 32 weight rows. Wider token
        // tiles add consumer warps, but must not duplicate this producer work.
        if (threadIdx.x < 256) {
            const int row_local = threadIdx.x / 8;
            const int row = row_begin + row_local;
            const int inner = threadIdx.x & 7;
            const std::uint64_t descriptor = row < n ? descriptors[row] : 0;
            const bool q6 = (descriptor & 1u) != 0;
            const int block_bytes = q6 ? 210 : 144;
            const unsigned char* block =
                rows + (descriptor >> 1) + (base >> 8) * block_bytes;
            const int half = (base & 255) >> 7;
            if (row < n && !q6) {
                const float d = __half2float(*reinterpret_cast<const __half*>(block));
                const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
                const unsigned scale_low = load_u32_maybe_unaligned(block + 4);
                const unsigned min_low = load_u32_maybe_unaligned(block + 8);
                const unsigned high = load_u32_maybe_unaligned(block + 12);
#pragma unroll
                for (int local_segment = 0; local_segment < 2; ++local_segment) {
                    const int segment = half * 2 + local_segment;
                    constexpr unsigned mask = 63;
                    const int shift = (segment & 1) * 16;
                    const unsigned s = scale_low >> shift;
                    const unsigned m = min_low >> shift;
                    const unsigned h = high >> shift;
                    const int scale0 = segment < 2 ? s & mask : (h & 15) | ((s >> 6 & 3) << 4);
                    const int scale1 = segment < 2 ? (s >> 8) & mask
                                                   : (h >> 8 & 15) | ((s >> 14 & 3) << 4);
                    const int min0 = segment < 2 ? m & mask : (h >> 4 & 15) | ((m >> 6 & 3) << 4);
                    const int min1 = segment < 2 ? (m >> 8) & mask
                                                 : (h >> 12 & 15) | ((m >> 14 & 3) << 4);
                    const float scale_f0 = d * static_cast<float>(scale0);
                    const float scale_f1 = d * static_cast<float>(scale1);
                    const float min_f0 = dmin * static_cast<float>(min0);
                    const float min_f1 = dmin * static_cast<float>(min1);
                    const unsigned packed = load_u32_maybe_unaligned(
                        block + 16 + segment * 32 + inner * 4);
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        const unsigned code = (packed >> (byte * 8)) & 255;
                        a[row_local * kStride + local_segment * 64 + inner * 4 + byte] =
                            __float2half_rn(scale_f0 * static_cast<float>(code & 15) - min_f0);
                        a[row_local * kStride + local_segment * 64 + 32 + inner * 4 + byte] =
                            __float2half_rn(scale_f1 * static_cast<float>(code >> 4) - min_f1);
                    }
                }
            } else if (row < n) {
                const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
                const unsigned lo0 = load_u32_maybe_unaligned(
                    block + half * 64 + inner * 4);
                const unsigned lo1 = load_u32_maybe_unaligned(
                    block + half * 64 + 32 + inner * 4);
                const unsigned hi = load_u32_maybe_unaligned(
                    block + 128 + half * 32 + inner * 4);
#pragma unroll
                for (int section = 0; section < 4; ++section) {
                    const int scale = static_cast<std::int8_t>(
                        __ldg(block + 192 + half * 8 + section * 2 + (inner >> 2)));
                    const float scale_f = d * static_cast<float>(scale);
                    const unsigned lo = (section & 1) == 0 ? lo0 : lo1;
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        const int code = static_cast<int>(
                            ((lo >> (byte * 8 + (section >> 1) * 4)) & 15) |
                            (((hi >> (byte * 8 + section * 2)) & 3) << 4)) - 32;
                        a[row_local * kStride + section * 32 + inner * 4 + byte] =
                            __float2half_rn(scale_f * static_cast<float>(code));
                    }
                }
            } else {
#pragma unroll
                for (int column = inner; column < kTile; column += 8) {
                    a[row_local * kStride + column] = __float2half_rn(0.0f);
                }
            }
        }
        for (int index = threadIdx.x; index < TokenTile * kTile; index += blockDim.x) {
            const int t = token_begin + index / kTile;
            const int column = base + index % kTile;
            b[(index / kTile) * kStride + index % kTile] = t < tokens
                ? __float2half_rn(__bfloat162float(x[t * k + input_column<TiledGdn>(column, k)]))
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        if (warp < 2 * token_warps) {
#pragma unroll
            for (int offset = 0; offset < kTile; offset += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, a + warp_row * kStride + offset, kStride);
                wmma::load_matrix_sync(fb, b + warp_token * kStride + offset, kStride);
                wmma::mma_sync(acc, fa, fb, acc);
            }
        }
        __syncthreads();
    }
    auto* c = reinterpret_cast<float*>(ac_storage);
    if (warp < 2 * token_warps) {
        wmma::store_matrix_sync(c + warp_token * 32 + warp_row, acc, 32, wmma::mem_col_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < 32 * TokenTile; index += blockDim.x) {
        const int row = row_begin + index % 32;
        const int t = token_begin + index / 32;
        if (row < n && t < tokens) { out.store(row, t, c[index]); }
    }
}
#endif // NINFER_PASCAL_BUILD (WMMA tile kernel)

__global__ void embedding(const std::int32_t* ids, const unsigned char* rows,
                          const std::uint64_t* descriptors, __nv_bfloat16* out, int k) {
    const int t = blockIdx.x;
    const std::uint64_t descriptor = descriptors[ids[t]];
    for (int column = threadIdx.x; column < k; column += blockDim.x) {
        out[t * k + column] = __float2bfloat16_rn(ggml_k_value(rows, descriptor, column));
    }
}

void validate_weight(const Weight& weight) {
    if (weight.qtype != QType::GGML_K || weight.layout != QuantLayout::GgmlK256 ||
        weight.n <= 0 || weight.k <= 0 || weight.k % 256 != 0 ||
        weight.qdata == nullptr || weight.qhigh == nullptr) {
        throw std::invalid_argument("GGML K projection: invalid row-block weight");
    }
}

} // namespace

void ggml_k_linear(const Tensor& x, const Weight& weight, Tensor& out,
                   WorkspaceArena* workspace, cudaStream_t stream) {
    ggml_k_project_split(x, weight, &out, 1, false, stream, false, workspace);
}

template <bool TiledGdn>
void project(const Tensor& x, const Weight& weight, const Tensor* outputs,
             int count, bool add, cudaStream_t stream, WorkspaceArena* workspace) {
    validate_weight(weight);
    if (count < 1 || count > 4 || x.dtype != DType::BF16 || !x.is_contiguous() ||
        x.ne[0] != weight.k || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("GGML K projection: invalid input or output sections");
    }
    Outputs output;
    output.count = count;
    output.add = add;
    int total_rows = 0;
    for (int i = 0; i < count; ++i) {
        const Tensor& o = outputs[i];
        if ((o.dtype != DType::BF16 && o.dtype != DType::FP32) || o.ne[0] <= 0 ||
            o.ne[1] != x.ne[1] || o.ne[2] != 1 || o.ne[3] != 1 || !o.data ||
            o.nb[0] != static_cast<std::int64_t>(dtype_size(o.dtype))) {
            throw std::invalid_argument("GGML K projection: invalid output section");
        }
        output.data[i] = o.data;
        output.rows[i] = o.ne[0];
        output.stride[i] = o.nb[1];
        output.fp32[i] = o.dtype == DType::FP32;
        total_rows += o.ne[0];
    }
    if (total_rows != weight.n) {
        throw std::invalid_argument("GGML K projection: output sections do not cover rows");
    }
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* rows = static_cast<const unsigned char*>(weight.qdata);
    const auto* descriptors = static_cast<const std::uint64_t*>(weight.qhigh);
    const int n = weight.n;
    const int k = weight.k;
    const int tokens = x.ne[1];
#ifdef NINFER_PASCAL_BUILD
    // No CUTLASS staging route on GP100; the HFMA2 decode gemv carries every width.
    (void)workspace;
    if (tokens > 4) {
        // The tile GEMM decodes each weight once for eight staged tokens. It covers the plain
        // single-bf16-section projection; section-split (qkv+gate fusion), add epilogues and the
        // TiledGdn column permutation go to the chunked HFMA2 gemv.
        if constexpr (!TiledGdn) {
            if (output.count == 1 && !output.add && !output.fp32[0]) {
                ggml_k_launch_tile_pascal(x, weight, outputs[0], stream);
                return;
            }
        }
        launch_gemv_hfma2<8, TiledGdn>(input, rows, descriptors, output, n, k, tokens, false,
                                       stream);
        return;
    }
    launch_gemv_hfma2_decode<TiledGdn>(input, rows, descriptors, output, n, k, tokens,
                                       weight.ggml_k_first_q6, weight.ggml_k_type_change, stream);
    return;
#else
    if (tokens > 64 && workspace != nullptr) {
        cutlass_project<TiledGdn>(input, rows, descriptors, output, n, k, tokens,
                                  *workspace, stream);
        return;
    }
#endif
    switch (tokens) {
    case 1: launch_gemv<1, TiledGdn>(input, rows, descriptors, output, n, k,
                                     weight.ggml_k_first_q6, weight.ggml_k_type_change, stream); break;
    case 2: launch_gemv<2, TiledGdn>(input, rows, descriptors, output, n, k,
                                     weight.ggml_k_first_q6, weight.ggml_k_type_change, stream); break;
    case 3: launch_gemv<3, TiledGdn>(input, rows, descriptors, output, n, k,
                                     weight.ggml_k_first_q6, weight.ggml_k_type_change, stream); break;
    case 4: launch_gemv<4, TiledGdn>(input, rows, descriptors, output, n, k,
                                     weight.ggml_k_first_q6, weight.ggml_k_type_change, stream); break;
    default:
#ifdef NINFER_PASCAL_BUILD
        // Unreachable: Pascal returned through the HFMA2 launchers above.
        throw std::invalid_argument("GGML K projection: unhandled token width");
#else
        if (tokens <= 16) {
            gemm<16, TiledGdn><<<dim3((n + 31) / 32, (tokens + 15) / 16), 256, 0, stream>>>(
                input, rows, descriptors, output, n, k, tokens);
        } else if (tokens <= 32) {
            gemm<32, TiledGdn><<<dim3((n + 31) / 32, (tokens + 31) / 32), 256, 0, stream>>>(
                input, rows, descriptors, output, n, k, tokens);
        } else if (tokens <= 64) {
            gemm<64, TiledGdn><<<dim3((n + 31) / 32, (tokens + 63) / 64), 256, 0, stream>>>(
                input, rows, descriptors, output, n, k, tokens);
        } else {
            gemm<96, TiledGdn><<<dim3((n + 31) / 32, (tokens + 95) / 96), 384, 0, stream>>>(
                input, rows, descriptors, output, n, k, tokens);
        }
#endif // NINFER_PASCAL_BUILD (tokens > 4 handled by the HFMA2 launchers above)
    }
}

void ggml_k_project_split(const Tensor& x, const Weight& weight, const Tensor* outputs,
                          int count, bool add, cudaStream_t stream, bool tiled_gdn_input,
                          WorkspaceArena* workspace) {
    if (tiled_gdn_input) {
        if (weight.k != 6144 && weight.k != 3072) {
            throw std::invalid_argument("GGML K GDN output requires K=6144 or TP2 K=3072");
        }
        project<true>(x, weight, outputs, count, add, stream, workspace);
    } else {
        project<false>(x, weight, outputs, count, add, stream, workspace);
    }
}

std::size_t ggml_k_cutlass_workspace_bytes(std::int32_t n, std::int32_t k,
                                           std::int32_t tokens) {
#ifdef NINFER_PASCAL_BUILD
    (void)n;
    (void)k;
    (void)tokens;
    return 0;  // the HFMA2 gemv route needs no workspace
#else
    if (n <= 0 || k <= 0 || tokens <= 64) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP16, {k, tokens});
    (void)layout.alloc(DType::FP16, {k, std::min(n, kCutlassRowTile)});
    (void)layout.alloc(DType::FP32, {n, tokens});
    return layout.peak_bytes(1);
#endif // NINFER_PASCAL_BUILD
}

void ggml_k_embedding(const Tensor& ids, const Weight& weight, Tensor& out, cudaStream_t stream) {
    validate_weight(weight);
    embedding<<<ids.numel(), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data),
        static_cast<const unsigned char*>(weight.qdata),
        static_cast<const std::uint64_t*>(weight.qhigh),
        static_cast<__nv_bfloat16*>(out.data), weight.k);
}

} // namespace ninfer::ops::detail
