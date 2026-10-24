// Standalone micro-probe for the sm_60 GGUF-native Q6_K vector route (ops/linear/gguf
// vector_kernel). nsys on the NAS shows decode is dominated by vector_kernel<Q6_K, MaxTokens 8,
// Rows 2> at ~1.9x the MT4 cost: instruction/LSU-bound, not weight-bandwidth-bound. This probe
// replicates the production kernel (V0) and two candidate Pascal variants so the winner can be
// measured before touching production code:
//   V0 : faithful replica of vector_kernel Q6_K A16 (fp32 decode8 + 8 FFMA per value).
//   V1 : V0 with the 16 scattered LDS.U8 reads replaced by two LDS.64 loads + register shifts
//        (bit-identical math, quarter the LSU instruction count).
//   V2 : magic-bits HFMA2 decode (half2 = (v, v+1) adjacent pairs), fp16 pair accumulators
//        flushed per block with the row scale in fp32.
// Build (PowerShell after vcvars64 import):
//   nvcc -O3 -std=c++17 -arch=sm_60 --ptxas-options=-v -Xcompiler=/MD \
//        tools/p100/p100_vector_probe.cu -o build-p100/vector_probe.exe
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { printf("cuda err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

namespace {

constexpr int kStageBlocks = 2;
constexpr int kBlockBytes  = 210;
constexpr int kWarps       = 4;

__device__ __forceinline__ unsigned u8(const unsigned char* b, int i) { return b[i]; }
__device__ __forceinline__ float f16(const unsigned char* b, int i) {
    __half h;
    unsigned short w = static_cast<unsigned short>(u8(b, i)) |
                       (static_cast<unsigned short>(u8(b, i + 1)) << 8);
    *reinterpret_cast<unsigned short*>(&h) = w;
    return __half2float(h);
}

__device__ __forceinline__ void load_bf16x8(const __nv_bfloat16* p, float (&v)[8]) {
    const uint2 a = *reinterpret_cast<const uint2*>(p);
    const uint2 b = *reinterpret_cast<const uint2*>(p + 4);
    const unsigned w[4] = {a.x, a.y, b.x, b.y};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = __uint_as_float((w[j >> 1] >> (16 * (j & 1))) << 16);
    }
}

__device__ __forceinline__ void decode8_q6(const unsigned char* b, int j8, float (&w)[8]) {
    const int i = 8 * j8;
    const int half = i >> 7;
    const int within = i & 127;
    const int section = within >> 5;
    const int lane = within & 31;
    const float scale = f16(b, 208) * static_cast<float>(static_cast<signed char>(u8(b, 192 + i / 16)));
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const unsigned low =
            (u8(b, half * 64 + (section & 1) * 32 + lane + v) >> ((section >> 1) * 4)) & 15U;
        const unsigned high = (u8(b, 128 + half * 32 + lane + v) >> (section * 2)) & 3U;
        w[v] = scale * static_cast<float>(static_cast<int>(low | (high << 4)) - 32);
    }
}

// Shared staging machinery (identical geometry for all variants).
template <int MaxTokens, int Rows>
struct Stage {
    static constexpr int kRowWords   = kStageBlocks * kBlockBytes / 4;  // 105
    static constexpr int kStageWords = Rows * kRowWords;
    static constexpr int kLaneWords  = (kStageWords + 31) / 32;
};

// V1/V2 stage every 210-byte block into its own 216-byte slot (54 words) so the block base and
// the in-block 8-byte reads are all naturally aligned; row r occupies two slots.
template <int MaxTokens, int Rows>
struct StageP {
    static constexpr int kSlotWords  = 54;
    static constexpr int kStageWords = Rows * kStageBlocks * kSlotWords;
    static constexpr int kLaneWords  = (kStageWords + 31) / 32;
};

// ---- V0: faithful production replica ----
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v0_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = Stage<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int r = word / S::kRowWords;
            const int w = word - r * S::kRowWords;
            prefetch[i] = word < S::kStageWords && r < live_rows
                              ? __ldg(base + r * row_stride + stage_block * kBlockBytes / 4 + w)
                              : 0U;
        }
    };
    float sum[Rows][MaxTokens] = {};
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            float a[MaxTokens][8];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                load_bf16x8(x + static_cast<std::int64_t>(t < tokens ? t : 0) * k + stored, a[t]);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + r * S::kRowWords * 4 + b * kBlockBytes;
                float w[8];
                decode8_q6(block, lane, w);
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t][j], sum[r][t]); }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum[r][t] += __shfl_xor_sync(0xffffffffU, sum[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = sum[r][t]; }
                }
            }
        }
    }
}

// ---- V1: V0 with LDS.64 word reads (bit-identical decode arithmetic) ----
__device__ __forceinline__ void decode8_q6_w64(const unsigned char* b, int j8, float (&w)[8]) {
    const int i = 8 * j8;
    const int half = i >> 7;
    const int within = i & 127;
    const int section = within >> 5;
    const int lane = within & 31;
    const float scale = f16(b, 208) * static_cast<float>(static_cast<signed char>(b[192 + i / 16]));
    const uint2 ql = *reinterpret_cast<const uint2*>(b + half * 64 + (section & 1) * 32 + lane);
    const uint2 qh = *reinterpret_cast<const uint2*>(b + 128 + half * 32 + lane);
    const int nsh = (section >> 1) * 4;
    const int hsh = section * 2;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const unsigned qw = v < 4 ? ql.x : ql.y;
        const unsigned hw = v < 4 ? qh.x : qh.y;
        const int byte = 8 * (v & 3);
        const unsigned low = (qw >> (byte + nsh)) & 15U;
        const unsigned high = (hw >> (byte + hsh)) & 3U;
        w[v] = scale * static_cast<float>(static_cast<int>(low | (high << 4)) - 32);
    }
}

template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v1_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = StageP<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int slot = word / S::kSlotWords;
            const int r    = slot >> 1;
            const int blk  = slot & 1;
            const int off  = word - slot * S::kSlotWords;
            const unsigned* pair_base =
                base + r * row_stride + (stage_block >> 1) * (kStageBlocks * kBlockBytes / 4);
            if (word >= S::kStageWords || r >= live_rows || off >= 52) { prefetch[i] = 0U; }
            else if (blk == 0) { prefetch[i] = __ldg(pair_base + off); }
            else { prefetch[i] = __byte_perm(__ldg(pair_base + 52 + off),
                                             __ldg(pair_base + 53 + off), 0x5432); }
        }
    };
    float sum[Rows][MaxTokens] = {};
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            float a[MaxTokens][8];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                load_bf16x8(x + static_cast<std::int64_t>(t < tokens ? t : 0) * k + stored, a[t]);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + (r * kStageBlocks + b) * S::kSlotWords * 4;
                float w[8];
                decode8_q6_w64(block, lane, w);
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t][j], sum[r][t]); }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum[r][t] += __shfl_xor_sync(0xffffffffU, sum[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = sum[r][t]; }
                }
            }
        }
    }
}

// ---- V2: magic-bits HFMA2 with fp16 pair accumulators, flushed per block ----
// Activation is read as half (pre-converted), so no bf16 unpack in the hot loop.
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v2_kernel(const __half* __restrict__ xh,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = StageP<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int slot = word / S::kSlotWords;
            const int r    = slot >> 1;
            const int blk  = slot & 1;
            const int off  = word - slot * S::kSlotWords;
            prefetch[i] = word < S::kStageWords && r < live_rows && off < 53
                              ? __ldg(base + r * row_stride + (stage_block + blk) * kBlockBytes / 4 + off)
                              : 0U;
        }
    };
    float acc[Rows][MaxTokens] = {};
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    const int half_field = lane >> 4;              // i = 8*lane: half = i>>7
    const int within = (8 * lane) & 127;
    const int section = within >> 5;
    const int ln = within & 31;
    const int ql_off = half_field * 64 + (section & 1) * 32 + ln;
    const int qh_off = 128 + half_field * 32 + ln;
    const int nsh = (section >> 1) * 4;
    const int hsh = section * 2;
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            half2 xa[MaxTokens][4];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                const uint2 h = *reinterpret_cast<const uint2*>(
                    xh + static_cast<std::int64_t>(t < tokens ? t : 0) * k + stored);
                xa[t][0] = *reinterpret_cast<const half2*>(&h.x);
                xa[t][1] = *reinterpret_cast<const half2*>(&h.y);
                const uint2 h2 = *reinterpret_cast<const uint2*>(
                    xh + static_cast<std::int64_t>(t < tokens ? t : 0) * k + stored + 4);
                xa[t][2] = *reinterpret_cast<const half2*>(&h2.x);
                xa[t][3] = *reinterpret_cast<const half2*>(&h2.y);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + (r * kStageBlocks + b) * S::kSlotWords * 4;
                const uint2 ql = *reinterpret_cast<const uint2*>(block + ql_off);
                const uint2 qh = *reinterpret_cast<const uint2*>(block + qh_off);
                const float scale =
                    f16(block, 208) * static_cast<float>(static_cast<signed char>(block[192 + (8 * lane) / 16]));
                half2 s0[MaxTokens], s1[MaxTokens];
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    s0[t] = __float2half2_rn(0.0F);
                    s1[t] = s0[t];
                }
#pragma unroll
                for (int p = 0; p < 4; ++p) {  // pairs of adjacent values (2p, 2p+1)
                    const unsigned qwl = p < 2 ? ql.x : ql.y;
                    const unsigned qwh = p < 2 ? qh.x : qh.y;
                    const int byte = 8 * (p & 1);
                    const unsigned code_lo = ((qwl >> (byte + nsh)) & 15U) |
                                             (((qwh >> (byte + hsh)) & 3U) << 4);
                    const unsigned code_hi = ((qwl >> (byte + 8 + nsh)) & 15U) |
                                             (((qwh >> (byte + 8 + hsh)) & 3U) << 4);
                    const unsigned bits = 0x64006400u | code_lo | (code_hi << 16);
                    const half2 c2 = __hsub2(*reinterpret_cast<const half2*>(&bits),
                                             __float2half2_rn(1056.0F));
#pragma unroll
                    for (int t = 0; t < MaxTokens; ++t) {
                        if (p < 2) { s0[t] = __hfma2(c2, xa[t][p], s0[t]); }
                        else       { s1[t] = __hfma2(c2, xa[t][p], s1[t]); }
                    }
                }
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    const half2 s = __hadd2(s0[t], s1[t]);
                    const float2 f = __half22float2(s);
                    acc[r][t] += scale * (f.x + f.y);
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                acc[r][t] += __shfl_xor_sync(0xffffffffU, acc[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = acc[r][t]; }
                }
            }
        }
    }
}

// ---- V4: scale folded into the weight half2 (hmul2), fp16 accumulators live across 4 blocks. ----
// Per (b,r,t): 4 HFMA2 only; flush once per 4 blocks (~1.25 instructions per (r,t,b)).
// Product range: w = d*s8*(code-32) is a real weight value, so d*s8 stays near bf16 weight scale;
// the fp16 scale conversion and the 32-value accumulator span are the points the oracle must clear.
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v4_kernel(const __half* __restrict__ xh,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = StageP<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int slot = word / S::kSlotWords;
            const int r    = slot >> 1;
            const int blk  = slot & 1;
            const int off  = word - slot * S::kSlotWords;
            prefetch[i] = word < S::kStageWords && r < live_rows && off < 53
                              ? __ldg(base + r * row_stride + (stage_block + blk) * kBlockBytes / 4 + off)
                              : 0U;
        }
    };
    float acc[Rows][MaxTokens] = {};
    const half2 zero2 = __float2half2_rn(0.0F);
    half2 hp[Rows][MaxTokens][2];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) { hp[r][t][0] = zero2; hp[r][t][1] = zero2; }
    }
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    const int half_field = lane >> 4;
    const int within = (8 * lane) & 127;
    const int section = within >> 5;
    const int ln = within & 31;
    const int ql_off = half_field * 64 + (section & 1) * 32 + ln;
    const int qh_off = 128 + half_field * 32 + ln;
    const int nsh = (section >> 1) * 4;
    const int hsh = section * 2;
    const half2 bias = __float2half2_rn(1056.0F);
    const int flush_period = (blocks % 4 == 0) ? 4 : 2;
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int blk = stage_block + b;
            const int stored = blk * 256 + lane * 8;
            half2 xa[MaxTokens][4];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                const int source = t < tokens ? t : 0;
                const __half* p = xh + static_cast<std::int64_t>(source) * k + stored;
                const uint2 h = *reinterpret_cast<const uint2*>(p);
                const uint2 h2 = *reinterpret_cast<const uint2*>(p + 4);
                xa[t][0] = *reinterpret_cast<const half2*>(&h.x);
                xa[t][1] = *reinterpret_cast<const half2*>(&h.y);
                xa[t][2] = *reinterpret_cast<const half2*>(&h2.x);
                xa[t][3] = *reinterpret_cast<const half2*>(&h2.y);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + (r * kStageBlocks + b) * S::kSlotWords * 4;
                const uint2 ql = *reinterpret_cast<const uint2*>(block + ql_off);
                const uint2 qh = *reinterpret_cast<const uint2*>(block + qh_off);
                const float scale = f16(block, 208) *
                    static_cast<float>(static_cast<signed char>(block[192 + lane / 2]));
                const half2 scale2 = __floats2half2_rn(scale, scale);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const unsigned qwl = p < 2 ? ql.x : ql.y;
                    const unsigned qwh = p < 2 ? qh.x : qh.y;
                    const int byte = 8 * (p & 1);
                    const unsigned code_lo = ((qwl >> (byte + nsh)) & 15U) |
                                             (((qwh >> (byte + hsh)) & 3U) << 4);
                    const unsigned code_hi = ((qwl >> (byte + 8 + nsh)) & 15U) |
                                             (((qwh >> (byte + 8 + hsh)) & 3U) << 4);
                    const unsigned bits = 0x64006400u | code_lo | (code_hi << 16);
                    const half2 w = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&bits), bias),
                                            scale2);
#pragma unroll
                    for (int t = 0; t < MaxTokens; ++t) {
                        hp[r][t][p & 1] = __hfma2(w, xa[t][p], hp[r][t][p & 1]);
                    }
                }
            }
            if ((blk + 1) % flush_period == 0 || blk == blocks - 1) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
#pragma unroll
                    for (int t = 0; t < MaxTokens; ++t) {
                        const half2 s = __hadd2(hp[r][t][0], hp[r][t][1]);
                        const float2 f = __half22float2(s);
                        acc[r][t] += f.x + f.y;
                        hp[r][t][0] = zero2; hp[r][t][1] = zero2;
                    }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                acc[r][t] += __shfl_xor_sync(0xffffffffU, acc[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = acc[r][t]; }
                }
            }
        }
    }
}

// ---- V5: V4 with a single half2 accumulator per (r,t) to hold registers under 4-5 CTAs/SM. ----
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v5_kernel(const __half* __restrict__ xh,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = StageP<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int slot = word / S::kSlotWords;
            const int r    = slot >> 1;
            const int blk  = slot & 1;
            const int off  = word - slot * S::kSlotWords;
            prefetch[i] = word < S::kStageWords && r < live_rows && off < 53
                              ? __ldg(base + r * row_stride + (stage_block + blk) * kBlockBytes / 4 + off)
                              : 0U;
        }
    };
    float acc[Rows][MaxTokens] = {};
    const half2 zero2 = __float2half2_rn(0.0F);
    half2 hp[Rows][MaxTokens];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) { hp[r][t] = zero2; }
    }
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    const int half_field = lane >> 4;
    const int within = (8 * lane) & 127;
    const int section = within >> 5;
    const int ln = within & 31;
    const int ql_off = half_field * 64 + (section & 1) * 32 + ln;
    const int qh_off = 128 + half_field * 32 + ln;
    const int nsh = (section >> 1) * 4;
    const int hsh = section * 2;
    const half2 bias = __float2half2_rn(1056.0F);
    const int flush_period = (blocks % 4 == 0) ? 4 : 2;
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int blk = stage_block + b;
            const int stored = blk * 256 + lane * 8;
            half2 w[Rows][4];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + (r * kStageBlocks + b) * S::kSlotWords * 4;
                const uint2 ql = *reinterpret_cast<const uint2*>(block + ql_off);
                const uint2 qh = *reinterpret_cast<const uint2*>(block + qh_off);
                const float scale = f16(block, 208) *
                    static_cast<float>(static_cast<signed char>(block[192 + lane / 2]));
                const half2 scale2 = __floats2half2_rn(scale, scale);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const unsigned qwl = p < 2 ? ql.x : ql.y;
                    const unsigned qwh = p < 2 ? qh.x : qh.y;
                    const int byte = 8 * (p & 1);
                    const unsigned code_lo = ((qwl >> (byte + nsh)) & 15U) |
                                             (((qwh >> (byte + hsh)) & 3U) << 4);
                    const unsigned code_hi = ((qwl >> (byte + 8 + nsh)) & 15U) |
                                             (((qwh >> (byte + 8 + hsh)) & 3U) << 4);
                    const unsigned bits = 0x64006400u | code_lo | (code_hi << 16);
                    w[r][p] = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&bits), bias), scale2);
                }
            }
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                const int source = t < tokens ? t : 0;
                const __half* p = xh + static_cast<std::int64_t>(source) * k + stored;
                const uint2 h = *reinterpret_cast<const uint2*>(p);
                const uint2 h2 = *reinterpret_cast<const uint2*>(p + 4);
                const half2 xa[4] = {*reinterpret_cast<const half2*>(&h.x),
                                     *reinterpret_cast<const half2*>(&h.y),
                                     *reinterpret_cast<const half2*>(&h2.x),
                                     *reinterpret_cast<const half2*>(&h2.y)};
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) { hp[r][t] = __hfma2(w[r][p], xa[p], hp[r][t]); }
                }
            }
            if ((blk + 1) % flush_period == 0 || blk == blocks - 1) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
#pragma unroll
                    for (int t = 0; t < MaxTokens; ++t) {
                        const float2 f = __half22float2(hp[r][t]);
                        acc[r][t] += f.x + f.y;
                        hp[r][t] = zero2;
                    }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                acc[r][t] += __shfl_xor_sync(0xffffffffU, acc[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = acc[r][t]; }
                }
            }
        }
    }
}

// ---- V6: V4 but consuming BF16 x directly (inline bf16 pair -> half2, no pre-pass) ----
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v6_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = StageP<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int slot = word / S::kSlotWords;
            const int r    = slot >> 1;
            const int blk  = slot & 1;
            const int off  = word - slot * S::kSlotWords;
            prefetch[i] = word < S::kStageWords && r < live_rows && off < 53
                              ? __ldg(base + r * row_stride + (stage_block + blk) * kBlockBytes / 4 + off)
                              : 0U;
        }
    };
    float acc[Rows][MaxTokens] = {};
    const half2 zero2 = __float2half2_rn(0.0F);
    half2 hp[Rows][MaxTokens][2];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) { hp[r][t][0] = zero2; hp[r][t][1] = zero2; }
    }
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    const int half_field = lane >> 4;
    const int within = (8 * lane) & 127;
    const int section = within >> 5;
    const int ln = within & 31;
    const int ql_off = half_field * 64 + (section & 1) * 32 + ln;
    const int qh_off = 128 + half_field * 32 + ln;
    const int nsh = (section >> 1) * 4;
    const int hsh = section * 2;
    const half2 bias = __float2half2_rn(1056.0F);
    const int flush_period = (blocks % 4 == 0) ? 4 : 2;
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int blk = stage_block + b;
            const int stored = blk * 256 + lane * 8;
            half2 xv[MaxTokens][4];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                const int source = t < tokens ? t : 0;
                const unsigned* wp = reinterpret_cast<const unsigned*>(
                    x + static_cast<std::int64_t>(source) * k + stored);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const unsigned pair = wp[p];  // two bf16 = the fp32 high halves
                    xv[t][p] = __floats2half2_rn(__uint_as_float(pair << 16),
                                                 __uint_as_float(pair & 0xFFFF0000u));
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + (r * kStageBlocks + b) * S::kSlotWords * 4;
                const uint2 ql = *reinterpret_cast<const uint2*>(block + ql_off);
                const uint2 qh = *reinterpret_cast<const uint2*>(block + qh_off);
                const float scale = f16(block, 208) *
                    static_cast<float>(static_cast<signed char>(block[192 + lane / 2]));
                const half2 scale2 = __floats2half2_rn(scale, scale);
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const unsigned qwl = p < 2 ? ql.x : ql.y;
                        const unsigned qwh = p < 2 ? qh.x : qh.y;
                        const int byte = 8 * (p & 1);
                        const unsigned code_lo = ((qwl >> (byte + nsh)) & 15U) |
                                                 (((qwh >> (byte + hsh)) & 3U) << 4);
                        const unsigned code_hi = ((qwl >> (byte + 8 + nsh)) & 15U) |
                                                 (((qwh >> (byte + 8 + hsh)) & 3U) << 4);
                        const unsigned bits = 0x64006400u | code_lo | (code_hi << 16);
                        const half2 w = __hmul2(
                            __hsub2(*reinterpret_cast<const half2*>(&bits), bias), scale2);
                        hp[r][t][p & 1] = __hfma2(w, xv[t][p], hp[r][t][p & 1]);
                    }
                }
            }
            if ((blk + 1) % flush_period == 0 || blk == blocks - 1) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
#pragma unroll
                    for (int t = 0; t < MaxTokens; ++t) {
                        const half2 s = __hadd2(hp[r][t][0], hp[r][t][1]);
                        const float2 f = __half22float2(s);
                        acc[r][t] += f.x + f.y;
                        hp[r][t][0] = zero2; hp[r][t][1] = zero2;
                    }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                acc[r][t] += __shfl_xor_sync(0xffffffffU, acc[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = acc[r][t]; }
                }
            }
        }
    }
}

// ---- V7: production fp32 math, ping-pong staging (LDG latency overlaps compute). ----
// Weight words for stage s+1 are issued as global loads BEFORE computing stage s, so the two
// syncwarp points bracket the long-latency LDGs. Decode uses the wide u64 loads (V1). Bit-identical
// arithmetic to production (bf16 -> fp32 shifts, fp32 FFMA accumulation).
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v7_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = StageP<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][2][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto issue_loads = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int slot = word / S::kSlotWords;
            const int r    = slot >> 1;
            const int blk  = slot & 1;
            const int off  = word - slot * S::kSlotWords;
            const unsigned* pair_base =
                base + r * row_stride + (stage_block >> 1) * (kStageBlocks * kBlockBytes / 4);
            if (word >= S::kStageWords || r >= live_rows || off >= 52) { prefetch[i] = 0U; }
            else if (blk == 0) { prefetch[i] = __ldg(pair_base + off); }
            else { prefetch[i] = __byte_perm(__ldg(pair_base + 52 + off),
                                             __ldg(pair_base + 53 + off), 0x5432); }
        }
    };
    float sum[Rows][MaxTokens] = {};
    issue_loads(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
        const int cur = (stage_block / kStageBlocks) & 1;
        // STS current, then start the next stage's LDGs, then compute the current stage.
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][cur][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { issue_loads(stage_block + kStageBlocks); }
        const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp][cur]);
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            float a[MaxTokens][8];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                load_bf16x8(x + static_cast<std::int64_t>(t < tokens ? t : 0) * k + stored, a[t]);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + (r * kStageBlocks + b) * S::kSlotWords * 4;
                float w[8];
                decode8_q6_w64(block, lane, w);
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t][j], sum[r][t]); }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum[r][t] += __shfl_xor_sync(0xffffffffU, sum[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = sum[r][t]; }
                }
            }
        }
    }
}

// ---- V8: V7 with 4-block stages (deeper prefetch, fewer sync points). ----
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v8_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    constexpr int kSB = 4;
    constexpr int kSlotWords = 54;  // 210B -> 216B slot: 8-byte aligned block bases
    constexpr int kStageWords = Rows * kSB * kSlotWords;
    constexpr int kLaneWords  = (kStageWords + 31) / 32;
    __shared__ __align__(16) unsigned stage[kWarps][kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[kLaneWords];
    const auto issue_loads = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int slot = word / kSlotWords;
            const int r    = slot / kSB;
            const int blk  = slot - r * kSB;
            const int off  = word - slot * kSlotWords;
            const unsigned* pair_base = base + r * row_stride + (stage_block + (blk >> 1) * 2) *
                                                  kBlockBytes / 4;
            if (word >= kStageWords || r < 0 || r >= live_rows || off >= 52) { prefetch[i] = 0U; }
            else if ((blk & 1) == 0) { prefetch[i] = __ldg(pair_base + off); }
            else { prefetch[i] = __byte_perm(__ldg(pair_base + 52 + off),
                                             __ldg(pair_base + 53 + off), 0x5432); }
        }
    };
    float sum[Rows][MaxTokens] = {};
    const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp]);
    issue_loads(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kSB) {
#pragma unroll
        for (int i = 0; i < kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < kStageWords) { stage[warp][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kSB < blocks) { issue_loads(stage_block + kSB); }
#pragma unroll
        for (int b = 0; b < kSB; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            float a[MaxTokens][8];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                load_bf16x8(x + static_cast<std::int64_t>(t < tokens ? t : 0) * k + stored, a[t]);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + (r * kSB + b) * kSlotWords * 4;
                float w[8];
                decode8_q6_w64(block, lane, w);
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t][j], sum[r][t]); }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum[r][t] += __shfl_xor_sync(0xffffffffU, sum[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = sum[r][t]; }
                }
            }
        }
    }
}

// ---- V9: production staging (105-word rows, byte decode) + ping-pong + Rows template. ----
template <int MaxTokens, int Rows>
__global__ void __launch_bounds__(kWarps * 32) v9_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const unsigned char* __restrict__ rows,
                                                         int n, int k, int tokens,
                                                         float* __restrict__ out) {
    using S = Stage<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][2][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int r = word / S::kRowWords;
            const int w = word - r * S::kRowWords;
            prefetch[i] = word < S::kStageWords && r < live_rows
                              ? __ldg(base + r * row_stride + stage_block * kBlockBytes / 4 + w)
                              : 0U;
        }
    };
    float sum[Rows][MaxTokens] = {};
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
        const int cur = (stage_block / kStageBlocks) & 1;
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][cur][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
        const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp][cur]);
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            float a[MaxTokens][8];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                load_bf16x8(x + static_cast<std::int64_t>(t < tokens ? t : 0) * k + stored, a[t]);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + r * S::kRowWords * 4 + b * kBlockBytes;
                float w[8];
                decode8_q6(block, lane, w);
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t][j], sum[r][t]); }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum[r][t] += __shfl_xor_sync(0xffffffffU, sum[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = sum[r][t]; }
                }
            }
        }
    }
}

// ---- V10: V9 + runtime valid-column break guard (production gguf.cu V10 shape). ----
template <int MaxTokens, int Rows, bool Pred>
__global__ void __launch_bounds__(kWarps * 32) v10_kernel(const __nv_bfloat16* __restrict__ x,
                                                          const unsigned char* __restrict__ rows,
                                                          int n, int k, int tokens,
                                                          float* __restrict__ out,
                                                          const int* __restrict__ valid_ends,
                                                          int valid_batch) {
    using S = Stage<MaxTokens, Rows>;
    __shared__ __align__(16) unsigned stage[kWarps][2][S::kStageWords];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * kWarps + warp) * Rows;
    if (first >= n) { return; }
    const int live_rows = min(Rows, n - first);
    int tokens_valid = tokens;
    if (valid_ends != nullptr) {
        const int stride = valid_batch > 0 ? tokens / valid_batch : tokens;
        int end = 0;
        if (lane < valid_batch) {
            end = lane * stride + min(max(__ldg(valid_ends + lane), 0), stride);
        }
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            end = max(end, __shfl_xor_sync(0xffffffffU, end, delta));
        }
        tokens_valid = min(tokens, end);
    }
    const int blocks = k / 256;
    const std::int64_t row_stride = static_cast<std::int64_t>(blocks) * kBlockBytes / 4;
    const auto* base = reinterpret_cast<const unsigned*>(rows) + first * row_stride;

    unsigned prefetch[S::kLaneWords];
    const auto load_stage = [&](int stage_block) {
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            const int r = word / S::kRowWords;
            const int w = word - r * S::kRowWords;
            prefetch[i] = word < S::kStageWords && r < live_rows
                              ? __ldg(base + r * row_stride + stage_block * kBlockBytes / 4 + w)
                              : 0U;
        }
    };
    float sum[Rows][MaxTokens] = {};
    load_stage(0);
    for (int stage_block = 0; stage_block < blocks; stage_block += kStageBlocks) {
        const int cur = (stage_block / kStageBlocks) & 1;
#pragma unroll
        for (int i = 0; i < S::kLaneWords; ++i) {
            const int word = lane + 32 * i;
            if (word < S::kStageWords) { stage[warp][cur][word] = prefetch[i]; }
        }
        __syncwarp();
        if (stage_block + kStageBlocks < blocks) { load_stage(stage_block + kStageBlocks); }
        const auto* staged = reinterpret_cast<const unsigned char*>(stage[warp][cur]);
#pragma unroll
        for (int b = 0; b < kStageBlocks; ++b) {
            const int stored = (stage_block + b) * 256 + lane * 8;
            float a[MaxTokens][8];
#pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                if (Pred) {
                    if (t < tokens_valid) {
                        load_bf16x8(x + static_cast<std::int64_t>(t) * k + stored, a[t]);
                    }
                } else {
                    if (t >= tokens_valid) { break; }
                    load_bf16x8(x + static_cast<std::int64_t>(t) * k + stored, a[t]);
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const unsigned char* block = staged + r * S::kRowWords * 4 + b * kBlockBytes;
                float w[8];
                decode8_q6(block, lane, w);
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (Pred) {
                        if (t < tokens_valid) {
#pragma unroll
                            for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t][j], sum[r][t]); }
                        }
                    } else {
                        if (t >= tokens_valid) { break; }
#pragma unroll
                        for (int j = 0; j < 8; ++j) { sum[r][t] = fmaf(w[j], a[t][j], sum[r][t]); }
                    }
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < MaxTokens; ++t) {
            if (t >= tokens_valid) { break; }
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum[r][t] += __shfl_xor_sync(0xffffffffU, sum[r][t], delta);
            }
        }
    }
    if (lane < live_rows) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == lane) {
#pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    if (t < tokens) { out[static_cast<std::int64_t>(first + r) * MaxTokens + t] = sum[r][t]; }
                }
            }
        }
    }
}

// ---- host ----
double timeit(const char* name, size_t bytes, int reps, void (*launch)()) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(); CK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(a); launch(); cudaEventRecord(b);
        CK(cudaDeviceSynchronize());
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    printf("%-22s %9.1f us  %6.1f GB/s\n", name, best * 1000.0, bytes / (best / 1000.0) / 1e9);
    return best;
}

int n_global, k_global;
unsigned char* g_rows;
__nv_bfloat16* g_x;
__half* g_xh;
float* g_out;
void l_v0_4()  { v0_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v0_8()  { v0_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v1_4()  { v1_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v1_8()  { v1_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v2_4()  { v2_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_xh, g_rows, n_global, k_global, 4, g_out); }
void l_v2_8()  { v2_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_xh, g_rows, n_global, k_global, 8, g_out); }
void l_v0_4x4(){ v0_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v1_4x4(){ v1_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v4_4()  { v4_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_xh, g_rows, n_global, k_global, 4, g_out); }
void l_v4_8()  { v4_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_xh, g_rows, n_global, k_global, 8, g_out); }
void l_v5_4()  { v5_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_xh, g_rows, n_global, k_global, 4, g_out); }
void l_v5_8()  { v5_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_xh, g_rows, n_global, k_global, 8, g_out); }
void l_v5_8r4(){ v5_kernel<8, 4><<<(n_global + 15) / 16, 128>>>(g_xh, g_rows, n_global, k_global, 8, g_out); }
void l_v6_4()  { v6_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v6_8()  { v6_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v6_4x4(){ v6_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v6_8x4(){ v6_kernel<8, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v7_4()  { v7_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v7_8()  { v7_kernel<8, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v7_4r2(){ v7_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v7_8r2(){ v7_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v8_4()  { v8_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v8_8()  { v8_kernel<8, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v8_8r2(){ v8_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v9_4()  { v9_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
void l_v9_8()  { v9_kernel<8, 4><<<(n_global + 15) / 16, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v9_8r2(){ v9_kernel<8, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out); }
void l_v9_4r2(){ v9_kernel<4, 2><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out); }
// TP2-shard row count (8704 = half of 17408): dispatch-grid sensitivity for R2 vs R4.
void l_v9_8r2_h(){ v9_kernel<8, 2><<<(n_global / 2 + 7) / 8, 128>>>(g_x, g_rows, n_global / 2, k_global, 8, g_out); }
void l_v9_8r4_h(){ v9_kernel<8, 4><<<(n_global / 2 + 15) / 16, 128>>>(g_x, g_rows, n_global / 2, k_global, 8, g_out); }
void l_v9_4r2_h(){ v9_kernel<4, 2><<<(n_global / 2 + 7) / 8, 128>>>(g_x, g_rows, n_global / 2, k_global, 4, g_out); }
void l_v9_4r4_h(){ v9_kernel<4, 4><<<(n_global / 2 + 15) / 16, 128>>>(g_x, g_rows, n_global / 2, k_global, 4, g_out); }
void l_v0_8r2_h(){ v0_kernel<8, 2><<<(n_global / 2 + 7) / 8, 128>>>(g_x, g_rows, n_global / 2, k_global, 8, g_out); }
void l_v0_4r2_h(){ v0_kernel<4, 2><<<(n_global / 2 + 7) / 8, 128>>>(g_x, g_rows, n_global / 2, k_global, 4, g_out); }
void l_v2_4x4() { v2_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_xh, g_rows, n_global, k_global, 4, g_out); }
void l_v4_4x4() { v4_kernel<4, 4><<<(n_global + 15) / 16, 128>>>(g_xh, g_rows, n_global, k_global, 4, g_out); }
void l_v4_8x4() { v4_kernel<8, 4><<<(n_global + 15) / 16, 128>>>(g_xh, g_rows, n_global, k_global, 8, g_out); }

int* g_ends = nullptr;
void l_v10_8r2_nb() { v10_kernel<8, 2, false><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out, nullptr, 0); }
void l_v10_8r2_b3() { v10_kernel<8, 2, false><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out, g_ends, 1); }
void l_v10_8r2_b3p(){ v10_kernel<8, 2, true ><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 8, g_out, g_ends, 1); }
void l_v10_4r2_nb() { v10_kernel<4, 2, false><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out, nullptr, 0); }
void l_v10_8r2_b3_h(){ v10_kernel<8, 2, false><<<(n_global / 2 + 7) / 8, 128>>>(g_x, g_rows, n_global / 2, k_global, 8, g_out, g_ends, 1); }
void l_v10_8r2_nb_h(){ v10_kernel<8, 2, false><<<(n_global / 2 + 7) / 8, 128>>>(g_x, g_rows, n_global / 2, k_global, 8, g_out, nullptr, 0); }
void l_v10_4r2_b3() { v10_kernel<4, 2, false><<<(n_global + 7) / 8, 128>>>(g_x, g_rows, n_global, k_global, 4, g_out, g_ends, 1); }

} // namespace

int main() {
    n_global = 17408; k_global = 5120;   // TP2 gate_up shard
    const size_t rows_bytes = static_cast<size_t>(n_global) * (k_global / 256) * kBlockBytes;
    CK(cudaMalloc(&g_rows, rows_bytes + 16));  // pad for the 2-byte over-read of slot word 52
    CK(cudaMemset(g_rows, 0x55, rows_bytes));
    CK(cudaMalloc(&g_x, sizeof(__nv_bfloat16) * k_global * 8));
    CK(cudaMemset(g_x, 0x3e, sizeof(__nv_bfloat16) * k_global * 8));
    CK(cudaMalloc(&g_xh, sizeof(__half) * k_global * 8));
    CK(cudaMemset(g_xh, 0x3c, sizeof(__half) * k_global * 8));
    CK(cudaMalloc(&g_out, sizeof(float) * n_global * 8));
    CK(cudaMalloc(&g_ends, sizeof(int)));
    const int h_ends = 3;
    CK(cudaMemcpy(g_ends, &h_ends, sizeof(int), cudaMemcpyHostToDevice));
    printf("shape %dx%d, weights %.1f MiB\n", n_global, k_global, rows_bytes / 1048576.0);
    timeit("V0 MT4 R2 (prod)",  rows_bytes, 20, l_v0_4);
    timeit("V1 MT4 R2 (u64)",   rows_bytes, 20, l_v1_4);
    timeit("V2 MT4 R2 (hfma2)", rows_bytes, 20, l_v2_4);
    timeit("V4 MT4 R2 (sc2hf)", rows_bytes, 20, l_v4_4);
    timeit("V0 MT8 R2 (prod)",  rows_bytes, 20, l_v0_8);
    timeit("V1 MT8 R2 (u64)",   rows_bytes, 20, l_v1_8);
    timeit("V2 MT8 R2 (hfma2)", rows_bytes, 20, l_v2_8);
    timeit("V4 MT8 R2 (sc2hf)", rows_bytes, 20, l_v4_8);
    timeit("V5 MT4 R2 (1acc)",  rows_bytes, 20, l_v5_4);
    timeit("V5 MT8 R2 (1acc)",  rows_bytes, 20, l_v5_8);
    timeit("V5 MT8 R4 (1acc)",  rows_bytes, 20, l_v5_8r4);
    timeit("V6 MT4 R2 (bf16)",  rows_bytes, 20, l_v6_4);
    timeit("V6 MT8 R2 (bf16)",  rows_bytes, 20, l_v6_8);
    timeit("V6 MT4 R4 (bf16)",  rows_bytes, 20, l_v6_4x4);
    timeit("V6 MT8 R4 (bf16)",  rows_bytes, 20, l_v6_8x4);
    timeit("V7 MT4 R4 (pipe)",  rows_bytes, 20, l_v7_4);
    timeit("V7 MT8 R4 (pipe)",  rows_bytes, 20, l_v7_8);
    timeit("V7 MT4 R2 (pipe)",  rows_bytes, 20, l_v7_4r2);
    timeit("V7 MT8 R2 (pipe)",  rows_bytes, 20, l_v7_8r2);
    timeit("V8 MT4 R4 (4b)",    rows_bytes, 20, l_v8_4);
    timeit("V8 MT8 R4 (4b)",    rows_bytes, 20, l_v8_8);
    timeit("V8 MT8 R2 (4b)",    rows_bytes, 20, l_v8_8r2);
    timeit("V9 MT4 R4 (pp)",    rows_bytes, 20, l_v9_4);
    timeit("V9 MT8 R4 (pp)",    rows_bytes, 20, l_v9_8);
    timeit("V9 MT8 R2 (pp)",    rows_bytes, 20, l_v9_8r2);
    timeit("V9 MT4 R2 (pp)",    rows_bytes, 20, l_v9_4r2);
    // V10 codegen attribution: nb = guard compiled but unbound (should match V9),
    // b3 = bound to 3/8 columns (the real gain), b3p = predicated variant.
    timeit("V10 MT8 R2 nb",     rows_bytes, 20, l_v10_8r2_nb);
    timeit("V10 MT8 R2 b3",     rows_bytes, 20, l_v10_8r2_b3);
    timeit("V10 MT8 R2 b3p",    rows_bytes, 20, l_v10_8r2_b3p);
    timeit("V10 MT4 R2 nb",     rows_bytes, 20, l_v10_4r2_nb);
    timeit("V10 MT4 R2 b3",     rows_bytes, 20, l_v10_4r2_b3);
    timeit("V10 MT8 R2 nb h8704", rows_bytes / 2, 20, l_v10_8r2_nb_h);
    timeit("V10 MT8 R2 b3 h8704", rows_bytes / 2, 20, l_v10_8r2_b3_h);
    // Down-projection-like half row count: R2 vs R4 grid sensitivity at TP2 shard sizes.
    timeit("V0 MT8 R2 h8704",   rows_bytes / 2, 20, l_v0_8r2_h);
    timeit("V0 MT4 R2 h8704",   rows_bytes / 2, 20, l_v0_4r2_h);
    timeit("V9 MT8 R2 h8704",   rows_bytes / 2, 20, l_v9_8r2_h);
    timeit("V9 MT8 R4 h8704",   rows_bytes / 2, 20, l_v9_8r4_h);
    timeit("V9 MT4 R2 h8704",   rows_bytes / 2, 20, l_v9_4r2_h);
    timeit("V9 MT4 R4 h8704",   rows_bytes / 2, 20, l_v9_4r4_h);
    timeit("V0 MT4 R4 tall",    rows_bytes, 20, l_v0_4x4);
    timeit("V1 MT4 R4 tall",    rows_bytes, 20, l_v1_4x4);
    timeit("V2 MT4 R4 tall",    rows_bytes, 20, l_v2_4x4);
    timeit("V4 MT4 R4 tall",    rows_bytes, 20, l_v4_4x4);
    timeit("V4 MT8 R4 tall",    rows_bytes, 20, l_v4_8x4);
    CK(cudaGetLastError());
    return 0;
}
