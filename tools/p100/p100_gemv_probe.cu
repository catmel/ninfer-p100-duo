// Standalone micro-probe for the P100 GGML_K gemv access pattern. No ninfer libs: kernels are
// self-contained so variants can be added cheaply. Answers: what caps the warp-per-row Q6_K
// stream at ~120 GB/s on sm_60?
//   k_stream  : loads only (no staging, no sync, no decode) -> raw pattern ceiling
//   k_v1      : loads + smem x staging + HFMA2 decode + fp32 merge (replicates production V1)
//   k_8rows   : k_v1 with 256-thread CTAs (8 rows)
//   k_2rowwarp: k_v1 with 2 rows per warp (T=1), same CTA size
//   k_q4      : aligned-only Q4_K pattern as a misalignment control
// Build (PowerShell after vcvars64 import):
//   nvcc -O3 -std=c++17 -arch=sm_60 --ptxas-options=-v -Xcompiler=/MD tools/p100/p100_gemv_probe.cu -o build-p100/gemv_probe.exe
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { printf("cuda err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

__device__ __forceinline__ uint32_t ld_u32_mis(const unsigned char* p) {
    if (((uintptr_t)p & 3u) == 0u) { return *reinterpret_cast<const uint32_t*>(p); }
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
__device__ __forceinline__ float bf2f(uint32_t word, int half_index) {
    return __uint_as_float(((word >> (16 * half_index)) & 0xffffu) << 16);
}

// ---- k_stream: pure loads, 1 warp/row, 128-thread CTA ----
__global__ void __launch_bounds__(128) k_stream(const unsigned char* rows, int n, int k, uint32_t* sink) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & 31;
    const unsigned char* b = rows + (size_t)row * (k / 256) * 210;
    const int qr = lane >> 3, ig = lane & 7;
    uint32_t s = 0;
    for (int blk = 0; blk < k / 256; ++blk) {
        const unsigned char* cb = b + blk * 210 + (qr >> 1) * 64 + ig * 4;
        s += ld_u32_mis(cb);
        s += ld_u32_mis(cb + 32);
        s += ld_u32_mis(b + blk * 210 + 128 + (qr >> 1) * 32 + ig * 4);
        s += __ldg(b + blk * 210 + 192 + 4 * qr + (ig >> 2));
        s += (uint32_t)*reinterpret_cast<const uint16_t*>(b + blk * 210 + 208);
    }
    if (s == 0xdeadbeefu) sink[0] = s;
}

// ---- shared staging + decode helper used by v1 family ----
__device__ __forceinline__ uint4 stage_pair(const __nv_bfloat16* src) {
    const uint2 a = *reinterpret_cast<const uint2*>(src);
    const uint2 b = *reinterpret_cast<const uint2*>(src + 32);
    half2 p0 = __floats2half2_rn(bf2f(a.x, 0), bf2f(b.x, 0));
    half2 p1 = __floats2half2_rn(bf2f(a.x, 1), bf2f(b.x, 1));
    half2 p2 = __floats2half2_rn(bf2f(a.y, 0), bf2f(b.y, 0));
    half2 p3 = __floats2half2_rn(bf2f(a.y, 1), bf2f(b.y, 1));
    return make_uint4(*(uint32_t*)&p0, *(uint32_t*)&p1, *(uint32_t*)&p2, *(uint32_t*)&p3);
}

template <int TPB>  // threads per block = rows per CTA
__device__ __forceinline__ void v1_body(const __nv_bfloat16* x, const unsigned char* rows,
                                        float* out, int n, int k, int row, int lane,
                                        half2 (*st)[128]) {
    const unsigned char* b = rows + (size_t)row * (k / 256) * 210;
    const int qr = lane >> 3, ig = lane & 7;
    const int nib_shift = 4 * (qr & 1);
    const half2 bias = __float2half2_rn(1056.0F);
    float acc = 0.0F;
    for (int blk = 0; blk < k / 256; ++blk) {
#pragma unroll
        for (int job = threadIdx.x; job < 32; job += TPB) {
            const int col = blk * 256 + 64 * (job >> 3) + 4 * (job & 7);
            reinterpret_cast<uint4*>(&st[0][0])[job] = stage_pair(x + col);
        }
        __syncthreads();
        const unsigned char* cb = b + blk * 210 + (qr >> 1) * 64 + ig * 4;
        uint32_t p0 = ld_u32_mis(cb), p1 = ld_u32_mis(cb + 32);
        uint32_t hi = ld_u32_mis(b + blk * 210 + 128 + (qr >> 1) * 32 + ig * 4);
        float d = __half2float(*reinterpret_cast<const __half*>(b + blk * 210 + 208));
        float s0 = d * (float)(int8_t)__ldg(b + blk * 210 + 192 + 4 * qr + (ig >> 2));
        float s1 = d * (float)(int8_t)__ldg(b + blk * 210 + 192 + 4 * qr + (ig >> 2) + 2);
        half2 sums = __float2half2_rn(0.0F);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int sh = 8 * p + nib_shift;
            const uint32_t a = ((p0 >> sh) & 0xFu) | (((hi >> sh) & 3u) << 4);
            const uint32_t bb = ((p1 >> sh) & 0xFu) | (((hi >> (sh + 2)) & 3u) << 4);
            const uint32_t bits = 0x64006400u | a | (bb << 16);
            half2 c2 = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
            sums = __hfma2(c2, st[0][32 * qr + 4 * ig + p], sums);
        }
        float2 s = __half22float2(sums);
        acc += s0 * s.x + s1 * s.y;
        __syncthreads();
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) acc += __shfl_down_sync(~0u, acc, delta);
    if (lane == 0) out[row] = acc;
}

__global__ void __launch_bounds__(128) k_v1(const __nv_bfloat16* x, const unsigned char* rows,
                                            float* out, int n, int k) {
    __shared__ half2 st[1][128];
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= n) return;
    v1_body<128>(x, rows, out, n, k, row, threadIdx.x & 31, st);
}

__global__ void __launch_bounds__(256) k_8rows(const __nv_bfloat16* x, const unsigned char* rows,
                                               float* out, int n, int k) {
    __shared__ half2 st[1][128];
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= n) return;
    v1_body<256>(x, rows, out, n, k, row, threadIdx.x & 31, st);
}

// ---- k_2rowwarp: one warp owns 2 rows, interleaved streams (T=1) ----
__global__ void __launch_bounds__(128) k_2rowwarp(const __nv_bfloat16* x, const unsigned char* rows,
                                                  float* out, int n, int k) {
    __shared__ half2 st[1][128];
    const int pair = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int r0 = pair * 2, r1 = r0 + 1;
    if (r0 >= n) return;
    const int lane = threadIdx.x & 31;
    const int qr = lane >> 3, ig = lane & 7;
    const int nib_shift = 4 * (qr & 1);
    const half2 bias = __float2half2_rn(1056.0F);
    const size_t stride = (size_t)(k / 256) * 210;
    const unsigned char* b0 = rows + (size_t)r0 * stride;
    const unsigned char* b1 = rows + (r1 < n ? (size_t)r1 : (size_t)r0) * stride;
    float acc0 = 0.0F, acc1 = 0.0F;
    for (int blk = 0; blk < k / 256; ++blk) {
#pragma unroll
        for (int job = threadIdx.x; job < 32; job += 128) {
            const int col = blk * 256 + 64 * (job >> 3) + 4 * (job & 7);
            reinterpret_cast<uint4*>(&st[0][0])[job] = stage_pair(x + col);
        }
        __syncthreads();
        const unsigned char* c0 = b0 + blk * 210;
        const unsigned char* c1 = b1 + blk * 210;
        uint32_t p0a = ld_u32_mis(c0 + (qr >> 1) * 64 + ig * 4), p1a = ld_u32_mis(c0 + 32 + (qr >> 1) * 64 + ig * 4);
        uint32_t hia = ld_u32_mis(c0 + 128 + (qr >> 1) * 32 + ig * 4);
        uint32_t p0b = ld_u32_mis(c1 + (qr >> 1) * 64 + ig * 4), p1b = ld_u32_mis(c1 + 32 + (qr >> 1) * 64 + ig * 4);
        uint32_t hib = ld_u32_mis(c1 + 128 + (qr >> 1) * 32 + ig * 4);
        float da = __half2float(*reinterpret_cast<const __half*>(c0 + 208));
        float db = __half2float(*reinterpret_cast<const __half*>(c1 + 208));
        float sa0 = da * (float)(int8_t)__ldg(c0 + 192 + 4 * qr + (ig >> 2));
        float sa1 = da * (float)(int8_t)__ldg(c0 + 192 + 4 * qr + (ig >> 2) + 2);
        float sb0 = db * (float)(int8_t)__ldg(c1 + 192 + 4 * qr + (ig >> 2));
        float sb1 = db * (float)(int8_t)__ldg(c1 + 192 + 4 * qr + (ig >> 2) + 2);
        half2 suma = __float2half2_rn(0.0F), sumb = suma;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int sh = 8 * p + nib_shift;
            half2 xv = st[0][32 * qr + 4 * ig + p];
            uint32_t aa = ((p0a >> sh) & 0xFu) | (((hia >> sh) & 3u) << 4);
            uint32_t ab = ((p1a >> sh) & 0xFu) | (((hia >> (sh + 2)) & 3u) << 4);
            uint32_t bits_a = 0x64006400u | aa | (ab << 16);
            uint32_t ba = ((p0b >> sh) & 0xFu) | (((hib >> sh) & 3u) << 4);
            uint32_t bb = ((p1b >> sh) & 0xFu) | (((hib >> (sh + 2)) & 3u) << 4);
            uint32_t bits_b = 0x64006400u | ba | (bb << 16);
            suma = __hfma2(__hsub2(*reinterpret_cast<const half2*>(&bits_a), bias), xv, suma);
            sumb = __hfma2(__hsub2(*reinterpret_cast<const half2*>(&bits_b), bias), xv, sumb);
        }
        float2 fa = __half22float2(suma), fb = __half22float2(sumb);
        acc0 += sa0 * fa.x + sa1 * fa.y;
        acc1 += sb0 * fb.x + sb1 * fb.y;
        __syncthreads();
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
        acc0 += __shfl_down_sync(~0u, acc0, delta);
        acc1 += __shfl_down_sync(~0u, acc1, delta);
    }
    if (lane == 0) { out[r0] = acc0; if (r1 < n) out[r1] = acc1; }
}

// ---- k_q4: aligned Q4_K pattern control (1 warp/row, staging, decode) ----
__global__ void __launch_bounds__(128) k_q4(const __nv_bfloat16* x, const unsigned char* rows,
                                            float* out, int n, int k) {
    __shared__ half2 st[1][128];
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= n) return;
    const int lane = threadIdx.x & 31;
    const unsigned char* b = rows + (size_t)row * (k / 256) * 144;
    const int qr = lane >> 3, ig = lane & 7;
    const half2 bias = __float2half2_rn(1024.0F);
    float acc = 0.0F;
    for (int blk = 0; blk < k / 256; ++blk) {
#pragma unroll
        for (int job = threadIdx.x; job < 32; job += 128) {
            const int col = blk * 256 + 64 * (job >> 3) + 4 * (job & 7);
            reinterpret_cast<uint4*>(&st[0][0])[job] = stage_pair(x + col);
        }
        __syncthreads();
        const unsigned char* cb = b + blk * 144;
        uint32_t p0 = *reinterpret_cast<const uint32_t*>(cb + 16 + qr * 32 + ig * 4);
        float d = __half2float(*reinterpret_cast<const __half*>(cb));
        float dmin = __half2float(*reinterpret_cast<const __half*>(cb + 2));
        int low = qr & 3;
        float s0 = d * (int)(cb[4 + low] & 63), m0 = dmin * (int)(cb[8 + low] & 63);
        float s1 = d * (int)(cb[4 + ((qr & 3) | 1)] & 63) * 0.5f;  // rough scale probe
        s1 = d * (int)(cb[4 + low + 1] & 63); m0 = m0;  // lane pair scale for c+32 group
        half2 sums = __float2half2_rn(0.0F), sumv = sums;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            uint32_t w = p0 >> (8 * p);
            uint32_t bits = 0x64006400u | (w & 0xFu) | (((w >> 4) & 0xFu) << 16);
            half2 c2 = __hsub2(*reinterpret_cast<const half2*>(&bits), bias);
            half2 xv = st[0][32 * qr + 4 * ig + p];
            sums = __hfma2(c2, xv, sums);
            sumv = __hadd2(sumv, xv);
        }
        float2 s = __half22float2(sums), v = __half22float2(sumv);
        acc += s0 * s.x + s1 * s.y - m0 * v.x - m0 * v.y;
        __syncthreads();
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) acc += __shfl_down_sync(~0u, acc, delta);
    if (lane == 0) out[row] = acc;
}

// ---- driver ----
template <typename F>
double timeit(const char* name, size_t bytes, int reps, F f) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); CK(cudaDeviceSynchronize());  // warm
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(a); f(); cudaEventRecord(b);
        CK(cudaDeviceSynchronize());
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    printf("%-14s %8.1f us   %6.1f GB/s\n", name, best * 1000.0, bytes / (best / 1000.0) / 1e9);
    return best;
}

int main() {
    const int n = 34816, k = 5120;           // gate_up full shape
    const size_t q6_bytes = (size_t)n * (k / 256) * 210;
    const size_t q4_bytes = (size_t)n * (k / 256) * 144;
    void *d_q6, *d_q4, *d_x, *d_out; uint32_t* d_sink;
    CK(cudaMalloc(&d_q6, q6_bytes));
    CK(cudaMalloc(&d_q4, q4_bytes));
    CK(cudaMalloc(&d_x, (size_t)k * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));
    CK(cudaMalloc(&d_sink, 4));
    CK(cudaMemset(d_q6, 0x55, q6_bytes));
    CK(cudaMemset(d_q4, 0x55, q4_bytes));
    CK(cudaMemset(d_x, 0x3f, (size_t)k * sizeof(__nv_bfloat16)));

    const __nv_bfloat16* x = (const __nv_bfloat16*)d_x;
    const unsigned char* q6 = (const unsigned char*)d_q6;
    const unsigned char* q4 = (const unsigned char*)d_q4;
    float* out = (float*)d_out;

    printf("shape n=%d k=%d, Q6 row bytes %.1f MiB total\n", n, k, q6_bytes / 1e6);
    timeit("stream_only", q6_bytes, 10, [&]{ k_stream<<<(n + 3) / 4, 128>>>(q6, n, k, d_sink); });
    timeit("v1_production", q6_bytes, 10, [&]{ k_v1<<<(n + 3) / 4, 128>>>(x, q6, out, n, k); });
    timeit("v1_8rows", q6_bytes, 10, [&]{ k_8rows<<<(n + 7) / 8, 256>>>(x, q6, out, n, k); });
    timeit("v1_2rowwarp", q6_bytes, 10, [&]{ k_2rowwarp<<<(n / 2 + 3) / 4, 128>>>(x, q6, out, n, k); });
    timeit("q4_aligned", q4_bytes, 10, [&]{ k_q4<<<(n + 3) / 4, 128>>>(x, q4, out, n, k); });
    // lm_head shape too
    const int nl = 248320; const size_t lb = (size_t)nl * (k / 256) * 210;
    void* d_lm; if (cudaMalloc(&d_lm, lb) == cudaSuccess) {
        cudaMemset(d_lm, 0x55, lb);
        const unsigned char* lmq = (const unsigned char*)d_lm;
        printf("lm_head n=%d: %.2f GiB\n", nl, lb / 1073741824.0);
        timeit("lm_stream", lb, 5, [&]{ k_stream<<<(nl + 3) / 4, 128>>>(lmq, nl, k, d_sink); });
        timeit("lm_v1", lb, 5, [&]{ k_v1<<<(nl + 3) / 4, 128>>>(x, lmq, out, nl, k); });
        timeit("lm_2rowwarp", lb, 5, [&]{ k_2rowwarp<<<(nl / 2 + 3) / 4, 128>>>(x, lmq, out, nl, k); });
        cudaFree(d_lm);
    }
    CK(cudaGetLastError());
    return 0;
}
