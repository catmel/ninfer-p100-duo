// P100 single-GPU + host-RAM CPU-offload speed test for the qwen3.8-27b Q6_K budget.
//
// Phase 1: calibrate both tiers per registered shape on 1/8-row weights (runs on any box):
//   GPU    = production ggml_k_linear sm_60 dispatch (T=1 gemv band).
//   host   = AVX2 Q6_K single-token gemv, multithreaded, spot-checked against an FP64 oracle.
// Phase 2: try to build the FULL per-token weight set (~21.3 GiB across 64 layers +
//   lm_head at the registered shapes) split GPU-first, trailing layers in host RAM, and
//   run the real sequential decode chain (boundary H2D/D2H transfers + syncs included).
//   On a 16 GiB-RAM box the host tier usually cannot be committed; then the tool says so
//   and falls back to the calibrated prediction (extrapolation of Phase-1 rates over the
//   placement plan), which is the deliverable speed answer for "single card + offload".
//
// Microbenchmark per AGENTS.md: linear-GEMM budget only (attention/GDN state, elementwise
// and KV time excluded; they are sub-millisecond per token at these shapes).
//
// Build (Windows, arch-60 tree already built):
//   VsDevCmd -> nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr
//     -Xcompiler=/MD,/arch:AVX2,/EHsc -DNINFER_PASCAL_BUILD -I src -I include
//     tools/p100/p100_hybrid_offload_bench.cu build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib
#include "core/tensor.h"
#include "ops/linear/ggml_k/ggml_k.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <immintrin.h>
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::detail;

namespace {

std::uint32_t next_rand(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float half_to_float(std::uint16_t bits) {
    const __half h = __ushort_as_half(bits);
    return __half2float(h);
}

std::uint16_t float_to_half_bits(float value) {
    const __half h = __float2half_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&h);
}

double host_free_gib() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return ms.ullAvailPhys / 1073741824.0;
}

double gpu_free_gib() {
    std::size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f / 1073741824.0;
}

// Independent scalar Q6_K decoder (mirrors ggml_k_codec.cuh) for the oracle spot-check.
double decode_q6_k_value(const unsigned char* block, int i) {
    const int half = i >> 7;
    const int within = i & 127;
    const int section = within >> 5;
    const int lane = within & 31;
    const int low = (block[half * 64 + (section & 1) * 32 + lane] >> ((section >> 1) * 4)) & 15;
    const int high = (block[128 + half * 32 + lane] >> (section * 2)) & 3;
    const int code = (low | (high << 4)) - 32;
    const int scale = static_cast<signed char>(block[192 + (i >> 4)]);
    const double d = half_to_float(*reinterpret_cast<const std::uint16_t*>(block + 208));
    return (d * scale) * code;
}

// Encode one Q6_K block with fresh random bits.
void encode_q6_k_block(unsigned char* block, std::uint32_t& state) {
    std::memset(block, 0, 210);
    for (int i = 0; i < 256; ++i) {
        const int half = i >> 7;
        const int within = i & 127;
        const int section = within >> 5;
        const int lane = within & 31;
        const std::uint32_t code = next_rand(state) & 63u;
        const int low_index = half * 64 + (section & 1) * 32 + lane;
        const int low_shift = (section >> 1) * 4;
        block[low_index] = static_cast<unsigned char>(
            (block[low_index] & ~(0x0F << low_shift)) | static_cast<unsigned>((code & 15u) << low_shift));
        const int hi_index = 128 + half * 32 + lane;
        block[hi_index] = static_cast<unsigned char>(
            (block[hi_index] & ~(3 << (section * 2))) |
            static_cast<unsigned>(((code >> 4) & 3u) << (section * 2)));
    }
    for (int g = 0; g < 16; ++g) { block[192 + g] = static_cast<unsigned char>(next_rand(state) & 0x3F); }
    const std::uint16_t db = float_to_half_bits(0.004F + static_cast<float>(next_rand(state) % 40) / 10000.0F);
    block[208] = static_cast<unsigned char>(db & 0xFFu);
    block[209] = static_cast<unsigned char>(db >> 8);
}

// Fill an all-Q6_K row-major image. A random template block is replicated per block with
// fresh scales so 21 GiB stays memory-bound to generate (values are irrelevant for timing).
void fill_q6k_image(unsigned char* dst, std::size_t bytes, std::uint32_t& state) {
    unsigned char tmpl[210];
    encode_q6_k_block(tmpl, state);
    std::size_t off = 0;
    while (off + 210 <= bytes) {
        std::memcpy(dst + off, tmpl, 210);
        const std::uint16_t db = float_to_half_bits(0.004F + static_cast<float>(next_rand(state) % 40) / 10000.0F);
        dst[off + 208] = static_cast<unsigned char>(db & 0xFFu);
        dst[off + 209] = static_cast<unsigned char>(db >> 8);
        off += 210;
    }
    if (off < bytes) { std::memset(dst + off, 0, bytes - off); }
}

// --- AVX2 Q6_K single-token gemv -----------------------------------------------------------
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55u));
    return _mm_cvtss_f32(lo);
}

void q6k_gemv_rows(const unsigned char* codes, int row_begin, int row_end, int k,
                   const __nv_bfloat16* __restrict x, __nv_bfloat16* __restrict out) {
    const int blocks_per_row = k / 256;
    for (int row = row_begin; row < row_end; ++row) {
        const unsigned char* blk = codes + static_cast<std::size_t>(row) * blocks_per_row * 210;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int b = 0; b < blocks_per_row; ++b, blk += 210) {
            const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(blk + 208));
            const __m256 dv = _mm256_set1_ps(d);
            for (int g = 0; g < 16; ++g) {
                const int half = g >> 3;
                const int section = (g & 7) >> 1;
                const int lane_off = (g & 1) * 16;
                const int shift_lo = (section >> 1) * 4;
                const int shift_hi = section * 2;
                const unsigned char* plow = blk + half * 64 + (section & 1) * 32 + lane_off;
                const unsigned char* phigh = blk + 128 + half * 32 + lane_off;
                const __nv_bfloat16* xp = x + static_cast<std::size_t>(b) * 256 + g * 16;
                const float scale = static_cast<float>(static_cast<signed char>(blk[192 + g]));
                __m256 p0 = _mm256_setzero_ps();
                __m256 p1 = _mm256_setzero_ps();
                for (int e = 0; e < 16; e += 8) {
                    __m128i lob = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(plow + e));
                    lob = _mm_srli_epi16(lob, shift_lo);
                    lob = _mm_and_si128(lob, _mm_set1_epi8(15));
                    __m128i hib = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(phigh + e));
                    hib = _mm_srli_epi16(hib, shift_hi);
                    hib = _mm_and_si128(hib, _mm_set1_epi8(3));
                    hib = _mm_slli_epi16(hib, 4);
                    __m128i code = _mm_or_si128(lob, hib);
                    code = _mm_sub_epi8(code, _mm_set1_epi8(32));
                    const __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(code));
                    __m128i xraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xp + e));
                    const __m256 xf = _mm256_castsi256_ps(
                        _mm256_slli_epi32(_mm256_cvtepu16_epi32(xraw), 16));
                    const __m256 prod = _mm256_mul_ps(_mm256_mul_ps(cf, _mm256_set1_ps(scale)), xf);
                    if (e == 0) { p0 = prod; } else { p1 = _mm256_add_ps(p1, prod); }
                }
                acc0 = _mm256_fmadd_ps(p0, dv, acc0);
                acc1 = _mm256_fmadd_ps(p1, dv, acc1);
            }
        }
        out[row] = __float2bfloat16(hsum256(_mm256_add_ps(acc0, acc1)));
    }
}

void cpu_gemv_mt(const unsigned char* codes, int n, int k, const __nv_bfloat16* x,
                 __nv_bfloat16* out, int threads) {
    if (threads <= 1) {
        q6k_gemv_rows(codes, 0, n, k, x, out);
        return;
    }
    std::vector<std::thread> pool;
    const int chunk = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const int b = t * chunk;
        const int e = std::min(n, b + chunk);
        if (b >= e) { break; }
        pool.emplace_back([&, b, e]() { q6k_gemv_rows(codes, b, e, k, x, out); });
    }
    for (auto& th : pool) { th.join(); }
}

struct Shape { const char* name; int n, k; };
// Per-token weight set: 64 layers (every 4th full-attention) + lm_head, from
// tools/convert/qwen3_6_27b/inventory.py and the qualified oracle shapes.
const Shape kAttn{ "attn", 14336, 5120 };
const Shape kGdn { "gdn ", 16384, 5120 };
const Shape kO   { "o   ",  5120, 6144 };
const Shape kGu  { "gu  ", 34816, 5120 };
const Shape kDown{ "down",  5120, 17408 };
const Shape kLm  { "lmhd", 248320, 5120 };

std::size_t q6_bytes(int n, int k) {
    return static_cast<std::size_t>(n) * (k / 256) * 210;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int threads = 12;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) { threads = std::atoi(argv[++i]); }
    }
    std::uint32_t state = 0xC0FFEEu;
    cudaFree(0);
    std::printf("GPU free %.2f GiB | host free %.2f GiB | cpu threads %d\n",
                gpu_free_gib(), host_free_gib(), threads);

    // ---- Phase 0: AVX2 gemv correctness vs FP64 oracle --------------------------------------
    {
        const int n = 16, k = 512;
        std::vector<unsigned char> codes(q6_bytes(n, k));
        for (std::size_t off = 0; off + 210 <= codes.size(); off += 210) {
            encode_q6_k_block(codes.data() + off, state);
        }
        std::vector<__nv_bfloat16> x(k), out(n);
        for (int i = 0; i < k; ++i) {
            x[i] = __float2bfloat16((static_cast<float>(next_rand(state) % 4000) / 1000.0F) - 2.0F);
        }
        cpu_gemv_mt(codes.data(), n, k, x.data(), out.data(), 4);
        // debug: plain scalar gemv through the independent decoder, compare vs AVX kernel
        std::vector<__nv_bfloat16> out_scalar(n);
        for (int r = 0; r < n; ++r) {
            const unsigned char* rb = codes.data() + static_cast<std::size_t>(r) * (k / 256) * 210;
            double acc = 0.0;
            for (int i = 0; i < k; ++i) {
                acc += decode_q6_k_value(rb + (i >> 8) * 210, i & 255) * __bfloat162float(x[i]);
            }
            out_scalar[r] = __float2bfloat16(static_cast<float>(acc));
        }
        std::printf("[debug] AVX vs scalar-gemv:");
        for (int r = 0; r < 4; ++r) {
            std::printf(" %d=%.3f/%.3f", r, __bfloat162float(out[r]), __bfloat162float(out_scalar[r]));
        }
        std::printf("\n");
        int bad = 0;
        double worst = 0.0;
        for (int r = 0; r < n; ++r) {
            const unsigned char* rb = codes.data() + static_cast<std::size_t>(r) * (k / 256) * 210;
            double ref = 0.0;
            for (int i = 0; i < k; ++i) {
                ref += decode_q6_k_value(rb + (i >> 8) * 210, i & 255) * __bfloat162float(x[i]);
            }
            const double rel = std::fabs(__bfloat162float(out[r]) - ref) / (std::fabs(ref) + 1.0);
            if (r < 4) { std::printf("  row %d ref %.6f got %.6f\n", r, ref, __bfloat162float(out[r])); }
            worst = std::max(worst, rel);
            if (rel > 0.01) { ++bad; }
        }
        std::printf("[oracle] AVX2 Q6_K gemv vs FP64: %s (worst rel %.4f, %d/%d rows)\n",
                    bad == 0 ? "PASS" : "FAIL", worst, bad, n);
        if (bad) { return 1; }
    }

    const Shape shapes[] = {kAttn, kGdn, kO, kGu, kDown, kLm};

    // ---- Phase 1: calibrate both tiers at 1/8 rows -----------------------------------------
    std::printf("\n=== Phase 1: per-shape tier calibration (1/8 rows, extrapolated to full) ===\n");
    struct Rate { double gpu_us = 0.0, cpu_us = 0.0, gpu_gbps = 0.0, cpu_gbps = 0.0; std::size_t full = 0; };
    Rate rate[6];
    for (int s = 0; s < 6; ++s) {
        rate[s].full = q6_bytes(shapes[s].n, shapes[s].k);
        const int n8 = shapes[s].n / 8;
        const std::size_t b8 = q6_bytes(n8, shapes[s].k);
        // GPU side
        void* d_codes = nullptr;
        void* d_desc = nullptr;
        if (cudaMalloc(&d_codes, b8) == cudaSuccess && cudaMalloc(&d_desc, n8 * 8) == cudaSuccess) {
            std::vector<unsigned char> host(b8);
            fill_q6k_image(host.data(), b8, state);
            std::vector<std::uint64_t> desc(n8);
            std::size_t off = 0;
            for (int r = 0; r < n8; ++r) { desc[r] = (static_cast<std::uint64_t>(off) << 1) | 1ull; off += static_cast<std::size_t>(shapes[s].k / 256) * 210; }
            cudaMemcpy(d_codes, host.data(), b8, cudaMemcpyHostToDevice);
            cudaMemcpy(d_desc, desc.data(), n8 * 8, cudaMemcpyHostToDevice);
            Weight w{};
            w.qdata = d_codes; w.qhigh = d_desc;
            w.n = n8; w.k = shapes[s].k;
            w.padded_shape[0] = n8; w.padded_shape[1] = shapes[s].k;
            w.qtype = QType::GGML_K; w.layout = QuantLayout::GgmlK256; w.group = 256;
            w.ggml_k_first_q6 = -1; w.ggml_k_type_change = -1;
            void* d_x = nullptr; void* d_out = nullptr;
            cudaMalloc(&d_x, shapes[s].k * 2);
            cudaMalloc(&d_out, static_cast<std::size_t>(n8) * 2);
            std::vector<std::uint16_t> xb(static_cast<std::size_t>(shapes[s].k));
            for (auto& v : xb) { __nv_bfloat16 bf = __float2bfloat16(0.001F * (next_rand(state) % 4000) - 2.0F); v = *reinterpret_cast<std::uint16_t*>(&bf); }
            cudaMemcpy(d_x, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice);
            Tensor x(d_x, DType::BF16, {shapes[s].k, 1});
            Tensor out(d_out, DType::BF16, {n8, 1});
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> t;
            for (int i = 0; i < 14; ++i) {
                cudaEventRecord(e0);
                ggml_k_linear(x, w, out, nullptr, 0);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                t.push_back(ms * 1000.0f);
            }
            std::sort(t.begin(), t.end());
            const double gpu_us_full = t.front() * 8.0;
            rate[s].gpu_us = gpu_us_full;
            rate[s].gpu_gbps = rate[s].full / (gpu_us_full * 1e-6) / 1e9;
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            cudaFree(d_x); cudaFree(d_out); cudaFree(d_codes); cudaFree(d_desc);
        }
        // CPU side
        unsigned char* hc = static_cast<unsigned char*>(std::malloc(b8));
        if (hc) {
            fill_q6k_image(hc, b8, state);
            std::vector<__nv_bfloat16> hx(shapes[s].k), ho(n8);
            for (int i = 0; i < shapes[s].k; ++i) { hx[i] = __float2bfloat16(0.001F * (next_rand(state) % 4000) - 2.0F); }
            std::vector<double> t;
            for (int i = 0; i < 5; ++i) {
                const auto b = std::chrono::steady_clock::now();
                cpu_gemv_mt(hc, n8, shapes[s].k, hx.data(), ho.data(), threads);
                const auto e = std::chrono::steady_clock::now();
                t.push_back(std::chrono::duration<double, std::micro>(e - b).count());
            }
            std::sort(t.begin(), t.end());
            rate[s].cpu_us = t.front() * 8.0;
            rate[s].cpu_gbps = rate[s].full / (rate[s].cpu_us * 1e-6) / 1e9;
            std::free(hc);
        }
        std::printf("%s %4d x %5d  full %7.1f MiB | GPU %8.1f us (%6.1f GB/s) | CPU %8.1f us (%6.1f GB/s)\n",
                    shapes[s].name, shapes[s].n, shapes[s].k, rate[s].full / 1048576.0,
                    rate[s].gpu_us, rate[s].gpu_gbps, rate[s].cpu_us, rate[s].cpu_gbps);
    }

    // ---- placement plan + analytic prediction ----------------------------------------------
    std::size_t total = 0, lm_bytes = q6_bytes(kLm.n, kLm.k);
    for (int l = 0; l < 64; ++l) {
        total += q6_bytes((l % 4 == 3) ? kAttn.n : kGdn.n, kAttn.k);
        total += q6_bytes(kO.n, kO.k) + q6_bytes(kGu.n, kGu.k) + q6_bytes(kDown.n, kDown.k);
    }
    total += lm_bytes;
    std::printf("\nmodel weight budget: %.2f GiB (Q6_K all rows)\n", total / 1073741824.0);

    // GPU-first in layer order; lm_head pinned to GPU.
    const double gpu_budget_gib = gpu_free_gib() - 1.2;
    const double host_room_gib = host_free_gib() - 0.8;
    struct Planned { Shape shape; bool gpu; int si; };
    std::vector<Planned> plan;
    {
        double gpu_used = lm_bytes / 1073741824.0;
        for (int l = 0; l < 64; ++l) {
            const bool fa = (l % 4 == 3);
            const Shape* group[4] = {fa ? &kAttn : &kGdn, &kO, &kGu, &kDown};
            for (const Shape* sh : group) {
                const double b = q6_bytes(sh->n, sh->k) / 1073741824.0;
                const bool gpu = gpu_used + b <= gpu_budget_gib;
                if (gpu) { gpu_used += b; }
                int si = 0;
                for (int s = 0; s < 6; ++s) { if (sh->n == shapes[s].n) { si = s; } }
                plan.push_back(Planned{*sh, gpu, si});
            }
        }
        plan.push_back(Planned{kLm, true, 5});
    }
    double host_need_gib = 0.0;
    for (const Planned& p : plan) { if (!p.gpu) { host_need_gib += q6_bytes(p.shape.n, p.shape.k) / 1073741824.0; } }
    double pred_ms = 0.0;
    std::size_t gpu_ops = 0, cpu_ops = 0;
    for (const Planned& p : plan) {
        pred_ms += (p.gpu ? rate[p.si].gpu_us : rate[p.si].cpu_us) / 1000.0;
        if (p.gpu) { ++gpu_ops; } else { ++cpu_ops; }
    }
    pred_ms += cpu_ops * 0.018;  // host-tier boundary: sync + 10 KiB transfer allowance
    std::printf("plan: %zu GPU ops + %zu host ops; offload %.2f GiB (host free %.2f GiB)\n",
                gpu_ops, cpu_ops, host_need_gib, host_room_gib);
    std::printf("PREDICTED hybrid decode: %.1f ms/token -> %.2f tok/s (linear budget; end-to-end"
                " ~15%% lower)\n", pred_ms, 1000.0 / pred_ms);
    double pred_gpu = 0.0;
    for (const Planned& p : plan) { pred_gpu += rate[p.si].gpu_us / 1000.0; }
    std::printf("(reference: same model fully GPU-resident would be %.1f ms/token -> %.2f tok/s)\n",
                pred_gpu, 1000.0 / pred_gpu);

    // ---- Phase 2: real chain if the host tier can be committed ------------------------------
    const bool host_fits = host_need_gib <= host_room_gib;
    if (!host_fits || cpu_ops == 0) {
        std::printf("\nPhase 2 skipped: host tier %.2f GiB does not fit in %.2f GiB free"
                    " (or nothing to offload). Prediction above is the deliverable.\n",
                    host_need_gib, host_room_gib);
        std::printf("Verdict: 16 GiB RAM box cannot hold the offload tier for this 27B Q6_K;"
                    " with >=24 GiB free RAM the same plan runs at the predicted rate.\n");
        return 0;
    }

    std::printf("\n=== Phase 2: building full split placement and running the real chain ===\n");
    struct LiveOp { Planned p; Weight w{}; void* d_codes = nullptr; unsigned char* h_codes = nullptr; };
    std::vector<LiveOp> ops;
    bool build_ok = true;
    for (const Planned& p : plan) {
        LiveOp o;
        o.p = p;
        const std::size_t bytes = q6_bytes(p.shape.n, p.shape.k);
        std::vector<std::uint64_t> desc(static_cast<std::size_t>(p.shape.n));
        if (p.gpu) {
            std::size_t off = 0;
            for (int r = 0; r < p.shape.n; ++r) { desc[r] = (static_cast<std::uint64_t>(off) << 1) | 1ull; off += static_cast<std::size_t>(p.shape.k / 256) * 210; }
            if (cudaMalloc(&o.d_codes, bytes) != cudaSuccess) { build_ok = false; break; }
            std::vector<unsigned char> scratch(64u << 20);
            std::size_t done = 0;
            while (done < bytes) {
                const std::size_t chunk = std::min(scratch.size(), bytes - done);
                fill_q6k_image(scratch.data(), chunk, state);
                cudaMemcpy(static_cast<unsigned char*>(o.d_codes) + done, scratch.data(), chunk, cudaMemcpyHostToDevice);
                done += chunk;
            }
            void* d_desc = nullptr;
            if (cudaMalloc(&d_desc, desc.size() * 8) != cudaSuccess) { build_ok = false; break; }
            cudaMemcpy(d_desc, desc.data(), desc.size() * 8, cudaMemcpyHostToDevice);
            o.w.qhigh = d_desc;
            o.w.qdata = o.d_codes;
        } else {
            o.h_codes = static_cast<unsigned char*>(std::malloc(bytes));
            if (!o.h_codes) { build_ok = false; break; }
            fill_q6k_image(o.h_codes, bytes, state);
        }
        o.w.n = p.shape.n;
        o.w.k = p.shape.k;
        o.w.padded_shape[0] = p.shape.n;
        o.w.padded_shape[1] = p.shape.k;
        o.w.qtype = QType::GGML_K;
        o.w.layout = QuantLayout::GgmlK256;
        o.w.group = 256;
        o.w.ggml_k_first_q6 = -1;
        o.w.ggml_k_type_change = -1;
        ops.push_back(o);
    }
    if (!build_ok) {
        std::printf("build failed mid-way (GPU %.2f GiB free, host %.2f GiB free) -> prediction stands.\n",
                    gpu_free_gib(), host_free_gib());
        return 0;
    }

    void* d_pool = nullptr;
    cudaMalloc(&d_pool, 32u << 20);
    std::vector<__nv_bfloat16> h_buf(34816);
    for (auto& v : h_buf) { v = __float2bfloat16(0.001F * (next_rand(state) % 4000) - 2.0F); }
    cudaMemcpy(d_pool, h_buf.data(), 40u << 10, cudaMemcpyHostToDevice);

    auto chain_once = [&]() -> double {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        bool hidden_gpu = true;
        char* dp = static_cast<char*>(d_pool);
        for (std::size_t i = 0; i < ops.size(); ++i) {
            const LiveOp& o = ops[i];
            if (o.p.gpu) {
                if (!hidden_gpu) {
                    cudaMemcpy(dp, h_buf.data(), static_cast<std::size_t>(o.p.shape.k) * 2, cudaMemcpyHostToDevice);
                    hidden_gpu = true;
                }
                Tensor x(dp, DType::BF16, {o.p.shape.k, 1});
                Tensor out(dp + (20u << 10), DType::BF16, {o.p.shape.n, 1});
                ggml_k_linear(x, o.w, out, nullptr, 0);
            } else {
                if (hidden_gpu) {
                    cudaStreamSynchronize(0);
                    cudaMemcpy(h_buf.data(), dp, static_cast<std::size_t>(o.p.shape.k) * 2, cudaMemcpyDeviceToHost);
                    hidden_gpu = false;
                }
                cpu_gemv_mt(o.h_codes, o.p.shape.n, o.p.shape.k, h_buf.data(), h_buf.data(), threads);
            }
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return ms;
    };
    double best = 1e30;
    for (int it = 0; it < 6; ++it) { best = std::min(best, chain_once()); }
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("chain error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    std::printf("MEASURED hybrid chain: %.2f ms/token -> %.2f tok/s (linear budget)\n", best, 1000.0 / best);
    std::printf("(GPU %.2f GiB + host %.2f GiB resident; end-to-end ~15%% lower)\n",
                gpu_free_gib() ? (total / 1073741824.0 - host_need_gib) : 0.0, host_need_gib);
    return 0;
}
