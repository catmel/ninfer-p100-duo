// P100 speed microbenchmark on the production GGML-K linear route at the real
// qwen3.8-27b per-layer shapes (16 full-attention layers every 4th, 48 GDN layers,
// 64 layers total; shapes from tools/convert/qwen3_6_27b/inventory.py and the oracle
// suite). Times the sm_60 route selection end to end (gemv for small T, HFMA2 tile for
// wide T), then extrapolates decode tokens/s per resident weight budget. Per AGENTS.md
// this is an Op microbenchmark: it establishes the linear-route cost model, not an
// end-to-end token rate (attention/state/elementwise/KV time are not included).
//
// Build (Windows, arch-60 tree already built):
//   VsDevCmd -> nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -Xcompiler=/MD
//     -DNINFER_PASCAL_BUILD -I src -I include tools/p100/p100_speed_bench.cu
//     build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib
#include "core/tensor.h"
#include "ops/linear/ggml_k/ggml_k.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::uint16_t float_to_half_bits(float value) {
    const __half h = __float2half_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&h);
}

// Fill one Q6_K 256-weight block with random valid bits (210 bytes). Values need not be
// small: timing is data-independent for these kernels.
void fill_q6_k_block(unsigned char* block, std::uint32_t& state) {
    const std::uint16_t dbits = float_to_half_bits(0.004F + static_cast<float>(next_rand(state) % 60) / 10000.0F);
    block[208] = static_cast<unsigned char>(dbits & 0xFFu);
    block[209] = static_cast<unsigned char>(dbits >> 8);
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
            (block[hi_index] & ~(3 << (section * 2))) | static_cast<unsigned>(((code >> 4) & 3u) << (section * 2)));
    }
    for (int g = 0; g < 16; ++g) { block[192 + g] = static_cast<unsigned char>(next_rand(state) & 0x3F); }
}

void fill_q4_k_block(unsigned char* block, std::uint32_t& state) {
    const std::uint16_t dbits = float_to_half_bits(0.004F + static_cast<float>(next_rand(state) % 60) / 10000.0F);
    const std::uint16_t mbits = float_to_half_bits(0.002F + static_cast<float>(next_rand(state) % 40) / 10000.0F);
    block[0] = static_cast<unsigned char>(dbits & 0xFFu);
    block[1] = static_cast<unsigned char>(dbits >> 8);
    block[2] = static_cast<unsigned char>(mbits & 0xFFu);
    block[3] = static_cast<unsigned char>(mbits >> 8);
    for (int g = 0; g < 8; ++g) {
        const std::uint32_t s = next_rand(state) & 63u;
        const std::uint32_t m = next_rand(state) & 63u;
        if (g < 4) {
            block[4 + g] = static_cast<unsigned char>(s | ((next_rand(state) & 0x30u) << 2));
            block[8 + g] = static_cast<unsigned char>(m | ((next_rand(state) & 0x30u) << 2));
            block[12 + g] = 0;
        } else {
            block[4 + g] = static_cast<unsigned char>(s);
            block[8 + g] = static_cast<unsigned char>(m);
            block[12 + g] = static_cast<unsigned char>((s & 0x0Fu) | ((m & 0x0Fu) << 4));
        }
    }
    std::memset(block + 16, 0, 32);
    for (int i = 0; i < 256; ++i) {
        const int group = i >> 5;
        const std::uint32_t code = next_rand(state) & 15u;
        const int byte = 16 + (i >> 6) * 32 + (i & 31);
        const int shift = (group & 1) * 4;
        block[byte] = static_cast<unsigned char>((block[byte] & ~(0x0Fu << shift)) | static_cast<unsigned>(code << shift));
    }
}

enum Kind { KindQ6, KindMixed };  // all-Q6_K rows (the downloaded artifact) vs Q4/Q6 halves

struct Built {
    void* codes = nullptr;
    void* desc = nullptr;
    std::size_t weight_bytes = 0;
    Weight w{};
};

Built build_weight(int n, int k, Kind kind, std::uint32_t& state) {
    Built b;
    const int blocks_per_row = k / 256;
    std::vector<int> row_type(n, 0);
    for (int row = 0; row < n; ++row) {
        row_type[row] = (kind == KindQ6 || row >= n / 2) ? 1 : 0;
    }
    std::size_t total = 0;
    for (int row = 0; row < n; ++row) { total += static_cast<std::size_t>(blocks_per_row) * (row_type[row] ? 210 : 144); }
    std::vector<unsigned char> codes(total);
    std::vector<std::uint64_t> descriptors(n);
    std::size_t offset = 0;
    for (int row = 0; row < n; ++row) {
        descriptors[row] = (static_cast<std::uint64_t>(offset) << 1) | static_cast<std::uint64_t>(row_type[row]);
        unsigned char* row_base = codes.data() + offset;
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            unsigned char* block = row_base + blk * (row_type[row] ? 210 : 144);
            if (row_type[row]) { fill_q6_k_block(block, state); } else { fill_q4_k_block(block, state); }
        }
        offset += static_cast<std::size_t>(blocks_per_row) * (row_type[row] ? 210 : 144);
    }
    b.weight_bytes = total;
    cudaMalloc(&b.codes, total);
    cudaMemcpy(b.codes, codes.data(), total, cudaMemcpyHostToDevice);
    cudaMalloc(&b.desc, descriptors.size() * 8);
    cudaMemcpy(b.desc, descriptors.data(), descriptors.size() * 8, cudaMemcpyHostToDevice);
    b.w.qdata = b.codes;
    b.w.qhigh = b.desc;
    b.w.n = n;
    b.w.k = k;
    b.w.padded_shape[0] = n;
    b.w.padded_shape[1] = k;
    b.w.qtype = QType::GGML_K;
    b.w.layout = QuantLayout::GgmlK256;
    b.w.group = 256;
    b.w.ggml_k_first_q6 = -1;
    b.w.ggml_k_type_change = -1;
    return b;
}

struct Result {
    double min_us = 0.0;
    double med_us = 0.0;
    double w_gbps = 0.0;
};

// Runs the production route at (n,k,tokens). Returns min/median over reps.
Result bench(Built& b, int n, int k, int tokens, int reps) {
    Result r;
    void* d_x = nullptr;
    void* d_out = nullptr;
    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * tokens);
    std::uint32_t state = 0x9E3779B9u ^ static_cast<std::uint32_t>(tokens * 7919 + n);
    for (auto& v : x_bits) {
        const float f = (static_cast<float>(next_rand(state) % 4000) / 1000.0F) - 2.0F;  // LayerNorm-ish ±2
        const __half h = __float2half_rn(f);
        const __nv_bfloat16 bfb = __float2bfloat16(__half2float(h));
        __nv_bfloat16 bf = bfb;
        v = *reinterpret_cast<std::uint16_t*>(&bf);
    }
    if (cudaMalloc(&d_x, x_bits.size() * 2) != cudaSuccess ||
        cudaMalloc(&d_out, static_cast<std::size_t>(n) * tokens * 2) != cudaSuccess) {
        cudaFree(d_x); cudaFree(d_out);
        r.min_us = -1.0;
        return r;
    }
    cudaMemcpy(d_x, x_bits.data(), x_bits.size() * 2, cudaMemcpyHostToDevice);
    Tensor x(d_x, DType::BF16, {k, tokens});
    Tensor out(d_out, DType::BF16, {n, tokens});
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    std::vector<float> times;
    for (int it = 0; it < reps + 2; ++it) {
        cudaEventRecord(ev0);
        ggml_k_linear(x, b.w, out, nullptr, 0);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms = 0.0F;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (it >= 2) { times.push_back(ms * 1000.0F); }
        if (cudaGetLastError() != cudaSuccess) {
            std::printf("  !! launch error at n=%d k=%d T=%d: %s\n", n, k, tokens, cudaGetErrorString(cudaGetLastError()));
            r.min_us = -1.0;
            return r;
        }
    }
    std::sort(times.begin(), times.end());
    r.min_us = times.front();
    r.med_us = times[times.size() / 2];
    r.w_gbps = (static_cast<double>(b.weight_bytes) / (r.min_us * 1e-6)) / 1e9;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_x);
    cudaFree(d_out);
    return r;
}

void free_weight(Built& b) {
    cudaFree(b.codes);
    cudaFree(b.desc);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool want_mixed = argc > 1 && std::strcmp(argv[1], "--mixed") == 0;
    Kind kind = want_mixed ? KindMixed : KindQ6;
    const char* kind_name = want_mixed ? "Q4_K/Q6_K mixed" : "Q6_K all rows";

    int dev = 0;
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, dev) != cudaSuccess) { std::printf("no device\n"); return 1; }
    std::size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::printf("device: %s cc %d.%d  free %.2f GiB  [route: ggml_k_linear production dispatch, %s]\n",
                props.name, props.major, props.minor, free_b / 1073741824.0, kind_name);

    struct Shape {
        const char* name;
        int n, k, layers;
        bool lm_head;
    };
    const Shape shapes[] = {
        {"attn_input   14336x5120 ", 14336, 5120, 16, false},
        {"gdn_input    16384x5120 ", 16384, 5120, 48, false},
        {"o_proj        5120x6144 ", 5120, 6144, 64, false},
        {"gate_up      34816x5120 ", 34816, 5120, 64, false},
        {"down         5120x17408 ", 5120, 17408, 64, false},
        {"lm_head     248320x5120 ", 248320, 5120, 1, true},
    };
    const int decode_ts[] = {1, 2, 4};
    const int prefill_ts[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048};

    std::uint32_t state = 0x12345678u;
    double decode_total_us = 0.0;  // T=1 per-token sum over layers + lm_head
    std::vector<double> prefill_us(std::size(prefill_ts), 0.0);

    std::printf("\n=== decode-band shapes (us/call, min over 15 reps) ===\n");
    std::printf("%-24s %8s %8s %8s %10s\n", "shape", "T=1", "T=2", "T=4", "GB/s@T1");
    for (const Shape& s : shapes) {
        Built b = build_weight(s.n, s.k, kind, state);
        double row_us[3] = {0.0, 0.0, 0.0};
        double gbps = 0.0;
        for (int i = 0; i < 3; ++i) {
            const Result r = bench(b, s.n, s.k, decode_ts[i], 15);
            row_us[i] = r.min_us;
            if (i == 0) { gbps = r.w_gbps; }
            if (r.min_us < 0.0) { std::printf("%-24s SKIP(OOM/err)\n", s.name); break; }
        }
        std::printf("%-24s %8.1f %8.1f %8.1f %10.1f\n", s.name, row_us[0], row_us[1], row_us[2], gbps);
        if (row_us[0] > 0.0 && s.layers > 0) { decode_total_us += row_us[0] * s.layers; }
        if (s.lm_head) { decode_total_us += 0.0; }  // already counted via layers==1
        free_weight(b);
    }
    // Re-count lm_head precisely (single call, layers==1 already included above).
    std::printf("\nper-token linear-only estimate (T=1): %.2f ms -> %.1f tok/s weight-bound (linear GEMMs only)\n",
                decode_total_us / 1000.0, 1e6 / decode_total_us);

    std::printf("\n=== prefill-band shapes (us/call) ===\n");
    for (const int t : prefill_ts) {
        double sum = 0.0;
        std::vector<double> per;
        bool ok = true;
        for (const Shape& s : shapes) {
            if (s.lm_head && t > 4) { continue; }  // lm_head runs once on the last token in prefill
            Built b = build_weight(s.n, s.k, kind, state);
            const int reps = t >= 512 ? 7 : 11;
            const Result r = bench(b, s.n, s.k, t, reps);
            if (r.min_us < 0.0) { ok = false; free_weight(b); break; }
            per.push_back(r.min_us);
            sum += r.min_us * (s.lm_head ? 1 : s.layers);
            free_weight(b);
        }
        if (!ok) { std::printf("T=%-5d SKIP (OOM)\n", t); continue; }
        std::printf("T=%-5d  all-layer linear sum: %10.2f ms  -> %7.1f prefill tok/s (linear only)\n",
                    t, sum / 1000.0, t * 1e6 / sum);
        for (std::size_t i = 0; i < per.size(); ++i) { std::printf("      #%zu %.1f us\n", i, per[i]); }
    }

    std::printf("\n=== TP2 shard shapes (T=1) for dual-P100 extrapolation ===\n");
    const Shape shards[] = {
        {"shard attn   7168x5120  ", 7168, 5120, 16, false},
        {"shard gdn    8192x5120  ", 8192, 5120, 48, false},
        {"shard o_proj 5120x3072  ", 5120, 3072, 64, false},
        {"shard gateup 17408x5120 ", 17408, 5120, 64, false},
        {"shard down   5120x8704  ", 5120, 8704, 64, false},
        {"shard lm    124160x5120 ", 124160, 5120, 1, false},
    };
    double shard_total = 0.0;
    std::printf("%-24s %8s %10s\n", "shape", "T=1", "GB/s@T1");
    for (const Shape& s : shards) {
        Built b = build_weight(s.n, s.k, kind, state);
        const Result r = bench(b, s.n, s.k, 1, 15);
        if (r.min_us < 0.0) { std::printf("%-24s SKIP\n", s.name); free_weight(b); continue; }
        std::printf("%-24s %8.1f %10.1f\n", s.name, r.min_us, r.w_gbps);
        shard_total += r.min_us * (s.layers > 0 ? s.layers : 1);
        free_weight(b);
    }
    std::printf("\nTP2 per-card linear-only (T=1): %.2f ms -> %.1f tok/s (excludes allreduce/state)\n",
                shard_total / 1000.0, 1e6 / shard_total);
    return 0;
}
