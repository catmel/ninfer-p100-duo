// Qualification of the GGUF-native (QType::GGUF) Q6_K vector route against an independent FP64
// oracle, at the shapes and dispatch buckets the sm_60 decode relies on: wide/tall/narrow row
// counts, the 1/4/8-column token buckets, the tiled GDN input column order, multi-segment weights,
// and split output sections. allow_a8 is passed true exactly like the deployed bindings; the sm_60
// build must keep the FP32-exact A16 route, so an int8-activation regression would bust the budget.
//
// Build (local sm_60) links ninfer_ops.lib, which already contains the patched gguf.cu:
//   nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -Xcompiler=/MD -DNINFER_PASCAL_BUILD \
//        -I src -I include tools/p100/oracle_gguf_vector_check.cu \
//        build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib -o build-p100/oracle_gguf.exe
#include "core/tensor.h"
#include "ops/linear/gguf/gguf.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace ninfer;
using ninfer::ops::detail::gguf_project;

#define CK(call)                                                                                   \
    do {                                                                                           \
        const cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("cuda error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,            \
                        __LINE__);                                                                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

static std::uint32_t next_rand(std::uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static float f16_bits_to_float(std::uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

static std::uint16_t float_to_f16_bits(float f) {
    const __half h = __float2half_rn(f);
    return *reinterpret_cast<const std::uint16_t*>(&h);
}

// Encodes one llama-layout Q6_K block (210 bytes, 256 values) with random codes, and returns the
// exact double values it represents: w[i] = d * s8[i / 16] * (code[i] - 32). The packing mirrors
// the decode8<GgufType::Q6_K> contract: low nibbles at half*64 + (section&1)*32 + ln (two sections
// per byte, 4-bit apart), high 2 bits at 128 + half*32 + ln (four sections per byte, 2-bit apart).
static void encode_q6_k_block(unsigned char* block, std::uint32_t& state, double* decoded) {
    const std::uint16_t d_bits = float_to_f16_bits(0.008F + static_cast<float>(next_rand(state) % 24u) * 0.001F);
    const float d = f16_bits_to_float(d_bits);
    std::memset(block, 0, 210);
    signed char s8[16];
    for (int g = 0; g < 16; ++g) {
        s8[g] = static_cast<signed char>(next_rand(state) % 17u - 8);
        block[192 + g] = static_cast<unsigned char>(s8[g]);
    }
    block[208] = static_cast<unsigned char>(d_bits & 0xFFu);
    block[209] = static_cast<unsigned char>(d_bits >> 8);
    for (int i = 0; i < 256; ++i) {
        const unsigned code = next_rand(state) % 64u;
        const int half    = i >> 7;
        const int within  = i & 127;
        const int section = within >> 5;
        const int ln      = within & 31;
        const unsigned low  = code & 15U;
        const unsigned high = (code >> 4) & 3U;
        block[half * 64 + (section & 1) * 32 + ln] =
            static_cast<unsigned char>(block[half * 64 + (section & 1) * 32 + ln] | (low << ((section >> 1) * 4)));
        block[128 + half * 32 + ln] =
            static_cast<unsigned char>(block[128 + half * 32 + ln] | (high << (section * 2)));
        decoded[i] = static_cast<double>(d) * static_cast<double>(static_cast<int>(s8[i / 16]) - 0) *
                     static_cast<double>(static_cast<int>(code) - 32);
    }
}

// Encodes one llama-layout Q5_K block (176 bytes, 256 values): half d, half dmin,
// u8 scales[12] (Q4_K's 6-bit scale/min pairs), u8 qh[32] (5th bit, bit index = 32-value
// sub-block), u8 qs[128] (low nibbles, sub-block pairs sharing 32-byte chunks).
static void encode_q5_k_block(unsigned char* block, std::uint32_t& state, double* decoded) {
    const std::uint16_t d_bits = float_to_f16_bits(0.008F + static_cast<float>(next_rand(state) % 24u) * 0.001F);
    const std::uint16_t dm_bits = float_to_f16_bits(0.002F + static_cast<float>(next_rand(state) % 9u) * 0.0005F);
    const double d = f16_bits_to_float(d_bits);
    const double dmin = f16_bits_to_float(dm_bits);
    std::memset(block, 0, 176);
    block[0] = static_cast<unsigned char>(d_bits & 0xFFu);
    block[1] = static_cast<unsigned char>(d_bits >> 8);
    block[2] = static_cast<unsigned char>(dm_bits & 0xFFu);
    block[3] = static_cast<unsigned char>(dm_bits >> 8);
    unsigned sc6[8];
    unsigned mn6[8];
    for (int g = 0; g < 8; ++g) {
        sc6[g] = next_rand(state) % 64u;
        mn6[g] = next_rand(state) % 64u;
    }
    for (int g = 0; g < 4; ++g) {
        block[4 + g]  = static_cast<unsigned char>(sc6[g] | ((sc6[g + 4] >> 4) << 6));
        block[8 + g]  = static_cast<unsigned char>(mn6[g] | ((mn6[g + 4] >> 4) << 6));
        block[12 + g] = static_cast<unsigned char>((sc6[g + 4] & 15u) | ((mn6[g + 4] & 15u) << 4));
    }
    for (int i = 0; i < 256; ++i) {
        const unsigned code = next_rand(state) % 32u;
        const int g = i >> 5;
        const int l = i & 31;
        block[48 + 32 * (g >> 1) + l] =
            static_cast<unsigned char>(block[48 + 32 * (g >> 1) + l] | ((code & 15U) << (4 * (g & 1))));
        if (code >> 4) { block[16 + l] = static_cast<unsigned char>(block[16 + l] | (1U << g)); }
        const double scale = d * static_cast<double>(sc6[g]);
        const double min   = dmin * static_cast<double>(mn6[g]);
        decoded[i] = scale * static_cast<double>(code) - min;
    }
}

// Tiled GDN value-head column order (mirrors the documented production mapping, re-derived here
// so the oracle is independent of the kernel source).
static int tiled_input_column(int column, int k) {
    const int head    = column >> 7;
    const int grouped = k == 3072 ? (head & 7) * 3 + (head >> 3) : (head & 15) * 3 + (head >> 4);
    return grouped * 128 + (column & 127);
}

// bf16 ulp band, same acceptance machinery as oracle_decode_check.cu.
static double bf16_ulp(double magnitude) {
    double exponent = 0.0;
    double m = magnitude;
    if (m <= 0.0) { m = 1e-6; }
    while (m >= 2.0) { m *= 0.5; exponent += 1.0; }
    while (m < 1.0) { m *= 2.0; exponent -= 1.0; }
    return std::pow(2.0, exponent - 7.0);
}
static double budget_for(double magnitude) {
    const double budget = 2.0 * bf16_ulp(magnitude);
    return budget < 1.0 ? 1.0 : budget;
}

struct GgufCase {
    const char* name;
    int n;
    int k;
    int tokens;
    bool tiled        = false;
    int seg1          = 0;  // first-segment rows for a two-segment weight; 0 = single segment
    int split_out     = 0;  // section-0 rows for a two-output split; 0 = single output
    GgufType type     = GgufType::Q6_K;
};

static bool run(const GgufCase& c) {
    std::uint32_t state = 0x243F6A88u ^ static_cast<std::uint32_t>(c.n * 131 + c.tokens);
    const int n = c.n, k = c.k, tokens = c.tokens;
    const int block_bytes = c.type == GgufType::Q5_K ? 176 : 210;

    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * tokens);
    std::vector<double> x_val(static_cast<std::size_t>(k) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < k; ++i) {
            const float v = (static_cast<float>(next_rand(state) % 4001u) / 1000.0F) - 2.0F;
            const __nv_bfloat16 b = __float2bfloat16_rn(v);
            x_bits[static_cast<std::size_t>(t) * k + i] = *reinterpret_cast<const std::uint16_t*>(&b);
            x_val[static_cast<std::size_t>(t) * k + i] = static_cast<double>(__bfloat162float(b));
        }
    }

    const int blocks_per_row = k / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * block_bytes;
    const int seg1_rows = c.seg1 != 0 ? c.seg1 : n;
    const int seg2_rows = n - seg1_rows;
    std::vector<unsigned char> host1(static_cast<std::size_t>(seg1_rows) * row_bytes);
    std::vector<unsigned char> host2(static_cast<std::size_t>(seg2_rows) * row_bytes);
    std::vector<double> reference(static_cast<std::size_t>(n) * tokens, 0.0);
    std::vector<double> decoded(256);
    for (int row = 0; row < n; ++row) {
        unsigned char* row_base = (row < seg1_rows ? host1.data() : host2.data()) +
                                  (row < seg1_rows ? static_cast<std::size_t>(row) * row_bytes
                                                   : static_cast<std::size_t>(row - seg1_rows) * row_bytes);
        for (int b = 0; b < blocks_per_row; ++b) {
            unsigned char* blk = row_base + static_cast<std::size_t>(b) * block_bytes;
            if (c.type == GgufType::Q5_K) { encode_q5_k_block(blk, state, decoded.data()); }
            else { encode_q6_k_block(blk, state, decoded.data()); }
            for (int i = 0; i < 256; ++i) {
                const int stored = b * 256 + i;
                const int src    = c.tiled ? tiled_input_column(stored, k) : stored;
                if (src < 0 || src >= k) {
                    std::printf("%-24s TILED MAPPING OUT OF RANGE col %d -> %d\n", c.name, stored, src);
                    return false;
                }
                for (int t = 0; t < tokens; ++t) {
                    reference[static_cast<std::size_t>(t) * n + row] +=
                        decoded[i] * x_val[static_cast<std::size_t>(t) * k + src];
                }
            }
        }
    }

    void *d_x = nullptr, *d_w1 = nullptr, *d_w2 = nullptr, *d_out = nullptr;
    if (getenv("ORACLE_TRACE")) { std::printf("  [pre-malloc] host1=%zu host2=%zu\n", host1.size(), host2.size()); std::fflush(stdout); }
    CK(cudaMalloc(&d_x, x_bits.size() * 2));
    CK(cudaMalloc(&d_w1, host1.size()));
    if (seg2_rows > 0) { CK(cudaMalloc(&d_w2, host2.size())); }
    CK(cudaMalloc(&d_out, static_cast<std::size_t>(n) * tokens * 2));
    if (getenv("ORACLE_TRACE")) { std::printf("  [post-malloc]\n"); std::fflush(stdout); }
    CK(cudaMemcpy(d_x, x_bits.data(), x_bits.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w1, host1.data(), host1.size(), cudaMemcpyHostToDevice));
    if (seg2_rows > 0) { CK(cudaMemcpy(d_w2, host2.data(), host2.size(), cudaMemcpyHostToDevice)); }
    if (getenv("ORACLE_TRACE")) { std::printf("  [post-memcpy]\n"); std::fflush(stdout); }

    Tensor x(d_x, DType::BF16, {k, tokens});
    Weight w{};
    w.qtype              = QType::GGUF;
    w.layout             = QuantLayout::GgufBlocks;
    w.n                  = n;
    w.k                  = k;
    w.group              = 256;
    w.padded_shape[0]    = n;
    w.padded_shape[1]    = k;
    w.gguf_segment_count = seg2_rows > 0 ? 2 : 1;
    w.gguf_segments[0]   = GgufSegment{d_w1, seg1_rows, c.type};
    if (seg2_rows > 0) { w.gguf_segments[1] = GgufSegment{d_w2, seg2_rows, c.type}; }

    Tensor out(d_out, DType::BF16, {n, tokens});
    const std::vector<Tensor> outs = c.split_out > 0
        ? std::vector<Tensor>{
              Tensor(d_out, DType::BF16, {c.split_out, tokens}),
              Tensor(static_cast<std::uint8_t*>(d_out) + static_cast<std::size_t>(c.split_out) * tokens * 2,
                     DType::BF16, {n - c.split_out, tokens})}
        : std::vector<Tensor>{out};

    // Sized workspace + allow_a8: exactly the deployed binding's contract. On sm_60 (this build)
    // project() ignores the workspace and takes the FP32-exact A16 route, so the ulp budget below
    // proves the override. We pass nullptr here to keep the qualification free of a large
    // tensor-core workspace allocation; allow_a8 is still true, so a route-selection regression
    // toward int8 would still bust the budget wherever the codepath exists.
    if (getenv("ORACLE_TRACE")) { std::printf("  [pre-project] seg=%d outs=%d k=%d T=%d tiled=%d\n", w.gguf_segment_count, (int)outs.size(), k, tokens, (int)c.tiled); std::fflush(stdout); }
    bool ok = false;
    try {
        gguf_project(x, w, outs.data(), static_cast<int>(outs.size()), /*add=*/false, c.tiled,
                     /*allow_a8=*/true, /*workspace=*/nullptr, /*stream=*/nullptr);
        if (getenv("ORACLE_TRACE")) { std::printf("  [post-project]\n"); std::fflush(stdout); }
        CK(cudaDeviceSynchronize());
        if (getenv("ORACLE_TRACE")) { std::printf("  [post-sync]\n"); std::fflush(stdout); }

    std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * tokens);
    CK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
    double worst = 0.0;
    std::size_t worst_idx = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        std::size_t ridx = i;
        if (c.split_out > 0) {
            // Each section dumps as [token][row] contiguously; map back to reference [t*n+row].
            const std::size_t sec_rows = static_cast<std::size_t>(c.split_out);
            const std::size_t per_sec = sec_rows * tokens;
            const std::size_t base = i / per_sec;
            const std::size_t j = i - base * per_sec;
            const std::size_t t = j / sec_rows;
            const std::size_t row = base * sec_rows + j % sec_rows;
            ridx = t * static_cast<std::size_t>(n) + row;
        }
        const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&got[i]);
        const double err = std::fabs(static_cast<double>(__bfloat162float(b)) - reference[ridx]);
        const double ratio = err / budget_for(std::fabs(reference[i]));
        if (ratio > worst) { worst = ratio; worst_idx = i; }
    }
    ok = worst <= 1.0;
    std::printf("%-24s n=%6d k=%6d T=%2d seg=%d out=%d %s worst=%5.2fx budget (idx %zu)\n", c.name, n,
                k, tokens, w.gguf_segment_count, static_cast<int>(outs.size()), ok ? "PASS" : "FAIL",
                worst, worst_idx);
    } catch (const std::exception& error) {
        std::printf("%-24s EXCEPTION %s\n", c.name, error.what());
    }
    cudaFree(d_x);
    cudaFree(d_w1);
    if (d_w2) { cudaFree(d_w2); }
    cudaFree(d_out);
    return ok;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* filter = argc > 1 ? argv[1] : nullptr;
    cudaSetDevice(0);
    const GgufCase cases[] = {
        {"wide-t1",     17408, 5120, 1},
        {"wide-t2",     17408, 5120, 2},
        {"wide-t4",     17408, 5120, 4},
        {"wide-t5",     17408, 5120, 5},
        {"wide-t8",     17408, 5120, 8},
        {"gate-t4",     14336, 5120, 4},
        {"narrow-t8",    1024, 5120, 8},
        {"narrow-t3",    1536, 3072, 3},
        {"tall-t4",     65536,  512, 4},
        {"tall-t1",     65536,  512, 1},
        {"tiled6144-t4", 2048, 6144, 4, true},
        {"tiled3072-t8", 2048, 3072, 8, true},
        {"two-seg-t3",   2048,  512, 3, false, 1024},
        {"split-out-t6", 2048,  512, 6, false, 0, 1024},
        {"q5k-wide-t1", 17408, 5120, 1, false, 0, 0, GgufType::Q5_K},
        {"q5k-wide-t4", 17408, 5120, 4, false, 0, 0, GgufType::Q5_K},
        {"q5k-tall-t1", 65536,  512, 1, false, 0, 0, GgufType::Q5_K},
        {"q5k-tiled-t8", 2048, 3072, 8, true, 0, 0, GgufType::Q5_K},
    };
    bool all = true;
    for (const GgufCase& c : cases) {
        if (filter != nullptr && std::strstr(c.name, filter) == nullptr) { continue; }
        std::printf("begin %s\n", c.name);
        all = run(c) && all;
    }
    std::printf("%s\n", all ? "ALL PASS" : "FAILURES PRESENT");
    return all ? 0 : 1;
}
