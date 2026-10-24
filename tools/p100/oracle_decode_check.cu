// Oracle check for the Pascal decode/GGML-K routes introduced by the P100 migration.
//
// Generates deterministic synthetic packed weights with their stored scales, decodes them
// independently on the host (GGML-K through byte-for-byte copies of the pure-arithmetic
// ggml_k_codec.cuh decoder; W8/Q4 directly from the group scale streams), computes the dot
// products in FP64, and runs the sm_60 device routes at the registered shapes.
//
// Build (Windows, arch-60 tree already built):
//   VsDevCmd -> nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -Xcompiler=/MD
//     -DNINFER_PASCAL_BUILD -I src -I include tools/p100/oracle_decode_check.cu
//     build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib
// Run on the target card. Qualification evidence per docs/maintainer/op-development.md,
// recorded in docs/p100-migration.md section 7.
#include "core/tensor.h"
#include "ops/linear/ggml_k/ggml_k.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/w8/w8_launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

float half_to_float(std::uint16_t bits) {
    const __half h = __ushort_as_half(bits);
    return __half2float(h);
}

std::uint16_t float_to_half_bits(float value) {
    const __half h = __float2half_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&h);
}

// --- GGML-K host encoders (produce exactly the layouts ggml_k_codec.cuh reads) -------------
void encode_q4_k_block(unsigned char* block, std::uint32_t& state, std::vector<double>& decoded,
                       int base_col) {
    const float d_raw = 0.004F + static_cast<float>(next_rand(state) % 60) / 10000.0F;
    const float dmin_raw = 0.002F + static_cast<float>(next_rand(state) % 40) / 10000.0F;
    const std::uint16_t dbits = float_to_half_bits(d_raw);
    const std::uint16_t mbits = float_to_half_bits(dmin_raw);
    // the block stores fp16; the reference must use the rounded-back values, not the inputs
    const float d = half_to_float(dbits);
    const float dmin = half_to_float(mbits);
    block[0] = static_cast<unsigned char>(dbits & 0xFFu);
    block[1] = static_cast<unsigned char>(dbits >> 8);
    block[2] = static_cast<unsigned char>(mbits & 0xFFu);
    block[3] = static_cast<unsigned char>(mbits >> 8);
    std::uint32_t s[8], m[8];
    for (int g = 0; g < 8; ++g) {
        s[g] = next_rand(state) & 63u;
        m[g] = next_rand(state) & 63u;
    }
    for (int j = 0; j < 4; ++j) {
        block[4 + j] = static_cast<unsigned char>((s[j] & 63u) | ((s[j + 4] & 0x30u) << 2));
        block[8 + j] = static_cast<unsigned char>((m[j] & 63u) | ((m[j + 4] & 0x30u) << 2));
        block[12 + j] = static_cast<unsigned char>((s[j + 4] & 0x0Fu) | ((m[j + 4] & 0x0Fu) << 4));
    }
    std::memset(block + 16, 0, 32);
    for (int i = 0; i < 256; ++i) {
        const int group = i >> 5;
        const std::uint32_t code = next_rand(state) & 15u;
        const int byte = 16 + (i >> 6) * 32 + (i & 31);
        const int shift = (group & 1) * 4;
        block[byte] = static_cast<unsigned char>((block[byte] & ~(0x0Fu << shift)) |
                                                 static_cast<unsigned>(code << shift));
        decoded[base_col + i] = (static_cast<double>(d) * s[group]) * code -
                                static_cast<double>(dmin) * m[group];
    }
}

void encode_q6_k_block(unsigned char* block, std::uint32_t& state, std::vector<double>& decoded,
                       int base_col) {
    const float d_raw = 0.004F + static_cast<float>(next_rand(state) % 60) / 10000.0F;
    const std::uint16_t d_bits_stored = float_to_half_bits(d_raw);
    const float d = half_to_float(d_bits_stored);
    std::int8_t scales[16];
    for (int g = 0; g < 16; ++g) { scales[g] = static_cast<std::int8_t>(next_rand(state) & 0x3F); }
    std::memset(block, 0, 208);
    for (int i = 0; i < 256; ++i) {
        const int half = i >> 7;
        const int within = i & 127;
        const int section = within >> 5;
        const int lane = within & 31;
        const std::uint32_t code = next_rand(state) & 63u;  // 6-bit, value = code - 32
        const int low_index = half * 64 + (section & 1) * 32 + lane;
        const int low_shift = (section >> 1) * 4;
        block[low_index] = static_cast<unsigned char>(
            (block[low_index] & ~(0x0F << low_shift)) | static_cast<unsigned>((code & 15u)
                                                                              << low_shift));
        const int hi_index = 128 + half * 32 + lane;
        block[hi_index] = static_cast<unsigned char>(
            (block[hi_index] & ~(3 << (section * 2))) |
            static_cast<unsigned>(((code >> 4) & 3u) << (section * 2)));
        decoded[base_col + i] = static_cast<double>(d) * static_cast<double>(scales[i >> 4]) *
                                static_cast<double>(static_cast<int>(code) - 32);
    }
    std::memcpy(block + 192, scales, 16);
    const std::uint16_t dbits = float_to_half_bits(d);
    block[208] = static_cast<unsigned char>(dbits & 0xFFu);
    block[209] = static_cast<unsigned char>(dbits >> 8);
}

// --- independent GGML-K decoders (arithmetic mirrored from ggml_k_codec.cuh) --------------
double decode_q4_k_value(const unsigned char* block, int i) {
    const unsigned char* scales = block + 4;
    const int group = i >> 5;
    int scale, minimum;
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    const unsigned char q = block[16 + (i >> 6) * 32 + (i & 31)];
    const int code = (q >> ((group & 1) * 4)) & 15;
    const double d = half_to_float(*reinterpret_cast<const std::uint16_t*>(block));
    const double dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(block + 2));
    return (d * scale) * code - dmin * minimum;
}

double decode_q6_k_value(const unsigned char* block, int i) {
    const int half = i >> 7;
    const int within = i & 127;
    const int section = within >> 5;
    const int lane = within & 31;
    const int low_index = half * 64 + (section & 1) * 32 + lane;
    const int low = (block[low_index] >> ((section >> 1) * 4)) & 15;
    const int high = (block[128 + half * 32 + lane] >> (section * 2)) & 3;
    const int code = (low | (high << 4)) - 32;
    const int scale = static_cast<signed char>(block[192 + (i >> 4)]);
    const double d = half_to_float(*reinterpret_cast<const std::uint16_t*>(block + 208));
    return (d * scale) * code;
}

enum Kind { KindW8, KindQ4, KindQ5, KindQ6, KindGgmlQ4, KindGgmlQ6, KindGgmlMixed };

struct Case {
    const char* name;
    Kind kind;
    int n;
    int k;
    int tokens;
    double max_ulp;  // error budget in units of the bf16 output ulp at the reference magnitude
    bool with_reference_route;  // also run the qualified float route for comparison
    // Activation half-range in units of the base uniform (8.0). Production prefill/decode
    // activations are LayerNorm-shaped with |x| typically below ~2; the wide fp16-input band
    // (uniform ±8) exceeds the represented public-input domain and only stresses the fp16
    // accumulator range, so HFMA2 routes run at the representative amplitude.
    double x_half = 8.0;
};

// bf16 has 8 fraction bits: one ulp at magnitude m is 2^(floor(log2 m) - 7).
double bf16_ulp(double magnitude) {
    double exponent = 0.0;
    double m = magnitude;
    if (m <= 0.0) { m = 1e-6; }
    while (m >= 2.0) { m *= 0.5; exponent += 1.0; }
    while (m < 1.0) { m *= 2.0; exponent -= 1.0; }
    return std::pow(2.0, exponent - 7.0);
}

// Mixed acceptance band: absolute floor (covers near-zero references, where a relative ulp is
// meaningless) plus a multiple of the bf16 ulp at the reference magnitude.
double budget_for(double reference_magnitude, double ulp_multiple) {
    const double budget = ulp_multiple * bf16_ulp(reference_magnitude);
    return budget < 1.0 ? 1.0 : budget;
}

bool is_ggml(Kind kind) { return kind == KindGgmlQ4 || kind == KindGgmlQ6 || kind == KindGgmlMixed; }

bool run_case(const Case& c, std::uint32_t seed) {
    const int n = c.n, k = c.k, tokens = c.tokens;
    std::uint32_t state = seed;

    // activation [k x tokens] bf16, row-contiguous over k
    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * tokens);
    std::vector<double> x_double(static_cast<std::size_t>(k) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < k; ++i) {
            const float v = ((static_cast<float>(next_rand(state) % 16001) / 1000.0F) - 8.0F) *
                            static_cast<float>(c.x_half / 8.0);
            const __nv_bfloat16 b = __float2bfloat16_rn(v);
            x_bits[static_cast<std::size_t>(t) * k + i] = *reinterpret_cast<const std::uint16_t*>(&b);
            x_double[static_cast<std::size_t>(t) * k + i] = static_cast<double>(__bfloat162float(b));
        }
    }

    // weights + oracle
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> high_plane;  // q5/q6 bit-5/bit-5:6 plane
    std::vector<std::uint16_t> scales;
    std::vector<std::uint64_t> descriptors;
    std::vector<double> reference(static_cast<std::size_t>(n) * tokens, 0.0);

    if (!is_ggml(c.kind)) {
        const int group = c.kind == KindW8 ? 32 : 64;
        const int groups_per_row = k / group;
        const int bits_per_value = c.kind == KindW8 ? 8 : (c.kind == KindQ4 ? 4 : (c.kind == KindQ5 ? 5 : 6));
        codes.assign(static_cast<std::size_t>(n) * k * bits_per_value / 8, 0);
        if (c.kind == KindQ5) { high_plane.assign(static_cast<std::size_t>(n) * k / 8, 0); }
        if (c.kind == KindQ6) { high_plane.assign(static_cast<std::size_t>(n) * k / 4, 0); }
        scales.assign(static_cast<std::size_t>(n) * groups_per_row, 0);
        std::vector<int> values(k);
        for (int row = 0; row < n; ++row) {
            for (int g = 0; g < groups_per_row; ++g) {
                const float mantissa = 1.0F + static_cast<float>(next_rand(state) & 0xFF) / 255.0F;
                scales[static_cast<std::size_t>(row) * groups_per_row + g] =
                    float_to_half_bits(mantissa * 0.005F);
            }
            for (int i = 0; i < k; ++i) {
                const int r = static_cast<int>(next_rand(state) & 0xFFFFu);
                values[i] = c.kind == KindW8   ? (r % 256) - 128
                            : c.kind == KindQ4 ? (r % 16) - 8
                            : c.kind == KindQ5 ? (r % 32) - 16
                                               : (r % 64) - 32;
                // two's-complement storage: Q4 nibble, Q5 5 bits (nibble + bit5 plane),
                // Q6 6 bits (nibble + 2-bit plane); W8 stores the raw byte.
                const std::uint32_t stored = static_cast<std::uint32_t>(values[i]) &
                                             (bits_per_value == 8 ? 0xFFu
                                              : c.kind == KindQ4 ? 0x0Fu
                                              : c.kind == KindQ5 ? 0x1Fu : 0x3Fu);
                std::size_t base = static_cast<std::size_t>(row) * k;
                if (c.kind == KindW8) {
                    codes[base + i] = static_cast<std::uint8_t>(stored);
                } else {
                    const std::size_t byte_index = (base + i) / 2;  // row-wide nibble stream
                    const int nibble = static_cast<int>(stored & 15u);
                    if (((base + i) & 1) == 0) { codes[byte_index] = static_cast<std::uint8_t>(nibble); }
                    else { codes[byte_index] |= static_cast<std::uint8_t>(nibble << 4); }
                    if (c.kind == KindQ5) {
                        const std::size_t hi_byte = (base + i) / 8;
                        high_plane[hi_byte] = static_cast<std::uint8_t>(
                            high_plane[hi_byte] | (((stored >> 4) & 1u) << (i & 7)));
                    } else if (c.kind == KindQ6) {
                        const int bit = ((base + i) & 7) * 2;
                        high_plane[(base + i) / 8 * 2 + (bit >> 3)] = static_cast<std::uint8_t>(
                            high_plane[(base + i) / 8 * 2 + (bit >> 3)] |
                            (((stored >> 4) & 3u) << (bit & 7)));
                    }
                }
            }
            for (int t = 0; t < tokens; ++t) {
                double total = 0.0;
                for (int g = 0; g < groups_per_row; ++g) {
                    const double scale =
                        half_to_float(scales[static_cast<std::size_t>(row) * groups_per_row + g]);
                    double part = 0.0;
                    for (int i = g * group; i < (g + 1) * group; ++i) {
                        part += values[i] * x_double[static_cast<std::size_t>(t) * k + i];
                    }
                    total += scale * part;
                }
                reference[static_cast<std::size_t>(t) * n + row] = total;
            }
        }
    } else {
        const int blocks_per_row = k / 256;
        std::vector<int> row_type(n, 0);
        for (int row = 0; row < n; ++row) {
            row_type[row] = c.kind == KindGgmlQ6 ? 1
                            : c.kind == KindGgmlMixed ? (row >= n / 2 ? 1 : 0) : 0;
        }
        std::size_t total_bytes = 0;
        for (int row = 0; row < n; ++row) {
            total_bytes += static_cast<std::size_t>(blocks_per_row) * (row_type[row] ? 210 : 144);
        }
        codes.resize(total_bytes);
        std::vector<double> decoded(k);
        std::size_t offset = 0;
        for (int row = 0; row < n; ++row) {
            descriptors.push_back((static_cast<std::uint64_t>(offset) << 1) |
                                  static_cast<std::uint64_t>(row_type[row]));
            unsigned char* row_base = codes.data() + offset;
            for (int b = 0; b < blocks_per_row; ++b) {
                unsigned char* block = row_base + b * (row_type[row] ? 210 : 144);
                if (row_type[row]) { encode_q6_k_block(block, state, decoded, b * 256); }
                else { encode_q4_k_block(block, state, decoded, b * 256); }
            }
            for (int t = 0; t < tokens; ++t) {
                double total = 0.0;
                for (int i = 0; i < k; ++i) {
                    total += decoded[i] * x_double[static_cast<std::size_t>(t) * k + i];
                }
                reference[static_cast<std::size_t>(t) * n + row] = total;
            }
            // cross-check the encoder against the mirrored decoder, once per row
            for (int i = 0; i < k; ++i) {
                const double again = row_type[row] ? decode_q6_k_value(row_base + (i >> 8) * 210,
                                                                       i & 255)
                                                   : decode_q4_k_value(row_base + (i >> 8) * 144,
                                                                       i & 255);
                if (std::fabs(again - decoded[i]) > 1e-9) {
                    std::printf("%-30s encoder/decoder mismatch at row %d col %d\n", c.name, row,
                                i);
                    return false;
                }
            }
            offset += static_cast<std::size_t>(blocks_per_row) * (row_type[row] ? 210 : 144);
        }
    }

    // --- device run ---
    void* d_x = nullptr;
    void* d_codes = nullptr;
    void* d_scales = nullptr;
    void* d_high = nullptr;
    void* d_desc = nullptr;
    void* d_out = nullptr;
    void* d_out_ref = nullptr;
    const std::size_t out_bytes = static_cast<std::size_t>(n) * tokens * 2;
    const std::size_t scales_bytes = scales.size() * 2;
    const std::size_t high_bytes = high_plane.size();
    const std::size_t desc_bytes = descriptors.size() * 8;
    cudaMalloc(&d_x, x_bits.size() * 2);
    cudaMalloc(&d_codes, codes.size());
    cudaMalloc(&d_out, out_bytes);
    cudaMalloc(&d_out_ref, out_bytes);
    cudaMemcpy(d_x, x_bits.data(), x_bits.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    if (scales_bytes) {
        cudaMalloc(&d_scales, scales_bytes);
        cudaMemcpy(d_scales, scales.data(), scales_bytes, cudaMemcpyHostToDevice);
    }
    if (high_bytes) {
        cudaMalloc(&d_high, high_bytes);
        cudaMemcpy(d_high, high_plane.data(), high_bytes, cudaMemcpyHostToDevice);
    }
    if (desc_bytes) {
        cudaMalloc(&d_desc, desc_bytes);
        cudaMemcpy(d_desc, descriptors.data(), desc_bytes, cudaMemcpyHostToDevice);
    }

    Tensor x(d_x, DType::BF16, {k, tokens});
    Tensor out(d_out, DType::BF16, {n, tokens});
    Tensor out_ref(d_out_ref, DType::BF16, {n, tokens});
    Weight w{};
    w.qdata = d_codes;
    w.scales = d_scales;
    w.qhigh = is_ggml(c.kind) ? d_desc : d_high;
    w.n = n;
    w.k = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    if (is_ggml(c.kind)) {
        w.qtype = QType::GGML_K;
        w.layout = QuantLayout::GgmlK256;
        w.group = 256;
        // Pure-Q6 fixtures carry the truthful uniform-matrix hint, so the T<=4 launches take
        // the specialized Q6 kernel exactly like the production Q6_K artifact; mixed fixtures
        // keep (-1, -1) and exercise the per-row dynamic branch.
        if (c.kind == KindGgmlQ6) {
            w.ggml_k_first_q6 = 1;
            w.ggml_k_type_change = n;
        } else {
            w.ggml_k_first_q6 = -1;
            w.ggml_k_type_change = -1;
        }
    }

    bool ok = true;
    double max_err = 0.0;
    double max_err_ref = 0.0;
    double err_ulp = 0.0;
    std::int64_t w_err_idx[8];
    for (int i = 0; i < 8; ++i) { w_err_idx[i] = -1; }
    std::vector<std::uint16_t> got_dump;
    try {
        if (is_ggml(c.kind)) {
            ggml_k_project_split(x, w, &out, 1, false, nullptr, false, nullptr);
            if (tokens > 4 && (n % 2) == 0) {
                // Reference route: a two-section split bypasses the tile GEMM and runs the
                // chunked HFMA2 gemv, so tile-vs-gemv drift is measurable per case.
                Weight w2 = w;  // descriptors and rows stay global; only the output splits
                Tensor s0(static_cast<std::uint8_t*>(d_out_ref), DType::BF16, {n / 2, tokens});
                Tensor s1(static_cast<std::uint8_t*>(d_out_ref) +
                              static_cast<std::size_t>(n / 2) * tokens * 2,
                          DType::BF16, {n / 2, tokens});
                const Tensor outs[2] = {s0, s1};
                ggml_k_project_split(x, w2, outs, 2, false, nullptr, false, nullptr);
            }
        } else if (c.kind == KindQ4) {
            if (tokens == 1) { launch_q4_decode_hfma2_pascal(x, w, out, nullptr); }
            else { launch_q4_decode_tile_pascal(x, w, out, nullptr); }
            if (c.with_reference_route && tokens == 1) {
                launch_q4_gemv_r1_w8_direct(x, w, out_ref, nullptr);
            }
        } else if (c.kind == KindQ5) {
            if (tokens == 1) { launch_q5_decode_hfma2_pascal(x, w, out, nullptr); }
            else { launch_q5_decode_tile_pascal(x, w, out, nullptr); }
            if (c.with_reference_route && tokens == 1) {
                launch_q5_simt_r8_c4(x, w, out_ref, nullptr);
            }
        } else if (c.kind == KindQ6) {
            if (tokens == 1) { launch_q6_decode_hfma2_pascal(x, w, out, nullptr); }
            else { launch_q6_decode_tile_pascal(x, w, out, nullptr); }
            if (c.with_reference_route && tokens == 1) {
                launch_q6_simt_r8_c4(x, w, out_ref, nullptr);
            }
        } else {
            if (tokens == 1) { launch_w8_decode_hfma2_pascal(x, w, out, nullptr); }
            else { launch_w8_decode_tile_pascal(x, w, out, nullptr); }
            if (c.with_reference_route && tokens == 1) {
                if (k == 16384 && n == 2048) { launch_w8_decode_r4(x, w, out_ref, nullptr); }
                else { launch_w8_simt_r8_c4(x, w, out_ref, nullptr); }
            }
        }
        cudaDeviceSynchronize();
        got_dump.assign(static_cast<std::size_t>(n) * tokens, 0);
        std::vector<std::uint16_t>& got = got_dump;
        cudaMemcpy(got.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
        double worst_ulp_ratio = 0.0;
        double max_err_ulp = 0.0;  // absolute error at the worst-ulp element, for reporting
        double w_err_ratio[8];
        for (int i = 0; i < 8; ++i) { w_err_ratio[i] = -1.0; }
        auto record = [&](std::size_t i, double ratio) {
            for (int s = 0; s < 8; ++s) {
                if (ratio > w_err_ratio[s]) {
                    for (int m = 7; m > s; --m) { w_err_idx[m] = w_err_idx[m - 1]; w_err_ratio[m] = w_err_ratio[m - 1]; }
                    w_err_idx[s] = static_cast<std::int64_t>(i); w_err_ratio[s] = ratio; return;
                }
            }
        };
        for (std::size_t i = 0; i < got.size(); ++i) {
            const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&got[i]);
            const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - reference[i]);
            const double ratio = e / budget_for(std::fabs(reference[i]), c.max_ulp);
            record(i, ratio);
            if (ratio > worst_ulp_ratio) { worst_ulp_ratio = ratio; max_err_ulp = e; }
        }
        max_err = max_err_ulp;
        err_ulp = worst_ulp_ratio;
        if ((c.with_reference_route && tokens == 1 && !is_ggml(c.kind)) ||
            (is_ggml(c.kind) && tokens > 4 && (n % 2) == 0)) {
            std::vector<std::uint16_t> got_ref(static_cast<std::size_t>(n) * tokens);
            cudaMemcpy(got_ref.data(), d_out_ref, out_bytes, cudaMemcpyDeviceToHost);
            double worst_ref_ratio = 0.0;
            const auto ref_at = [&](std::size_t i) -> std::uint16_t {
                if (!is_ggml(c.kind)) { return got_ref[i]; }
                const int row = static_cast<int>(i) % n;
                const int t = static_cast<int>(i) / n;
                const std::size_t half = static_cast<std::size_t>(n / 2) * tokens;
                const std::size_t idx = row < n / 2
                                            ? static_cast<std::size_t>(t) * (n / 2) + row
                                            : half + static_cast<std::size_t>(t) * (n / 2) +
                                                  (row - n / 2);
                return got_ref[idx];
            };
            for (std::size_t i = 0; i < got_ref.size(); ++i) {
                const std::uint16_t bits = ref_at(i);
                const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&bits);
                const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - reference[i]);
                const double ratio = e / budget_for(std::fabs(reference[i]), c.max_ulp);
                if (ratio > worst_ref_ratio) { worst_ref_ratio = ratio; }
            }
            max_err_ref = worst_ref_ratio;
        } else {
            max_err_ref = err_ulp;  // ggml/wide-T cases compare against their own band
        }
    } catch (const std::exception& error) {
        std::printf("%-30s EXCEPTION %s\n", c.name, error.what());
        ok = false;
    }

    std::printf("%-30s n=%7d k=%6d T=%3d  max|err|=%9.5f  worst=%6.2fx budget (ref-route %6.2fx, "
                "cap 1.00x over %4.1fulp+1.0 floor)  %s\n",
                c.name, n, k, tokens, max_err, err_ulp, max_err_ref, c.max_ulp,
                (ok && err_ulp <= 1.0) ? "PASS" : "FAIL");
    if (ok == false || (getenv("ORACLE_DEBUG") != nullptr)) {
        const char* env = getenv("ORACLE_DEBUG");
        const int show = env ? atoi(env) : 8;
        for (int i = 0; i < show; ++i) {
            const int idx = static_cast<int>(w_err_idx[i]);
            if (idx < 0 || idx >= static_cast<int>(got_dump.size())) { break; }
            const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&got_dump[idx]);
            std::printf("   worst[%d] row=%d t=%d got=%f ref=%f err=%f\n", i, idx % n, idx / n,
                        __bfloat162float(b), reference[idx],
                        std::fabs(__bfloat162float(b) - reference[idx]));
        }
    }
    if (ok && err_ulp > 1.0) { ok = false; }

    cudaFree(d_x);
    cudaFree(d_codes);
    if (d_scales) { cudaFree(d_scales); }
    if (d_high) { cudaFree(d_high); }
    if (d_desc) { cudaFree(d_desc); }
    cudaFree(d_out);
    cudaFree(d_out_ref);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;  // substring case-name filter
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1) {
        std::printf("no CUDA device\n");
        return 2;
    }
    cudaDeviceProp d{};
    cudaGetDeviceProperties(&d, 0);
    std::printf("oracle_decode_check: device %s sm_%d%d\n", d.name, d.major, d.minor);

    const Case cases[] = {
        {"w8 vocab head", KindW8, 248320, 5120, 1, 2.5, true},
        {"w8 gate/up", KindW8, 34816, 5120, 1, 2.5, true},
        {"w8 down", KindW8, 5120, 17408, 1, 2.5, true},
        {"w8 qkv/out", KindW8, 5120, 5120, 1, 2.5, true},
        {"w8 mtp input", KindW8, 2048, 16384, 1, 2.5, true},
        {"q4 gate/up", KindQ4, 34816, 5120, 1, 2.5, true},
        {"q4 down", KindQ4, 5120, 17408, 1, 2.5, false},
        {"q4 qkv", KindQ4, 7168, 5120, 1, 2.5, true},
        {"q4 small gdn", KindQ4, 4096, 5120, 1, 2.5, true},
        {"q5 gate/up", KindQ5, 34816, 5120, 1, 2.5, true},
        {"q5 down", KindQ5, 5120, 17408, 1, 2.5, true},
        {"q5 qkv 6144", KindQ5, 6144, 5120, 1, 2.5, true},
        {"q6 vocab head", KindQ6, 248320, 5120, 1, 2.5, true},
        {"q6 small", KindQ6, 4096, 5120, 1, 2.5, true},
        {"ggml q4 decode T1", KindGgmlQ4, 512, 512, 1, 2.5, false, 2.0},
        {"ggml q6 decode T1", KindGgmlQ6, 512, 512, 1, 2.5, false, 2.0},
        {"ggml mixed T4", KindGgmlMixed, 256, 512, 4, 2.5, false, 2.0},
        {"ggml q4 ragged T7", KindGgmlQ4, 256, 512, 7, 2.5, false, 2.0},
        {"ggml q4 multi T16", KindGgmlQ4, 1024, 1024, 16, 2.5, false, 2.0},
        {"ggml q6 multi T64", KindGgmlQ6, 512, 512, 64, 2.5, false, 2.0},
        {"w8 tile T5", KindW8, 512, 1024, 5, 2.5, false},
        {"w8 tile T16", KindW8, 1024, 512, 16, 2.5, false},
        {"w8 tile k5120 T1024", KindW8, 128, 5120, 1024, 2.5, false},
        {"q4 tile T8", KindQ4, 512, 1024, 8, 2.5, false},
        {"q4 tile T13 ragged", KindQ4, 256, 512, 13, 2.5, false},
        {"q4 tile k2048 T64", KindQ4, 1024, 2048, 64, 2.5, false},
        {"q5 tile T8", KindQ5, 512, 1024, 8, 2.5, false},
        {"q6 tile T9 ragged", KindQ6, 512, 512, 9, 2.5, false},
        {"q4 tile k5120 T1024", KindQ4, 128, 5120, 1024, 2.5, false},
        {"q5 tile k17408 T12 multi", KindQ5, 256, 17408, 12, 2.5, false},
        {"q6 tile k5120 T33 multi", KindQ6, 256, 5120, 33, 2.5, false},
        {"ggml mixed T33 tile", KindGgmlMixed, 512, 512, 33, 2.5, false, 2.0},
        {"ggml q4 tile k5120 T97", KindGgmlQ4, 256, 5120, 97, 2.5, false, 2.0},
        {"ggml q6 tile k5120 T8", KindGgmlQ6, 256, 5120, 8, 2.5, false, 2.0},
        {"ggml q4 tile T4096", KindGgmlQ4, 64, 512, 4096, 2.5, false, 2.0},
    };
    bool all = true;
    for (const Case& c : cases) {
        if (filter != nullptr && strstr(c.name, filter) == nullptr) { continue; }
        all = run_case(c, 0x9E3779B9u) && all;
    }
    std::printf(all ? "ALL PASS\n" : "FAILURES PRESENT\n");
    return all ? 0 : 1;
}
