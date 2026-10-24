// Oracle check for the Pascal NVFP4 A16 routes (ops::linear nvfp4 dispatch, normalized to the
// dequantizing SIMT paths on sm_60 -- W4A4 needs the Blackwell-only mxf4nvf4 mma and is stubbed
// out of the arch-60 build; see docs/p100-migration.md decision 6).
//
// Generates deterministic synthetic packed NVFP4 weights -- row-major e2m1 nibble codes plus the
// BlockScaleK16M128x4 swizzled e4m3 scale plane -- decodes them independently on the host from
// the stored scales, computes the dot products in FP64, and runs the production nvfp4_dispatch
// route (AllowA4 policy) at the registered shapes and token widths, including the bands that
// previously resolved to W4A4 (T>=4..8 depending on problem).
//
// Build (Windows, arch-60 tree already built):
//   VsDevCmd -> nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -Xcompiler=/MD
//     -DNINFER_PASCAL_BUILD -I src -I include tools/p100/nvfp4_linear_oracle.cu
//     build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib
// Run on the target card.
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// --- NVFP4 element formats, decoded independently of the device codec ----------------------
double e2m1_value(unsigned nibble) {
    static const double kMag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude      = kMag[nibble & 7u];
    return (nibble & 8u) ? -magnitude : magnitude;
}

double e4m3_value(unsigned byte) {
    const unsigned exponent = (byte >> 3) & 0x0Fu;
    const unsigned mantissa = byte & 0x07u;
    if (exponent == 15u && mantissa == 7u) { return std::nan(""); }  // e4m3fn NaN
    const double sign = (byte & 0x80u) ? -1.0 : 1.0;
    if (exponent == 0u) { return sign * static_cast<double>(mantissa) / 8.0 * std::pow(2.0, -6.0); }
    return sign * (1.0 + static_cast<double>(mantissa) / 8.0) * std::pow(2.0, int(exponent) - 7);
}

// Byte-for-byte mirror of nvfp4_gemv.cuh's nvfp4_scale_offset swizzle for the host side:
// tiles of 128 rows x k/64 scale-groups, (row_mod32, row_quartile, group_lane) packing.
std::size_t host_scale_offset(int row, int group, int k) {
    const int m_tile           = row / 128;
    const int row_inner        = row - m_tile * 128;
    const int scale_tile       = group / 4;
    const int scale_lane       = group & 3;
    const int row_mod32        = row_inner & 31;
    const int row_quartile     = row_inner >> 5;
    const int tiles_per_row    = k / 64;
    return static_cast<std::size_t>(m_tile * tiles_per_row + scale_tile) * 512 +
           static_cast<std::size_t>(row_mod32) * 16 + row_quartile * 4 + scale_lane;
}

double bf16_ulp(double magnitude) {
    double exponent = 0.0;
    double m = magnitude;
    if (m <= 0.0) { m = 1e-6; }
    while (m >= 2.0) { m *= 0.5; exponent += 1.0; }
    while (m < 1.0) { m *= 2.0; exponent -= 1.0; }
    return std::pow(2.0, exponent - 7.0);
}

double budget_for(double reference_magnitude, double ulp_multiple) {
    const double budget = ulp_multiple * bf16_ulp(reference_magnitude);
    return budget < 1.0 ? 1.0 : budget;
}

struct Case {
    const char* name;
    int n;
    int k;
    int tokens;
    double max_ulp;
};

constexpr float kWeightScaleDivisor = 16.0F;

bool run_case(const Case& c, std::uint32_t seed) {
    const int n      = c.n;
    const int k      = c.k;
    const int tokens = c.tokens;
    std::uint32_t state = seed;

    // activation [k x tokens] bf16 (token-major storage, k contiguous per token row)
    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * tokens);
    std::vector<double> x_double(static_cast<std::size_t>(k) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < k; ++i) {
            // LayerNorm-shaped activations: |x| below ~2, the represented public-input band.
            const float v = ((static_cast<float>(next_rand(state) % 3201) / 800.0F) - 2.0F);
            const __nv_bfloat16 b = __float2bfloat16_rn(v);
            x_bits[static_cast<std::size_t>(t) * k + i] =
                *reinterpret_cast<const std::uint16_t*>(&b);
            x_double[static_cast<std::size_t>(t) * k + i] =
                static_cast<double>(__bfloat162float(b));
        }
    }

    // weight payload: codes plane (n*k/2 bytes) + 256-align + scale plane (n*k/16 bytes) + 4B
    const std::size_t code_bytes  = static_cast<std::size_t>(n) * k / 2;
    const std::size_t scale_off   = (code_bytes + 255) / 256 * 256;
    const std::size_t scale_bytes = static_cast<std::size_t>(n) * k / 16;
    std::vector<std::uint8_t> payload(scale_off + scale_bytes + 4, 0);
    std::vector<double> decoded_weight(static_cast<std::size_t>(n) * k, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int i = 0; i < k; ++i) {
            const unsigned nibble = next_rand(state) & 0x0Fu;
            const std::size_t byte = static_cast<std::size_t>(row) * (k / 2) + (i >> 1);
            if ((i & 1) == 0) { payload[byte] = static_cast<std::uint8_t>(nibble); }
            else { payload[byte] = static_cast<std::uint8_t>((payload[byte] & 0x0Fu) |
                                                             (nibble << 4)); }
            decoded_weight[static_cast<std::size_t>(row) * k + i] = e2m1_value(nibble);
        }
        // one e4m3 scale per 16-element block, stored through the swizzle
        for (int g = 0; g < k / 16; ++g) {
            // representative positive scale band: exponent 6..10, few mantissa steps
            unsigned byte = 0;
            const unsigned exponent = 6u + (next_rand(state) % 5u);
            const unsigned mantissa = next_rand(state) % 8u;
            byte                      = (exponent << 3) | mantissa;
            if (next_rand(state) & 1u) { byte |= 0x80u; }  // half the blocks negative
            payload[scale_off + host_scale_offset(row, g, k)] = static_cast<std::uint8_t>(byte);
            const double coefficient = e4m3_value(byte) / static_cast<double>(kWeightScaleDivisor);
            for (int i = g * 16; i < (g + 1) * 16; ++i) {
                decoded_weight[static_cast<std::size_t>(row) * k + i] *= coefficient;
            }
        }
    }

    // reference: out[t][row] = sum_k weight[row][k] * x[t][k]  (FP64)
    std::vector<double> reference(static_cast<std::size_t>(n) * tokens, 0.0);
    for (int t = 0; t < tokens; ++t) {
        for (int row = 0; row < n; ++row) {
            double total = 0.0;
            const double* wrow = decoded_weight.data() + static_cast<std::size_t>(row) * k;
            const double* xcol = x_double.data() + static_cast<std::size_t>(t) * k;
            for (int i = 0; i < k; ++i) { total += wrow[i] * xcol[i]; }
            reference[static_cast<std::size_t>(t) * n + row] = total;
        }
    }

    // --- device run through the production route ---
    void* d_payload = nullptr;
    void* d_x       = nullptr;
    void* d_out     = nullptr;
    cudaMalloc(&d_payload, payload.size());
    cudaMalloc(&d_x, x_bits.size() * 2);
    cudaMalloc(&d_out, static_cast<std::size_t>(n) * tokens * 2);
    cudaMemcpy(d_payload, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_bits.data(), x_bits.size() * 2, cudaMemcpyHostToDevice);

    Tensor x(d_x, DType::BF16, {k, tokens});
    Tensor out(d_out, DType::BF16, {n, tokens});
    Weight w{};
    w.payload              = d_payload;
    w.qdata                = d_payload;
    w.scales               = static_cast<std::uint8_t*>(d_payload) + scale_off;
    w.qhigh                = nullptr;  // NVFP4 requires no high plane (validate_nvfp4_weight)
    w.payload_bytes        = payload.size();
    w.high_plane_bytes     = 0;
    w.qtype                = QType::NVFP4;
    w.layout               = QuantLayout::BlockScaleK16M128x4;
    w.scale_dtype          = DType::FP8_E4M3FN;
    w.group                = 16;
    w.group_size           = 16;
    w.ndim                 = 2;
    w.n                    = n;
    w.k                    = k;
    w.shape[0]             = n;
    w.shape[1]             = k;
    w.padded_shape[0]      = n;
    w.padded_shape[1]      = k;
    w.weight_scale_divisor = kWeightScaleDivisor;
    w.input_scale_divisor  = 1.0F;

    bool ok = true;
#ifdef NVFP4_PROBE
    {
        const std::uint64_t elements = static_cast<std::uint64_t>(w.n) * w.k;
        const std::uint64_t code_b   = elements / 2;
        const std::uint64_t off      = (code_b + 255) / 256 * 256;
        const std::uint64_t req      = off + elements / 16 + 4;
        std::printf("PROBE %s: qtype=%d layout=%d sdt=%d grp=%u/%d ndim=%u shape=%d,%d pad=%d,%d\n",
                    c.name, static_cast<int>(w.qtype), static_cast<int>(w.layout),
                    static_cast<int>(w.scale_dtype), w.group_size, w.group, w.ndim, w.shape[0],
                    w.shape[1], w.padded_shape[0], w.padded_shape[1]);
        std::printf("  payload=%d qdata=%d scales=%d qhigh=%d highb=%llu pb=%llu req=%llu\n",
                    w.payload != nullptr, w.qdata != nullptr, w.scales != nullptr,
                    w.qhigh != nullptr, (unsigned long long)w.high_plane_bytes,
                    (unsigned long long)w.payload_bytes, (unsigned long long)req);
        std::printf("  ws=%f is=%f qd==pl=%d sc==pl+off=%d al16=%d/%d n%%128=%d k%%64=%d\n",
                    w.weight_scale_divisor, w.input_scale_divisor, w.qdata == w.payload,
                    w.scales == static_cast<const std::byte*>(w.payload) + off,
                    (reinterpret_cast<std::uintptr_t>(w.qdata) & 15) == 0,
                    (reinterpret_cast<std::uintptr_t>(w.scales) & 15) == 0, w.n % 128, w.k % 64);
    }
#endif
    try {
        // AllowA4 on purpose: on Pascal this must resolve to the A16 chunked routes, at every
        // width that previously picked W4A4 (the stub would throw at runtime otherwise).
        nvfp4_dispatch(x, w, out, ops::LinearPolicy::AllowA4, nullptr, nullptr);
        cudaDeviceSynchronize();
        std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * tokens, 0);
        cudaMemcpy(got.data(), d_out, static_cast<std::size_t>(n) * tokens * 2,
                   cudaMemcpyDeviceToHost);
        double worst_ratio = 0.0;
        double max_err     = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&got[i]);
            const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - reference[i]);
            const double ratio = e / budget_for(std::fabs(reference[i]), c.max_ulp);
            if (ratio > worst_ratio) { worst_ratio = ratio; max_err = e; }
        }
        ok = worst_ratio <= 1.0;
        std::printf("%-28s n=%7d k=%6d T=%4d  max|err|=%9.5f  worst=%6.2fx budget  %s\n", c.name,
                    n, k, tokens, max_err, worst_ratio, ok ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
        std::printf("%-28s n=%7d k=%6d T=%4d  THREW: %s  FAIL\n", c.name, n, k, tokens, e.what());
        ok = false;
    }
    cudaFree(d_payload);
    cudaFree(d_x);
    cudaFree(d_out);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("nvfp4_linear_oracle: %s sm_%d%d\n", prop.name, prop.major, prop.minor);

    // registered tp1 + tp2-column shapes; token bands chosen to hit every route that the
    // Pascal normalization changed (T=1 gemv, 2..32 small_t single pass, >32 chunk loop,
    // and the former W4A4 entry bands 4/5/7/8).
    const std::vector<Case> cases = {
        {"nvfp4 attn input T1", 14336, 5120, 1, 2.5},
        {"nvfp4 gdn input T1", 16384, 5120, 1, 2.5},
        {"nvfp4 gate_up T1", 34816, 5120, 1, 2.5},
        {"nvfp4 residual 6144 T1", 5120, 6144, 1, 2.5},
        {"nvfp4 residual 17408 T1", 5120, 17408, 1, 2.5},
        {"nvfp4 gate_up T4 wasW4A4", 34816, 5120, 4, 2.5},
        {"nvfp4 attn input T5 wasW4A4", 14336, 5120, 5, 2.5},
        {"nvfp4 residual 6144 T7", 5120, 6144, 7, 2.5},
        {"nvfp4 gdn input T8 wasW4A4", 16384, 5120, 8, 2.5},
        {"nvfp4 gate_up T32", 34816, 5120, 32, 2.5},
        {"nvfp4 gate_up T33 chunk", 34816, 5120, 33, 2.5},
        {"nvfp4 attn input T128 chunk", 14336, 5120, 128, 2.5},
        {"nvfp4 down tp2 col T9", 17408, 5120, 9, 2.5},
        {"nvfp4 residual 17408 T256", 5120, 17408, 256, 2.5},
    };

    const char* filter = argc > 1 ? argv[1] : nullptr;
    std::vector<bool> results;
    for (const Case& c : cases) {
        if (filter && std::strstr(c.name, filter) == nullptr) { continue; }
        results.push_back(run_case(c, 0x9E3779B9u));
    }
    bool all = !results.empty();
    for (const bool r : results) { all = all && r; }
    std::printf("%s\n", all ? "ALL PASS" : "FAILURES PRESENT");
    return all ? 0 : 1;
}
