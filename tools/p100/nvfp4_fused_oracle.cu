// End-to-end oracle for the Pascal NVFP4 *fused* op families (attn_input_proj, gdn_input_proj,
// linear_swiglu, linear_add) running through their production dispatchers under the AllowA4
// policy, which the PASCAL build normalizes to the A16 dequantizing routes (W4A4 is the
// Blackwell-only mxf4nvf4 mma, stubbed out of the arch-60 build; docs/p100-migration.md).
//
// Same synthetic-weight recipe as tools/p100/nvfp4_linear_oracle.cu: deterministic row-major
// e2m1 nibble codes + BlockScaleK16M128x4 swizzled e4m3 scale plane, decoded independently on
// the host with the stored scales, dot products in FP64, then compared against each fused
// dispatcher's own section split (Q|K|Gate|V, Q|K|V|Z, silu(gate)*up, +residual).
//
// Build (Windows, arch-60 tree already built):
//   VsDevCmd -> nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -Xcompiler=/MD
//     -DNINFER_PASCAL_BUILD -I src -I include tools/p100/nvfp4_fused_oracle.cu
//     build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <cuda_bf16.h>

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

double e2m1_value(unsigned nibble) {
    static const double kMag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude      = kMag[nibble & 7u];
    return (nibble & 8u) ? -magnitude : magnitude;
}

double e4m3_value(unsigned byte) {
    const unsigned exponent = (byte >> 3) & 0x0Fu;
    const unsigned mantissa = byte & 0x07u;
    if (exponent == 15u && mantissa == 7u) { return std::nan(""); }
    const double sign = (byte & 0x80u) ? -1.0 : 1.0;
    if (exponent == 0u) { return sign * static_cast<double>(mantissa) / 8.0 * std::pow(2.0, -6.0); }
    return sign * (1.0 + static_cast<double>(mantissa) / 8.0) * std::pow(2.0, int(exponent) - 7);
}

std::size_t host_scale_offset(int row, int group, int k) {
    const int m_tile        = row / 128;
    const int row_inner     = row - m_tile * 128;
    const int scale_tile    = group / 4;
    const int scale_lane    = group & 3;
    const int row_mod32     = row_inner & 31;
    const int row_quartile  = row_inner >> 5;
    const int tiles_per_row = k / 64;
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

constexpr float kWeightScaleDivisor = 16.0F;

// Deterministic synthetic activations + NVFP4 weight payload + FP64-decoded weight matrix.
struct Fixture {
    std::vector<std::uint16_t> x_bits;
    std::vector<double> x;
    std::vector<std::uint8_t> payload;
    std::size_t scale_off = 0;
    std::vector<double> w;  // decoded weight [n*k], = e2m1 * e4m3 / divisor
};

Fixture make_fixture(int n, int k, int tokens, std::uint32_t seed) {
    Fixture f;
    std::uint32_t state = seed;
    f.x_bits.assign(static_cast<std::size_t>(k) * tokens, 0);
    f.x.assign(static_cast<std::size_t>(k) * tokens, 0.0);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < k; ++i) {
            const float v = (static_cast<float>(next_rand(state) % 3201) / 800.0F) - 2.0F;
            const __nv_bfloat16 b = __float2bfloat16_rn(v);
            f.x_bits[static_cast<std::size_t>(t) * k + i] = *reinterpret_cast<const std::uint16_t*>(&b);
            f.x[static_cast<std::size_t>(t) * k + i] = static_cast<double>(__bfloat162float(b));
        }
    }
    const std::size_t code_bytes  = static_cast<std::size_t>(n) * k / 2;
    f.scale_off                   = (code_bytes + 255) / 256 * 256;
    const std::size_t scale_bytes = static_cast<std::size_t>(n) * k / 16;
    f.payload.assign(f.scale_off + scale_bytes + 4, 0);
    f.w.assign(static_cast<std::size_t>(n) * k, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int i = 0; i < k; ++i) {
            const unsigned nibble = next_rand(state) & 0x0Fu;
            const std::size_t byte = static_cast<std::size_t>(row) * (k / 2) + (i >> 1);
            if ((i & 1) == 0) { f.payload[byte] = static_cast<std::uint8_t>(nibble); }
            else { f.payload[byte] = static_cast<std::uint8_t>((f.payload[byte] & 0x0Fu) | (nibble << 4)); }
            f.w[static_cast<std::size_t>(row) * k + i] = e2m1_value(nibble);
        }
        for (int g = 0; g < k / 16; ++g) {
            const unsigned exponent = 6u + (next_rand(state) % 5u);
            const unsigned mantissa = next_rand(state) % 8u;
            unsigned byte           = (exponent << 3) | mantissa;
            if (next_rand(state) & 1u) { byte |= 0x80u; }
            f.payload[f.scale_off + host_scale_offset(row, g, k)] = static_cast<std::uint8_t>(byte);
            const double c = e4m3_value(byte) / static_cast<double>(kWeightScaleDivisor);
            for (int i = g * 16; i < (g + 1) * 16; ++i) {
                f.w[static_cast<std::size_t>(row) * k + i] *= c;
            }
        }
    }
    return f;
}

void* upload(const std::vector<std::uint8_t>& bytes) {
    void* d = nullptr;
    cudaMalloc(&d, bytes.size());
    cudaMemcpy(d, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
    return d;
}

void* upload_x(const std::vector<std::uint16_t>& bits) {
    void* d = nullptr;
    cudaMalloc(&d, bits.size() * 2);
    cudaMemcpy(d, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice);
    return d;
}

Weight make_weight(void* d_payload, std::size_t payload_size, std::size_t scale_off, int n, int k) {
    Weight w{};
    w.payload              = d_payload;
    w.qdata                = d_payload;
    w.scales               = static_cast<std::uint8_t*>(d_payload) + scale_off;
    w.qhigh                = nullptr;
    w.payload_bytes        = payload_size;
    w.high_plane_bytes     = 0;
    w.qtype                = QType::NVFP4;
    w.layout               = QuantLayout::BlockScaleK16M128x4;
    w.scale_dtype          = DType::FP8_E4M3FN;
    w.group              = 16;
    w.group_size         = 16;
    w.ndim               = 2;
    w.n                  = n;
    w.k                  = k;
    w.shape[0]           = n;
    w.shape[1]           = k;
    w.padded_shape[0]    = n;
    w.padded_shape[1]    = k;
    w.weight_scale_divisor = kWeightScaleDivisor;
    w.input_scale_divisor  = 1.0F;
    return w;
}

// Downloads one fused output tensor and scores it against the FP64 reference rows.
struct OutputSlot {
    void* d_out;
    int rows;         // rows of this section
    int row_base;     // first parent row of the section
    std::vector<std::uint16_t> bits;
};

bool score(const char* name, const Fixture& f, int tokens, OutputSlot& slot, double max_ulp) {
    const int k = static_cast<int>(f.x.size()) / tokens;
    slot.bits.assign(static_cast<std::size_t>(slot.rows) * tokens, 0);
    cudaMemcpy(slot.bits.data(), slot.d_out, slot.bits.size() * 2, cudaMemcpyDeviceToHost);
    double worst = 0.0;
    double max_err = 0.0;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < slot.rows; ++r) {
            const int parent_row = slot.row_base + r;
            double total = 0.0;
            const double* xcol = f.x.data() + static_cast<std::size_t>(t) * k;
            for (int i = 0; i < k; ++i) {
                total += f.w[static_cast<std::size_t>(parent_row) * k + i] * xcol[i];
            }
            const __nv_bfloat16 b =
                *reinterpret_cast<const __nv_bfloat16*>(&slot.bits[static_cast<std::size_t>(t) *
                                                                   slot.rows + r]);
            const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - total);
            const double ratio = e / budget_for(std::fabs(total), max_ulp);
            if (ratio > worst) { worst = ratio; max_err = e; }
        }
    }
    const bool ok = worst <= 1.0;
    std::printf("%-30s T=%4d rows=%6d  max|err|=%9.5f  worst=%6.2fx budget  %s\n", name, tokens,
                slot.rows, max_err, worst, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_attn(int tokens, double max_ulp) {
    const int n = 14336, k = 5120;
    Fixture f = make_fixture(n, k, tokens, 0xB16B00B5u ^ static_cast<std::uint32_t>(tokens));
    void* d_payload = upload(f.payload);
    void* d_x       = upload_x(f.x_bits);
    const int qrows = 6144, krows = 1024;
    OutputSlot q{nullptr, qrows, 0, {}}, key{nullptr, krows, 6144, {}},
        gate{nullptr, qrows, 7168, {}}, value{nullptr, krows, 13312, {}};
    cudaMalloc(&q.d_out, static_cast<std::size_t>(qrows) * tokens * 2);
    cudaMalloc(&key.d_out, static_cast<std::size_t>(krows) * tokens * 2);
    cudaMalloc(&gate.d_out, static_cast<std::size_t>(qrows) * tokens * 2);
    cudaMalloc(&value.d_out, static_cast<std::size_t>(krows) * tokens * 2);
    Tensor x(d_x, DType::BF16, {k, tokens});
    Tensor tq(q.d_out, DType::BF16, {qrows, tokens});
    Tensor tg(gate.d_out, DType::BF16, {qrows, tokens});
    Tensor tk(key.d_out, DType::BF16, {krows, tokens});
    Tensor tv(value.d_out, DType::BF16, {krows, tokens});
    Weight w = make_weight(d_payload, f.payload.size(), f.scale_off, n, k);
    bool ok = true;
    try {
        nvfp4_attn_input_dispatch(x, w, tq, tg, tk, tv, ops::LinearPolicy::AllowA4, nullptr, nullptr);
        cudaDeviceSynchronize();
        ok = score("nvfp4 attn input q", f, tokens, q, max_ulp) && ok;
        ok = score("nvfp4 attn input k", f, tokens, key, max_ulp) && ok;
        ok = score("nvfp4 attn input gate", f, tokens, gate, max_ulp) && ok;
        ok = score("nvfp4 attn input v", f, tokens, value, max_ulp) && ok;
    } catch (const std::exception& e) {
        std::printf("nvfp4 attn input T%d THREW: %s  FAIL\n", tokens, e.what());
        ok = false;
    }
    cudaFree(d_payload); cudaFree(d_x);
    cudaFree(q.d_out); cudaFree(key.d_out); cudaFree(gate.d_out); cudaFree(value.d_out);
    return ok;
}

bool run_gdn(int tokens, double max_ulp) {
    const int n = 16384, k = 5120;
    Fixture f = make_fixture(n, k, tokens, 0x9E3779B9u ^ static_cast<std::uint32_t>(tokens));
    void* d_payload = upload(f.payload);
    void* d_x       = upload_x(f.x_bits);
    const int qkvrows = 10240, zrows = 6144;
    OutputSlot qkv{nullptr, qkvrows, 0, {}}, z{nullptr, zrows, 10240, {}};
    cudaMalloc(&qkv.d_out, static_cast<std::size_t>(qkvrows) * tokens * 2);
    cudaMalloc(&z.d_out, static_cast<std::size_t>(zrows) * tokens * 2);
    Tensor x(d_x, DType::BF16, {k, tokens});
    Tensor t_qkv(qkv.d_out, DType::BF16, {qkvrows, tokens});
    Tensor t_z(z.d_out, DType::BF16, {zrows, tokens});
    Weight w = make_weight(d_payload, f.payload.size(), f.scale_off, n, k);
    bool ok = true;
    try {
        nvfp4_gdn_input_dispatch(x, w, t_qkv, t_z, ops::LinearPolicy::AllowA4, nullptr, nullptr);
        cudaDeviceSynchronize();
        ok = score("nvfp4 gdn input qkv", f, tokens, qkv, max_ulp) && ok;
        ok = score("nvfp4 gdn input z", f, tokens, z, max_ulp) && ok;
    } catch (const std::exception& e) {
        std::printf("nvfp4 gdn input T%d THREW: %s  FAIL\n", tokens, e.what());
        ok = false;
    }
    cudaFree(d_payload); cudaFree(d_x); cudaFree(qkv.d_out); cudaFree(z.d_out);
    return ok;
}

bool run_swiglu(int tokens, double max_ulp) {
    const int n = 34816, k = 5120, half = 17408;
    Fixture f = make_fixture(n, k, tokens, 0x85EBCA1Eu ^ static_cast<std::uint32_t>(tokens));
    void* d_payload = upload(f.payload);
    void* d_x       = upload_x(f.x_bits);
    OutputSlot out{nullptr, half, 0, {}};
    cudaMalloc(&out.d_out, static_cast<std::size_t>(half) * tokens * 2);
    Tensor x(d_x, DType::BF16, {k, tokens});
    Tensor t_out(out.d_out, DType::BF16, {half, tokens});
    Weight w = make_weight(d_payload, f.payload.size(), f.scale_off, n, k);
    bool ok = true;
    try {
        const std::size_t cap = nvfp4_linear_swiglu_workspace_capacity_bytes(
            ops::LinearPolicy::AllowA4, 1, tokens);
        WorkspaceArena arena(cap + 4096);
        nvfp4_linear_swiglu_dispatch(x, w, t_out, ops::LinearPolicy::AllowA4, arena, nullptr);
        cudaDeviceSynchronize();
        out.bits.assign(static_cast<std::size_t>(half) * tokens, 0);
        cudaMemcpy(out.bits.data(), out.d_out, out.bits.size() * 2, cudaMemcpyDeviceToHost);
        double worst = 0.0, max_err = 0.0;
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < half; ++r) {
                double gate = 0.0, up = 0.0;
                const double* xcol = f.x.data() + static_cast<std::size_t>(t) * k;
                const double* g = f.w.data() + static_cast<std::size_t>(r) * k;
                const double* u = f.w.data() + static_cast<std::size_t>(r + half) * k;
                for (int i = 0; i < k; ++i) { gate += g[i] * xcol[i]; up += u[i] * xcol[i]; }
                const double total = gate / (1.0 + std::exp(-gate)) * up;
                const __nv_bfloat16 b =
                    *reinterpret_cast<const __nv_bfloat16*>(&out.bits[static_cast<std::size_t>(t) * half + r]);
                const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - total);
                const double ratio = e / budget_for(std::fabs(total), max_ulp);
                if (ratio > worst) { worst = ratio; max_err = e; }
            }
        }
        ok = worst <= 1.0;
        std::printf("%-30s T=%4d rows=%6d  max|err|=%9.5f  worst=%6.2fx budget  %s\n",
                    "nvfp4 swiglu silu(gate)*up", tokens, half, max_err, worst,
                    ok ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
        std::printf("nvfp4 swiglu T%d THREW: %s  FAIL\n", tokens, e.what());
        ok = false;
    }
    cudaFree(d_payload); cudaFree(d_x); cudaFree(out.d_out);
    return ok;
}

bool run_add(int tokens, double max_ulp, int k) {
    const int n = 5120;
    Fixture f = make_fixture(n, k, tokens, 0xC0FFEE01u ^ static_cast<std::uint32_t>(k + tokens));
    void* d_payload = upload(f.payload);
    void* d_x       = upload_x(f.x_bits);
    OutputSlot res{nullptr, n, 0, {}};
    // residual starts as its own deterministic bf16 tensor; uploaded through res.bits
    std::uint32_t state = 0x5EED5EEDu;
    res.bits.assign(static_cast<std::size_t>(n) * tokens, 0);
    std::vector<double> residual(static_cast<std::size_t>(n) * tokens, 0.0);
    for (std::size_t i = 0; i < res.bits.size(); ++i) {
        const float v = (static_cast<float>(next_rand(state) % 2001) / 500.0F) - 2.0F;
        const __nv_bfloat16 b = __float2bfloat16_rn(v);
        res.bits[i]           = *reinterpret_cast<const std::uint16_t*>(&b);
        residual[i]           = static_cast<double>(__bfloat162float(b));
    }
    cudaMalloc(&res.d_out, res.bits.size() * 2);
    cudaMemcpy(res.d_out, res.bits.data(), res.bits.size() * 2, cudaMemcpyHostToDevice);
    Tensor x(d_x, DType::BF16, {k, tokens});
    Tensor t_res(res.d_out, DType::BF16, {n, tokens});
    Weight w = make_weight(d_payload, f.payload.size(), f.scale_off, n, k);
    bool ok = true;
    try {
        nvfp4_linear_add_dispatch(x, w, t_res, ops::LinearPolicy::AllowA4, nullptr, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(res.bits.data(), res.d_out, res.bits.size() * 2, cudaMemcpyDeviceToHost);
        double worst = 0.0, max_err = 0.0;
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < n; ++r) {
                double total = 0.0;
                const double* xcol = f.x.data() + static_cast<std::size_t>(t) * k;
                const double* wrow = f.w.data() + static_cast<std::size_t>(r) * k;
                for (int i = 0; i < k; ++i) { total += wrow[i] * xcol[i]; }
                total += residual[static_cast<std::size_t>(t) * n + r];
                const __nv_bfloat16 b =
                    *reinterpret_cast<const __nv_bfloat16*>(&res.bits[static_cast<std::size_t>(t) * n + r]);
                const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - total);
                const double ratio = e / budget_for(std::fabs(total), max_ulp);
                if (ratio > worst) { worst = ratio; max_err = e; }
            }
        }
        ok = worst <= 1.0;
        std::printf("%-30s T=%4d rows=%6d  max|err|=%9.5f  worst=%6.2fx budget  %s\n",
                    "nvfp4 linear_add residual", tokens, n, max_err, worst, ok ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
        std::printf("nvfp4 linear_add T%d k%d THREW: %s  FAIL\n", tokens, k, e.what());
        ok = false;
    }
    cudaFree(d_payload); cudaFree(d_x); cudaFree(res.d_out);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("nvfp4_fused_oracle: %s sm_%d%d\n", prop.name, prop.major, prop.minor);
    const char* filter = argc > 1 ? argv[1] : nullptr;
    auto maybe = [&](const char* key, bool value) {
        if (filter && std::strstr(key, filter) == nullptr) { return true; }
        return value;
    };
    std::vector<bool> results;
    // T bands chosen so every normalized route runs: T=1 decode, 2..32 small_t single pass,
    // >32 chunk loop, plus the old W4A4 entry widths (4/5/8/7) and the swiglu compose band
    // (T>16 -> LinearW4A4Post = linear + silu_mul; budget relaxed to 6 ulp there because the
    // projected BF16 intermediate is a real semantic round on that composition -- the fused
    // small-T bands at 2.5 ulp stay strict).
    results.push_back(maybe("attn1", run_attn(1, 2.5)));
    results.push_back(maybe("attn5", run_attn(5, 2.5)));
    results.push_back(maybe("attn33", run_attn(33, 2.5)));
    results.push_back(maybe("attn128", run_attn(128, 2.5)));
    results.push_back(maybe("gdn1", run_gdn(1, 2.5)));
    results.push_back(maybe("gdn8", run_gdn(8, 2.5)));
    results.push_back(maybe("gdn33", run_gdn(33, 2.5)));
    results.push_back(maybe("swiglu1", run_swiglu(1, 2.5)));
    results.push_back(maybe("swiglu4", run_swiglu(4, 2.5)));
    results.push_back(maybe("swiglu16", run_swiglu(16, 2.5)));
    results.push_back(maybe("swiglu64", run_swiglu(64, 6.0)));
    results.push_back(maybe("swiglu256", run_swiglu(256, 6.0)));
    results.push_back(maybe("add6144-1", run_add(1, 2.5, 6144)));
    results.push_back(maybe("add6144-8", run_add(8, 2.5, 6144)));
    results.push_back(maybe("add17408-1", run_add(1, 2.5, 17408)));
    results.push_back(maybe("add17408-33", run_add(33, 2.5, 17408)));
    bool all = !results.empty();
    for (const bool r : results) { all = all && r; }
    std::printf("%s\n", all ? "ALL PASS" : "FAILURES PRESENT");
    return all ? 0 : 1;
}
