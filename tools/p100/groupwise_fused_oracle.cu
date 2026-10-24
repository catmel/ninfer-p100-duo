// End-to-end oracle for the groupwise-int fused input-projection / MLP families on Pascal:
//   q4_q5_attn_input  tp1 (ParentSplitFixed SIMT) + tp2 column shard (rowsplit_grouped_mma
//                     SIMT body below sm_80 -- added by the Pascal migration)
//   q4_q5_gdn_input   tp1 (IndependentDirectFixed) + tp2 shard (GroupedMixedMma, SIMT body)
//   q4_linear_swiglu  tp1 (GemvPair / SmallTExact / Materialized-via-linear)
//   q5_linear_add     tp1 (GemvResidual / Split2Exact / SimtWideT) + the TP2 shard table
//                     (MmaResidual* = SIMT-backed entry points in q5_linear_add_gemm_simt.cu)
//
// Weights are synthetic Q4/Q5 row-split planes (per 64-value group: 32B nibble codes, 2B f16
// scale, 8B q5 high plane; row-major over groups) decoded independently on the host; dots in
// FP64; production dispatchers called at their admitted shapes and token bands.
//
// Build (Windows, arch-60 tree already built):
//   VsDevCmd -> nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -Xcompiler=/MD
//     -DNINFER_PASCAL_BUILD -I src -I include tools/p100/groupwise_fused_oracle.cu
//     build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"
#include "quantized_weight.h"  // tests/ops (authoritative patterned fixture + fp64 logical decode)

#include <cuda_bf16.h>

#include <functional>
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

std::uint16_t float_to_half_bits(float value) {
    const __half h = __float2half_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&h);
}
double half_to_float_bits(std::uint16_t bits) {
    return static_cast<double>(__half2float(__ushort_as_half(bits)));
}

// --- weight from the authoritative patterned fixture (tests/ops/quantized_weight.h) -----------
// Same producer the 5090/Volta split tests qualify these kernels with; the FP64 reference reads
// the packed planes back through logical_weight_fp64(), exactly as the conformance suites do.
struct IntWeight {
    ninfer::test::quantized_weight::PackedWeight packed;
    std::vector<double> w;  // n*k logical decode for the dot reference
    void* d = nullptr;

    void build(QType qtype, int rows, int cols, std::uint32_t seed) {
        ninfer::test::quantized_weight::PatternedWeightOptions options;
        options.row_split_scale         = ninfer::test::quantized_weight::RowSplitScalePattern::Small;
        options.row_split_codes         = ninfer::test::quantized_weight::RowSplitCodePattern::Coordinate;
        options.decorrelate_coordinates = true;
        packed = ninfer::test::quantized_weight::make_patterned_weight(qtype, rows, cols, seed,
                                                                       options);
        w.assign(static_cast<std::size_t>(rows) * cols, 0.0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                w[static_cast<std::size_t>(r) * cols + c] =
                    ninfer::test::quantized_weight::logical_weight_fp64(packed, r, c);
            }
        }
        cudaMalloc(&d, packed.payload.size());
        cudaMemcpy(d, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    }
    int n() const { return packed.weight.n; }
    int k() const { return packed.weight.k; }
    Weight view() const { return packed.device_weight(d); }
    void free() { cudaFree(d); }
};

// --- activations (column-major [k x tokens], k contiguous per token) --------------------------
struct Act {
    int k = 0;
    int tokens = 0;
    std::vector<std::uint16_t> bits;
    std::vector<double> d;
    void* d_ptr = nullptr;
    void build(int kk, int t, std::uint32_t seed) {
        k = kk; tokens = t;
        std::uint32_t state = seed;
        bits.assign(static_cast<std::size_t>(k) * tokens, 0);
        d.assign(bits.size(), 0.0);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const float v = (static_cast<float>(next_rand(state) % 3201) / 800.0F) - 2.0F;
            const __nv_bfloat16 b = __float2bfloat16_rn(v);
            bits[i] = *reinterpret_cast<const std::uint16_t*>(&b);
            d[i]    = static_cast<double>(__bfloat162float(b));
        }
        cudaMalloc(&d_ptr, bits.size() * 2);
        cudaMemcpy(d_ptr, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice);
    }
    void free() { cudaFree(d_ptr); }
};

double dot(const IntWeight& w, const Act& a, int parent_row, int t) {
    const double* wrow = w.w.data() + static_cast<std::size_t>(parent_row) * w.k();
    const double* xcol = a.d.data() + static_cast<std::size_t>(t) * a.k;
    double total = 0.0;
    for (int i = 0; i < w.k(); ++i) { total += wrow[i] * xcol[i]; }
    return total;
}

void* new_out(std::size_t elems) {
    void* d = nullptr;
    cudaMalloc(&d, elems * 2);
    cudaMemset(d, 0xFF, elems * 2);  // poison: unwritten rows show up as NaN
    return d;
}

// Score a [rows x tokens] section living inside a column-major tensor whose COLUMN STRIDE is
// out_ld (qkv is shared by two sections, so out_ld can exceed the section's own rows).
// section output rows [out_row_offset, out_row_offset+rows); weight rows [weight_base, +rows).
bool score_section(const char* name, const IntWeight& w, const Act& a, void* d_out, int rows,
                   int weight_base, int out_row_offset, int out_ld, double max_ulp) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(out_ld) * a.tokens);
    cudaMemcpy(bits.data(), d_out, bits.size() * 2, cudaMemcpyDeviceToHost);
    double worst = 0.0, max_err = 0.0;
    int printed = 0;
    const bool dbg = std::getenv("ORACLE_DEBUG") != nullptr;
    for (int t = 0; t < a.tokens; ++t) {
        for (int r = 0; r < rows; ++r) {
            const double ref = dot(w, a, weight_base + r, t);
            const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(
                &bits[static_cast<std::size_t>(t) * out_ld + out_row_offset + r]);
            const double got = static_cast<double>(__bfloat162float(b));
            const double e = std::fabs(got - ref);
            const double ratio = e / budget_for(std::fabs(ref), max_ulp);
            if (ratio > worst) { worst = ratio; max_err = e; }
            if (dbg && ratio > 1.0 && printed < 12) {
                std::printf("  bad (row=%d tok=%d) got=%.4f ref=%.4f\n", r, t, got, ref);
                ++printed;
            }
        }
    }
    const bool ok = worst <= 1.0;
    std::printf("%-34s T=%4d rows=%6d  max|err|=%9.5f  worst=%6.2fx  %s\n", name, a.tokens, rows,
                max_err, worst, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_attn(int parent_rows, int qrows, int kvrows, int tokens, bool shard, std::uint32_t seed) {
    const int k = 5120;
    Act a; a.build(k, tokens, seed);
    IntWeight qk, gv;
    qk.build(QType::Q4G64_F16S, parent_rows, k, seed ^ 0x1111u);
    gv.build(QType::Q5G64_F16S, parent_rows, k, seed ^ 0x2222u);
    void *dq = new_out(static_cast<std::size_t>(qrows) * tokens);
    void *dg = new_out(static_cast<std::size_t>(qrows) * tokens);
    void *dk = new_out(static_cast<std::size_t>(kvrows) * tokens);
    void *dv = new_out(static_cast<std::size_t>(kvrows) * tokens);
    Tensor x(a.d_ptr, DType::BF16, {k, tokens});
    Tensor tq(dq, DType::BF16, {qrows, tokens});
    Tensor tg(dg, DType::BF16, {qrows, tokens});
    Tensor tk(dk, DType::BF16, {kvrows, tokens});
    Tensor tv(dv, DType::BF16, {kvrows, tokens});
    const std::size_t cap = shard
                                ? q4_q5_attn_input_shard_capacity_workspace_bytes(1, tokens)
                                : q4_q5_attn_input_capacity_workspace_bytes(1, tokens);
    WorkspaceArena arena(cap + (1u << 20));
    bool ok = true;
    try {
        if (shard) {
            q4_q5_attn_input_dispatch_shard(x, qk.view(), gv.view(), tq, tg, tk, tv, arena,
                                            nullptr);
        } else {
            q4_q5_attn_input_dispatch(x, qk.view(), gv.view(), tq, tg, tk, tv, arena, nullptr);
        }
        cudaDeviceSynchronize();
        if (std::getenv("ORACLE_DEBUG") && !shard && tokens == 1) {
            std::vector<std::uint16_t> g(qrows);
            cudaMemcpy(g.data(), dg, qrows * 2, cudaMemcpyDeviceToHost);
            std::printf("  probe gate rows0..3: got=%.4f %.4f %.4f %.4f  ref=%.4f %.4f %.4f %.4f\n",
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&g[0])),
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&g[1])),
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&g[2])),
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&g[3])),
                        dot(gv, a, 0, 0), dot(gv, a, 1, 0), dot(gv, a, 2, 0), dot(gv, a, 3, 0));
            std::vector<std::uint16_t> qq(qrows);
            cudaMemcpy(qq.data(), dq, qrows * 2, cudaMemcpyDeviceToHost);
            std::printf("  probe q    rows0..3: got=%.4f %.4f %.4f %.4f  ref=%.4f %.4f %.4f %.4f\n",
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&qq[0])),
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&qq[1])),
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&qq[2])),
                        __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&qq[3])),
                        dot(qk, a, 0, 0), dot(qk, a, 1, 0), dot(qk, a, 2, 0), dot(qk, a, 3, 0));
        }
        ok = score_section("attn q", qk, a, dq, qrows, 0, 0, qrows, 2.5) && ok;
        ok = score_section("attn k", qk, a, dk, kvrows, qrows, 0, kvrows, 2.5) && ok;
        ok = score_section("attn gate", gv, a, dg, qrows, 0, 0, qrows, 2.5) && ok;
        ok = score_section("attn v", gv, a, dv, kvrows, qrows, 0, kvrows, 2.5) && ok;
    } catch (const std::exception& e) {
        std::printf("attn %s T%d THREW: %s  FAIL\n", shard ? "shard" : "tp1", tokens, e.what());
        ok = false;
    }
    a.free(); qk.free(); gv.free();
    cudaFree(dq); cudaFree(dg); cudaFree(dk); cudaFree(dv);
    return ok;
}

bool run_gdn(int tokens, bool shard, std::uint32_t seed) {
    const int k = 5120;
    const int qk_rows = shard ? 2048 : 4096;
    const int vz_rows = shard ? 6144 : 12288;
    const int z_rows  = vz_rows / 2;
    const int qkv_rows = qk_rows + z_rows;
    Act a; a.build(k, tokens, seed);
    IntWeight qk_w, vz_w;
    qk_w.build(QType::Q4G64_F16S, qk_rows, k, seed ^ 0x3333u);
    vz_w.build(QType::Q5G64_F16S, vz_rows, k, seed ^ 0x4444u);
    void* dqkv = new_out(static_cast<std::size_t>(qkv_rows) * tokens);
    void* dz   = new_out(static_cast<std::size_t>(z_rows) * tokens);
    Tensor x(a.d_ptr, DType::BF16, {k, tokens});
    Tensor t_qkv(dqkv, DType::BF16, {qkv_rows, tokens});
    Tensor t_z(dz, DType::BF16, {z_rows, tokens});
    const std::size_t cap = q4_q5_gdn_input_capacity_workspace_bytes(1, tokens);
    WorkspaceArena arena(cap + (1u << 20));
    bool ok = true;
    try {
        q4_q5_gdn_input_dispatch(x, qk_w.view(), vz_w.view(), t_qkv, t_z, arena, nullptr);
        cudaDeviceSynchronize();
        if (std::getenv("ORACLE_DEBUG")) {
            std::vector<std::uint16_t> head(static_cast<std::size_t>(qkv_rows) * a.tokens);
            cudaMemcpy(head.data(), dqkv, head.size() * 2, cudaMemcpyDeviceToHost);
            auto at = [&](int row, int t) {
                return static_cast<double>(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                    &head[static_cast<std::size_t>(t) * qkv_rows + row])));
            };
            for (int t = 0; t < std::min(a.tokens, 3); ++t) {
                std::printf("  probe gdn qk row0 t%d: got(ld)=%.4f ref=%.4f got(flat)=%.4f\n", t,
                            at(0, t), dot(qk_w, a, 0, t),
                            static_cast<double>(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                                &head[static_cast<std::size_t>(t) * qk_rows]))));
            }
        }
        // grouped jobs: qk -> qkv rows [0, qk_rows); value = vz[0, vz/2) -> qkv [qk_rows, ...);
        // z = vz[vz/2, vz) -> z tensor. Column-major sections: row offset = offset in the
        // shared qkv tensor; the column stride is the FULL qkv row count.
        ok = score_section("gdn qk", qk_w, a, dqkv, qk_rows, 0, 0, qkv_rows, 2.5) && ok;
        ok = score_section("gdn value", vz_w, a, dqkv, z_rows, 0, qk_rows, qkv_rows, 2.5) && ok;
        ok = score_section("gdn z", vz_w, a, dz, z_rows, z_rows, 0, z_rows, 2.5) && ok;
    } catch (const std::exception& e) {
        std::printf("gdn %s T%d THREW: %s  FAIL\n", shard ? "shard" : "tp1", tokens, e.what());
        ok = false;
    }
    a.free(); qk_w.free(); vz_w.free();
    cudaFree(dqkv); cudaFree(dz);
    return ok;
}

bool run_swiglu(int tokens, std::uint32_t seed) {
    const int n = 34816, k = 5120, half = 17408;
    Act a; a.build(k, tokens, seed);
    IntWeight gate_up; gate_up.build(QType::Q4G64_F16S, n, k, seed ^ 0x5555u);
    void* dout = new_out(static_cast<std::size_t>(half) * tokens);
    Tensor x(a.d_ptr, DType::BF16, {k, tokens});
    Tensor t_out(dout, DType::BF16, {half, tokens});
    const std::size_t cap =
        q4_linear_swiglu_capacity_workspace_bytes(n, half, k, k, 1, tokens);
    WorkspaceArena arena(cap + (1u << 20));
    bool ok = true;
    try {
        q4_linear_swiglu_dispatch(x, gate_up.view(), t_out, arena, nullptr);
        cudaDeviceSynchronize();
        // The Materialized band (>32) materializes both projections to BF16 before silu*mul, so
        // that observable intermediate round widens the tolerance (same justification as the
        // nvfp4 compose band); the fused bands stay at 2.5 ulp.
        const double ulp = (tokens > 32) ? 6.0 : 2.5;
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(half) * tokens);
        cudaMemcpy(bits.data(), dout, bits.size() * 2, cudaMemcpyDeviceToHost);
        double worst = 0.0, max_err = 0.0;
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < half; ++r) {
                const double gate = dot(gate_up, a, r, t);
                const double up   = dot(gate_up, a, r + half, t);
                const double ref  = gate / (1.0 + std::exp(-gate)) * up;
                const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(
                    &bits[static_cast<std::size_t>(t) * half + r]);
                const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - ref);
                const double ratio = e / budget_for(std::fabs(ref), ulp);
                if (ratio > worst) { worst = ratio; max_err = e; }
            }
        }
        ok = worst <= 1.0;
        std::printf("%-34s T=%4d rows=%6d  max|err|=%9.5f  worst=%6.2fx  %s\n",
                    "swiglu q4 silu(gate)*up", tokens, half, max_err, worst, ok ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
        std::printf("swiglu q4 T%d THREW: %s  FAIL\n", tokens, e.what());
        ok = false;
    }
    a.free(); gate_up.free(); cudaFree(dout);
    return ok;
}

bool run_add_q5(int tokens, int k, bool shard, std::uint32_t seed) {
    const int n = 5120;
    Act a; a.build(k, tokens, seed);
    IntWeight w; w.build(QType::Q5G64_F16S, n, k, seed ^ 0x6666u);
    std::uint32_t state = seed ^ 0x7777u;
    std::vector<std::uint16_t> res_bits(static_cast<std::size_t>(n) * tokens);
    std::vector<double> res(res_bits.size(), 0.0);
    for (std::size_t i = 0; i < res_bits.size(); ++i) {
        const float v = (static_cast<float>(next_rand(state) % 2001) / 500.0F) - 2.0F;
        const __nv_bfloat16 b = __float2bfloat16_rn(v);
        res_bits[i] = *reinterpret_cast<const std::uint16_t*>(&b);
        res[i]      = static_cast<double>(__bfloat162float(b));
    }
    void* dres = new_out(res_bits.size());
    cudaMemcpy(dres, res_bits.data(), res_bits.size() * 2, cudaMemcpyHostToDevice);
    Tensor x(a.d_ptr, DType::BF16, {k, tokens});
    Tensor t_res(dres, DType::BF16, {n, tokens});
    bool ok = true;
    try {
        q5_linear_add_dispatch(x, w.view(), t_res, nullptr, nullptr);  // ws never read (plan.h)
        cudaDeviceSynchronize();
        std::vector<std::uint16_t> bits(res_bits.size());
        cudaMemcpy(bits.data(), dres, bits.size() * 2, cudaMemcpyDeviceToHost);
        double worst = 0.0, max_err = 0.0;
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < n; ++r) {
                const std::size_t idx = static_cast<std::size_t>(t) * n + r;
                const double ref = dot(w, a, r, t) + res[idx];
                const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&bits[idx]);
                const double e = std::fabs(static_cast<double>(__bfloat162float(b)) - ref);
                const double ratio = e / budget_for(std::fabs(ref), 2.5);
                if (ratio > worst) { worst = ratio; max_err = e; }
            }
        }
        ok = worst <= 1.0;
        std::printf("%-34s T=%4d  k=%6d  max|err|=%9.5f  worst=%6.2fx  %s\n",
                    shard ? "linear_add q5 tp2 shard" : "linear_add q5 tp1", tokens, k, max_err,
                    worst, ok ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
        std::printf("add q5 %s T%d k%d THREW: %s  FAIL\n", shard ? "shard" : "tp1", tokens, k,
                    e.what());
        ok = false;
    }
    a.free(); w.free(); cudaFree(dres);
    return ok;
}

} // namespace

// --- bf16 (A16) wide-T routes: PASCAL SIMT fallback family ------------------------------------
// Plain row-major [n, k] bf16 weights; the reference is an exact FP64 dot on the stored
// (bf16-rounded) values. Covers bf16_attn_input_dispatch above its SIMT small-T band, the
// bf16 linear dispatch (14336/7168 x 5120 + 5120 x 6144), and the bf16 linear_add SIMT route.
struct Bf16Dense {
    std::vector<double> w;  // n*k stored-precision values
    void* d = nullptr;
    int n_ = 0;
    int k_ = 0;
    void build(int rows, int cols, std::uint32_t seed) {
        n_ = rows; k_ = cols;
        std::uint32_t state = seed;
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * cols);
        w.assign(bits.size(), 0.0);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const float v = (static_cast<float>(next_rand(state) % 8001) - 4000.0F) / 200000.0F;
            const __nv_bfloat16 b = __float2bfloat16_rn(v);
            bits[i] = *reinterpret_cast<const std::uint16_t*>(&b);
            w[i]    = static_cast<double>(__bfloat162float(b));
        }
        cudaMalloc(&d, bits.size() * 2);
        cudaMemcpy(d, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice);
    }
    Weight view() const {
        Weight wv{};
        wv.payload        = d;
        wv.payload_bytes  = w.size() * 2;
        wv.qdata          = d;
        wv.n              = n_;
        wv.k              = k_;
        wv.ndim           = 2;
        wv.shape[0]        = n_;
        wv.shape[1]        = k_;
        wv.padded_shape[0] = n_;
        wv.padded_shape[1] = k_;
        return wv;
    }
    void free() { cudaFree(d); }
};

// extra (optional) = host residual terms for linear_add (same [rows x tokens] column-major).
bool score_dense(const char* name, const Bf16Dense& w, const Act& a, void* d_out, int rows,
                 int weight_base, const double* extra, double max_ulp) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * a.tokens);
    cudaMemcpy(bits.data(), d_out, bits.size() * 2, cudaMemcpyDeviceToHost);
    double worst = 0.0, max_err = 0.0;
    int printed = 0;
    const bool dbg = std::getenv("ORACLE_DEBUG") != nullptr;
    for (int t = 0; t < a.tokens; ++t) {
        const double* xcol = a.d.data() + static_cast<std::size_t>(t) * a.k;
        for (int r = 0; r < rows; ++r) {
            const double* wrow = w.w.data() + static_cast<std::size_t>(weight_base + r) * w.k_;
            double ref = 0.0;
            for (int i = 0; i < w.k_; ++i) { ref += wrow[i] * xcol[i]; }
            if (extra) { ref += extra[static_cast<std::size_t>(t) * rows + r]; }
            const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(
                &bits[static_cast<std::size_t>(t) * rows + r]);
            const double got = static_cast<double>(__bfloat162float(b));
            const double e = std::fabs(got - ref);
            const double ratio = e / budget_for(std::fabs(ref), max_ulp);
            if (ratio > worst) { worst = ratio; max_err = e; }
            if (dbg && ratio > 1.0 && printed < 6) {
                std::printf("  bad (row=%d tok=%d) got=%.4f ref=%.4f dot=%.4f resid=%.4f\n", r, t,
                            got, ref, ref - (extra ? extra[static_cast<std::size_t>(t) * rows + r] : 0.0),
                            extra ? extra[static_cast<std::size_t>(t) * rows + r] : 0.0);
                ++printed;
            }
        }
    }
    std::printf("%-34s T=%4d rows=%6d max|err|=%10.5f worst=%7.2fx  %s\n", name, a.tokens, rows,
                max_err, worst, worst <= 1.0 ? "PASS" : "FAIL");
    return worst <= 1.0;
}

bool run_bf16_attn(int tokens, std::uint32_t seed) {
    const int k = 5120;
    Act a; a.build(k, tokens, seed);
    Bf16Dense w; w.build(14336, k, seed ^ 0x7777u);
    void* dq = new_out(6144 * tokens); void* dk = new_out(1024 * tokens);
    void* dg = new_out(6144 * tokens); void* dv = new_out(1024 * tokens);
    Tensor x(a.d_ptr, DType::BF16, {k, tokens});
    Tensor tq(dq, DType::BF16, {6144, tokens}), tk(dk, DType::BF16, {1024, tokens});
    Tensor tg(dg, DType::BF16, {6144, tokens}), tv(dv, DType::BF16, {1024, tokens});
    bool ok = true;
    try {
        bf16_attn_input_dispatch(x, w.view(), tq, tg, tk, tv, nullptr);
        cudaDeviceSynchronize();
        ok = score_dense("bf16 attn q", w, a, dq, 6144, 0, nullptr, 2.5) && ok;
        ok = score_dense("bf16 attn k", w, a, dk, 1024, 6144, nullptr, 2.5) && ok;
        ok = score_dense("bf16 attn gate", w, a, dg, 6144, 7168, nullptr, 2.5) && ok;
        ok = score_dense("bf16 attn v", w, a, dv, 1024, 13312, nullptr, 2.5) && ok;
    } catch (const std::exception& e) {
        std::printf("  THREW: %s\n", e.what());
        ok = false;
    }
    cudaFree(dq); cudaFree(dk); cudaFree(dg); cudaFree(dv);
    a.free(); w.free();
    return ok;
}

bool run_bf16_linear(int rows, int k, int tokens, std::uint32_t seed) {
    Act a; a.build(k, tokens, seed);
    Bf16Dense w; w.build(rows, k, seed ^ 0x8888u);
    void* dout = new_out(static_cast<std::size_t>(rows) * tokens);
    Tensor x(a.d_ptr, DType::BF16, {k, tokens});
    Tensor out(dout, DType::BF16, {rows, tokens});
    bool ok = true;
    try {
        bf16_dispatch(x, w.view(), out, ops::LinearPolicy::A16Only, nullptr);
        cudaDeviceSynchronize();
        char name[64];
        std::snprintf(name, sizeof(name), "bf16 linear %dx%d", rows, k);
        ok = score_dense(name, w, a, dout, rows, 0, nullptr, 2.5);
    } catch (const std::exception& e) {
        std::printf("  THREW: %s\n", e.what());
        ok = false;
    }
    cudaFree(dout);
    a.free(); w.free();
    return ok;
}

bool run_bf16_add(int tokens, std::uint32_t seed) {
    const int k = 6144;
    const int rows = 5120;
    Act a; a.build(k, tokens, seed);
    Bf16Dense w; w.build(rows, k, seed ^ 0x9999u);
    std::uint32_t state = seed ^ 0xABABu;
    std::vector<std::uint16_t> rbits(static_cast<std::size_t>(rows) * tokens);
    std::vector<double> rhost(rbits.size());
    for (std::size_t i = 0; i < rbits.size(); ++i) {
        const float v = (static_cast<float>(next_rand(state) % 3201) / 400.0F) - 4.0F;
        const __nv_bfloat16 b = __float2bfloat16_rn(v);
        rbits[i] = *reinterpret_cast<const std::uint16_t*>(&b);
        rhost[i] = static_cast<double>(__bfloat162float(b));
    }
    void* dres = nullptr;
    cudaMalloc(&dres, rbits.size() * 2);
    cudaMemcpy(dres, rbits.data(), rbits.size() * 2, cudaMemcpyHostToDevice);
    Tensor x(a.d_ptr, DType::BF16, {k, tokens});
    Tensor res(dres, DType::BF16, {rows, tokens});
    bool ok = true;
    try {
        bf16_linear_add_dispatch(x, w.view(), res, nullptr);
        cudaDeviceSynchronize();
        ok = score_dense("bf16 linear_add residual", w, a, dres, rows, 0, rhost.data(), 2.5);
    } catch (const std::exception& e) {
        std::printf("  THREW: %s\n", e.what());
        ok = false;
    }
    cudaFree(dres);
    a.free(); w.free();
    return ok;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("groupwise_fused_oracle: %s sm_%d%d\n", prop.name, prop.major, prop.minor);
    std::fflush(stdout);
    const char* filter = argc > 1 ? argv[1] : nullptr;
    std::vector<bool> results;
    const std::uint32_t base = 0xC0FFEE00u;
    auto maybe = [&](const char* key, const std::function<bool()>& fn) {
        if (filter && std::strstr(key, filter) == nullptr) { return true; }
        std::printf("--- %s\n", key);
        const cudaError_t pre = cudaGetLastError();
        if (pre != cudaSuccess) { std::printf("pre-existing cuda error: %s\n", cudaGetErrorName(pre)); }
        const bool v = fn();
        const cudaError_t post = cudaGetLastError();
        if (post != cudaSuccess) { std::printf("[%s] post dispatch cuda error: %s\n", key, cudaGetErrorName(post)); }
        return v;
    };
    // tp1 attn (ParentSplitFixed) and tp2 shard (grouped SIMT body below sm_80)
    results.push_back(maybe("attn-tp1-T1", [&] { return run_attn(7168, 6144, 1024, 1, false, base); }));
    results.push_back(maybe("attn-tp1-T5", [&] { return run_attn(7168, 6144, 1024, 5, false, base + 1); }));
    results.push_back(maybe("attn-tp1-T17", [&] { return run_attn(7168, 6144, 1024, 17, false, base + 2); }));
    results.push_back(maybe("attn-tp1-T33", [&] { return run_attn(7168, 6144, 1024, 33, false, base + 3); }));
    results.push_back(maybe("attn-shard-T1", [&] { return run_attn(3584, 3072, 512, 1, true, base + 4); }));
    results.push_back(maybe("attn-shard-T33", [&] { return run_attn(3584, 3072, 512, 33, true, base + 5); }));
    // gdn tp1 + shard
    results.push_back(maybe("gdn-tp1-T1", [&] { return run_gdn(1, false, base + 6); }));
    results.push_back(maybe("gdn-tp1-T33", [&] { return run_gdn(33, false, base + 7); }));
    results.push_back(maybe("gdn-shard-T1", [&] { return run_gdn(1, true, base + 8); }));
    results.push_back(maybe("gdn-shard-T9", [&] { return run_gdn(9, true, base + 9); }));
    // swiglu q4 bands (GemvPair / SmallTExact / Materialized)
    results.push_back(maybe("swiglu-T1", [&] { return run_swiglu(1, base + 10); }));
    results.push_back(maybe("swiglu-T16", [&] { return run_swiglu(16, base + 11); }));
    results.push_back(maybe("swiglu-T33", [&] { return run_swiglu(33, base + 12); }));
    results.push_back(maybe("swiglu-T256", [&] { return run_swiglu(256, base + 13); }));
    // q5 linear_add tp1 bands + tp2 shard table
    results.push_back(maybe("add-T1-k6144", [&] { return run_add_q5(1, 6144, false, base + 14); }));
    results.push_back(maybe("add-T13-k6144", [&] { return run_add_q5(13, 6144, false, base + 15); }));
    results.push_back(maybe("add-T17-k6144", [&] { return run_add_q5(17, 6144, false, base + 16); }));
    results.push_back(maybe("add-T64-k17408", [&] { return run_add_q5(64, 17408, false, base + 17); }));
    results.push_back(maybe("add-shard-T1-k3072", [&] { return run_add_q5(1, 3072, true, base + 18); }));
    results.push_back(maybe("add-shard-T40-k8704", [&] { return run_add_q5(40, 8704, true, base + 19); }));

    results.push_back(maybe("bf16-attn-T23", [&] { return run_bf16_attn(23, base + 20); }));
    results.push_back(maybe("bf16-attn-T33", [&] { return run_bf16_attn(33, base + 21); }));
    results.push_back(maybe("bf16-lin-T40", [&] { return run_bf16_linear(14336, 5120, 40, base + 22); }));
    results.push_back(maybe("bf16-lin-shard-T40", [&] { return run_bf16_linear(7168, 5120, 40, base + 23); }));
    results.push_back(maybe("bf16-lin-o-T40", [&] { return run_bf16_linear(5120, 6144, 40, base + 24); }));
    results.push_back(maybe("bf16-add-T1", [&] { return run_bf16_add(1, base + 25); }));
    results.push_back(maybe("bf16-add-T2", [&] { return run_bf16_add(2, base + 26); }));
    results.push_back(maybe("bf16-add-T9", [&] { return run_bf16_add(9, base + 27); }));
    bool all = !results.empty();
    for (const bool r : results) { all = all && r; }
    std::printf("%s\n", all ? "ALL PASS" : "FAILURES PRESENT");
    return all ? 0 : 1;
}
