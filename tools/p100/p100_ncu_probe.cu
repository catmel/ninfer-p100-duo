// Single-call kernel probe for ncu profiling of the Pascal GGML_K gemv routes on real shapes.
// Build (Linux, inside ninfer:p100-builder, repo tree mounted at /work):
//   nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -DNINFER_PASCAL_BUILD \
//        -I /work/src -I /work/include /work/tools/p100/p100_ncu_probe.cu \
//        /work/build-p100-duo/src/libninfer_ops.a /work/build-p100-duo/src/libninfer_core.a \
//        -lcudart -o /tmp/ncu_probe
// Usage: ncu_probe <shape> <T> [mode]  shape in {attn,gdn,o,gateup,down,lm}, mode gemv|gemv_multi
#include "core/tensor.h"
#include "ops/linear/ggml_k/ggml_k.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::detail;

namespace {

struct Shape { const char* name; int n; int k; };
const Shape shapes[] = {
    {"attn", 14336, 5120}, {"gdn", 16384, 5120}, {"o", 5120, 6144},
    {"gateup", 34816, 5120}, {"down", 5120, 17408}, {"lm", 248320, 5120},
};

std::uint32_t next_rand(std::uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: ncu_probe <shape> <T>\n");
        return 2;
    }
    const Shape* s = nullptr;
    for (const Shape& sh : shapes) {
        if (std::strcmp(sh.name, argv[1]) == 0) { s = &sh; break; }
    }
    if (s == nullptr) { std::printf("unknown shape\n"); return 2; }
    const int T = std::atoi(argv[2]);

    // Synthetic all-Q6_K weights: rows of k/256 blocks, each 210 B, deterministic pattern.
    const int nb = s->k / 256;
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(s->n) * nb * 210u);
    std::vector<std::uint64_t> desc(s->n);
    std::uint32_t st = 0x9E3779B9u;
    std::size_t off = 0;
    for (int r = 0; r < s->n; ++r) {
        desc[r] = (static_cast<std::uint64_t>(off) << 1) | 1ull;  // q6 flag
        for (int b = 0; b < nb; ++b) {
            std::uint8_t* blk = codes.data() + off + static_cast<std::size_t>(b) * 210u;
            for (int i = 0; i < 208; ++i) blk[i] = static_cast<std::uint8_t>(next_rand(st));
            const std::uint16_t dh = static_cast<std::uint16_t>(0x3C00 + (next_rand(st) & 0x3FF));
            std::memcpy(blk + 208, &dh, 2);
        }
        off += static_cast<std::size_t>(nb) * 210u;
    }
    std::vector<std::uint16_t> xb(static_cast<std::size_t>(s->k) * T);
    for (auto& v : xb) v = static_cast<std::uint16_t>(0x3F00 + (next_rand(st) & 0x0FFF));

    void *d_x, *d_c, *d_d, *d_o;
    cudaMalloc(&d_x, xb.size() * 2);
    cudaMalloc(&d_c, codes.size());
    cudaMalloc(&d_d, desc.size() * 8);
    cudaMalloc(&d_o, static_cast<std::size_t>(s->n) * T * 2);
    cudaMemcpy(d_x, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, desc.data(), desc.size() * 8, cudaMemcpyHostToDevice);

    Tensor x(d_x, DType::BF16, {s->k, T});
    Tensor out(d_o, DType::BF16, {s->n, T});
    Weight w{};
    w.qdata = d_c;
    w.qhigh = d_d;
    w.n = s->n;
    w.k = s->k;
    w.qtype = QType::GGML_K;
    w.layout = QuantLayout::GgmlK256;
    w.group = 256;
    w.padded_shape[0] = s->n;
    w.padded_shape[1] = s->k;
    w.ggml_k_first_q6 = -1;  // production artifact carries no hint: profile the dynamic route
    w.ggml_k_type_change = -1;

    ggml_k_project_split(x, w, &out, 1, false, nullptr, false, nullptr);
    cudaDeviceSynchronize();
    std::printf("probe %s T=%d done: %s\n", s->name, T,
                cudaGetLastError() == cudaSuccess ? "ok" : "error");
    return 0;
}
