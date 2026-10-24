// Oracle check for the Pascal SIMT fallback body of the vision attention flash kernel.
//
// Produces deterministic BF16 q/k/v at the registered [72,16,P] layout (including a padded
// token-stride case), evaluates packed (cu_seqlens) and uniform-segment routes with uneven
// and partial-tile segments, and compares the BF16 device output against a naive FP64
// softmax(QK^T/sqrt(72))V reference with the same mixed budget as the linear oracle:
// max(1.0, 2.5 * bf16_ulp(|ref|)).
//
// Build (Windows, arch-60 tree already built):
//   VsDevCmd -> nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -Xcompiler=/MD
//     -DNINFER_PASCAL_BUILD -I src -I include tools/p100/vision_attention_oracle.cu
//     build-p100/src/ninfer_ops.lib build-p100/src/ninfer_core.lib
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/vision_attention.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

constexpr int kD    = 72;
constexpr int kH    = 16;

std::uint32_t next_rand(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

double bf16_ulp(double magnitude) {
    if (!(magnitude > 0.0)) { return 0.0; }
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    return std::ldexp(1.0, exponent - 8);
}

double budget_for(double reference_magnitude, double ulp_multiple) {
    const double budget = ulp_multiple * bf16_ulp(reference_magnitude);
    return budget < 1.0 ? 1.0 : budget;
}

#define CUDA_OK(call)                                                                \
    do {                                                                             \
        const cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                                  \
            std::fprintf(stderr, "cuda failure at %d: %s\n", __LINE__,               \
                         cudaGetErrorString(err__));                                 \
            std::exit(2);                                                            \
        }                                                                            \
    } while (0)

struct Inputs {
    std::vector<std::uint16_t> q;  // [P][H][D] packed feature-major, token outer
    std::vector<std::uint16_t> k;
    std::vector<std::uint16_t> v;
    std::int32_t patches = 0;
    std::int64_t token_stride_elems = 0;  // elements per token in device layout
};

// Feature-major storage: index = d + h*D + t*stride, stride = kD*kH (or padded).
std::size_t elem_index(const Inputs& in, int d, int h, int t) {
    return static_cast<std::size_t>(d) + static_cast<std::size_t>(h) * kD +
           static_cast<std::size_t>(t) * in.token_stride_elems;
}

Inputs make_inputs(std::uint32_t seed, std::int32_t patches, std::int64_t token_stride_elems) {
    Inputs in;
    in.patches             = patches;
    in.token_stride_elems  = token_stride_elems;
    const std::size_t total =
        static_cast<std::size_t>(token_stride_elems) * static_cast<std::size_t>(patches);
    in.q.resize(total);
    in.k.resize(total);
    in.v.resize(total);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < total; ++i) {
        const auto unit = [&state, i](std::vector<std::uint16_t>& dst) {
            const double value = (static_cast<double>(next_rand(state) % 2001) - 1000.0) / 500.0;
            dst[i] = [](__nv_bfloat16 b) {
                return *reinterpret_cast<const std::uint16_t*>(&b);
            }(__float2bfloat16_rn(static_cast<float>(value)));
        };
        unit(in.q);
        unit(in.k);
        unit(in.v);
    }
    return in;
}

void upload(const Inputs& in, void** dq, void** dk, void** dv) {
    const std::size_t bytes = in.token_stride_elems * static_cast<std::size_t>(in.patches) * 2;
    CUDA_OK(cudaMalloc(dq, bytes));
    CUDA_OK(cudaMalloc(dk, bytes));
    CUDA_OK(cudaMalloc(dv, bytes));
    CUDA_OK(cudaMemcpy(*dq, in.q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(*dk, in.k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(*dv, in.v.data(), bytes, cudaMemcpyHostToDevice));
}

Tensor view(void* data, const Inputs& in, std::int32_t patches) {
    Tensor t;
    t.data         = data;
    t.dtype        = DType::BF16;
    t.ne[0]        = kD;
    t.ne[1]        = kH;
    t.ne[2]        = patches;
    t.ne[3]        = 1;
    t.nb[0]        = 2;
    t.nb[1]        = 2 * kD;
    t.nb[2]        = 2 * in.token_stride_elems;
    t.nb[3]        = t.nb[2] * patches;
    return t;
}

// Naive FP64 reference over the represented BF16 inputs.
std::vector<double> reference(const Inputs& in, const std::vector<std::int32_t>& bounds) {
    std::vector<double> out(static_cast<std::size_t>(kD) * kH * in.patches, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kD));
    for (std::size_t s = 0; s + 1 < bounds.size(); ++s) {
        const int begin = bounds[s];
        const int end   = bounds[s + 1];
        for (int h = 0; h < kH; ++h) {
            std::vector<double> score(end - begin), prob(end - begin);
            for (int i = begin; i < end; ++i) {
                double max_score = -1.0e300;
                for (int j = begin; j < end; ++j) {
                    double acc = 0.0;
                    for (int d = 0; d < kD; ++d) {
                        const double qv = __bfloat162float(
                            *reinterpret_cast<const __nv_bfloat16*>(&in.q[elem_index(in, d, h, i)]));
                        const double kv = __bfloat162float(
                            *reinterpret_cast<const __nv_bfloat16*>(&in.k[elem_index(in, d, h, j)]));
                        acc += qv * kv;
                    }
                    score[j - begin] = acc * scale;
                    if (score[j - begin] > max_score) { max_score = score[j - begin]; }
                }
                double sum = 0.0;
                for (int j = begin; j < end; ++j) {
                    prob[j - begin] = std::exp(score[j - begin] - max_score);
                    sum += prob[j - begin];
                }
                for (int j = begin; j < end; ++j) { prob[j - begin] /= sum; }
                for (int d = 0; d < kD; ++d) {
                    double acc = 0.0;
                    for (int j = begin; j < end; ++j) {
                        const double vv = __bfloat162float(
                            *reinterpret_cast<const __nv_bfloat16*>(&in.v[elem_index(in, d, h, j)]));
                        acc += prob[j - begin] * vv;
                    }
                    out[static_cast<std::size_t>(d) + static_cast<std::size_t>(h) * kD +
                        static_cast<std::size_t>(i) * kD * kH] = acc;
                }
            }
        }
    }
    return out;
}

bool run_case(const char* name, std::uint32_t seed, std::int32_t patches, std::int64_t pad_elems,
              const std::vector<std::int32_t>* cu_bounds, std::int32_t uniform_segment,
              double ulp_multiple) {
    Inputs in = make_inputs(seed, patches, static_cast<std::int64_t>(kD) * kH + pad_elems);
    void *dq = nullptr, *dk = nullptr, *dv = nullptr;
    upload(in, &dq, &dk, &dv);

    Tensor q = view(dq, in, patches);
    Tensor k = view(dk, in, patches);
    Tensor v = view(dv, in, patches);

    std::vector<std::int32_t> bounds;
    if (cu_bounds != nullptr) {
        bounds = *cu_bounds;
    } else {
        for (std::int32_t b = 0; b <= patches; b += uniform_segment) { bounds.push_back(b); }
    }

    Tensor out = view(nullptr, in, patches);
    const std::size_t out_bytes = static_cast<std::size_t>(kD) * kH * patches * 2;
    void* dout                   = nullptr;
    CUDA_OK(cudaMalloc(&dout, out_bytes));
    CUDA_OK(cudaMemset(dout, 0, out_bytes));
    out.data = dout;
    Tensor out_contig = out;
    out_contig.nb[2]  = 2 * kD * kH;  // out must be contiguous per the contract
    out_contig.nb[3]  = out_contig.nb[2] * patches;

    const std::int32_t segments = static_cast<std::int32_t>(bounds.size()) - 1;
    const std::size_t capacity  = vision_attention_workspace_capacity_bytes(
        patches, patches, segments, segments);
    WorkspaceArena workspace(capacity + 4096);
    CUDA_OK(cudaDeviceSynchronize());

    if (cu_bounds != nullptr) {
        Tensor cu;
        std::vector<std::int32_t> host_cu = bounds;
        void* dcu                          = nullptr;
        CUDA_OK(cudaMalloc(&dcu, host_cu.size() * sizeof(std::int32_t)));
        CUDA_OK(cudaMemcpy(dcu, host_cu.data(), host_cu.size() * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice));
        cu.data  = dcu;
        cu.dtype = DType::I32;
        cu.ne[0] = static_cast<std::int32_t>(host_cu.size());
        cu.ne[1] = cu.ne[2] = cu.ne[3] = 1;
        cu.nb[0] = 4;
        cu.nb[1] = cu.nb[2] = cu.nb[3] =
            static_cast<std::int64_t>(host_cu.size()) * 4;
        vision_attention(q, k, v, cu, workspace, out_contig, nullptr);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaFree(dcu));
    } else {
        vision_attention(q, k, v, uniform_segment, workspace, out_contig, nullptr);
        CUDA_OK(cudaDeviceSynchronize());
    }

    std::vector<std::uint16_t> got(static_cast<std::size_t>(kD) * kH * patches);
    CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));

    const std::vector<double> ref = reference(in, bounds);

    double worst_ratio = 0.0;
    double err_at_worst = 0.0;
    int worst_t = -1, worst_h = -1, worst_d = -1;
    double worst_ref = 0.0, worst_got = 0.0;
    for (int t = 0; t < patches; ++t) {
        for (int h = 0; h < kH; ++h) {
            for (int d = 0; d < kD; ++d) {
                const std::size_t flat =
                    static_cast<std::size_t>(d) + static_cast<std::size_t>(h) * kD +
                    static_cast<std::size_t>(t) * kD * kH;
                const double g = __bfloat162float(
                    *reinterpret_cast<const __nv_bfloat16*>(&got[flat]));
                const double e = std::fabs(g - ref[flat]);
                const double ratio = e / budget_for(std::fabs(ref[flat]), ulp_multiple);
                if (ratio > worst_ratio) {
                    worst_ratio = ratio;
                    err_at_worst = e;
                    worst_t      = t;
                    worst_h      = h;
                    worst_d      = d;
                    worst_ref    = ref[flat];
                    worst_got    = g;
                }
            }
        }
    }

    const bool pass = worst_ratio <= 1.0;
    std::printf("%-24s P=%5d segs=%3d pad=%2d  worst=%6.2fx budget  max|err|=%8.5f  %s\n", name,
                patches, segments, static_cast<int>(pad_elems), worst_ratio, err_at_worst,
                pass ? "PASS" : "FAIL");
    if (!pass && std::getenv("ORACLE_DEBUG") != nullptr) {
        std::fprintf(stderr, "  worst at t=%d h=%d d=%d  got=%.6f ref=%.6f\n", worst_t, worst_h,
                     worst_d, worst_got, worst_ref);
        // per-d vectors for the worst (t,h), plus the segment-uniform mean of v as a
        // pattern signature (a degenerate softmax shows got == uniform mean)
        int b0 = 0, e0 = patches;
        for (std::size_t s = 0; s + 1 < bounds.size(); ++s) {
            if (worst_t >= bounds[s] && worst_t < bounds[s + 1]) {
                b0 = bounds[s];
                e0 = bounds[s + 1];
            }
        }
        std::fprintf(stderr, "  seg=[%d,%d) row dump t=%d h=%d\n", b0, e0, worst_t, worst_h);
        for (int d = 0; d < 8; ++d) {
            const std::size_t flat =
                static_cast<std::size_t>(d) + static_cast<std::size_t>(worst_h) * kD +
                static_cast<std::size_t>(worst_t) * kD * kH;
            const double g  = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&got[flat]));
            std::fprintf(stderr, "    d=%2d got=%9.5f ref=%9.5f\n", d, g, ref[flat]);
        }
        double vmean = 0.0;
        for (int j = b0; j < e0; ++j) {
            vmean += __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&in.v[elem_index(in, worst_d, worst_h, j)]));
        }
        std::fprintf(stderr, "    uniform v mean at d=%d: %.5f\n", worst_d, vmean / (e0 - b0));
        // distribution of per-token worst ratios, to separate "one tile broken" from
        // "uniform noise"
        for (int t = 0; t < patches; ++t) {
            double worst_row = 0.0;
            for (int h = 0; h < kH; ++h) {
                for (int d = 0; d < kD; ++d) {
                    const std::size_t flat =
                        static_cast<std::size_t>(d) + static_cast<std::size_t>(h) * kD +
                        static_cast<std::size_t>(t) * kD * kH;
                    const double g = __bfloat162float(
                        *reinterpret_cast<const __nv_bfloat16*>(&got[flat]));
                    const double e = std::fabs(g - ref[flat]);
                    const double r = e / budget_for(std::fabs(ref[flat]), ulp_multiple);
                    if (r > worst_row) { worst_row = r; }
                }
            }
            if (worst_row > 1.0) { std::fprintf(stderr, "  t=%d ratio=%.2f\n", t, worst_row); }
        }
    }

    CUDA_OK(cudaFree(dq));
    CUDA_OK(cudaFree(dk));
    CUDA_OK(cudaFree(dv));
    CUDA_OK(cudaFree(dout));
    return pass;
}

} // namespace

int main(int argc, char** argv) {
    CUDA_OK(cudaSetDevice(0));
    const char* filter = argc > 1 ? argv[1] : nullptr;
    bool all           = true;
    const auto maybe   = [&](const char* name, bool result) {
        if (filter == nullptr || std::strstr(name, filter) != nullptr) { all = all && result; }
        return result;
    };

    const std::vector<std::int32_t> two{0, 33, 128};
    const std::vector<std::int32_t> uneven{0, 17, 80, 171};
    const std::vector<std::int32_t> single{0, 64};
    const std::vector<std::int32_t> halves{0, 256, 512};

    const std::vector<bool> results = {
        maybe("packed-2seg", run_case("packed-2seg", 0xC0FFEE1u, 128, 0, &two, 0, 2.5)),
        maybe("packed-uneven", run_case("packed-uneven", 0xC0FFEE2u, 171, 0, &uneven, 0, 2.5)),
        maybe("packed-single", run_case("packed-single", 0xC0FFEE3u, 64, 0, &single, 0, 2.5)),
        maybe("packed-halves", run_case("packed-halves", 0xC0FFEE4u, 512, 0, &halves, 0, 2.5)),
        maybe("uniform-16", run_case("uniform-16", 0xC0FFEE5u, 128, 0, nullptr, 16, 2.5)),
        maybe("uniform-64", run_case("uniform-64", 0xC0FFEE6u, 256, 0, nullptr, 64, 2.5)),
        maybe("uniform-100", run_case("uniform-100", 0xC0FFEE7u, 200, 0, nullptr, 100, 2.5)),
        maybe("uniform-32-pad",
              run_case("uniform-32-pad", 0xC0FFEE8u, 96, 8, nullptr, 32, 2.5)),
    };
    for (const bool r : results) { all = all && r; }

    std::printf(all ? "ALL PASS\n" : "FAILURES PRESENT\n");
    return all ? 0 : 1;
}
