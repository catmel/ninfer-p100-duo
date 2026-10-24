#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include "ops/linear/bf16/bf16_launch.h"

namespace ninfer::ops::detail {

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        bf16_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (x.ne[1] <= kBf16AttnInputSmallTDispatchEnd) {
        bf16_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    bf16_attn_input_cutlass_sm70_launch(x, weight, q, gate, k, v, stream);
#elif defined(NINFER_PASCAL_BUILD)
    // Pascal (sm_60): bf16_attn_input_mma_launch drives the bf16 tensor-core kernel
    // (ldmatrix + mma.m16n8k16, trap-stubbed below sm_80 in ops/common/mma.cuh) and no CUTLASS
    // sm70 TU is built. The decode/small-T routes above are SIMT and stay; above them the
    // generic SIMT A16 GEMM runs per output section of the [14336, 5120] parent (one row-range
    // per q|k|gate|v tensor). docs/p100-migration.md.
    struct Section {
        std::int32_t row_begin;
        std::int32_t rows;
        Tensor*      out;
    };
    constexpr std::int32_t kHidden = 5120;
    const Section sections[4] = {
        {0, 6144, &q}, {6144, 1024, &k}, {7168, 6144, &gate}, {13312, 1024, &v},
    };
    for (const Section& section : sections) {
        Weight part   = weight;
        part.n        = section.rows;
        part.qdata    = static_cast<const unsigned char*>(weight.qdata) +
                        static_cast<std::int64_t>(section.row_begin) * kHidden * 2;  // bf16 = 2B
        bf16_gemm_simt_launch(x, part, *section.out, stream);
    }
#else
    bf16_attn_input_mma_launch(x, weight, q, gate, k, v, stream);
#endif
}

} // namespace ninfer::ops::detail
