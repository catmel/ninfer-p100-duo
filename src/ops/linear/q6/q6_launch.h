#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q6Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q6_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c5(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c6(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c7(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
#ifdef NINFER_PASCAL_BUILD
// sm_60 half2 decode GEMV, T=1 (q6_decode_hfma2_pascal.cu).
void launch_q6_decode_hfma2_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream);
// sm_60 wide-T (T>=2) decode tile GEMM (linear_tile_pascal.cu).
void launch_q6_decode_tile_pascal(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
#endif
void launch_q6_mma_r64_c16_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c24_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c32_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c40_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c48_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c56_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c64_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c72_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c88_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c112_partial(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
