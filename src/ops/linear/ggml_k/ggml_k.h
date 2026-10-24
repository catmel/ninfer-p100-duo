#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The artifact preserves each original Q4_K/Q6_K row and its embedded scales.
// The row descriptor low bit selects Q6_K; the remaining bits locate the row.
void ggml_k_linear(const Tensor& x, const Weight& weight, Tensor& out,
                   WorkspaceArena* workspace, cudaStream_t stream);
[[nodiscard]] std::size_t ggml_k_cutlass_workspace_bytes(std::int32_t n, std::int32_t k,
                                                          std::int32_t tokens);
void ggml_k_project_split(const Tensor& x, const Weight& weight, const Tensor* outputs,
                          int count, bool add, cudaStream_t stream,
                          bool tiled_gdn_input = false, WorkspaceArena* workspace = nullptr);
void ggml_k_embedding(const Tensor& ids, const Weight& weight, Tensor& out,
                      cudaStream_t stream);
#ifdef NINFER_PASCAL_BUILD
// sm_60 wide-token tile route (ggml_k_tile_pascal.cu): one HFMA2 decode per weight feeding
// eight staged tokens; single bf16 output section, non-tiled input only.
void ggml_k_launch_tile_pascal(const Tensor& x, const Weight& weight, const Tensor& out,
                               cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
