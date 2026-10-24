#pragma once

#include <cstdint>

// `__dp4a` needs sm_61. Tesla P100 (GP100, sm_60) has no DP4A instruction, so every consumer
// goes through this helper: the hardware intrinsic on sm_61+, and on sm_60 a bit-exact
// emulation of the four signed-byte products and their accumulation. The emulation matches
// `mma.sync...s8` / `dp4a.s32.s8` semantics exactly (signed s8 x s8 -> s32), because the
// arithmetic below is plain two's-complement integer multiply-add in the same order.
//
// Phase-B of the P100 migration replaces the GGUF A8 *dot* route with an fp16 HFMA2 unpack
// (sm_60 decode is instruction-bound and the HFMA2 path costs fewer issue slots); this header
// keeps the legacy route compiling and numerically identical until that rewrite lands.

namespace ninfer::ops {

__device__ __forceinline__ int dp4a_s8(int accumulator, unsigned a, unsigned b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(static_cast<int>(a), static_cast<int>(b), accumulator);
#else
    int result = accumulator;
    result += static_cast<int>(static_cast<signed char>(a)) *
              static_cast<int>(static_cast<signed char>(b));
    result += static_cast<int>(static_cast<signed char>(a >> 8)) *
              static_cast<int>(static_cast<signed char>(b >> 8));
    result += static_cast<int>(static_cast<signed char>(a >> 16)) *
              static_cast<int>(static_cast<signed char>(b >> 16));
    result += static_cast<int>(static_cast<signed char>(a >> 24)) *
              static_cast<int>(static_cast<signed char>(b >> 24));
    return result;
#endif
}

// Sum of the eight signed bytes packed into (a0, a1): the 0x01010101 multiplier form used by
// the GGUF A8 activation path to recover the code-sum for the minimum term.
__device__ __forceinline__ int byte8_sum_s8(unsigned a0, unsigned a1) {
    return dp4a_s8(dp4a_s8(0, a0, 0x01010101u), a1, 0x01010101u);
}

} // namespace ninfer::ops
