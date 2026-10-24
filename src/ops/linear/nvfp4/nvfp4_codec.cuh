#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#ifndef NINFER_PASCAL_BUILD
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_PASCAL_BUILD
// CUDA 12.4 (the Pascal toolchain window) ships no cuda_fp4.h and GP100 has no FP4 hardware:
// the codec boundary here becomes a bit-exact table implementation instead of the Blackwell
// conversions. Every finite E2M1 magnitude {0, 1/2, 1, 3/2, 2, 3, 4, 6} is exactly
// representable in FP16 and FP32, and the encoder applies round-to-nearest-even across the
// midpoint ladder with SATFINITE magnitude clamping, matching cvt.rn.satfinite.e2m1x2.f32 for
// finite inputs (NaN saturates to the largest code, as on hardware). The low nibble of a
// packed byte is the first element, exactly like the CUDA type this shim replaces.
struct __nv_fp4x2_e2m1 {
    unsigned char __x;

    __device__ explicit operator __half2() const {
        static constexpr unsigned short kMagnitudeBits[8] = {
            0x0000u, 0x3800u, 0x3C00u, 0x3E00u, 0x4000u, 0x4200u, 0x4400u, 0x4600u};
        const unsigned lo = static_cast<unsigned>(__x) & 0xFu;
        const unsigned hi = static_cast<unsigned>(__x) >> 4u;
        __half2 out;
        out.x = __ushort_as_half(static_cast<unsigned short>(kMagnitudeBits[lo & 7u] |
                                                              ((lo & 8u) << 12)));
        out.y = __ushort_as_half(static_cast<unsigned short>(kMagnitudeBits[hi & 7u] |
                                                              ((hi & 8u) << 12)));
        return out;
    }

    __device__ explicit operator float2() const {
        return __half22float2(static_cast<__half2>(*this));
    }
};

__device__ __forceinline__ std::uint8_t nvfp4_e2m1x2_encode(float lo_value, float hi_value) {
    const auto one = [](float v) -> unsigned {
        const unsigned sign     = __float_as_uint(v) >> 31;
        const float a           = fabsf(v);
        unsigned code;
        if (a > 5.0F) { code = 7u; }
        else if (a >= 3.5F) { code = 6u; }
        else if (a > 2.5F) { code = 5u; }
        else if (a >= 1.75F) { code = 4u; }
        else if (a > 1.25F) { code = 3u; }
        else if (a >= 0.75F) { code = 2u; }
        else if (a > 0.25F) { code = 1u; }
        else { code = __isnanf(a) ? 7u : 0u; }
        return code | (sign << 3u);
    };
    return static_cast<std::uint8_t>(one(lo_value) | (one(hi_value) << 4u));
}
#endif

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
// cvt.e2m1x2 is native FP4 hardware, Blackwell-only -- Volta has no FP4 hardware at all,
// so unlike the mma.cuh helpers this isn't "no SIMT replacement built yet", it's simply
// never going to have one. Already permanently out of scope (see the V100 performance summary);
// this only needs a guard so ptxas doesn't reject the PTX outright at compile time.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
#elif defined(NINFER_PASCAL_BUILD)
    // sm_60 device pass: table encoder (the Blackwell asm above still serves the host pass and
    // sm_80+ builds). Byte order matches the mov.b32 packs: pairs 0..3 in codes_lo, 4..7 in
    // codes_hi.
    codes_lo = static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[0].x, values[0].y)) |
               (static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[1].x, values[1].y)) << 8u) |
               (static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[2].x, values[2].y)) << 16u) |
               (static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[3].x, values[3].y)) << 24u);
    codes_hi = static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[4].x, values[4].y)) |
               (static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[5].x, values[5].y)) << 8u) |
               (static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[6].x, values[6].y)) << 16u) |
               (static_cast<std::uint32_t>(nvfp4_e2m1x2_encode(values[7].x, values[7].y)) << 24u);
#else
    (void)values;
    codes_lo = 0;
    codes_hi = 0;
    __trap();
#endif
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail
