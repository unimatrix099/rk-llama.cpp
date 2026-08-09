#include <arm_neon.h>
#include <cmath>
#include <algorithm>

#include "rknpu2-quantization.h"

namespace rknpu2_quantization {

// --- Conversion from FP32 ---

void convert_fp32_to_fp16(const float * src, uint16_t * dst, size_t n_elements) {
    size_t i = 0;
#ifdef __ARM_NEON
    for (; i + 7 < n_elements; i += 8) {
        float32x4_t f32_vec_0 = vld1q_f32(src + i);
        float32x4_t f32_vec_1 = vld1q_f32(src + i + 4);
        float16x8_t f16_vec = vcombine_f16(vcvt_f16_f32(f32_vec_0), vcvt_f16_f32(f32_vec_1));
        vst1q_u16(dst + i, (uint16x8_t)f16_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = GGML_FP32_TO_FP16(src[i]);
    }
}

void quantize_fp32_to_int8(const float * src, int8_t * dst, size_t n_elements, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    size_t i = 0;
#ifdef __ARM_NEON
    // vcvtaq rounds half away from zero, matching roundf exactly
    const float32x4_t vi = vdupq_n_f32(iscale);
    for (; i + 16 <= n_elements; i += 16) {
        int32x4_t a = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i),      vi));
        int32x4_t b = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i + 4),  vi));
        int32x4_t c = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i + 8),  vi));
        int32x4_t d = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i + 12), vi));
        int16x8_t ab = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
        int16x8_t cd = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        vst1q_s8(dst + i, vcombine_s8(vmovn_s16(ab), vmovn_s16(cd)));
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (int8_t)roundf(src[i] * iscale);
    }
}

void quantize_fp32_to_int4_packed(const float * src, uint8_t * dst, size_t n_elements, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    size_t i2 = 0;   // element index (pairs consumed below as i2/2)
#ifdef __ARM_NEON
    const float32x4_t vi = vdupq_n_f32(iscale);
    const int32x4_t lo = vdupq_n_s32(-7), hi = vdupq_n_s32(7);
    for (; i2 + 16 <= n_elements; i2 += 16) {
        int32x4_t a = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i2),      vi))));
        int32x4_t b = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i2 + 4),  vi))));
        int32x4_t c = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i2 + 8),  vi))));
        int32x4_t d = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + i2 + 12), vi))));
        int16x8_t ab = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
        int16x8_t cd = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        uint8x16_t u = vreinterpretq_u8_s8(vcombine_s8(vmovn_s16(ab), vmovn_s16(cd)));
        uint8x8x2_t z = vuzp_u8(vget_low_u8(u), vget_high_u8(u));
        uint8x8_t packed = vorr_u8(vand_u8(z.val[0], vdup_n_u8(0x0F)),
                                   vshl_n_u8(vand_u8(z.val[1], vdup_n_u8(0x0F)), 4));
        vst1_u8(dst + i2 / 2, packed);
    }
#endif
    for (size_t i = i2 / 2; i < n_elements / 2; ++i) {
        float v0_f = src[i * 2 + 0] * iscale;
        float v1_f = src[i * 2 + 1] * iscale;

        // clamp in int32 before narrowing (the previous int8 cast wrapped
        // for |v| >= 127.5, sign-flipping extreme outliers before the
        // clamp; matches the NEON body above)
        int32_t v0_i = std::max(-7, std::min(7, (int32_t)roundf(v0_f)));
        int32_t v1_i = std::max(-7, std::min(7, (int32_t)roundf(v1_f)));

        dst[i] = ((uint8_t)v0_i & 0x0F) | (((uint8_t)v1_i & 0x0F) << 4);
    }
}


// --- Row helpers for the activation prep path ---

float amax_fp32(const float * src, size_t n_elements) {
    float m = 0.0f;
    size_t i = 0;
#ifdef __ARM_NEON
    float32x4_t vm = vdupq_n_f32(0.0f);
    for (; i + 4 <= n_elements; i += 4) {
        vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(src + i)));
    }
    m = vmaxvq_f32(vm);
#endif
    for (; i < n_elements; ++i) {
        m = std::max(m, std::abs(src[i]));
    }
    return m;
}

void mul_fp32(float * dst, const float * a, const float * b, size_t n_elements) {
    size_t i = 0;
#ifdef __ARM_NEON
    for (; i + 4 <= n_elements; i += 4) {
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = a[i] * b[i];
    }
}

void dequant_acc_int16_to_fp32(float * dst, const int16_t * src, size_t n_elements, float scale) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t vs = vdupq_n_f32(scale);
    for (; i + 8 <= n_elements; i += 8) {
        int16x8_t s16 = vld1q_s16(src + i);
        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16)));
        vst1q_f32(dst + i,     vfmaq_f32(vld1q_f32(dst + i),     f0, vs));
        vst1q_f32(dst + i + 4, vfmaq_f32(vld1q_f32(dst + i + 4), f1, vs));
    }
#endif
    for (; i < n_elements; ++i) {
        // explicitly fused to match the vector body (vfmaq) at any
        // optimization level
        dst[i] = fmaf((float)src[i], scale, dst[i]);
    }
}

void dequant_acc_int16_tiled(float * dst, const int16_t * src_native,
                             int32_t m, int32_t m_stride, int32_t outer, int32_t sub,
                             int32_t n_limit, float scale) {
    for (int32_t t = 0; t < outer; ++t) {
        const int16_t * cell = src_native + ((size_t)t * m_stride + m) * sub;
        const int32_t n0 = t * sub;
        const int32_t lim = std::min(sub, n_limit - n0);
        if (lim <= 0) {
            break;
        }
        dequant_acc_int16_to_fp32(dst + n0, cell, (size_t)lim, scale);
    }
}

// --- Dequantization to FP32 ---

void dequantize_int16_to_fp32(const int16_t * src, float * dst, size_t n_elements, float scale) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 3 < n_elements; i += 4) {
        int16x4_t i16_vec = vld1_s16(src + i);
        int32x4_t i32_vec = vmovl_s16(i16_vec);
        float32x4_t f32_vec = vcvtq_f32_s32(i32_vec);
        f32_vec = vmulq_f32(f32_vec, scale_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (float)src[i] * scale;
    }
}

void dequantize_int32_to_fp32(const int32_t * src, float * dst, size_t n_elements, float scale) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 3 < n_elements; i += 4) {
        int32x4_t i32_vec = vld1q_s32(src + i);
        float32x4_t f32_vec = vcvtq_f32_s32(i32_vec);
        f32_vec = vmulq_f32(f32_vec, scale_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (float)src[i] * scale;
    }
}

} // namespace rknpu2_quantization