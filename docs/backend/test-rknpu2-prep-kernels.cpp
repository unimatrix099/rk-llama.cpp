// Exactness tests for the RKNPU2 CPU prep kernels (quantization,
// Hadamard transform, dequantization) against scalar reference
// implementations embedded here.
//
// Written test-first for the NEON vectorization work
// (RKNPU2-neon-prep-plan.md): the references pin the semantics of the
// existing scalar code — including rounding-tie behaviour (roundf =
// half-away-from-zero) and the zero-scale path — so the NEON versions
// must reproduce them bit-exactly, tails and all.
//
// Build & run (from docs/backend):
//   make -f Makefile.rknpu2-tools check-prep

#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        printf("FAIL %s:%d  ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

static uint32_t g_seed = 0xBADC0DE5u;
static uint32_t prng(void) { g_seed = g_seed * 1664525u + 1013904223u; return g_seed; }
static float frand(void) { return (float)((int)(prng() % 4001) - 2000) / 173.0f; }

// ------------------------------------------------- scalar references
// Semantics copied from the pre-NEON backend code, verbatim.

static void ref_q8(const float* src, int8_t* dst, size_t n, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    for (size_t i = 0; i < n; ++i) dst[i] = (int8_t)roundf(src[i] * iscale);
}

static void ref_q4(const float* src, uint8_t* dst, size_t n, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    for (size_t i = 0; i < n / 2; ++i) {
        float v0f = src[i*2+0] * iscale, v1f = src[i*2+1] * iscale;
        // clamp in int32 before narrowing. NOTE: the pre-NEON backend cast
        // to int8 first, wrapping for |v| >= 127.5 and sign-flipping
        // extreme outliers to the wrong clamp bound (reachable on the B
        // side under entropy calibration). That was a bug; the corrected
        // semantics below are what the implementation must match.
        int32_t v0 = std::max(-7, std::min(7, (int32_t)roundf(v0f)));
        int32_t v1 = std::max(-7, std::min(7, (int32_t)roundf(v1f)));
        dst[i] = ((uint8_t)v0 & 0x0F) | (((uint8_t)v1 & 0x0F) << 4);
    }
}

static float ref_amax(const float* v, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::abs(v[i]));
    return m;
}

static void ref_mul(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] * b[i];
}

static void ref_dq16_acc(float* dst, const int16_t* src, size_t n, float scale) {
    // fused multiply-add, explicitly: the production build contracts the
    // original scalar loop to FMA (-ffp-contract=fast), and the NEON path
    // uses vfmaq. fmaf keeps this reference exact at any optimization level.
    for (size_t i = 0; i < n; ++i) dst[i] = fmaf((float)src[i], scale, dst[i]);
}

static void ref_fwht(float* data, int size) {
    for (int h = 1; h < size; h <<= 1)
        for (int i = 0; i < size; i += h * 2)
            for (int j = i; j < i + h; ++j) {
                float x = data[j], y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
}

// ------------------------------------------------------------ tests

static const size_t SIZES[] = {2, 8, 16, 30, 62, 100, 1536, 2048, 8960};
static const size_t N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static void test_quantize_int8(void) {
    for (size_t si = 0; si < N_SIZES; ++si) {
        size_t n = SIZES[si];
        std::vector<float> src(n);
        for (auto& v : src) v = frand();
        std::vector<int8_t> a(n), b(n);
        for (float scale : {0.37f, 1.0f, 0.0f}) {
            ref_q8(src.data(), a.data(), n, scale);
            rknpu2_quantization::quantize_fp32_to_int8(src.data(), b.data(), n, scale);
            CHECK(memcmp(a.data(), b.data(), n) == 0, "q8 n=%zu scale=%.2f", n, scale);
        }
    }
    // rounding ties: roundf is half-away-from-zero
    float ties[32];
    for (int i = 0; i < 32; ++i) ties[i] = (i - 16) * 0.5f;
    int8_t a[32], b[32];
    ref_q8(ties, a, 32, 1.0f);
    rknpu2_quantization::quantize_fp32_to_int8(ties, b, 32, 1.0f);
    CHECK(memcmp(a, b, 32) == 0, "q8 ties");

    // int8 has no clamp; out-of-range values wrap identically in the
    // scalar cast and the NEON narrowing (cannot occur in the backend,
    // where scale = amax/127 - pinned for safety)
    float wrap[16] = {201.0f, -201.0f, 300.6f, -300.6f, 127.4f, 127.6f,
                      -127.6f, 128.0f, 1000.0f, -1000.0f, 0.5f, -0.5f,
                      42.0f, -42.0f, 7.7f, -7.7f};
    int8_t wa[16], wb[16];
    ref_q8(wrap, wa, 16, 1.0f);
    rknpu2_quantization::quantize_fp32_to_int8(wrap, wb, 16, 1.0f);
    CHECK(memcmp(wa, wb, 16) == 0, "q8 wrap equality");
}

static void test_quantize_int4(void) {
    for (size_t si = 0; si < N_SIZES; ++si) {
        size_t n = SIZES[si] & ~(size_t)1;   // pairs
        std::vector<float> src(n);
        for (auto& v : src) v = frand() * 3.0f;   // exercise the +-7 clamp
        std::vector<uint8_t> a(n/2), b(n/2);
        for (float scale : {0.37f, 1.0f, 0.0f}) {
            ref_q4(src.data(), a.data(), n, scale);
            rknpu2_quantization::quantize_fp32_to_int4_packed(src.data(), b.data(), n, scale);
            CHECK(memcmp(a.data(), b.data(), n/2) == 0, "q4 n=%zu scale=%.2f", n, scale);
        }
    }
    float ties[32];
    for (int i = 0; i < 32; ++i) ties[i] = (i - 16) * 0.5f;
    uint8_t a[16], b[16];
    ref_q4(ties, a, 32, 1.0f);
    rknpu2_quantization::quantize_fp32_to_int4_packed(ties, b, 32, 1.0f);
    CHECK(memcmp(a, b, 16) == 0, "q4 ties");

    // extreme outliers past the old int8 wrap threshold: must clamp to the
    // CORRECT sign (+7/-7), in both the vector body and the scalar tail
    float extreme[18] = {200.4f, -200.4f, 127.6f, -127.6f, 128.0f, -128.0f,
                         300.0f, -300.0f, 1e6f, -1e6f, 6.9f, -6.9f,
                         127.4f, -127.4f, 55.5f, -55.5f,
                         500.0f, -500.0f};   // last pair lands in the tail
    uint8_t ea[9], eb[9];
    ref_q4(extreme, ea, 18, 1.0f);
    rknpu2_quantization::quantize_fp32_to_int4_packed(extreme, eb, 18, 1.0f);
    CHECK(memcmp(ea, eb, 9) == 0, "q4 extremes (clamp sign)");
    CHECK((eb[0] & 0x0F) == 0x7 && (eb[0] >> 4) == 0x9,
          "q4 extreme signs: +200 -> +7, -200 -> -7, got %02x", eb[0]);
}

static void test_amax(void) {
    for (size_t si = 0; si < N_SIZES; ++si) {
        size_t n = SIZES[si];
        std::vector<float> v(n);
        for (auto& x : v) x = frand();
        v[n / 2] = -123.5f;                          // ensure a negative peak
        float a = ref_amax(v.data(), n);
        float b = rknpu2_quantization::amax_fp32(v.data(), n);
        CHECK(a == b, "amax n=%zu: %f vs %f", n, a, b);
    }
    CHECK(rknpu2_quantization::amax_fp32(nullptr, 0) == 0.0f, "amax n=0");
}

static void test_mul(void) {
    for (size_t si = 0; si < N_SIZES; ++si) {
        size_t n = SIZES[si];
        std::vector<float> x(n), s(n), a(n), b(n);
        for (size_t i = 0; i < n; ++i) { x[i] = frand(); s[i] = (prng() & 1) ? 1.0f : -1.0f; }
        ref_mul(a.data(), x.data(), s.data(), n);
        rknpu2_quantization::mul_fp32(b.data(), x.data(), s.data(), n);
        CHECK(memcmp(a.data(), b.data(), n * 4) == 0, "mul n=%zu", n);
    }
}

static void test_dequant_acc_int16(void) {
    for (size_t si = 0; si < N_SIZES; ++si) {
        size_t n = SIZES[si];
        std::vector<int16_t> src(n);
        for (auto& v : src) v = (int16_t)(prng() % 4001 - 2000);
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) a[i] = b[i] = frand();   // must ACCUMULATE
        ref_dq16_acc(a.data(), src.data(), n, 0.0173f);
        rknpu2_quantization::dequant_acc_int16_to_fp32(b.data(), src.data(), n, 0.0173f);
        CHECK(memcmp(a.data(), b.data(), n * 4) == 0, "dq16acc n=%zu", n);
    }
}

// tiled variant used by the native-layout C path: gathers [outer, m_stride,
// sub] int16 cells for row m while dequant-accumulating
static void test_dequant_acc_tiled(void) {
    struct { int outer, m_stride, sub, n_limit; } cases[] = {
        {8, 4, 8, 64}, {256, 32, 8, 2048}, {3, 1, 8, 20},  // n_limit < outer*sub tail
        {1, 5, 8, 8},
    };
    for (auto& c : cases) {
        size_t total = (size_t)c.outer * c.m_stride * c.sub;
        std::vector<int16_t> native(total);
        for (auto& v : native) v = (int16_t)(prng() % 4001 - 2000);
        for (int m = 0; m < c.m_stride; ++m) {
            std::vector<float> a(c.n_limit), b(c.n_limit);
            for (int i = 0; i < c.n_limit; ++i) a[i] = b[i] = frand();
            // reference: walk cells, accumulate
            for (int t = 0; t < c.outer; ++t) {
                const int16_t* cell = native.data() + ((size_t)t * c.m_stride + m) * c.sub;
                for (int j = 0; j < c.sub && t * c.sub + j < c.n_limit; ++j)
                    a[t * c.sub + j] = fmaf((float)cell[j], 0.0173f, a[t * c.sub + j]);
            }
            rknpu2_quantization::dequant_acc_int16_tiled(
                b.data(), native.data(), m, c.m_stride, c.outer, c.sub, c.n_limit, 0.0173f);
            CHECK(memcmp(a.data(), b.data(), (size_t)c.n_limit * 4) == 0,
                  "dq16tiled outer=%d ms=%d m=%d", c.outer, c.m_stride, m);
        }
    }
}

// per-output-channel variants (INT4 per-channel weight scales)
static void test_dequant_acc_perchan(void) {
    const float common = 0.0173f;
    for (size_t si = 0; si < N_SIZES; ++si) {
        size_t n = SIZES[si];
        std::vector<int16_t> src(n);
        for (auto& v : src) v = (int16_t)(prng() % 4001 - 2000);
        std::vector<float> chan(n);
        for (auto& v : chan) v = 0.001f + std::fabs(frand());
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) a[i] = b[i] = frand();
        for (size_t i = 0; i < n; ++i) {
            a[i] = fmaf((float)src[i], chan[i] * common, a[i]);   // mul-then-fma, as impl
        }
        rknpu2_quantization::dequant_acc_int16_to_fp32_perchan(b.data(), src.data(), n, common, chan.data());
        CHECK(memcmp(a.data(), b.data(), n * 4) == 0, "dq16acc-pc n=%zu", n);
    }
}

static void test_dequant_acc_tiled_perchan(void) {
    const float common = 0.0173f;
    struct { int outer, m_stride, sub, n_limit; } cases[] = {
        {8, 4, 8, 64}, {256, 32, 8, 2048}, {3, 1, 8, 20},
        {1, 5, 8, 8},
    };
    for (auto& c : cases) {
        size_t total = (size_t)c.outer * c.m_stride * c.sub;
        std::vector<int16_t> native(total);
        for (auto& v : native) v = (int16_t)(prng() % 4001 - 2000);
        std::vector<float> chan(c.n_limit);
        for (auto& v : chan) v = 0.001f + std::fabs(frand());
        for (int m = 0; m < c.m_stride; ++m) {
            std::vector<float> a(c.n_limit), b(c.n_limit);
            for (int i = 0; i < c.n_limit; ++i) a[i] = b[i] = frand();
            for (int t = 0; t < c.outer; ++t) {
                const int16_t* cell = native.data() + ((size_t)t * c.m_stride + m) * c.sub;
                for (int j = 0; j < c.sub && t * c.sub + j < c.n_limit; ++j) {
                    int nn = t * c.sub + j;
                    a[nn] = fmaf((float)cell[j], chan[nn] * common, a[nn]);
                }
            }
            rknpu2_quantization::dequant_acc_int16_tiled_perchan(
                b.data(), native.data(), m, c.m_stride, c.outer, c.sub, c.n_limit, common, chan.data());
            CHECK(memcmp(a.data(), b.data(), (size_t)c.n_limit * 4) == 0,
                  "dq16tiled-pc outer=%d ms=%d m=%d", c.outer, c.m_stride, m);
        }
    }
}

// existing conversion/dequant functions: regression-guard them too
static void test_existing_conversions(void) {
    for (size_t si = 0; si < N_SIZES; ++si) {
        size_t n = SIZES[si];
        std::vector<float> src(n);
        for (auto& v : src) v = frand();

        std::vector<uint16_t> h(n);
        rknpu2_quantization::convert_fp32_to_fp16(src.data(), h.data(), n);
        int ok = 1;
        for (size_t i = 0; i < n; ++i)
            ok &= (h[i] == (uint16_t)GGML_FP32_TO_FP16(src[i]));
        CHECK(ok, "fp16 n=%zu", n);

        std::vector<int16_t> s16(n);
        std::vector<int32_t> s32(n);
        for (size_t i = 0; i < n; ++i) { s16[i] = (int16_t)(prng() % 9001 - 4500); s32[i] = (int32_t)(prng() % 90001) - 45000; }
        std::vector<float> d16(n), d32(n), r(n);
        rknpu2_quantization::dequantize_int16_to_fp32(s16.data(), d16.data(), n, 0.01f);
        rknpu2_quantization::dequantize_int32_to_fp32(s32.data(), d32.data(), n, 0.001f);
        ok = 1;
        for (size_t i = 0; i < n; ++i) ok &= (d16[i] == (float)s16[i] * 0.01f);
        CHECK(ok, "dq16 n=%zu", n);
        ok = 1;
        for (size_t i = 0; i < n; ++i) ok &= (d32[i] == (float)s32[i] * 0.001f);
        CHECK(ok, "dq32 n=%zu", n);
    }
}

static void test_hadamard(void) {
    // cases where the default (pure block-diagonal) mode degenerates to
    // one full-length FWHT: power-of-two K
    struct { int K, padded; } cases[] = {
        {4, 4}, {8, 8}, {32, 32}, {2048, 2048},
        {1, 1}, {2, 2},                          // scalar fallback path
    };
    for (auto& c : cases) {
        std::vector<float> src(c.K);
        for (auto& v : src) v = frand();

        std::vector<float> ref(c.padded, 0.0f);
        memcpy(ref.data(), src.data(), c.K * sizeof(float));
        ref_fwht(ref.data(), c.padded);

        std::vector<float> out(c.padded, -777.0f);
        rknpu2_calibration::hadamard_transform(out.data(), src.data(), c.K, c.padded);

        CHECK(memcmp(ref.data(), out.data(), c.padded * 4) == 0,
              "hadamard K=%d padded=%d", c.K, c.padded);
    }
}

static void test_hadamard_blocked(void) {
    // Default mode (RKNPU_HADAMARD_BLOCK unset = pure block-diagonal):
    // block = largest pow2 divisor of K, K_op == K, no padding. E4B dims:
    // 2560 -> 512-blocks, 10240 -> 2048-blocks, 10752 -> 512-blocks.
    // Element-exact vs a per-block scalar reference.
    struct { int K, block, k_op; } cases[] = {
        {2560, 512, 2560}, {10240, 2048, 10240}, {10752, 512, 10752},
        {1536, 512, 1536}, {256, 256, 256}, {96, 32, 96},
        {12, 4, 12}, {5, 1, 5}, {2048, 2048, 2048},
    };
    for (auto& c : cases) {
        CHECK(rknpu2_calibration::hadamard_block_len(c.K) == c.block,
              "block_len K=%d", c.K);
        CHECK(rknpu2_calibration::hadamard_k_op(c.K) == c.k_op, "k_op K=%d", c.K);

        std::vector<float> src(c.K);
        for (auto& v : src) v = frand();

        std::vector<float> ref(c.k_op, 0.0f);
        memcpy(ref.data(), src.data(), c.K * sizeof(float));
        for (int off = 0; off < c.k_op; off += c.block) {
            ref_fwht(ref.data() + off, c.block);
        }
        std::vector<float> out(c.k_op, -777.0f);
        rknpu2_calibration::hadamard_transform(out.data(), src.data(), c.K, c.k_op);
        CHECK(memcmp(ref.data(), out.data(), c.k_op * 4) == 0,
              "blocked hadamard K=%d block=%d", c.K, c.block);

        // semantic sanity: H(H(x)) = block * x per block (float-tolerant:
        // the two passes reassociate the sums). Only valid when the padded
        // length keeps the same block (K=1536 pads to 2048 whose block is
        // 2048, so a second transform would use a different H).
        if (rknpu2_calibration::hadamard_block_len(c.k_op) == c.block) {
            std::vector<float> twice(c.k_op, -777.0f);
            rknpu2_calibration::hadamard_transform(twice.data(), out.data(), c.k_op, c.k_op);
            bool ok = true;
            for (int i = 0; i < c.K; ++i) {
                ok &= std::fabs(twice[i] - src[i] * (float)c.block) <= 1e-3f * c.block;
            }
            CHECK(ok, "involution K=%d block=%d", c.K, c.block);
        }
    }
}

int main(void) {
    test_quantize_int8();
    test_quantize_int4();
    test_amax();
    test_mul();
    test_dequant_acc_int16();
    test_dequant_acc_tiled();
    test_dequant_acc_perchan();
    test_dequant_acc_tiled_perchan();
    test_existing_conversions();
    test_hadamard();
    test_hadamard_blocked();
    CHECK(rknpu2_calibration::next_power_of_two(0) == 1, "npot 0");
    CHECK(rknpu2_calibration::next_power_of_two(1) == 1, "npot 1");
    CHECK(rknpu2_calibration::next_power_of_two(3) == 4, "npot 3");
    CHECK(rknpu2_calibration::next_power_of_two(2048) == 2048, "npot 2048");
    CHECK(rknpu2_calibration::next_power_of_two(2049) == 4096, "npot 2049");
    printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
