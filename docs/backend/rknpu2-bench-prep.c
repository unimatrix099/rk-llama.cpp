// Microbenchmark + NEON prototypes for the W4A4 CPU-side activation prep.
//
// Measures the current scalar cost of each per-row prep step at
// model-realistic K sizes, then times NEON prototypes of the same steps and
// verifies them element-exact against the scalar versions. This is the
// research basis for vectorizing the backend's prep path (see
// RKNPU2-neon-prep-plan.md).
//
// Prep steps per activation row per W4A4 matmul node (ggml-rknpu2.cpp
// A-matrix pass): sign multiply -> FWHT (K padded to pow2) -> amax scan ->
// INT4 quantize+pack. C side: INT16 dequant-accumulate.
//
// Build & run on the board (from docs/backend):
//   gcc bench-prep.c -o bench-prep -O2 -march=armv8.2-a+fp16
//   ./bench-prep

#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// ------------------------------------------------------------------ scalar
// (verbatim shapes of the backend's current code)

static void fwht_scalar(float* data, int size) {
    for (int h = 1; h < size; h <<= 1)
        for (int i = 0; i < size; i += h * 2)
            for (int j = i; j < i + h; ++j) {
                float x = data[j], y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
}

static void signmul_scalar(float* dst, const float* src, const float* s, int n) {
    for (int k = 0; k < n; ++k) dst[k] = src[k] * s[k];
}

static float amax_scalar(const float* v, int n) {
    float m = 0.0f;
    for (int k = 0; k < n; ++k) m = fmaxf(m, fabsf(v[k]));
    return m;
}

static void q4pack_scalar(const float* src, uint8_t* dst, int n, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    for (int i = 0; i < n / 2; ++i) {
        float v0f = src[i*2+0] * iscale, v1f = src[i*2+1] * iscale;
        int8_t v0 = (int8_t)fmaxf(-7, fminf(7, roundf(v0f)));
        int8_t v1 = (int8_t)fmaxf(-7, fminf(7, roundf(v1f)));
        dst[i] = ((uint8_t)v0 & 0x0F) | (((uint8_t)v1 & 0x0F) << 4);
    }
}

static void c16dequant_scalar(float* dst, const int16_t* src, int n, float scale) {
    for (int i = 0; i < n; ++i) dst[i] += (float)src[i] * scale;
}

// ------------------------------------------------------------------- NEON

static void fwht_neon(float* data, int size) {
    // stage h=1: pairwise butterflies inside 4-lane vectors
    for (int i = 0; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t sw = vrev64q_f32(v);              // [b,a,d,c]
        float32x4_t sum = vaddq_f32(v, sw);           // [a+b, a+b, c+d, c+d]
        float32x4_t dif = vsubq_f32(v, sw);           // [a-b, b-a, c-d, d-c]
        // want [a+b, a-b, c+d, c-d]: even lanes from sum, odd lanes from dif
        float32x4_t r = vtrn1q_f32(sum, dif);
        vst1q_f32(data + i, r);
    }
    // stage h=2: butterflies between lane pairs of one vector
    for (int i = 0; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x2_t lo = vget_low_f32(v), hi = vget_high_f32(v);
        vst1_f32(data + i,     vadd_f32(lo, hi));
        vst1_f32(data + i + 2, vsub_f32(lo, hi));
    }
    // stages h>=4: contiguous 4-wide butterflies
    for (int h = 4; h < size; h <<= 1)
        for (int i = 0; i < size; i += h * 2)
            for (int j = i; j < i + h; j += 4) {
                float32x4_t x = vld1q_f32(data + j);
                float32x4_t y = vld1q_f32(data + j + h);
                vst1q_f32(data + j,     vaddq_f32(x, y));
                vst1q_f32(data + j + h, vsubq_f32(x, y));
            }
}

static void signmul_neon(float* dst, const float* src, const float* s, int n) {
    int k = 0;
    for (; k + 4 <= n; k += 4)
        vst1q_f32(dst + k, vmulq_f32(vld1q_f32(src + k), vld1q_f32(s + k)));
    for (; k < n; ++k) dst[k] = src[k] * s[k];
}

static float amax_neon(const float* v, int n) {
    float32x4_t m = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= n; k += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(v + k)));
    float mm = vmaxvq_f32(m);
    for (; k < n; ++k) mm = fmaxf(mm, fabsf(v[k]));
    return mm;
}

static void q4pack_neon(const float* src, uint8_t* dst, int n, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    const float32x4_t vi = vdupq_n_f32(iscale);
    const int32x4_t lo = vdupq_n_s32(-7), hi = vdupq_n_s32(7);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        int32x4_t a = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src+i),    vi))));
        int32x4_t b = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src+i+4),  vi))));
        int32x4_t c = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src+i+8),  vi))));
        int32x4_t d = vmaxq_s32(lo, vminq_s32(hi, vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src+i+12), vi))));
        int16x8_t ab = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
        int16x8_t cd = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        int8x16_t v16 = vcombine_s8(vmovn_s16(ab), vmovn_s16(cd));   // 16 int4 values as bytes
        // pack pairs: even lanes -> low nibble, odd lanes -> high nibble
        uint8x16_t u = vreinterpretq_u8_s8(v16);
        uint8x8x2_t z = vuzp_u8(vget_low_u8(u), vget_high_u8(u));
        uint8x8_t ev = z.val[0], od = z.val[1];
        uint8x8_t packed = vorr_u8(vand_u8(ev, vdup_n_u8(0x0F)), vshl_n_u8(vand_u8(od, vdup_n_u8(0x0F)), 4));
        vst1_u8(dst + i / 2, packed);
    }
    for (; i < n; i += 2) {
        float v0f = src[i] * iscale, v1f = src[i+1] * iscale;
        int8_t v0 = (int8_t)fmaxf(-7, fminf(7, roundf(v0f)));
        int8_t v1 = (int8_t)fmaxf(-7, fminf(7, roundf(v1f)));
        dst[i/2] = ((uint8_t)v0 & 0x0F) | (((uint8_t)v1 & 0x0F) << 4);
    }
}

static void c16dequant_neon(float* dst, const int16_t* src, int n, float scale) {
    const float32x4_t vs = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t s16 = vld1q_s16(src + i);
        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16)));
        vst1q_f32(dst + i,     vmlaq_f32(vld1q_f32(dst + i),     f0, vs));
        vst1q_f32(dst + i + 4, vmlaq_f32(vld1q_f32(dst + i + 4), f1, vs));
    }
    for (; i < n; ++i) dst[i] += (float)src[i] * scale;
}

// ------------------------------------------------------------------ driver

static float frand(void) { return (float)(rand() % 2000 - 1000) / 97.0f; }

static double bench_us(void (*fn)(void*), void* arg, int iters) {
    fn(arg); fn(arg);                       // warmup
    double t0 = now_us();
    for (int r = 0; r < iters; ++r) fn(arg);
    return (now_us() - t0) / iters;
}

typedef struct { float *a, *b, *s; uint8_t* p; int16_t* c16; int n; float scale; int which; } ctx_t;
static void run_one(void* v) {
    ctx_t* c = (ctx_t*)v;
    switch (c->which) {
        case 0: fwht_scalar(c->a, c->n); break;
        case 1: fwht_neon(c->a, c->n); break;
        case 2: signmul_scalar(c->b, c->a, c->s, c->n); break;
        case 3: signmul_neon(c->b, c->a, c->s, c->n); break;
        case 4: c->scale = amax_scalar(c->a, c->n); break;
        case 5: c->scale = amax_neon(c->a, c->n); break;
        case 6: q4pack_scalar(c->a, c->p, c->n, 3.7f); break;
        case 7: q4pack_neon(c->a, c->p, c->n, 3.7f); break;
        case 8: c16dequant_scalar(c->b, c->c16, c->n, 0.01f); break;
        case 9: c16dequant_neon(c->b, c->c16, c->n, 0.01f); break;
    }
}

int main(void) {
    const int Ks[] = {2048, 4096, 16384};   // K_op of Qwen attn / E4B attn / Qwen ffn
    const char* names[5] = {"fwht", "signmul", "amax", "q4pack", "c16dequant"};

    // correctness first: NEON must match scalar exactly
    srand(7);
    for (unsigned ki = 0; ki < 3; ++ki) {
        int n = Ks[ki];
        float *x = malloc(n*4), *y = malloc(n*4), *s = malloc(n*4);
        float *d0 = calloc(n,4), *d1 = calloc(n,4);
        uint8_t *p0 = malloc(n/2), *p1 = malloc(n/2);
        int16_t *c16 = malloc(n*2);
        for (int i = 0; i < n; ++i) { x[i]=frand(); s[i]=(rand()&1)?1.f:-1.f; c16[i]=(int16_t)(rand()%2000-1000); }

        memcpy(y, x, n*4); fwht_scalar(y, n);
        float* y2 = malloc(n*4); memcpy(y2, x, n*4); fwht_neon(y2, n);
        int ok_f = memcmp(y, y2, n*4) == 0;

        signmul_scalar(d0, x, s, n); signmul_neon(d1, x, s, n);
        int ok_s = memcmp(d0, d1, n*4) == 0;

        float a0 = amax_scalar(x, n), a1 = amax_neon(x, n);
        int ok_a = a0 == a1;

        q4pack_scalar(x, p0, n, 3.7f); q4pack_neon(x, p1, n, 3.7f);
        int ok_q = memcmp(p0, p1, n/2) == 0;

        memset(d0, 0, n*4); memset(d1, 0, n*4);
        c16dequant_scalar(d0, c16, n, 0.01f); c16dequant_neon(d1, c16, n, 0.01f);
        int ok_c = memcmp(d0, d1, n*4) == 0;

        // rounding ties: roundf is half-away-from-zero; the NEON path must
        // use vcvtaq (not vcvtnq, which rounds half-to-even) to match
        float ties[32];
        for (int i = 0; i < 32; ++i) ties[i] = (i - 16) * 0.5f;   // ..., -0.5, 0, 0.5, 1.5, ...
        uint8_t tp0[16], tp1[16];
        q4pack_scalar(ties, tp0, 32, 1.0f);
        q4pack_neon  (ties, tp1, 32, 1.0f);
        int ok_t = memcmp(tp0, tp1, 16) == 0;

        printf("K=%-6d exactness: fwht %s  signmul %s  amax %s  q4pack %s  c16dq %s  q4ties %s\n",
               n, ok_f?"OK":"DIFF", ok_s?"OK":"DIFF", ok_a?"OK":"DIFF",
               ok_q?"OK":"DIFF", ok_c?"OK":"DIFF", ok_t?"OK":"DIFF");
        free(x);free(y);free(y2);free(s);free(d0);free(d1);free(p0);free(p1);free(c16);
    }

    printf("\nper-call time (us), scalar vs NEON:\n");
    printf("%-12s", "step");
    for (unsigned ki = 0; ki < 3; ++ki) printf("  K=%-6d scal/neon (x)   ", Ks[ki]);
    printf("\n");
    for (int step = 0; step < 5; ++step) {
        printf("%-12s", names[step]);
        for (unsigned ki = 0; ki < 3; ++ki) {
            int n = Ks[ki];
            ctx_t c = {0};
            c.a = malloc(n*4); c.b = calloc(n,4); c.s = malloc(n*4);
            c.p = malloc(n/2); c.c16 = malloc(n*2); c.n = n;
            for (int i = 0; i < n; ++i) { c.a[i]=frand(); c.s[i]=(i&1)?1.f:-1.f; c.c16[i]=(int16_t)i; }
            int iters = n > 8192 ? 300 : 2000;
            c.which = step*2;     double t_s = bench_us(run_one, &c, iters);
            c.which = step*2 + 1; double t_n = bench_us(run_one, &c, iters);
            printf("  %7.2f/%-7.2f (%4.1fx)  ", t_s, t_n, t_s/t_n);
            free(c.a);free(c.b);free(c.s);free(c.p);free(c.c16);
        }
        printf("\n");
    }
    return 0;
}
