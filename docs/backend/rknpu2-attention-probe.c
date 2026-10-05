// Go/no-go probe for running prefill attention on the NPU (Gemma-4 E4B).
//
// Times FP16 rknn_matmul at the attention shapes of a 512-token prefill, with
// the 4 query heads that share a KV head (GQA) stacked into one matmul:
//   S = Q * K^T : M = 4*512, K = 256 (head dim), N = 512 (kv positions)
//   O = P * V   : M = 4*512, K = 512,            N = 256
// B (K^T or V) is runtime data, not a weight, so each call needs it in the
// NPU's native layout: timed both as B_layout = NORM (driver converts) and as
// a CPU-side rknn_B_normal_layout_to_native_layout call plus a NATIVE run.
// Compare the per-prefill estimate with the CPU flash-attention cost
// measured on the board (~2.6 ms/token at pp512, ~1.3 s per 512 tokens).
//
// Build and run on the target board:
//   gcc rknpu2-attention-probe.c -o attention-probe -O2 \
//       -I ../../ggml/src/ggml-rknpu2/libs/include \
//       ../../ggml/src/ggml-rknpu2/libs/librknnrt.so
//   ulimit -n 65536
//   LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs ./attention-probe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <rknn_matmul_api.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// best-of-n ms for one run; conv_ms gets the B conversion cost when native_b
static double bench(int M, int K, int N, int native_b, int ac_native, double* conv_ms) {
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = native_b ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;
    info.AC_layout = ac_native ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;
    rknn_matmul_io_attr io; rknn_matmul_ctx ctx = 0;
    if (rknn_matmul_create(&ctx, &info, &io) != 0) { printf("  create failed M=%d K=%d N=%d\n", M, K, N); return -1; }
    rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_0);
    rknn_tensor_mem *A = rknn_create_mem(ctx, io.A.size), *B = rknn_create_mem(ctx, io.B.size), *C = rknn_create_mem(ctx, io.C.size);
    memset(A->virt_addr, 0x11, io.A.size);
    unsigned short* b_norm = (unsigned short*)malloc((size_t)K * N * 2);
    for (size_t i = 0; i < (size_t)K * N; ++i) b_norm[i] = 0x3c00;   // 1.0h
    double conv = 0;
    if (native_b) {
        double best = 1e30;
        for (int i = 0; i < 10; ++i) {
            const double t0 = now_ms();
            rknn_B_normal_layout_to_native_layout(b_norm, B->virt_addr, K, N, &info);
            const double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        conv = best;
    } else {
        memcpy(B->virt_addr, b_norm, (size_t)K * N * 2);
    }
    rknn_matmul_set_io_mem(ctx, A, &io.A);
    rknn_matmul_set_io_mem(ctx, B, &io.B);
    rknn_matmul_set_io_mem(ctx, C, &io.C);
    for (int i = 0; i < 3; ++i) rknn_matmul_run(ctx);
    double best = 1e30;
    for (int i = 0; i < 20; ++i) {
        const double t0 = now_ms();
        rknn_matmul_run(ctx);
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    if (conv_ms) *conv_ms = conv;
    free(b_norm);
    rknn_destroy_mem(ctx, A); rknn_destroy_mem(ctx, B); rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    return best;
}

int main(void) {
    const int M = 4 * 512, D = 256, NKV = 512;
    const int layers = 42, kv_heads = 2;
    printf("E4B prefill attention on the NPU, FP16, one core, best of 20\n");
    for (int nb = 0; nb < 2; ++nb) {
        double c1 = 0, c2 = 0;
        const double s  = bench(M, D, NKV, nb, 1, &c1);    // S = Q K^T
        const double o  = bench(M, NKV, D, nb, 1, &c2);    // O = P V
        if (s < 0 || o < 0) continue;
        const double per_layer = kv_heads * (s + o + c1 + c2);
        printf("B %-6s: QK^T %.3f ms, PV %.3f ms, B conversion %.3f + %.3f ms per KV head\n",
               nb ? "NATIVE" : "NORM", s, o, c1, c2);
        printf("          per prefill (42 layers x 2 KV heads, one core): %.0f ms; spread on 3 cores: ~%.0f ms\n",
               layers * per_layer, layers * per_layer / 3.0);
    }
    printf("CPU flash attention today: ~1300 ms per 512-token prefill (2.6 ms/token)\n");
    return 0;
}
