// Sweep AC_layout (NORM vs NATIVE) x dtype x shape for rknn_matmul on RK3588.
// B is always NATIVE, single NPU core, contents irrelevant to timing.
//
// This is the benchmark that corrected the earlier conclusion in
// RKNPU2-int4-research.md: with AC_layout=NORM (what ggml-rknpu2 uses)
// INT4 is pinned at ~44 GOPS, but with AC_layout=NATIVE it reaches
// ~3.6-3.9 TOPS, i.e. ~2x INT8. The ~80x gap is an SDK layout-conversion
// penalty, not a hardware limit. Always sweep layout before drawing
// conclusions about this NPU.
//
// Build and run on the target board:
//   gcc rknpu2-matmul-layout-bench.c -o layout-bench -O2 \
//       -I ../../ggml/src/ggml-rknpu2/libs/include \
//       ../../ggml/src/ggml-rknpu2/libs/librknnrt.so
//   ulimit -n 65536
//   LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs ./layout-bench

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <rknn_matmul_api.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void bench(int type, const char* name, int M, int K, int N, int ac_native, int iters) {
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N;
    info.type = (rknn_matmul_type)type;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = ac_native ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;

    rknn_matmul_io_attr io; rknn_matmul_ctx ctx = 0;
    if (rknn_matmul_create(&ctx, &info, &io) != 0) {
        printf("  %-18s AC=%-6s -> create failed\n", name, ac_native ? "NATIVE" : "NORM");
        return;
    }
    rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_0);

    rknn_tensor_mem *A = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem *B = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem *C = rknn_create_mem(ctx, io.C.size);
    if (!A || !B || !C) { printf("  alloc failed\n"); rknn_matmul_destroy(ctx); return; }
    memset(A->virt_addr, 0x11, io.A.size);
    memset(B->virt_addr, 0x22, io.B.size);

    rknn_matmul_set_io_mem(ctx, A, &io.A);
    rknn_matmul_set_io_mem(ctx, B, &io.B);
    rknn_matmul_set_io_mem(ctx, C, &io.C);

    for (int i = 0; i < 5; ++i) rknn_matmul_run(ctx);

    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        rknn_matmul_run(ctx);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    printf("  %-18s AC=%-6s  %8.3f ms  %9.2f GOPS\n",
           name, ac_native ? "NATIVE" : "NORM", best, (2.0*M*K*N)/(best*1e6));

    rknn_destroy_mem(ctx, A); rknn_destroy_mem(ctx, B); rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
}

int main(void) {
    struct { int M, K, N; const char* tag; } shapes[] = {
        {128, 1024, 8192, "reference peak shape (marty1885)"},
        {128, 2048, 2048, "our original shape"},
        {128, 1024, 2048, "K=1024 N=2048"},
        {128, 2048, 8192, "K=2048 N=8192"},
        {1,   2048, 2048, "decode M=1"},
    };
    for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s) {
        printf("--- M=%d K=%d N=%d  (%s) ---\n",
               shapes[s].M, shapes[s].K, shapes[s].N, shapes[s].tag);
        int iters = 20;
        for (int ac = 0; ac < 2; ++ac) {
            bench(2,  "INT8xINT8", shapes[s].M, shapes[s].K, shapes[s].N, ac, iters);
            bench(10, "INT4xINT4", shapes[s].M, shapes[s].K, shapes[s].N, ac, iters);
        }
        printf("\n");
    }
    return 0;
}
