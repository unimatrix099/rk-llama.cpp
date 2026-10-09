// Measure raw rknn_matmul_run throughput for INT8xINT8 vs INT4xINT4 (and
// FP16xFP16 for reference) at identical shapes, pinned to one NPU core.
//
// Purpose: determine whether the ~4x prefill deficit of the W4A4 pipelines
// versus W8A8 in ggml-rknpu2 comes from the backend's INT4 plumbing (scalar
// quantiser/packer, scalar FWHT, extra segmentation and INT16 output pass)
// or from the NPU's INT4 matmul path itself. Nothing but rknn_matmul_run is
// inside the timed loop: no quantisation, no packing, no scaling.
//
// WARNING: this tool uses AC_layout=NORM (matching ggml-rknpu2's own
// configuration). That path carries a large SDK conversion penalty for INT4
// specifically (~80x). Do not draw conclusions about INT4 hardware capability
// from this tool alone -- use rknpu2-matmul-layout-bench.c, which sweeps the
// layout flag, and see RKNPU2-int4-research.md.
//
// Build and run on the target board:
//   gcc rknpu2-matmul-bench.c -o rknpu2-matmul-bench -O2 \
//       -I ../../ggml/src/ggml-rknpu2/libs/include \
//       ../../ggml/src/ggml-rknpu2/libs/librknnrt.so
//   ulimit -n 65536
//   LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs ./rknpu2-matmul-bench

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <rknn_matmul_api.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Returns median ms per matmul run, or -1 on failure.
static double bench(int type, const char* name, int M, int K, int N, int iters) {
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N;
    info.type = (rknn_matmul_type)type;
    info.B_layout = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;

    rknn_matmul_io_attr io;
    rknn_matmul_ctx ctx = 0;
    if (rknn_matmul_create(&ctx, &info, &io) != 0) {
        printf("  %-22s M=%-4d -> create failed\n", name, M);
        return -1;
    }

    // Pin to a single core so the comparison is per-core throughput
    rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_0);

    rknn_tensor_mem* A = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem* B = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem* C = rknn_create_mem(ctx, io.C.size);
    if (!A || !B || !C) {
        printf("  %-22s M=%-4d -> alloc failed\n", name, M);
        rknn_matmul_destroy(ctx);
        return -1;
    }

    // Deterministic non-zero payload; contents do not affect timing, but
    // zero-filled buffers could in principle hit data-dependent shortcuts.
    memset(A->virt_addr, 0x11, io.A.size);
    memset(B->virt_addr, 0x22, io.B.size);
    memset(C->virt_addr, 0x00, io.C.size);

    rknn_matmul_set_io_mem(ctx, A, &io.A);
    rknn_matmul_set_io_mem(ctx, B, &io.B);
    rknn_matmul_set_io_mem(ctx, C, &io.C);

    for (int i = 0; i < 5; ++i) rknn_matmul_run(ctx);   // warmup

    double best = 1e30, total = 0.0;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        rknn_matmul_run(ctx);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        total += dt;
    }
    double avg = total / iters;

    // 2*M*K*N ops per matmul
    double gops = (2.0 * M * K * N) / (best * 1e6);
    printf("  %-22s M=%-4d best %8.3f ms  avg %8.3f ms  %8.2f GOPS  (A:%u B:%u C:%u bytes)\n",
           name, M, best, avg, gops, io.A.size, io.B.size, io.C.size);

    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    return best;
}

int main(void) {
    const int K = 2048, N = 2048;
    const int Ms[] = {1, 32, 128, 512};
    const int n_ms = sizeof(Ms) / sizeof(Ms[0]);

    printf("Raw rknn_matmul_run throughput, single NPU core, K=%d N=%d\n\n", K, N);

    for (int i = 0; i < n_ms; ++i) {
        int M = Ms[i];
        int iters = (M >= 512) ? 20 : 50;
        printf("--- M=%d ---\n", M);
        double t_f16 = bench(1,  "FP16xFP16->FP32", M, K, N, iters);
        double t_i8  = bench(2,  "INT8xINT8->INT32", M, K, N, iters);
        double t_i4  = bench(10, "INT4xINT4->INT16", M, K, N, iters);
        if (t_i8 > 0 && t_i4 > 0) {
            printf("  => INT4 is %.2fx the time of INT8 (%s)\n",
                   t_i4 / t_i8,
                   t_i4 < t_i8 ? "INT4 faster" : "INT4 slower");
        }
        if (t_i8 > 0 && t_f16 > 0) {
            printf("  => FP16 is %.2fx the time of INT8\n", t_f16 / t_i8);
        }
        printf("\n");
    }
    return 0;
}
