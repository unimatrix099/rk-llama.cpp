// Per-run fixed cost of rknn_matmul_run at decode shape (M=1, INT4xINT4,
// AC_layout=NATIVE — the backend's default W4A4 decode configuration).
//
// Purpose: split the NPU time of a decoded token into a fixed per-run cost
// (driver submit, IRQ, wake-up) and a per-byte cost (weight streaming). The
// backend issues ~1,000 runs per Gemma-4 E4B token (341 nodes x 3 cores), so
// a fixed cost of tens of microseconds would be a large, addressable share.
// Sweeps N at fixed K on one core, then fits t = a + bytes / bw by least
// squares. Also times three cores in parallel at the E4B FFN shape.
//
// Build and run on the target board:
//   gcc rknpu2-run-latency-probe.c -o run-latency-probe -O2 -lpthread \
//       -I ../../ggml/src/ggml-rknpu2/libs/include \
//       ../../ggml/src/ggml-rknpu2/libs/librknnrt.so
//   ulimit -n 65536
//   LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs ./run-latency-probe

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <rknn_matmul_api.h>

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

typedef struct {
    rknn_matmul_ctx ctx;
    rknn_matmul_io_attr io;
    rknn_tensor_mem *A, *B, *C;
} mm_t;

static int mm_create(mm_t* m, int K, int N, rknn_core_mask core) {
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = 1; info.K = K; info.N = N;
    info.type = RKNN_INT4_MM_INT4_TO_INT16;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NATIVE;
    if (rknn_matmul_create(&m->ctx, &info, &m->io) != 0) return -1;
    rknn_matmul_set_core_mask(m->ctx, core);
    m->A = rknn_create_mem(m->ctx, m->io.A.size);
    m->B = rknn_create_mem(m->ctx, m->io.B.size);
    m->C = rknn_create_mem(m->ctx, m->io.C.size);
    if (!m->A || !m->B || !m->C) return -1;
    memset(m->A->virt_addr, 0x11, m->io.A.size);
    memset(m->B->virt_addr, 0x22, m->io.B.size);
    rknn_matmul_set_io_mem(m->ctx, m->A, &m->io.A);
    rknn_matmul_set_io_mem(m->ctx, m->B, &m->io.B);
    rknn_matmul_set_io_mem(m->ctx, m->C, &m->io.C);
    for (int i = 0; i < 5; ++i) rknn_matmul_run(m->ctx);
    return 0;
}

static void mm_destroy(mm_t* m) {
    rknn_destroy_mem(m->ctx, m->A); rknn_destroy_mem(m->ctx, m->B); rknn_destroy_mem(m->ctx, m->C);
    rknn_matmul_destroy(m->ctx);
}

// median of n run times (us)
static double time_runs(mm_t* m, int n) {
    double t[201];
    if (n > 201) n = 201;
    for (int i = 0; i < n; ++i) { double t0 = now_us(); rknn_matmul_run(m->ctx); t[i] = now_us() - t0; }
    for (int i = 1; i < n; ++i) { double v = t[i]; int j = i - 1; while (j >= 0 && t[j] > v) { t[j + 1] = t[j]; --j; } t[j + 1] = v; }
    return t[n / 2];
}

static void* run_thread(void* p) { rknn_matmul_run(((mm_t*)p)->ctx); return NULL; }

int main(void) {
    const int K = 2560;
    const int Ns[] = {64, 128, 256, 512, 1024, 2048, 3456};
    const int nN = sizeof(Ns) / sizeof(Ns[0]);
    double xs[16], ys[16];
    int np = 0;
    printf("one core, M=1, K=%d, INT4xINT4 NATIVE\n", K);
    printf("%6s %10s %10s %9s\n", "N", "B bytes", "median us", "GB/s");
    for (int i = 0; i < nN; ++i) {
        mm_t m;
        if (mm_create(&m, K, Ns[i], RKNN_NPU_CORE_0) != 0) { printf("%6d create failed\n", Ns[i]); continue; }
        const double us = time_runs(&m, 201);
        const double bytes = (double)m.io.B.size;
        xs[np] = bytes; ys[np] = us; ++np;
        printf("%6d %10.0f %10.1f %9.2f\n", Ns[i], bytes, us, bytes / us / 1e3);
        mm_destroy(&m);
    }
    // least squares t = a + b*bytes
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < np; ++i) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
    const double b = (np * sxy - sx * sy) / (np * sxx - sx * sx);
    const double a = (sy - b * sx) / np;
    printf("fit: fixed %.1f us per run, streaming %.2f GB/s (one core)\n", a, 1e-3 / b);

    // three cores in parallel at the E4B FFN shape (K=2560, N=10240 split 3 ways)
    const rknn_core_mask cores[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    mm_t mm[3];
    for (int c = 0; c < 3; ++c) if (mm_create(&mm[c], K, 3456, cores[c]) != 0) { printf("3-core create failed\n"); return 1; }
    double best = 1e30;
    for (int it = 0; it < 101; ++it) {
        pthread_t th[3];
        const double t0 = now_us();
        for (int c = 0; c < 3; ++c) pthread_create(&th[c], NULL, run_thread, &mm[c]);
        for (int c = 0; c < 3; ++c) pthread_join(th[c], NULL);
        const double dt = now_us() - t0;
        if (dt < best) best = dt;
    }
    const double bytes3 = 3.0 * mm[0].io.B.size;
    printf("3 cores, K=%d N=3x3456: best %.1f us (incl. thread create), %.2f GB/s aggregate\n", K, best, bytes3 / best / 1e3);
    for (int c = 0; c < 3; ++c) mm_destroy(&mm[c]);

    // one context spanning all three cores: does the driver split a single
    // matmul across them? (one run per node instead of three)
    const rknn_core_mask masks[2] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_0_1_2};
    const char* mask_name[2] = {"CORE_0", "CORE_0_1_2"};
    for (int k = 0; k < 2; ++k) {
        mm_t m;
        if (mm_create(&m, K, 10368, masks[k]) != 0) { printf("N=10368 %s create failed\n", mask_name[k]); continue; }
        const double us = time_runs(&m, 101);
        printf("one ctx, K=%d N=10368, mask %-10s: median %.1f us, %.2f GB/s\n", K, mask_name[k], us, m.io.B.size / us / 1e3);
        mm_destroy(&m);
    }
    return 0;
}
