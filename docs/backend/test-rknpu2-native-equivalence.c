// Hardware equivalence test: the same INT4 multiplication through
// AC_layout=NORM (row-major A/C, the backend's current path) and through
// AC_layout=NATIVE with A tiled by rknpu2-native-layout.c must produce
// bit-identical C. Integer in, integer MACs: any difference means the
// tiling contract is wrong.
//
// B uses B_layout=NATIVE in BOTH contexts, converted once from row-major by
// the SDK's own rknn_B_normal_layout_to_native_layout helper — identical
// bytes on both sides, so only the A/C layouts differ (the thing under
// test). B=NORM would be a simpler control but its INT4 path errors at
// set_io_mem time on librknnrt 2.3.2 ("Unsupport type bits 0").
//
// On mismatch the test retries with nibble-swapped A cells and reports
// whether that variant matches instead (diagnostic for byte-order surprises).
//
// Build & run on the board (from docs/backend):
//   make -f Makefile.rknpu2-tools test-rknpu2-native-equivalence
//   ulimit -n 65536
//   LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs ./test-rknpu2-native-equivalence

#include "rknpu2-native-layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rknn_matmul_api.h>

static uint32_t g_seed = 0xC0FFEEu;
static uint32_t prng(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed;
}

// Pack int4 values (range [-7,7]) two per byte, low nibble = even index —
// the convention of rknpu2_quantization::quantize_fp32_to_int4_packed.
static void pack_int4(uint8_t * dst, const int8_t * vals, int n) {
    for (int i = 0; i < n / 2; ++i) {
        dst[i] = (uint8_t)(((uint8_t)vals[2*i] & 0x0F) |
                           (((uint8_t)vals[2*i+1] & 0x0F) << 4));
    }
}

static void swap_nibbles(uint8_t * buf, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        buf[i] = (uint8_t)((buf[i] >> 4) | (buf[i] << 4));
    }
}

// Run one INT4 matmul with the given AC layout. A_packed is row-major
// nibble-packed M x K. B_vals is UNPACKED K x N (one int4 value per byte) —
// the input convention of rknn_B_normal_layout_to_native_layout, determined
// empirically: feeding it nibble-packed data makes it read K*N/2 bytes past
// the buffer (heap garbage -> nondeterministic wrong results). C_out
// receives M*N int16, already row-major (gathered when ac_native).
// Returns 0 on success.
static int run_matmul(int M, int K, int N, int ac_native,
                      const uint8_t * A_packed, const int8_t * B_vals,
                      int16_t * C_out, int swap_a_nibbles) {
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N;
    info.type = RKNN_INT4_MM_INT4_TO_INT16;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = ac_native ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;

    rknn_matmul_io_attr io;
    rknn_matmul_ctx ctx = 0;
    if (rknn_matmul_create(&ctx, &info, &io) != 0) {
        fprintf(stderr, "create failed (M=%d K=%d N=%d ac_native=%d)\n",
                M, K, N, ac_native);
        return -1;
    }
    rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_0);

    rknn_tensor_mem * A = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem * B = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem * C = rknn_create_mem(ctx, io.C.size);
    if (!A || !B || !C) { rknn_matmul_destroy(ctx); return -1; }

    const size_t row_bytes = (size_t)K / 2;

    if (!ac_native) {
        memcpy(A->virt_addr, A_packed, (size_t)M * row_bytes);
    } else {
        rknpu2_native_geom ga;
        if (rknpu2_native_geom_from_dims(io.A.dims, io.A.n_dims, &ga) != 0) {
            fprintf(stderr, "unexpected A dims (n_dims=%u)\n", io.A.n_dims);
            rknn_matmul_destroy(ctx);
            return -1;
        }
        // int4: sub elements per cell -> sub/2 bytes per cell
        const int cell_bytes = ga.sub / 2;
        const int row_cells  = ga.outer;
        uint8_t * scratch = malloc(row_bytes);
        for (int m = 0; m < M; ++m) {
            memcpy(scratch, A_packed + (size_t)m * row_bytes, row_bytes);
            if (swap_a_nibbles) swap_nibbles(scratch, row_bytes);
            rknpu2_native_scatter_row((uint8_t*)A->virt_addr, scratch,
                                      m, ga.m_stride, row_cells, cell_bytes);
        }
        free(scratch);
    }

    if (rknn_B_normal_layout_to_native_layout((void*)B_vals, B->virt_addr,
                                               K, N, &info) != 0) {
        fprintf(stderr, "B layout conversion failed\n");
        rknn_matmul_destroy(ctx);
        return -1;
    }

    if (rknn_matmul_set_io_mem(ctx, A, &io.A) != 0 ||
        rknn_matmul_set_io_mem(ctx, B, &io.B) != 0 ||
        rknn_matmul_set_io_mem(ctx, C, &io.C) != 0) {
        fprintf(stderr, "set_io_mem failed (ac_native=%d)\n", ac_native);
        rknn_matmul_destroy(ctx);
        return -1;
    }

    // CPU writes went through a cached mapping; flush them to the device
    // before the run (the backend does exactly this around every matmul).
    if (rknn_mem_sync(ctx, A, RKNN_MEMORY_SYNC_TO_DEVICE)   != 0 ||
        rknn_mem_sync(ctx, B, RKNN_MEMORY_SYNC_TO_DEVICE)   != 0) {
        fprintf(stderr, "mem_sync TO_DEVICE failed\n");
        rknn_matmul_destroy(ctx);
        return -1;
    }

    if (rknn_matmul_run(ctx) != 0) {
        fprintf(stderr, "run failed\n");
        rknn_matmul_destroy(ctx);
        return -1;
    }

    // ... and invalidate before the CPU reads the result back.
    if (rknn_mem_sync(ctx, C, RKNN_MEMORY_SYNC_FROM_DEVICE) != 0) {
        fprintf(stderr, "mem_sync FROM_DEVICE failed\n");
        rknn_matmul_destroy(ctx);
        return -1;
    }

    if (!ac_native) {
        memcpy(C_out, C->virt_addr, (size_t)M * N * sizeof(int16_t));
    } else {
        rknpu2_native_geom gc;
        if (rknpu2_native_geom_from_dims(io.C.dims, io.C.n_dims, &gc) != 0) {
            fprintf(stderr, "unexpected C dims (n_dims=%u)\n", io.C.n_dims);
            rknn_matmul_destroy(ctx);
            return -1;
        }
        const int cell_bytes = gc.sub * (int)sizeof(int16_t);
        for (int m = 0; m < M; ++m) {
            rknpu2_native_gather_row((uint8_t*)(C_out + (size_t)m * N),
                                     (const uint8_t*)C->virt_addr,
                                     m, gc.m_stride, gc.outer, cell_bytes);
        }
    }

    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    return 0;
}

static int test_shape(int M, int K, int N) {
    int8_t  * a_vals = malloc((size_t)M * K);
    int8_t  * b_vals = malloc((size_t)K * N);
    uint8_t * A_packed = malloc((size_t)M * K / 2);
    int16_t * C_norm   = malloc((size_t)M * N * sizeof(int16_t));
    int16_t * C_native = malloc((size_t)M * N * sizeof(int16_t));

    for (size_t i = 0; i < (size_t)M * K; ++i) a_vals[i] = (int8_t)(prng() % 15) - 7;
    for (size_t i = 0; i < (size_t)K * N; ++i) b_vals[i] = (int8_t)(prng() % 15) - 7;
    for (int m = 0; m < M; ++m)
        pack_int4(A_packed + (size_t)m * K / 2, a_vals + (size_t)m * K, K);

    // CPU ground truth (int32 accumulate, then both wrap and saturate views)
    int32_t * C_ref32 = malloc((size_t)M * N * sizeof(int32_t));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)a_vals[(size_t)m * K + k] * b_vals[(size_t)k * N + n];
            C_ref32[(size_t)m * N + n] = acc;
        }

    int rc = -1;
    if (run_matmul(M, K, N, 0, A_packed, b_vals, C_norm, 0)   != 0) goto out;
    if (run_matmul(M, K, N, 1, A_packed, b_vals, C_native, 0) != 0) goto out;

    // Determinism check: re-run each path with identical inputs.
    {
        int16_t * C2 = malloc((size_t)M * N * sizeof(int16_t));
        if (run_matmul(M, K, N, 0, A_packed, b_vals, C2, 0) == 0) {
            size_t d = 0;
            for (size_t i = 0; i < (size_t)M * N; ++i) d += (C_norm[i] != C2[i]);
            if (d) printf("      NORM nondeterministic: %zu/%d differ across runs\n", d, M*N);
        }
        if (run_matmul(M, K, N, 1, A_packed, b_vals, C2, 0) == 0) {
            size_t d = 0;
            for (size_t i = 0; i < (size_t)M * N; ++i) d += (C_native[i] != C2[i]);
            if (d) printf("      NATIVE nondeterministic: %zu/%d differ across runs\n", d, M*N);
        }
        free(C2);
    }

    {
        size_t norm_ok = 0, native_ok = 0;
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            int16_t wrap = (int16_t)C_ref32[i];
            norm_ok   += (C_norm[i]   == wrap);
            native_ok += (C_native[i] == wrap);
        }
        printf("      vs CPU ref (wrap16): NORM %zu/%d ok, NATIVE %zu/%d ok\n",
               norm_ok, M*N, native_ok, M*N);
        int shown = 0;
        for (size_t i = 0; i < (size_t)M * N && shown < 4; ++i) {
            if (C_norm[i] != C_native[i]) {
                printf("      [m=%zu n=%zu] ref32=%d wrap16=%d norm=%d native=%d\n",
                       i / N, i % N, C_ref32[i], (int)(int16_t)C_ref32[i],
                       (int)C_norm[i], (int)C_native[i]);
                ++shown;
            }
        }
    }

    if (memcmp(C_norm, C_native, (size_t)M * N * sizeof(int16_t)) == 0) {
        printf("PASS  M=%-4d K=%-5d N=%-5d  (bit-identical, %d elements)\n",
               M, K, N, M * N);
        rc = 0;
    } else {
        size_t diff = 0;
        for (size_t i = 0; i < (size_t)M * N; ++i) diff += (C_norm[i] != C_native[i]);
        printf("FAIL  M=%-4d K=%-5d N=%-5d  %zu/%d elements differ\n",
               M, K, N, diff, M * N);
        // Diagnostic: does a nibble-swapped A layout match instead?
        if (run_matmul(M, K, N, 1, A_packed, b_vals, C_native, 1) == 0 &&
            memcmp(C_norm, C_native, (size_t)M * N * sizeof(int16_t)) == 0) {
            printf("      -> nibble-SWAPPED variant matches: cell byte order "
                   "is high-nibble-first; fix the scatter contract\n");
        }
        rc = 1;
    }

out:
    free(C_ref32);
    free(a_vals); free(b_vals); free(A_packed);
    free(C_norm); free(C_native);
    return rc;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    // K must be 32-aligned, N 64-aligned for int4 (RK3588 constraints).
    // Covers: decode shape (M=1), non-power-of-2 M, K spanning multiple
    // tiles, N spanning multiple cells, and a prefill-sized case.
    struct { int M, K, N; } shapes[] = {
        {1,   256,  256},
        {5,   320,  256},
        {32,  2048, 2048},
        {128, 1024, 512},
    };
    int failures = 0;
    for (unsigned i = 0; i < sizeof(shapes)/sizeof(shapes[0]); ++i) {
        int rc = test_shape(shapes[i].M, shapes[i].K, shapes[i].N);
        if (rc != 0) ++failures;
    }
    printf("%s\n", failures ? "EQUIVALENCE: FAIL" : "EQUIVALENCE: PASS");
    return failures ? 1 : 0;
}
