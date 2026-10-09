// Which n_kv can the NPU attention contexts be created for (FP16, NATIVE A/C/B)?
#include <stdio.h>
#include <string.h>
#include "rknn_matmul_api.h"
static int try_create(int M, int K, int N, int b_layout) {
    rknn_matmul_info info; rknn_matmul_io_attr io; rknn_matmul_ctx ctx;
    memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N; info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = b_layout; info.AC_layout = RKNN_MM_LAYOUT_NATIVE;
    int r = rknn_matmul_create(&ctx, &info, &io);
    if (r != 0) return 0;
    rknn_tensor_mem *A = rknn_create_mem(ctx, io.A.size), *B = rknn_create_mem(ctx, io.B.size), *C = rknn_create_mem(ctx, io.C.size);
    int ok = A && B && C;
    if (ok) {
        rknn_matmul_set_io_mem(ctx, A, &io.A); rknn_matmul_set_io_mem(ctx, B, &io.B); rknn_matmul_set_io_mem(ctx, C, &io.C);
        ok = rknn_matmul_run(ctx) == 0 ? 1 : 3;
    }
    if (A) rknn_destroy_mem(ctx, A); if (B) rknn_destroy_mem(ctx, B); if (C) rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    return ok ? 1 : 2;
}
#include <stdlib.h>
int main(int argc, char ** argv) {
    const int nkvs[] = {16384, 17408, 18432, 20480, 24576, 28672, 32768};
    printf("%6s | QK D256 | QK D512 | PV D256 | PV D512   (1=ok, 0=create fail, 2=mem fail, 3=run fail)\n", "n_kv");
    for (int i = 1; i < argc; ++i) {
        int n = atoi(argv[i]); (void)nkvs;
        printf("%6d |    %d    |    %d    |    %d    |    %d\n", n,
               try_create(2048, 256, n, RKNN_MM_LAYOUT_NATIVE), try_create(2048, 512, n, RKNN_MM_LAYOUT_NATIVE),
               try_create(2048, n, 256, RKNN_MM_LAYOUT_NATIVE), try_create(2048, n, 512, RKNN_MM_LAYOUT_NATIVE));
        fflush(stdout);
    }
    return 0;
}
