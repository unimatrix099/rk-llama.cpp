// Print io_attr A/B/C dims for AC_layout=NATIVE vs NORM, INT4 and INT8,
// across M values — establishes the native tiling format the backend
// would need to produce (A) and consume (C).
#include <stdio.h>
#include <string.h>
#include <rknn_matmul_api.h>

static void probe(int type, const char* name, int M, int K, int N, int ac_native) {
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N;
    info.type = (rknn_matmul_type)type;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = ac_native ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;

    rknn_matmul_io_attr io;
    rknn_matmul_ctx ctx = 0;
    if (rknn_matmul_create(&ctx, &info, &io) != 0) {
        printf("%-10s AC=%-6s M=%-4d K=%-5d N=%-5d -> FAILED\n",
               name, ac_native?"NATIVE":"NORM", M, K, N);
        return;
    }
    printf("%-10s AC=%-6s M=%-4d K=%-5d N=%-5d  A(%u B):[", name,
           ac_native?"NATIVE":"NORM", M, K, N, io.A.size);
    for (unsigned i=0;i<io.A.n_dims;++i) printf("%s%d", i?",":"", io.A.dims[i]);
    printf("]  C(%u B):[", io.C.size);
    for (unsigned i=0;i<io.C.n_dims;++i) printf("%s%d", i?",":"", io.C.dims[i]);
    printf("]\n");
    rknn_matmul_destroy(ctx);
}

int main(void) {
    int Ms[] = {1, 2, 5, 32, 128};
    for (unsigned i=0;i<sizeof(Ms)/sizeof(Ms[0]);++i) {
        probe(10, "INT4xINT4", Ms[i], 2048, 2048, 1);
        probe(10, "INT4xINT4", Ms[i], 2048, 2048, 0);
    }
    printf("\n");
    for (unsigned i=0;i<sizeof(Ms)/sizeof(Ms[0]);++i) {
        probe(2,  "INT8xINT8", Ms[i], 2048, 2048, 1);
    }
    printf("\n-- K variation, INT4 native --\n");
    probe(10, "INT4xINT4", 32, 1024, 2048, 1);
    probe(10, "INT4xINT4", 32, 4096, 2048, 1);
    probe(10, "INT4xINT4", 32, 2048, 4096, 1);
    return 0;
}
