// Probe which rknn_matmul dtype combinations the runtime accepts on this
// platform, and print the resulting io_attr buffer sizes / native B layout.
//
// Used to establish that RK3588 + librknnrt 2.3.2 supports only the
// symmetric types (FP16xFP16, INT8xINT8, INT4xINT4) and rejects every
// mixed-precision combination, i.e. no weight-only (w4a16-style) NPU
// pipeline is possible on this chip. See RKNPU2-int4-research.md.
//
// Build and run on the target board:
//   gcc rknpu2-matmul-probe.c -o rknpu2-matmul-probe \
//       -I ../../ggml/src/ggml-rknpu2/libs/include \
//       ../../ggml/src/ggml-rknpu2/libs/librknnrt.so
//   ulimit -n 65536
//   LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs ./rknpu2-matmul-probe
//
// A rejected type prints "CREATE FAILED ret=-5" here and logs
// "unsupported matmul dtype ... in this platform" from the runtime itself.

#include <stdio.h>
#include <string.h>
#include <rknn_matmul_api.h>

static void probe(int type, const char* name, int M, int K, int N) {
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N;
    info.type = (rknn_matmul_type)type;
    info.B_layout = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;

    rknn_matmul_io_attr io;
    rknn_matmul_ctx ctx = 0;
    int ret = rknn_matmul_create(&ctx, &info, &io);
    if (ret != 0) {
        printf("%-36s M=%d K=%d N=%d -> CREATE FAILED ret=%d\n", name, M, K, N, ret);
        return;
    }
    printf("%-36s M=%d K=%d N=%d -> OK  A:%zu B:%zu C:%zu  B.dims=[", name, M, K, N,
           (size_t)io.A.size, (size_t)io.B.size, (size_t)io.C.size);
    for (unsigned i = 0; i < io.B.n_dims; ++i) printf("%s%d", i ? "," : "", io.B.dims[i]);
    printf("]\n");
    rknn_matmul_destroy(ctx);
}

int main(void) {
    const int M = 32, K = 2048, N = 2048;

    // Symmetric types: expected to succeed on RK3588
    probe(1,  "FLOAT16_MM_FLOAT16_TO_FLOAT32", M, K, N);
    probe(2,  "INT8_MM_INT8_TO_INT32",         M, K, N);
    probe(10, "INT4_MM_INT4_TO_INT16",         M, K, N);

    // Mixed / weight-only types: rejected on RK3588 (available from RK3576)
    probe(5,  "FLOAT16_MM_INT8_TO_FLOAT32",    M, K, N);
    probe(7,  "FLOAT16_MM_INT4_TO_FLOAT32",    M, K, N);
    probe(11, "INT8_MM_INT4_TO_INT32",         M, K, N);

    // Decode-shaped (M=1) variants of the interesting mixed types
    probe(5,  "FLOAT16_MM_INT8_TO_FLOAT32",    1, K, N);
    probe(7,  "FLOAT16_MM_INT4_TO_FLOAT32",    1, K, N);

    // Alignment sensitivity checks
    probe(7,  "FLOAT16_MM_INT4_TO_FLOAT32",    M, K, N + 32);
    probe(5,  "FLOAT16_MM_INT8_TO_FLOAT32",    M, K, N + 16);

    return 0;
}
