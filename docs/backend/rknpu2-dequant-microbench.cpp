// Microbenchmark: native INT16 C -> FP32 dequant (dequant_acc_int16_tiled_perchan_rows)
// at Gemma-4 E4B prefill shapes, 4 OpenMP threads, caches evicted before each run.
// Found the N=10240 slowdown behind decode research #1f iteration 23: ~1.0-1.2 ns per
// element at N=10240 against 0.27-0.33 at N<=2560, with read-only and write-only runs of
// the same access pattern far faster.
//
// Build and run on the board (from the repo root):
//   g++ -O3 -march=armv8.2-a+fp16+dotprod -fopenmp -I ggml/src/ggml-rknpu2 -I ggml/src \
//       -I ggml/include docs/backend/rknpu2-dequant-microbench.cpp \
//       ggml/src/ggml-rknpu2/rknpu2-quantization.cpp -o /tmp/dqbench
//   taskset -c 4-7 /tmp/dqbench N R mode [stride] [parts]
//     N      output width (10240 = ffn gate/up, 2560 = o/down, 2048 = q, 512 = k/v)
//     R      rows per block (the backend uses 4)
//     mode   0 store (first K-segment), 1 accumulate, 2 read C only, 3 write dst only
//     stride dst row stride in floats (default N; try N+16 to break 4 KB aliasing)
//     parts  column parts per row block (1 = whole rows, as in the backend)
#include "rknpu2-quantization.h"
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <arm_neon.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    const int MC = 256, sub = 8;
    const int N = argc > 1 ? atoi(argv[1]) : 10240;
    const int R = argc > 2 ? atoi(argv[2]) : 4;
    const int mode = argc > 3 ? atoi(argv[3]) : 0;
    const int S = argc > 4 ? atoi(argv[4]) : N;   // dst row stride (floats)
    const int P = argc > 5 ? atoi(argv[5]) : 1;   // column parts   // 0 store,1 acc, 2 read-only sum, 3 write-only
    const int outer = N / sub;
    int16_t* C = (int16_t*)aligned_alloc(64, (size_t)outer * MC * sub * 2);
    float* dst = (float*)aligned_alloc(64, ((size_t)MC * S * 4 + 63) / 64 * 64);
    std::vector<float> chan(N, 0.01f), common(MC, 0.5f);
    for (size_t i = 0; i < (size_t)outer * MC * sub; ++i) C[i] = (int16_t)(i * 7);
    memset(dst, 0, (size_t)MC * S * 4);
    std::vector<char> flush(64 << 20);
    double best = 1e9; volatile float sink = 0;
    for (int it = 0; it < 15; ++it) {
        memset(flush.data(), it, flush.size());   // evict caches
        const double t0 = now();
        const int nb = MC / R;
        #pragma omp parallel for num_threads(4) reduction(+:sink)
        for (int item = 0; item < nb * P; ++item) {
            const int part = item / nb, b = item % nb;
            const int t0 = outer * part / P, t1 = outer * (part + 1) / P, n0 = t0 * sub;
            if (mode <= 1) {
                rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(dst + (size_t)b * R * S + n0, S, C + (size_t)t0 * MC * sub, b * R, R, MC, t1 - t0, sub, (t1 - t0) * sub, common.data() + b * R, chan.data() + n0, mode == 0);
            } else if (mode == 2) {
                int32x4_t acc = vdupq_n_s32(0);
                for (int t = 0; t < outer; ++t) for (int r = 0; r < R; ++r) acc = vpadalq_s16(acc, vld1q_s16(C + ((size_t)t * MC + b * R + r) * sub));
                sink += vaddvq_s32(acc);
            } else {
                for (int r = 0; r < R; ++r) { float* d = dst + (size_t)(b * R + r) * N; for (int n = 0; n < N; n += 4) vst1q_f32(d + n, vdupq_n_f32(1.0f)); }
            }
        }
        const double dt = now() - t0;
        if (dt < best) best = dt;
    }
    const double el = (double)MC * N;
    printf("P=%d S=%d N=%d R=%d mode=%d: %.3f ms, %.2f ns/elem\n", P, S, N, R, mode, best * 1e3, best * 1e9 / el);
}
