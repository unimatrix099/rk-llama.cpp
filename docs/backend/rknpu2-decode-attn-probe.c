// Go/no-go probe: decode/verify attention of one Gemma-4 E4B global layer on the NPU,
// with K and V already resident in NPU-native layout (a persistent KV mirror).
// Per KV head (2 per layer, one NPU core each, run in parallel):
//   Q*K^T  A = Q (M x 512, FP16), B = K (512 x n_kv, native), C = S (M x n_kv, FP16)   one run
//   softmax on the CPU over M x n_kv (FP16 in, FP16 P out)
//   P*V    A = P chunk (M x 2048), B = V chunk (2048 x 512, native), C = M x 512 FP32   n_kv/2048 runs
// M = 4 (decode: 4 query heads per KV head) or 24 (6-token MTP verify).
// Also: cost of syncing one appended token in the persistent K/V buffers.
// build: gcc -O2 -o /tmp/dap rknpu2-decode-attn-probe.c -I../../ggml/src/ggml-rknpu2/libs/include -lrknnrt -lpthread -lm
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_matmul_api.h"

#define DK 512
#define DV 512
#define PVC 2048

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

typedef struct { rknn_matmul_ctx ctx; rknn_matmul_io_attr io; rknn_tensor_mem *A, *B, *C; } mm;

static int mm_create(mm * m, int M, int K, int N, int type, int core) {
    rknn_matmul_info info; memset(&info, 0, sizeof info); memset(m, 0, sizeof *m);
    info.M = M; info.K = K; info.N = N; info.type = type;
    info.B_layout = RKNN_MM_LAYOUT_NATIVE; info.AC_layout = RKNN_MM_LAYOUT_NATIVE;
    if (rknn_matmul_create(&m->ctx, &info, &m->io) != 0) return -1;
    rknn_matmul_set_core_mask(m->ctx, core == 0 ? RKNN_NPU_CORE_0 : core == 1 ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2);
    m->A = rknn_create_mem(m->ctx, m->io.A.size); m->B = rknn_create_mem(m->ctx, m->io.B.size); m->C = rknn_create_mem(m->ctx, m->io.C.size);
    if (!m->A || !m->B || !m->C) return -2;
    // non-zero, finite FP16 data (0x3c00 = 1.0, 0x2e66 ~ 0.1)
    uint16_t * a = m->A->virt_addr; for (size_t i = 0; i < m->io.A.size / 2; ++i) a[i] = 0x2e66;
    uint16_t * b = m->B->virt_addr; for (size_t i = 0; i < m->io.B.size / 2; ++i) b[i] = 0x2e66;
    rknn_matmul_set_io_mem(m->ctx, m->A, &m->io.A); rknn_matmul_set_io_mem(m->ctx, m->B, &m->io.B); rknn_matmul_set_io_mem(m->ctx, m->C, &m->io.C);
    rknn_mem_sync(m->ctx, m->A, RKNN_MEMORY_SYNC_TO_DEVICE); rknn_mem_sync(m->ctx, m->B, RKNN_MEMORY_SYNC_TO_DEVICE);
    return 0;
}
static void mm_free(mm * m) {
    if (!m->ctx) return;
    if (m->A) rknn_destroy_mem(m->ctx, m->A); if (m->B) rknn_destroy_mem(m->ctx, m->B); if (m->C) rknn_destroy_mem(m->ctx, m->C);
    rknn_matmul_destroy(m->ctx); m->ctx = 0;
}

// fp16 <-> fp32 via the compiler's __fp16 (aarch64)
static void softmax_rows(const __fp16 * s, __fp16 * p, int M, int n, int ld_s) {
    for (int r = 0; r < M; ++r) {
        const __fp16 * x = s + (size_t)r * ld_s; __fp16 * y = p + (size_t)r * n;
        float mx = -INFINITY; for (int j = 0; j < n; ++j) mx = fmaxf(mx, (float)x[j]);
        float sum = 0; for (int j = 0; j < n; ++j) { float e = expf((float)x[j] - mx); sum += e; y[j] = (__fp16)e; }
        const float inv = 1.0f / sum; for (int j = 0; j < n; ++j) y[j] = (__fp16)((float)y[j] * inv);
    }
}

typedef struct { mm qk; mm * pv; int npv; int M, n_kv; double t_qk, t_sm, t_pv, t_sync; __fp16 * p; } head;

// softmax split over SMT threads per head (rows, or halves of a row when M is small)
static int SMT = 1;
typedef struct { const __fp16 * s; __fp16 * p; int M, n, r0, r1; } smjob;
static void * sm_thread(void * arg) { smjob * j = arg; softmax_rows(j->s + (size_t)j->r0 * j->n, j->p + (size_t)j->r0 * j->n, j->r1 - j->r0, j->n, j->n); return NULL; }
static void softmax_mt(const __fp16 * s, __fp16 * p, int M, int n) {
    if (SMT <= 1) { softmax_rows(s, p, M, n, n); return; }
    pthread_t th[8]; smjob j[8];
    for (int t = 0; t < SMT; ++t) { j[t] = (smjob){ s, p, M, n, M * t / SMT, M * (t + 1) / SMT }; pthread_create(&th[t], NULL, sm_thread, &j[t]); }
    for (int t = 0; t < SMT; ++t) pthread_join(th[t], NULL);
}

static void * run_head(void * arg) {
    head * h = arg;
    double t0 = now_ms();
    rknn_mem_sync(h->qk.ctx, h->qk.A, RKNN_MEMORY_SYNC_TO_DEVICE);
    rknn_matmul_run(h->qk.ctx);
    rknn_mem_sync(h->qk.ctx, h->qk.C, RKNN_MEMORY_SYNC_FROM_DEVICE);
    double t1 = now_ms();
    // C is native [N/subN][M][subN]; the softmax here just reads it as M x n_kv (same cost, layout ignored)
    softmax_mt((const __fp16 *)h->qk.C->virt_addr, h->p, h->M, h->n_kv);
    double t2 = now_ms();
    for (int c = 0; c < h->npv; ++c) {
        mm * v = &h->pv[c];
        memcpy(v->A->virt_addr, h->p + (size_t)c * h->M * PVC, (size_t)h->M * PVC * 2);   // stand-in for the native A gather
        rknn_mem_sync(v->ctx, v->A, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_matmul_run(v->ctx);
        rknn_mem_sync(v->ctx, v->C, RKNN_MEMORY_SYNC_FROM_DEVICE);
    }
    double t3 = now_ms();
    h->t_qk += t1 - t0; h->t_sm += t2 - t1; h->t_pv += t3 - t2;
    return NULL;
}

// CHECK=1: (a) are rows bit-identical between M=4 and M=24 contexts (decode vs verify)?
// (b) does syncing a from_fd view make CPU writes to the K buffer visible to the NPU?
static int check(void) {
    const int N = 4096, Ms[2] = { 4, 24 };
    mm q[2], v[2];
    for (int i = 0; i < 2; ++i) {
        if (mm_create(&q[i], Ms[i], DK, N, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16, 0) || mm_create(&v[i], Ms[i], PVC, DV, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, 0)) { printf("create failed\n"); return 1; }
    }
    srand(1);
    // same B in both; random A rows; the M=24 context gets the M=4 rows first, then other rows
    __fp16 * bq = q[0].B->virt_addr; for (size_t i = 0; i < q[0].io.B.size / 2; ++i) bq[i] = (__fp16)((rand() % 2001 - 1000) / 1000.0f);
    memcpy(q[1].B->virt_addr, bq, q[0].io.B.size);
    __fp16 * bv = v[0].B->virt_addr; for (size_t i = 0; i < v[0].io.B.size / 2; ++i) bv[i] = (__fp16)((rand() % 2001 - 1000) / 1000.0f);
    memcpy(v[1].B->virt_addr, bv, v[0].io.B.size);
    // A is native [K/subK][M][subK]: fill row r of each context from the same logical rows
    for (int i = 0; i < 2; ++i) {
        const int M = Ms[i];
        const int sq = q[i].io.A.dims[2], sv = v[i].io.A.dims[2];
        __fp16 * aq = q[i].A->virt_addr; __fp16 * av = v[i].A->virt_addr;
        for (int r = 0; r < M; ++r) {
            srand(100 + r);
            for (int k = 0; k < DK; ++k) aq[((size_t)(k / sq) * M + r) * sq + k % sq] = (__fp16)((rand() % 2001 - 1000) / 4000.0f);
            for (int k = 0; k < PVC; ++k) av[((size_t)(k / sv) * M + r) * sv + k % sv] = (__fp16)((rand() % 1000) / 1000000.0f);
        }
        rknn_mem_sync(q[i].ctx, q[i].A, RKNN_MEMORY_SYNC_TO_DEVICE); rknn_mem_sync(q[i].ctx, q[i].B, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_mem_sync(v[i].ctx, v[i].A, RKNN_MEMORY_SYNC_TO_DEVICE); rknn_mem_sync(v[i].ctx, v[i].B, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_matmul_run(q[i].ctx); rknn_matmul_run(v[i].ctx);
        rknn_mem_sync(q[i].ctx, q[i].C, RKNN_MEMORY_SYNC_FROM_DEVICE); rknn_mem_sync(v[i].ctx, v[i].C, RKNN_MEMORY_SYNC_FROM_DEVICE);
    }
    // compare rows 0..3 (C native [N/subN][M][subN])
    int dq = 0, dv = 0;
    for (int r = 0; r < 4; ++r) {
        const int s0 = q[0].io.C.dims[2], s1 = q[1].io.C.dims[2];
        for (int n = 0; n < N; ++n) {
            uint16_t x = ((uint16_t *)q[0].C->virt_addr)[((size_t)(n / s0) * 4 + r) * s0 + n % s0];
            uint16_t y = ((uint16_t *)q[1].C->virt_addr)[((size_t)(n / s1) * 24 + r) * s1 + n % s1];
            dq += x != y;
        }
        const int t0 = v[0].io.C.dims[2], t1 = v[1].io.C.dims[2];
        for (int n = 0; n < DV; ++n) {
            float x = ((float *)v[0].C->virt_addr)[((size_t)(n / t0) * 4 + r) * t0 + n % t0];
            float y = ((float *)v[1].C->virt_addr)[((size_t)(n / t1) * 24 + r) * t1 + n % t1];
            dv += memcmp(&x, &y, 4) != 0;
        }
    }
    printf("batch invariance M=4 vs M=24, rows 0-3: Q*K^T %d / %d values differ, P*V %d / %d differ\n", dq, 4 * N, dv, 4 * DV);
    // (b) write K for the last 16 positions through the native B layout [N/subN][K/subK][subN][subK],
    // sync only the touched byte range through a view (then, as a control, the whole buffer), re-run, read C
    const int sN = q[0].io.B.dims[2], sK = q[0].io.B.dims[3], kb = DK / sK;
    printf("native B dims [%u %u %u %u]\n", q[0].io.B.dims[0], q[0].io.B.dims[1], q[0].io.B.dims[2], q[0].io.B.dims[3]);
    size_t lo = (size_t)-1, hi = 0;
    for (int n = N - 16; n < N; ++n) for (int k = 0; k < DK; ++k) {
        size_t idx = (((size_t)(n / sN) * kb + k / sK) * sN + n % sN) * sK + k % sK;
        bq[idx] = (__fp16)0.5f; if (idx < lo) lo = idx; if (idx > hi) hi = idx;
    }
    const uint32_t off = (uint32_t)(lo * 2) & ~4095u, vb = (uint32_t)(((hi + 1) * 2 - off + 4095) & ~4095u);
    const int sq = q[0].io.A.dims[2], s0 = q[0].io.C.dims[2];
    float e = 0; for (int k = 0; k < DK; ++k) e += 0.5f * (float)((__fp16 *)q[0].A->virt_addr)[((size_t)(k / sq) * 4) * sq + k % sq];
    rknn_tensor_mem * view = rknn_create_mem_from_fd(q[0].ctx, q[0].B->fd, (uint8_t *)q[0].B->virt_addr + off, vb, off);
    if (!view) { printf("view creation failed\n"); return 1; }
    for (int pass = 0; pass < 2; ++pass) {
        rknn_mem_sync(q[0].ctx, pass == 0 ? view : q[0].B, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_matmul_run(q[0].ctx);
        rknn_mem_sync(q[0].ctx, q[0].C, RKNN_MEMORY_SYNC_FROM_DEVICE);
        int ok = 0;
        for (int n = N - 16; n < N; ++n) { float c = (float)((__fp16 *)q[0].C->virt_addr)[((size_t)(n / s0) * 4) * s0 + n % s0]; ok += fabsf(c - e) < 0.05f + 0.01f * fabsf(e); }
        printf("%s sync (%u bytes at %u): %d / 16 updated positions as expected (%.3f)\n", pass == 0 ? "view" : "full", pass == 0 ? vb : (unsigned) q[0].B->size, pass == 0 ? off : 0, ok, e);
    }
    rknn_destroy_mem(q[0].ctx, view);
    // (c) non-cacheable B: CPU writes need no sync. Rebind q[0] to a non-cacheable copy, change K, run with no sync
    rknn_tensor_mem * nc = rknn_create_mem2(q[0].ctx, q[0].io.B.size, RKNN_FLAG_MEMORY_NON_CACHEABLE);
    if (!nc) { printf("non-cacheable alloc failed\n"); return 1; }
    memcpy(nc->virt_addr, q[0].B->virt_addr, q[0].io.B.size);
    rknn_matmul_set_io_mem(q[0].ctx, nc, &q[0].io.B);
    for (int n = N - 16; n < N; ++n) for (int k = 0; k < DK; ++k)
        ((__fp16 *)nc->virt_addr)[(((size_t)(n / sN) * kb + k / sK) * sN + n % sN) * sK + k % sK] = (__fp16)0.25f;
    if (getenv("DSB")) __asm__ volatile("dsb sy" ::: "memory");
    rknn_matmul_run(q[0].ctx);
    rknn_mem_sync(q[0].ctx, q[0].C, RKNN_MEMORY_SYNC_FROM_DEVICE);
    int ok = 0;
    for (int n = N - 16; n < N; ++n) { float c = (float)((__fp16 *)q[0].C->virt_addr)[((size_t)(n / s0) * 4) * s0 + n % s0]; ok += fabsf(c - e / 2) < 0.05f + 0.01f * fabsf(e); }
    printf("non-cacheable B, no sync: %d / 16 updated positions as expected (%.3f):", ok, e / 2);
    for (int n = N - 16; n < N; ++n) printf(" %.3f", (float)((__fp16 *)q[0].C->virt_addr)[((size_t)(n / s0) * 4) * s0 + n % s0]);
    printf("\n");
    double t0 = now_ms(); for (int it = 0; it < 20; ++it) rknn_matmul_run(q[0].ctx); double t1 = now_ms();
    rknn_matmul_set_io_mem(q[0].ctx, q[0].B, &q[0].io.B);
    double t2 = now_ms(); for (int it = 0; it < 20; ++it) rknn_matmul_run(q[0].ctx); double t3 = now_ms();
    printf("Q*K^T run (N=%d): non-cacheable B %.3f ms, cacheable B %.3f ms\n", N, (t1 - t0) / 20, (t3 - t2) / 20);
    double t4 = now_ms(); for (int it = 0; it < 1000; ++it) for (int k = 0; k < DK; ++k) ((__fp16 *)nc->virt_addr)[(((size_t)((N - 1) / sN) * kb + k / sK) * sN + (N - 1) % sN) * sK + k % sK] = (__fp16)(float)it; double t5 = now_ms();
    printf("append one position (512 scattered FP16 writes) into non-cacheable B: %.4f ms\n", (t5 - t4) / 1000);
    rknn_destroy_mem(q[0].ctx, nc);
    // (d) cacheable B, append by CPU writes, then clean only the touched cache lines (dc cvac) + dsb, no rknn sync
    int good = 0, trials = 20;
    double tc = 0;
    for (int tr = 0; tr < trials; ++tr) {
        const float val = 0.125f * (1 + tr % 4);
        for (int n = N - 16; n < N; ++n) for (int k = 0; k < DK; ++k)
            bq[(((size_t)(n / sN) * kb + k / sK) * sN + n % sN) * sK + k % sK] = (__fp16)val;
        double c0 = now_ms();
        for (int n = N - 16; n < N; ++n) for (int k = 0; k < DK; k += 32) {   // 32 FP16 = 64 bytes = one line
            void * a = &bq[(((size_t)(n / sN) * kb + k / sK) * sN + n % sN) * sK + k % sK];
            __asm__ volatile("dc cvac, %0" :: "r"(a) : "memory");
        }
        __asm__ volatile("dsb sy" ::: "memory");
        tc += now_ms() - c0;
        rknn_matmul_run(q[0].ctx);
        rknn_mem_sync(q[0].ctx, q[0].C, RKNN_MEMORY_SYNC_FROM_DEVICE);
        int okc = 0;
        for (int n = N - 16; n < N; ++n) { float c = (float)((__fp16 *)q[0].C->virt_addr)[((size_t)(n / s0) * 4) * s0 + n % s0]; okc += fabsf(c - e * val / 0.5f) < 0.05f + 0.01f * fabsf(e); }
        good += okc == 16;
    }
    printf("cacheable B + dc cvac on touched lines: %d / %d trials fully correct; clean of 16 positions %.4f ms\n", good, trials, tc / trials);
    return 0;
}

// CHECK=2: does a row's result depend on its index within M (M=32 context, same row at index 0 and at 1..31)?
static int check_rowpos(void) {
    const int N = 2048, M = 32;
    mm q, v;
    if (mm_create(&q, M, DK, N, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16, 0) || mm_create(&v, M, PVC, DV, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, 0)) { printf("create failed\n"); return 1; }
    srand(7);
    __fp16 * bq = q.B->virt_addr; for (size_t i = 0; i < q.io.B.size / 2; ++i) bq[i] = (__fp16)((rand() % 2001 - 1000) / 1000.0f);
    __fp16 * bv = v.B->virt_addr; for (size_t i = 0; i < v.io.B.size / 2; ++i) bv[i] = (__fp16)((rand() % 2001 - 1000) / 1000.0f);
    const int sq = q.io.A.dims[2], sv = v.io.A.dims[2], cq = q.io.C.dims[2], cv = v.io.C.dims[2];
    __fp16 * aq = q.A->virt_addr; __fp16 * av = v.A->virt_addr;
    // every row holds the same data (row 0's), so each output row must equal row 0's if position does not matter
    srand(11);
    __fp16 rq[DK], rv[PVC];
    for (int k = 0; k < DK; ++k) rq[k] = (__fp16)((rand() % 2001 - 1000) / 4000.0f);
    for (int k = 0; k < PVC; ++k) rv[k] = (__fp16)((rand() % 1000) / 1000000.0f);
    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < DK; ++k) aq[((size_t)(k / sq) * M + r) * sq + k % sq] = rq[k];
        for (int k = 0; k < PVC; ++k) av[((size_t)(k / sv) * M + r) * sv + k % sv] = rv[k];
    }
    rknn_mem_sync(q.ctx, q.A, RKNN_MEMORY_SYNC_TO_DEVICE); rknn_mem_sync(q.ctx, q.B, RKNN_MEMORY_SYNC_TO_DEVICE);
    rknn_mem_sync(v.ctx, v.A, RKNN_MEMORY_SYNC_TO_DEVICE); rknn_mem_sync(v.ctx, v.B, RKNN_MEMORY_SYNC_TO_DEVICE);
    rknn_matmul_run(q.ctx); rknn_matmul_run(v.ctx);
    rknn_mem_sync(q.ctx, q.C, RKNN_MEMORY_SYNC_FROM_DEVICE); rknn_mem_sync(v.ctx, v.C, RKNN_MEMORY_SYNC_FROM_DEVICE);
    int dq = 0, dv = 0;
    for (int r = 1; r < M; ++r) {
        for (int n = 0; n < N; ++n) dq += ((uint16_t *)q.C->virt_addr)[((size_t)(n / cq) * M + r) * cq + n % cq] != ((uint16_t *)q.C->virt_addr)[((size_t)(n / cq) * M) * cq + n % cq];
        for (int n = 0; n < DV; ++n) dv += memcmp(&((float *)v.C->virt_addr)[((size_t)(n / cv) * M + r) * cv + n % cv], &((float *)v.C->virt_addr)[((size_t)(n / cv) * M) * cv + n % cv], 4) != 0;
    }
    printf("row-position invariance (M=32, identical rows): Q*K^T %d / %d differ from row 0, P*V %d / %d\n", dq, 31 * N, dv, 31 * DV);
    return 0;
}

int main(int argc, char ** argv) {
    if (getenv("CHECK") && atoi(getenv("CHECK")) == 2) return check_rowpos();
    if (getenv("CHECK")) return check();
    const int reps = 10;
    if (getenv("SMT")) SMT = atoi(getenv("SMT"));
    int nkvs[] = { 4096, 16384, 32768 }, Ms[] = { 4, 24 };
    printf("per global layer (2 KV heads on 2 NPU cores in parallel), ms; mean of %d after warm-up\n", reps);
    printf("%6s %3s | %7s %7s %7s | %8s | %s\n", "n_kv", "M", "QK", "softmax", "PV", "layer", "append-sync K / V (one token)");
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 2; ++b) {
        const int n_kv = nkvs[a], M = Ms[b], npv = n_kv / PVC;
        head h[2];
        int ok = 1;
        for (int k = 0; k < 2; ++k) {
            memset(&h[k], 0, sizeof h[k]); h[k].M = M; h[k].n_kv = n_kv; h[k].npv = npv;
            if (mm_create(&h[k].qk, M, DK, n_kv, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16, k) != 0) { ok = 0; break; }
            h[k].pv = calloc(npv, sizeof(mm));
            for (int c = 0; c < npv; ++c) if (mm_create(&h[k].pv[c], M, PVC, DV, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, k) != 0) { ok = 0; break; }
            h[k].p = malloc((size_t)M * n_kv * 2);
        }
        if (!ok) { printf("%6d %3d | context creation failed\n", n_kv, M); fflush(stdout); continue; }
        double tl = 0;
        for (int it = -2; it < reps; ++it) {
            if (it == 0) for (int k = 0; k < 2; ++k) h[k].t_qk = h[k].t_sm = h[k].t_pv = 0;
            double t0 = now_ms();
            pthread_t th[2];
            for (int k = 0; k < 2; ++k) pthread_create(&th[k], NULL, run_head, &h[k]);
            for (int k = 0; k < 2; ++k) pthread_join(th[k], NULL);
            if (it >= 0) tl += now_ms() - t0;
        }
        // appending one token: CPU writes 512 values into K (one position) and V (last chunk); sync whole buffers
        double ts0 = now_ms();
        for (int it = 0; it < reps; ++it) rknn_mem_sync(h[0].qk.ctx, h[0].qk.B, RKNN_MEMORY_SYNC_TO_DEVICE);
        double ts1 = now_ms();
        for (int it = 0; it < reps; ++it) rknn_mem_sync(h[0].pv[npv - 1].ctx, h[0].pv[npv - 1].B, RKNN_MEMORY_SYNC_TO_DEVICE);
        double ts2 = now_ms();
        // sync only the touched block: a 16-position view of the K buffer made from its fd
        rknn_tensor_mem * kb = h[0].qk.B;
        const uint32_t vb = 16 * DK * 2, off = kb->size - vb;
        rknn_tensor_mem * view = rknn_create_mem_from_fd(h[0].qk.ctx, kb->fd, (uint8_t *)kb->virt_addr + off, vb, off);
        double ts3 = now_ms();
        for (int it = 0; it < reps && view; ++it) rknn_mem_sync(h[0].qk.ctx, view, RKNN_MEMORY_SYNC_TO_DEVICE);
        double ts4 = now_ms();
        if (view) rknn_destroy_mem(h[0].qk.ctx, view);
        printf("%6d %3d | %7.2f %7.2f %7.2f | %8.2f | %.3f / %.3f, K view %s %.3f\n", n_kv, M,
               (h[0].t_qk + h[1].t_qk) / 2 / reps, (h[0].t_sm + h[1].t_sm) / 2 / reps, (h[0].t_pv + h[1].t_pv) / 2 / reps,
               tl / reps, (ts1 - ts0) / reps, (ts2 - ts1) / reps, view ? "ok" : "FAILED", (ts4 - ts3) / reps);
        fflush(stdout);
        for (int k = 0; k < 2; ++k) { mm_free(&h[k].qk); for (int c = 0; c < npv; ++c) mm_free(&h[k].pv[c]); free(h[k].pv); free(h[k].p); }
    }
    return 0;
}
