// Cooperative CPU+NPU decode probe (the "2-node prototype" gate from
// RKNPU2-decode-research.md, avenue #2).
//
// Question it answers: at M=1 (token generation), do the CPU's and NPU's
// memory bandwidths actually SUM when each engine reads a different slice
// of the same weight matrix — and is the per-node coordination (thread
// wake + driver calls + join) cheap enough that a real backend
// integration would beat both solo paths on a ~250-matmul-node model?
//
// What it does, per shape (K x N):
//   - CPU solo:  Q4_0 x Q8 GEMV over all N output rows, T pinned threads
//                (this is the RKNPU_CPU_DECODE path in miniature: native
//                Q4_0 reads, NEON sdot).
//   - NPU solo:  the full backend per-node driver sequence at M=1 on the
//                packed INT4 copy, N split across 3 cores exactly like
//                ggml_backend_rknpu_graph_compute: set_io A x3, sync A
//                TO_DEVICE, set_io C x3, 3 parallel rknn_matmul_run, sync
//                C FROM_DEVICE x3.  B is pre-bound per context (b_bound),
//                as in the backend's context cache.
//   - Coop:      N split at fraction f: NPU gets the first align64(f*N)
//                rows, CPU threads the rest, both launched concurrently,
//                one join per node.  Sweeps f.
//
// Fidelity notes (kept deliberately, so numbers transfer to the backend):
//   - 4 independent weight sets are cycled node-to-node (distinct B
//     contexts with B pre-bound, distinct Q4_0 buffers) to model
//     consecutive layers and defeat cache reuse.
//   - A and C rknn buffers are shared across sets, matching the backend's
//     a_buffer_cache / c_buffer_cache keyed on geometry.
//   - AC_layout is NORM: decode is layout-neutral at M=1 (measured — see
//     RKNPU2-optimization-notes.md, native-layout section).
//   - NPU dispatch threads are pinned to LITTLE cores (they block in the
//     driver; they must not steal big cores from the GEMV threads).  GEMV
//     threads are pinned to the A76 big cores.
//
// What it does NOT model (mind these before extrapolating):
//   - the CPU-side graph ops between matmuls (norms/attention/AltUp) that
//     compete with GEMV threads in a real decode step;
//   - the per-node A-prep (sign-mul + FWHT + int4 pack, ~5-15 us NEON) and
//     the C dequant pass — both CPU work a real integration must schedule;
//   - K segmentation (probe shapes keep K <= 8192 = max_k_limit).
//
// Decision gate (from the research doc): coop is worth building into the
// backend only if best-f node throughput beats the better solo path by
// >= ~25% after subtracting nothing — the probe already pays the real
// join costs.  Below that, per-node overheads in the full graph will eat
// the difference.
//
// Build and run on the target board (needs the NPU):
//   cd docs/backend && make -f Makefile.rknpu2-tools rknpu2-coop-decode-probe
//   ulimit -n 65536            # and pin clocks (scripts/fix_freq_rk3588.sh)
//   LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs ./rknpu2-coop-decode-probe
//
// Env knobs: COOP_CPU_THREADS (default 4), COOP_BIG_CORES (default
// "4,5,6,7"), COOP_LITTLE_CORES (default "0,1,2"), COOP_NODES (default 64).

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <arm_neon.h>

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#define MAX_SEGS      3   // NPU cores, as in the backend's active_cores {0,1,2}
#define WEIGHT_SETS   4   // distinct weight tensors cycled node-to-node
#define MAX_CPU_THREADS 8
#define N_ALIGN       64  // W4A4 n_align (rknpu2-configuration.cpp)
#define QK            32  // Q4_0 block size

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int parse_core_list(const char* env, const char* dflt, int* out, int max) {
    const char* s = env ? env : dflt;
    int n = 0;
    while (*s && n < max) {
        char* end;
        long v = strtol(s, &end, 10);
        if (end == s) break;
        out[n++] = (int)v;
        s = (*end == ',') ? end + 1 : end;
    }
    return n;
}

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        // Non-fatal: report once so numbers are read with that in mind.
        fprintf(stderr, "warning: could not pin thread to cpu %d\n", cpu);
    }
}

// ---------------------------------------------------------------------------
// CPU side: Q4_0 x Q8 GEMV (llama.cpp block layout: 2-byte fp16 scale +
// 16 bytes of nibbles; low nibbles = elements 0..15, high = 16..31)
// ---------------------------------------------------------------------------

typedef struct {
    __fp16  d;
    uint8_t qs[QK / 2];
} block_q4_0;   // 18 bytes, matches GGUF Q4_0 read volume

static float gemv_row_q4(const block_q4_0* row, int nblocks,
                         const int8_t* act, const float* act_d) {
    float sum = 0.0f;
#if defined(__ARM_FEATURE_DOTPROD)
    for (int b = 0; b < nblocks; ++b) {
        const uint8x16_t q  = vld1q_u8(row[b].qs);
        const int8x16_t  lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
        const int8x16_t  hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)),              vdupq_n_s8(8));
        const int8x16_t  a0 = vld1q_s8(act + b * QK);
        const int8x16_t  a1 = vld1q_s8(act + b * QK + 16);
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, lo, a0);
        acc = vdotq_s32(acc, hi, a1);
        sum += (float)vaddvq_s32(acc) * (float)row[b].d * act_d[b];
    }
#else
    for (int b = 0; b < nblocks; ++b) {
        int32_t acc = 0;
        for (int j = 0; j < QK / 2; ++j) {
            const int lo = (row[b].qs[j] & 0x0F) - 8;
            const int hi = (row[b].qs[j] >> 4)   - 8;
            acc += lo * act[b * QK + j] + hi * act[b * QK + 16 + j];
        }
        sum += (float)acc * (float)row[b].d * act_d[b];
    }
#endif
    return sum;
}

// Scalar reference for the self-test below — always compiled, so the NEON
// path is checked for exactness on every start (project convention: prep
// kernels are validated element-exact against scalar references).
static float gemv_row_q4_ref(const block_q4_0* row, int nblocks,
                             const int8_t* act, const float* act_d) {
    float sum = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        int32_t acc = 0;
        for (int j = 0; j < QK / 2; ++j) {
            const int lo = (row[b].qs[j] & 0x0F) - 8;
            const int hi = (row[b].qs[j] >> 4)   - 8;
            acc += lo * act[b * QK + j] + hi * act[b * QK + 16 + j];
        }
        sum += (float)acc * (float)row[b].d * act_d[b];
    }
    return sum;
}

static int gemv_self_test(void) {
    enum { TK = 96, TB = TK / QK };
    block_q4_0 row[TB];
    int8_t act[TK];
    float  act_d[TB];
    unsigned rng = 0x5eed;
    for (int b = 0; b < TB; ++b) {
        row[b].d = (__fp16)0.013f;
        act_d[b] = 0.021f;
        for (int j = 0; j < QK / 2; ++j) { rng = rng * 1664525u + 1013904223u; row[b].qs[j] = (uint8_t)(rng >> 24); }
    }
    for (int i = 0; i < TK; ++i) { rng = rng * 1664525u + 1013904223u; act[i] = (int8_t)(rng >> 24); }
    const float got = gemv_row_q4(row, TB, act, act_d);
    const float ref = gemv_row_q4_ref(row, TB, act, act_d);
    if (got != ref) {   // integer dot + identical float ops: must be exact
        fprintf(stderr, "GEMV self-test FAILED: neon %.9g vs scalar %.9g\n", got, ref);
        return -1;
    }
    return 0;
}

// Persistent CPU worker pool, generation-counter handshake (the join cost
// is part of what we measure, so no per-node pthread_create).
typedef struct {
    // job description (set by main before bumping gen)
    const block_q4_0* W;        // weight set base
    int nblocks;                // K / QK
    int row0[MAX_CPU_THREADS], row1[MAX_CPU_THREADS];
    const int8_t* act;
    const float*  act_d;
    float* out;
    double end_ms[MAX_CPU_THREADS];

    _Atomic int gen;
    _Atomic int done;
    _Atomic int quit;
    int nthreads;
    int cpus[MAX_CPU_THREADS];
} cpu_pool_t;

typedef struct { cpu_pool_t* pool; int idx; } cpu_worker_arg_t;

static void* cpu_worker(void* argp) {
    cpu_worker_arg_t* arg = (cpu_worker_arg_t*)argp;
    cpu_pool_t* p = arg->pool;
    const int idx = arg->idx;
    pin_to_cpu(p->cpus[idx]);

    int seen = 0;
    for (;;) {
        while (atomic_load_explicit(&p->gen, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&p->quit, memory_order_relaxed)) return NULL;
            __asm__ volatile("yield");
        }
        seen = atomic_load_explicit(&p->gen, memory_order_acquire);

        const int r0 = p->row0[idx], r1 = p->row1[idx];
        for (int r = r0; r < r1; ++r) {
            p->out[r] = gemv_row_q4(p->W + (size_t)r * p->nblocks, p->nblocks,
                                    p->act, p->act_d);
        }
        p->end_ms[idx] = now_ms();
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// NPU side: persistent runner threads, one per matmul segment/core
// ---------------------------------------------------------------------------

typedef struct {
    rknn_matmul_ctx ctx[WEIGHT_SETS];   // this segment's ctx, per weight set
    rknn_matmul_io_attr io[WEIGHT_SETS];
    rknn_tensor_mem* B[WEIGHT_SETS];
    int size_n;
} npu_seg_t;

typedef struct {
    npu_seg_t* segs;
    _Atomic int nsegs;                  // active segments this config; runners beyond it idle
    _Atomic int set_idx;                // which weight set this node uses
    _Atomic int gen;
    _Atomic int done;
    _Atomic int quit;
    int little_cpus[MAX_SEGS];
    int n_little;
} npu_pool_t;

typedef struct { npu_pool_t* pool; int idx; } npu_worker_arg_t;

static void* npu_runner(void* argp) {
    npu_worker_arg_t* arg = (npu_worker_arg_t*)argp;
    npu_pool_t* p = arg->pool;
    const int idx = arg->idx;
    if (p->n_little > 0) pin_to_cpu(p->little_cpus[idx % p->n_little]);

    int seen = 0;
    for (;;) {
        while (atomic_load_explicit(&p->gen, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&p->quit, memory_order_relaxed)) return NULL;
            __asm__ volatile("yield");
        }
        seen = atomic_load_explicit(&p->gen, memory_order_acquire);
        if (idx >= atomic_load_explicit(&p->nsegs, memory_order_relaxed))
            continue;   // inactive this config: never touch done
        const int set = atomic_load_explicit(&p->set_idx, memory_order_relaxed);
        rknn_matmul_run(p->segs[idx].ctx[set]);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
}

// Backend's compute_n_segments logic: equal 64-aligned base, remainder in
// alignment chunks to the first segments.
static int split_n(int N, int sizes[MAX_SEGS]) {
    int base = (N / MAX_SEGS / N_ALIGN) * N_ALIGN;
    int rem  = N - base * MAX_SEGS;
    int nseg = 0;
    for (int i = 0; i < MAX_SEGS; ++i) sizes[i] = base;
    for (int i = 0; rem >= N_ALIGN && i < MAX_SEGS; ++i) { sizes[i] += N_ALIGN; rem -= N_ALIGN; }
    if (rem > 0) sizes[MAX_SEGS - 1] += rem;   // tail < 64, backend keeps it on the last core
    for (int i = 0; i < MAX_SEGS; ++i) if (sizes[i] > 0) nseg++;
    return nseg;
}

static const rknn_core_mask CORE_MASKS[MAX_SEGS] = { RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2 };

// Create the per-segment, per-weight-set contexts for N_npu output rows.
// Returns 0 on success. A and C memories are created once from the first
// context and shared across sets (mirrors the backend buffer caches).
static int npu_setup(int K, int N_npu, npu_seg_t* segs, int* nsegs_out,
                     rknn_tensor_mem** A_out, rknn_tensor_mem* C_out[MAX_SEGS]) {
    int sizes[MAX_SEGS];
    const int nsegs = split_n(N_npu, sizes);
    *nsegs_out = nsegs;
    if (nsegs == 0) return 0;

    memset(segs, 0, sizeof(npu_seg_t) * MAX_SEGS);
    *A_out = NULL;

    int seg_i = 0;
    for (int i = 0; i < MAX_SEGS; ++i) {
        if (sizes[i] == 0) continue;
        npu_seg_t* sg = &segs[seg_i];
        sg->size_n = sizes[i];
        for (int s = 0; s < WEIGHT_SETS; ++s) {
            rknn_matmul_info info;
            memset(&info, 0, sizeof(info));
            info.M = 1; info.K = K; info.N = sizes[i];
            info.type      = RKNN_INT4_MM_INT4_TO_INT16;
            info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
            info.AC_layout = RKNN_MM_LAYOUT_NORM;   // layout-neutral at M=1
            if (rknn_matmul_create(&sg->ctx[s], &info, &sg->io[s]) != 0) {
                fprintf(stderr, "matmul_create failed (K=%d N=%d seg=%d)\n", K, sizes[i], i);
                return -1;
            }
            rknn_matmul_set_core_mask(sg->ctx[s], CORE_MASKS[i]);

            sg->B[s] = rknn_create_mem(sg->ctx[s], sg->io[s].B.size);
            if (!sg->B[s]) { fprintf(stderr, "B alloc failed\n"); return -1; }
            memset(sg->B[s]->virt_addr, 0x22 + s, sg->io[s].B.size);
            // b_bound: B is set once per context, never per node
            if (rknn_matmul_set_io_mem(sg->ctx[s], sg->B[s], &sg->io[s].B) != 0) {
                fprintf(stderr, "set_io_mem B failed\n");
                return -1;
            }
        }
        // Shared A (per shape) and C (per segment geometry), from set 0's ctx
        if (*A_out == NULL) {
            *A_out = rknn_create_mem(segs[0].ctx[0], segs[0].io[0].A.size);
            if (!*A_out) { fprintf(stderr, "A alloc failed\n"); return -1; }
            memset((*A_out)->virt_addr, 0x11, segs[0].io[0].A.size);
        }
        C_out[seg_i] = rknn_create_mem(sg->ctx[0], sg->io[0].C.size);
        if (!C_out[seg_i]) { fprintf(stderr, "C alloc failed\n"); return -1; }
        seg_i++;
    }
    return 0;
}

static void npu_teardown(npu_seg_t* segs, int nsegs,
                         rknn_tensor_mem* A, rknn_tensor_mem* C[MAX_SEGS]) {
    for (int i = 0; i < nsegs; ++i) {
        if (C[i]) rknn_destroy_mem(segs[i].ctx[0], C[i]);
        for (int s = 0; s < WEIGHT_SETS; ++s) {
            if (segs[i].B[s]) rknn_destroy_mem(segs[i].ctx[s], segs[i].B[s]);
        }
    }
    if (A && nsegs > 0) rknn_destroy_mem(segs[0].ctx[0], A);
    for (int i = 0; i < nsegs; ++i)
        for (int s = 0; s < WEIGHT_SETS; ++s)
            if (segs[i].ctx[s]) rknn_matmul_destroy(segs[i].ctx[s]);
}

// ---------------------------------------------------------------------------
// The measured node loop
// ---------------------------------------------------------------------------

typedef struct {
    double node_avg_ms, node_best_ms;
    double cpu_part_ms, npu_part_ms;    // averages of each side's own span
} node_stats_t;

// Runs `nodes` consecutive matmul nodes at the given split. N_npu==0 means
// CPU solo; N_cpu==0 means NPU solo.
static node_stats_t run_nodes(int nodes, int K, int N,
                              int N_npu,
                              npu_pool_t* np, npu_seg_t* segs, int nsegs,
                              rknn_tensor_mem* A, rknn_tensor_mem* C[MAX_SEGS],
                              cpu_pool_t* cp,
                              const block_q4_0* Wsets[WEIGHT_SETS],
                              const int8_t* act, const float* act_d, float* out) {
    const int N_cpu   = N - N_npu;
    const int nblocks = K / QK;
    node_stats_t st = {0, 1e30, 0, 0};

    // Static row partition over the CPU threads for this split
    if (N_cpu > 0) {
        const int T = cp->nthreads;
        int r = N_npu;
        for (int t = 0; t < T; ++t) {
            int chunk = (N_cpu + T - 1 - t) / T;    // balanced
            cp->row0[t] = r;
            cp->row1[t] = r + chunk;
            r += chunk;
        }
        cp->nblocks = nblocks;
        cp->act = act; cp->act_d = act_d; cp->out = out;
    }

    for (int node = -4; node < nodes; ++node) {     // 4 warmup nodes
        const int set = ((unsigned)node) % WEIGHT_SETS;
        const double t0 = now_ms();
        double t_cpu = t0, t_npu = t0;

        // Launch CPU side first — it runs while we do the NPU driver calls
        if (N_cpu > 0) {
            cp->W = Wsets[set];
            atomic_store_explicit(&cp->done, 0, memory_order_relaxed);
            atomic_fetch_add_explicit(&cp->gen, 1, memory_order_release);
        }

        if (N_npu > 0) {
            // Per-node driver sequence, exactly as graph_compute does it
            for (int i = 0; i < nsegs; ++i)
                rknn_matmul_set_io_mem(segs[i].ctx[set], A, &segs[i].io[set].A);
            rknn_mem_sync(segs[0].ctx[set], A, RKNN_MEMORY_SYNC_TO_DEVICE);
            for (int i = 0; i < nsegs; ++i)
                rknn_matmul_set_io_mem(segs[i].ctx[set], C[i], &segs[i].io[set].C);

            atomic_store_explicit(&np->set_idx, set, memory_order_relaxed);
            atomic_store_explicit(&np->done, 0, memory_order_relaxed);
            atomic_fetch_add_explicit(&np->gen, 1, memory_order_release);
            while (atomic_load_explicit(&np->done, memory_order_acquire) < nsegs)
                __asm__ volatile("yield");
            for (int i = 0; i < nsegs; ++i)
                rknn_mem_sync(segs[i].ctx[set], C[i], RKNN_MEMORY_SYNC_FROM_DEVICE);
            t_npu = now_ms();
        }

        if (N_cpu > 0) {
            while (atomic_load_explicit(&cp->done, memory_order_acquire) < cp->nthreads)
                __asm__ volatile("yield");
            t_cpu = cp->end_ms[0];
            for (int t = 1; t < cp->nthreads; ++t)
                if (cp->end_ms[t] > t_cpu) t_cpu = cp->end_ms[t];
        }

        const double t1 = now_ms();
        if (node < 0) continue;
        const double dt = t1 - t0;
        st.node_avg_ms += dt;
        if (dt < st.node_best_ms) st.node_best_ms = dt;
        st.cpu_part_ms += (t_cpu - t0);
        st.npu_part_ms += (t_npu - t0);
    }
    st.node_avg_ms /= nodes;
    st.cpu_part_ms /= nodes;
    st.npu_part_ms /= nodes;
    return st;
}

int main(void) {
    if (gemv_self_test() != 0) return 1;

    int big_cores[MAX_CPU_THREADS], little_cores[MAX_SEGS];
    const int n_big    = parse_core_list(getenv("COOP_BIG_CORES"),    "4,5,6,7", big_cores,    MAX_CPU_THREADS);
    const int n_little = parse_core_list(getenv("COOP_LITTLE_CORES"), "0,1,2",   little_cores, MAX_SEGS);

    int nthreads = getenv("COOP_CPU_THREADS") ? atoi(getenv("COOP_CPU_THREADS")) : 4;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > MAX_CPU_THREADS) nthreads = MAX_CPU_THREADS;

    const int nodes = getenv("COOP_NODES") ? atoi(getenv("COOP_NODES")) : 64;

    // Main thread spin-waits on the pools; keep it off the big cores so it
    // never steals cycles from the GEMV threads (COOP_MAIN_CORE to change).
    const int main_core = getenv("COOP_MAIN_CORE") ? atoi(getenv("COOP_MAIN_CORE")) : 3;
    pin_to_cpu(main_core);

    // NPU fraction sweep; solos first so every coop row can be compared
    // against the better solo. 0 = CPU solo, 1 = NPU solo.
    const double fracs[] = {0.0, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125};
    const int nfracs = (int)(sizeof(fracs) / sizeof(fracs[0]));

    // COOP_CPU_ONLY=1: skip every NPU config (smoke-testing the pool and
    // GEMV kernel on a machine without the NPU)
    const int cpu_only = getenv("COOP_CPU_ONLY") ? atoi(getenv("COOP_CPU_ONLY")) : 0;

    // Representative decode shapes (K <= max_k_limit 8192 so no K split)
    const struct { int K, N; } shapes[] = {
        {2048, 2048}, {2048, 16384}, {8192, 2048}, {4096, 4096},
    };
    const int nshapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("Cooperative CPU+NPU decode probe: M=1, INT4 NPU (B NATIVE / AC NORM)"
           " vs Q4_0 CPU GEMV\n");
    printf("cpu threads=%d on cores", nthreads);
    for (int i = 0; i < nthreads; ++i) printf(" %d", big_cores[i % n_big]);
    printf("; npu runners on cores");
    for (int i = 0; i < n_little; ++i) printf(" %d", little_cores[i]);
    printf("; %d timed nodes per config, %d weight sets cycled\n\n", nodes, WEIGHT_SETS);
#if !defined(__ARM_FEATURE_DOTPROD)
    printf("WARNING: built without +dotprod — CPU GEMV numbers will undersell the CPU\n\n");
#endif

    // --- CPU worker pool (persistent across all configs)
    cpu_pool_t cpool;
    memset(&cpool, 0, sizeof(cpool));
    cpool.nthreads = nthreads;
    for (int t = 0; t < nthreads; ++t) cpool.cpus[t] = big_cores[t % n_big];
    pthread_t cpu_threads[MAX_CPU_THREADS];
    cpu_worker_arg_t cargs[MAX_CPU_THREADS];
    for (int t = 0; t < nthreads; ++t) {
        cargs[t].pool = &cpool; cargs[t].idx = t;
        pthread_create(&cpu_threads[t], NULL, cpu_worker, &cargs[t]);
    }

    // --- NPU runner pool (persistent; segs pointer swapped per config)
    npu_pool_t npool;
    memset(&npool, 0, sizeof(npool));
    npool.n_little = n_little;
    for (int i = 0; i < n_little; ++i) npool.little_cpus[i] = little_cores[i];
    pthread_t npu_threads[MAX_SEGS];
    npu_worker_arg_t nargs[MAX_SEGS];
    npu_seg_t segs[MAX_SEGS];
    npool.segs = segs;
    for (int i = 0; i < MAX_SEGS; ++i) {
        nargs[i].pool = &npool; nargs[i].idx = i;
        pthread_create(&npu_threads[i], NULL, npu_runner, &nargs[i]);
    }

    for (int sh = 0; sh < nshapes; ++sh) {
        const int K = shapes[sh].K, N = shapes[sh].N;
        const int nblocks = K / QK;
        const size_t cpu_row_bytes = (size_t)nblocks * sizeof(block_q4_0);

        // CPU-side buffers: WEIGHT_SETS full Q4_0 matrices, one activation
        block_q4_0* Wbufs[WEIGHT_SETS];
        const block_q4_0* Wsets[WEIGHT_SETS];
        for (int s = 0; s < WEIGHT_SETS; ++s) {
            Wbufs[s] = (block_q4_0*)malloc((size_t)N * cpu_row_bytes);
            if (!Wbufs[s]) { fprintf(stderr, "OOM cpu weights\n"); return 1; }
            memset(Wbufs[s], 0x35 + s, (size_t)N * cpu_row_bytes);
            for (size_t b = 0; b < (size_t)N * nblocks; ++b) Wbufs[s][b].d = (__fp16)0.01f;
            Wsets[s] = Wbufs[s];
        }
        int8_t* act   = (int8_t*)malloc(K);
        float*  act_d = (float*)malloc((size_t)nblocks * sizeof(float));
        float*  out   = (float*)malloc((size_t)N * sizeof(float));
        for (int i = 0; i < K; ++i) act[i] = (int8_t)((i * 7) % 17 - 8);
        for (int b = 0; b < nblocks; ++b) act_d[b] = 0.02f;

        printf("=== K=%d N=%d  (CPU bytes/node full-N: %.1f MB, NPU int4 full-N: %.1f MB) ===\n",
               K, N, (double)N * cpu_row_bytes / 1e6, (double)K * N / 2.0 / 1e6);
        printf("  %-9s %6s %6s | %9s %9s %9s | %8s %8s\n",
               "f_npu", "N_npu", "N_cpu", "node avg", "node best", "cpu|npu ms", "GB/s", "vs solo");

        double best_solo_ms = 1e30, best_coop_ms = 1e30;
        double solo_rows[2] = {-1, -1};   // [0]=cpu solo, [1]=npu solo avg ms

        for (int fi = 0; fi < nfracs; ++fi) {
            if (cpu_only && fracs[fi] > 0.0) continue;

            int N_npu = (int)(fracs[fi] * N + 0.5);
            N_npu = (N_npu / N_ALIGN) * N_ALIGN;
            if (fracs[fi] == 1.0) N_npu = N;
            if (fracs[fi] == 0.0) N_npu = 0;
            const int N_cpu = N - N_npu;

            int nsegs = 0;
            rknn_tensor_mem* A = NULL;
            rknn_tensor_mem* C[MAX_SEGS] = {NULL, NULL, NULL};
            if (N_npu > 0) {
                if (npu_setup(K, N_npu, segs, &nsegs, &A, C) != 0) return 1;
            }
            atomic_store_explicit(&npool.nsegs, nsegs, memory_order_release);

            node_stats_t st = run_nodes(nodes, K, N, N_npu,
                                        &npool, segs, nsegs, A, C,
                                        &cpool, Wsets, act, act_d, out);

            const double bytes = (double)N_cpu * cpu_row_bytes + (double)K * N_npu / 2.0;
            const double gbps  = bytes / (st.node_avg_ms * 1e6);

            const int is_solo = (N_npu == 0 || N_cpu == 0);
            if (is_solo) {
                if (st.node_avg_ms < best_solo_ms) best_solo_ms = st.node_avg_ms;
                solo_rows[N_npu == 0 ? 0 : 1] = st.node_avg_ms;
            } else if (st.node_avg_ms < best_coop_ms) {
                best_coop_ms = st.node_avg_ms;
            }

            printf("  %-9.3f %6d %6d | %8.3f  %8.3f  %4.2f|%4.2f | %8.2f",
                   fracs[fi], N_npu, N_cpu,
                   st.node_avg_ms, st.node_best_ms, st.cpu_part_ms, st.npu_part_ms, gbps);
            if (!is_solo && best_solo_ms < 1e29)
                printf("  %+7.1f%%", (best_solo_ms / st.node_avg_ms - 1.0) * 100.0);
            printf("\n");

            if (N_npu > 0) npu_teardown(segs, nsegs, A, C);
        }

        if (best_coop_ms < 1e29 && best_solo_ms < 1e29) {
            printf("  => best coop %.3f ms vs best solo %.3f ms: %+.1f%% "
                   "(cpu solo %.3f, npu solo %.3f)\n\n",
                   best_coop_ms, best_solo_ms,
                   (best_solo_ms / best_coop_ms - 1.0) * 100.0,
                   solo_rows[0], solo_rows[1]);
        }

        for (int s = 0; s < WEIGHT_SETS; ++s) free(Wbufs[s]);
        free(act); free(act_d); free(out);
    }

    atomic_store(&cpool.quit, 1);
    atomic_store(&npool.quit, 1);
    for (int t = 0; t < nthreads; ++t) pthread_join(cpu_threads[t], NULL);
    for (int i = 0; i < MAX_SEGS; ++i) pthread_join(npu_threads[i], NULL);
    return 0;
}
