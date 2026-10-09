#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"
#include "rknpu2-dispatch-pool.h"
#include "rknpu2-native-layout.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <omp.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <chrono>
#include <deque>
#include <unordered_set>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <random>
#include <limits>
#include <sys/mman.h>
#include <sstream>

#define UNUSED(x) (void)(x)

// --- IOMMU Domain Manager ---

// Helper function for parsing complex integer lists
static std::vector<int32_t> parse_domain_list(const std::string& str) {
    std::vector<int32_t> result;
    if (str.empty()) return result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        auto dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            int start = std::strtol(token.substr(0, dash_pos).c_str(), nullptr, 10);
            int end = std::strtol(token.substr(dash_pos + 1).c_str(), nullptr, 10);
            for (int i = start; i <= end; ++i) result.push_back(i);
        } else {
            result.push_back(std::strtol(token.c_str(), nullptr, 10));
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct IOMMUDomainManager {
    std::mutex mutex;

    // Max domain size for assigning
    const size_t max_domain_size = ((size_t) std::numeric_limits<int32_t>::max() - 65536);

    // Storage for domains and their sizes
    std::unordered_map<int32_t, size_t> domain_sizes;
    std::unordered_map<int32_t, rknn_matmul_ctx> allocator_contexts;

    // Allowed domain IDs defined by the user
    std::vector<int32_t> allowed_domains;

    IOMMUDomainManager() {
        // Read restricted domains from ENV variable
        const char* env_domains = std::getenv("RKNPU_DOMAINS");
        if (env_domains != nullptr) {
            allowed_domains = parse_domain_list(env_domains);

            if (!allowed_domains.empty()) {
                fprintf(stderr, "\n"
                    "RKNPU WARNING: Custom IOMMU domains detected via RKNPU_DOMAINS.\n"
                    "Due to Rockchip library limitations, concurrent execution of\n"
                    "multiple processes accessing the NPU simultaneously WILL LEAD\n"
                    "to a SYSTEM KERNEL PANIC and WILL FREEZE YOUR OPERATING SYSTEM.\n"
                    "Execute models SEQUENTIALLY if using multiple independent processes.\n");
            }
        }
    }

    // Function for assigning the domain for the tensor of given size
    int32_t assign_domain_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Allocate strictly within the allowed domains
        if (!allowed_domains.empty()) {
            for (int32_t d : allowed_domains) {
                if (domain_sizes[d] + size <= max_domain_size) {
                    domain_sizes[d] += size;
                    ensure_allocator_context(d);
                    return d;
                }
            }

            fprintf(stderr, "RKNPU ERROR: Out of memory in allowed IOMMU domains!\n");
            assert(false);
            return -1;
        // Allocate dynamically
        } else {
            for (int32_t i = 0; i <= 15; ++i) {
                if (domain_sizes[i] + size <= max_domain_size) {
                    domain_sizes[i] += size;
                    ensure_allocator_context(i);
                    return i;
                }
            }
            fprintf(stderr, "RKNPU ERROR: Out of memory in all IOMMU domains!\n");
            assert(false);
            return -1;
        }
    }

    // Function for releasing the given size of the domain memory
    void release_domain_memory(int32_t domain_id, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = domain_sizes.find(domain_id);
        if (it != domain_sizes.end()) {
            if (it->second >= size) {
                it->second -= size;
            } else {
                it->second = 0;
            }
        }
    }

    // Function for getting a new dummy context in the required domain
    rknn_matmul_ctx get_allocator_context(int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);
        ensure_allocator_context(domain_id);
        return allocator_contexts[domain_id];
    }

private:
    // Function for ensuring a dummy context existence in the required domain
    void ensure_allocator_context(int32_t domain_id) {
        if (allocator_contexts.find(domain_id) == allocator_contexts.end()) {
            rknn_matmul_info info;
            memset(&info, 0, sizeof(info));
            info.M = 32; info.K = 32; info.N = 32;
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
            info.iommu_domain_id = domain_id;

            rknn_matmul_io_attr io_attr;
            rknn_matmul_ctx ctx = 0;
            rknn_matmul_create(&ctx, &info, &io_attr);
            allocator_contexts[domain_id] = ctx;
        }
    }
};
static IOMMUDomainManager g_domain_manager;

// M-dependent routing threshold (RKNPU_CPU_DECODE env variable).
// When set to M > 0, MUL_MAT ops whose batch dimension is below M are
// rejected by supports_op so the scheduler runs them on the CPU backend.
// Token generation (M=1) is memory-bandwidth bound and the NPU re-reads
// weights at upconverted precision, so the CPU's native-quantization read
// path can be faster. Requires keeping the original weight bytes
// host-resident (see dual residency in set_tensor / get_alloc_size).
static int rknpu_cpu_decode_threshold() {
    static const int threshold = []() {
        const char* env = std::getenv("RKNPU_CPU_DECODE");
        return env ? std::atoi(env) : 0;
    }();
    return threshold;
}

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,         \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

// Matrix segment information for N dimension
struct MatrixSegmentN {
    int offset_n;
    int size_n;
    int core_id;
};

// Matrix segment information for K dimension
struct MatrixSegmentK {
    int offset_k;
    int size_k;
};

// Split B-matrix into N-segments for cores
static std::vector<MatrixSegmentN> compute_n_segments(int N, const std::vector<int>& active_cores, int alignment) {
    std::vector<MatrixSegmentN> segments;
    int num_cores = active_cores.size();

    if (num_cores == 0) return segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);

    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegmentN seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = active_cores[i];

        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }

        offset += seg.size_n;
        segments.push_back(seg);
    }
    return segments;
}

// Split B-matrix into K-segments for hardware limit
static std::vector<MatrixSegmentK> compute_k_segments(int K_op, int k_limit, int alignment) {
    std::vector<MatrixSegmentK> segments;

    if (k_limit <= 0 || K_op <= k_limit) {
        segments.push_back({0, K_op});
        return segments;
    }

    int k_limit_aligned = (k_limit / alignment) * alignment;
    int offset = 0;
    while (offset < K_op) {
        int size = std::min(k_limit_aligned, K_op - offset);
        segments.push_back({offset, size});
        offset += size;
    }
    return segments;
}

// --- Structs ---

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    void* virtual_base;
    size_t total_size;
    std::string name;

    // RKNN buffers allocations for each tensor
    struct TensorAllocation {
        rknn_tensor_mem* mem = nullptr;
        size_t size = 0;
        int32_t iommu_domain_id = 0;
    };
    std::unordered_map<size_t, TensorAllocation> tensor_allocs;

    // Per-block scaling factors for quantized weights
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    std::mutex mutex;

    // Function for the allocation of a RKNN buffer for the individual tensor
    TensorAllocation get_tensor_allocation(size_t tensor_offset, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Trying to find an existing buffer
        auto it = tensor_allocs.find(tensor_offset);
        if (it != tensor_allocs.end()) {
            if (it->second.size < size) {
                rknn_matmul_ctx old_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                rknn_destroy_mem(old_ctx, it->second.mem);
                g_domain_manager.release_domain_memory(it->second.iommu_domain_id, it->second.size);

                it->second.iommu_domain_id = g_domain_manager.assign_domain_memory(size);
                rknn_matmul_ctx new_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                it->second.mem = rknn_create_mem(new_ctx, size);
                it->second.size = size;
            }
            return it->second;
        }

        // Acquiring a domain for allocation
        int32_t domain_id = g_domain_manager.assign_domain_memory(size);
        rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(domain_id);

        // Allocating a new buffer for the tensor
        TensorAllocation alloc;
        alloc.mem = rknn_create_mem(alloc_ctx, size);
        alloc.size = size;
        alloc.iommu_domain_id = domain_id;

        GGML_ASSERT(alloc.mem != nullptr && "Failed to allocate tensor memory via RKNN API");
        tensor_allocs[tensor_offset] = alloc;

        return alloc;
    }
};


// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    bool b_bound = false;
    std::shared_ptr<rknn_tensor_mem> mem_B;

    // A/C buffers currently bound with rknn_matmul_set_io_mem. Both come from
    // shape-keyed caches whose entries live as long as the backend, so a
    // pointer match means the binding is already in place and the driver
    // call can be skipped (it was ~2,300 calls per decoded token).
    const rknn_tensor_mem* bound_A = nullptr;
    const rknn_tensor_mem* bound_C = nullptr;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type, rknn_matmul_layout ac_layout, int32_t domain_id) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = ac_layout;
        info.iommu_domain_id = domain_id;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) ctx = 0;
    }

    // The dispatch pool's segment action; see rknpu2-dispatch-pool.h
    void run() { rknn_matmul_run(ctx); }

    ~rknpu_matmul_context() {
        mem_B.reset();

        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};

using rknpu_dispatch_pool = rknpu_dispatch_pool_t<rknpu_matmul_context>;

// FP16 matmul with runtime A, B and C (attention: B is the K or V cache, not a
// weight), owning its own DMA buffers. B_layout NORM or TP_NORM: the driver
// takes row-major data directly (checked to fp32 precision, decode research
// #1e), so K rows feed Q*K^T as TP_NORM and V rows feed P*V as NORM.
struct rknpu_attn_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;
    rknn_tensor_mem *A = nullptr, *B = nullptr, *C = nullptr;
    // Native A/C: the CPU writes A and reads C in the NPU's tiling (in
    // parallel, inside loops it runs anyway) instead of the runtime
    // converting both on one thread inside every run
    bool native = false;
    rknpu2_native_geom a_geom = {0, 0, 0}, c_geom = {0, 0, 0};
    // Native B: the CPU writes K / V in the NPU's (N/subN, K/subK, subN,
    // subK) tiling and B stays bound, instead of the driver converting a
    // plain B on one thread at every re-bind
    bool b_native = false;
    int b_subN = 0, b_subK = 0;
    bool c_fp16 = false;   // C written as FP16 (RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16)
    rknpu_attn_context(int M, int K, int N, int b_layout, int core_id, bool fp16_c = false) : c_fp16(fp16_c) {
        static const bool want_native = []() {
            const char* env = std::getenv("RKNPU_FA_NATIVE");
            return env == nullptr || std::atoi(env) != 0;
        }();
        static const bool want_native_b = []() {
            const char* env = std::getenv("RKNPU_FA_NATIVE_B");
            return env == nullptr || std::atoi(env) != 0;
        }();
        memset(&info, 0, sizeof(info));
        info.M = M; info.K = K; info.N = N;
        info.type = fp16_c ? RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 : RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = (int16_t)(want_native && want_native_b ? RKNN_MM_LAYOUT_NATIVE : b_layout);
        info.AC_layout = want_native ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;
        if (rknn_matmul_create(&ctx, &info, &io_attr) < 0) { ctx = 0; return; }
        if (want_native) {
            native = rknpu2_native_geom_from_dims(io_attr.A.dims, io_attr.A.n_dims, &a_geom) == 0 &&
                     rknpu2_native_geom_from_dims(io_attr.C.dims, io_attr.C.n_dims, &c_geom) == 0 &&
                     a_geom.outer * a_geom.sub == K && c_geom.outer * c_geom.sub == N &&
                     a_geom.m_stride == M && c_geom.m_stride == M;
            if (native && info.B_layout == RKNN_MM_LAYOUT_NATIVE) {
                const auto& d = io_attr.B.dims;
                b_native = io_attr.B.n_dims == 4 && d[2] > 0 && d[3] > 0 && (int)(d[0] * d[2]) == N &&
                           (int)(d[1] * d[3]) == K && d[3] <= 64 && io_attr.B.size >= (uint32_t)K * N * 2;
                b_subN = (int)d[2]; b_subK = (int)d[3];
            }
            if (!native || (info.B_layout == RKNN_MM_LAYOUT_NATIVE && !b_native)) {   // unexpected geometry: plain layouts
                rknn_matmul_destroy(ctx);
                native = b_native = false;
                info.AC_layout = RKNN_MM_LAYOUT_NORM;
                info.B_layout = (int16_t)b_layout;
                if (rknn_matmul_create(&ctx, &info, &io_attr) < 0) { ctx = 0; return; }
            }
        }
        rknn_matmul_set_core_mask(ctx, core_id == 0 ? RKNN_NPU_CORE_0 : core_id == 1 ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2);
        A = rknn_create_mem(ctx, io_attr.A.size);
        B = rknn_create_mem(ctx, io_attr.B.size);
        C = rknn_create_mem(ctx, io_attr.C.size);
        if (!A || !B || !C) { release(); return; }
        rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
        rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
        rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
    }
    void release() {
        if (ctx) {
            if (A) rknn_destroy_mem(ctx, A);
            if (B) rknn_destroy_mem(ctx, B);
            if (C) rknn_destroy_mem(ctx, C);
            rknn_matmul_destroy(ctx);
        }
        ctx = 0; A = B = C = nullptr;
    }
    ~rknpu_attn_context() { release(); }
};

// Runs one node's per-core segments on dedicated threads and returns at once,
// so the caller's OpenMP team can do CPU work (the next chunk's A prep, the
// previous chunk's dequant) while the NPU computes. The threads spend their
// time blocked in rknn_matmul_run, so they do not compete for cores with the
// team. Used only by the pipelined prefill path, where a job lasts tens of
// milliseconds and condition-variable wake-up latency is irrelevant.
// CPU time spent blocked in rknpu_async_runner::wait (NPU work not hidden
// behind CPU work), reported by RKNPU_PROFILE
static std::atomic<uint64_t> g_rknpu_wait_ns{0};

struct rknpu_async_runner {
    void start(const std::vector<std::shared_ptr<rknpu_matmul_context>>& ctxs) {
        std::unique_lock<std::mutex> lock(mutex);
        // Seed new workers with the generation *before* this job's bump, so
        // they take this job (seeding after it would make a fresh worker
        // wait for the next job and deadlock wait() — the inverse of the
        // dispatch pool's stale-seed bug)
        while (threads.size() < ctxs.size()) {
            const int idx = (int)threads.size();
            const uint64_t seed = generation;
            threads.emplace_back([this, idx, seed] { worker(idx, seed); });
        }
        // the workers read this copy, never the caller's vector: a worker that
        // is not part of the job can wake after wait() returned, while the
        // caller already rebuilds its vector (crashed the server on prefill)
        job.clear();
        for (const auto& c : ctxs) job.push_back(c.get());
        pending = (int)ctxs.size();
        ++generation;
        cv_start.notify_all();
    }
    void wait() {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [this] { return pending == 0; });
        g_rknpu_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    ~rknpu_async_runner() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            quit = true;
            cv_start.notify_all();
        }
        for (auto& t : threads) t.join();
    }
  private:
    void worker(int idx, uint64_t seen) {
        for (;;) {
            rknpu_matmul_context* ctx = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_start.wait(lock, [&] { return quit || generation != seen; });
                if (quit) return;
                seen = generation;
                if (idx < (int)job.size()) ctx = job[idx];
            }
            if (ctx) {
                ctx->run();
                std::lock_guard<std::mutex> lock(mutex);
                if (--pending == 0) cv_done.notify_all();
            }
        }
    }
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv_start, cv_done;
    std::vector<rknpu_matmul_context*> job;   // the current job's contexts (cached for the backend's life)
    uint64_t generation = 0;
    int pending = 0;
    bool quit = false;
};

// A gate matmul whose dequant is deferred into the following up matmul's
// fused GEGLU collect ([gate, up, GLU], gate consumed only by the GLU): its
// INT16 C is kept per chunk and dequantized there into thread-local rows,
// so gate's FP32 output is never written to (and read back from) DRAM.
struct rknpu_deferred_gate {
    struct ggml_tensor* gate = nullptr;   // pending: gate's dst is not written
    const struct ggml_tensor* up = nullptr;
    int M = 0, N = 0, MC = 0;
    std::vector<std::vector<std::shared_ptr<rknn_tensor_mem>>> c;   // [chunk][n-segment]
    std::vector<rknpu2_native_geom> c_geom;
    std::vector<MatrixSegmentN> segs;
    std::vector<float> common;            // per row: scale_A / hadamard divisor
    const float* chan = nullptr;          // per-channel B scales (pointer-stable map value)
};

// The FFN down projection run as one job across two nodes. Its A rows are
// prepared (Hadamard + INT4, straight into per-chunk A buffers) inside the
// up matmul's fused gate/up/GEGLU collect, while each GEGLU row is still in
// cache, so the GLU output is never written; its first chunk is started on
// the NPU before up's last collect, which it then overlaps; and both of its
// K-segments run per chunk, dequantized together (dequant2). Element-exact
// vs the separate GLU write, prep and per-segment passes.
struct rknpu_w4a4_job {
    const struct ggml_tensor* node = nullptr;   // pending matmul
    int M = 0, N = 0, K = 0, MC = 0, n_chunks = 0;
    size_t nas = 0;
    std::vector<int> seg_off, seg_len;                                  // K-segments
    std::vector<MatrixSegmentN> nsegs;                                  // active N-segments
    std::vector<std::shared_ptr<rknpu_matmul_context>> cctx;            // [s * nas + idx]
    std::vector<std::vector<std::shared_ptr<rknn_tensor_mem>>> a;       // [chunk][s]
    std::vector<std::vector<std::shared_ptr<rknn_tensor_mem>>> c;       // [slot][s * nas + idx]
    std::vector<rknpu2_native_geom> a_geom, c_geom;                     // [s], [s * nas + idx]
    std::vector<std::vector<float>> scales_A;                           // [s][M]
    const float* s_vec = nullptr;                                       // Hadamard signs
    const float* chan = nullptr;                                        // per-channel B scales, [s * N + n]
    float a_clip = 1.0f, hdiv = 1.0f;
    int started = -1;                                                   // last chunk started
};

// A few persistent workers for blocking NPU calls that should overlap CPU
// work (attention runs of different KV-head groups, each on its own core)
struct rknpu_fn_pool {
    struct ticket { bool done = false; };
    void submit(ticket& t, std::function<void()> fn) {
        std::unique_lock<std::mutex> lock(mutex);
        while (threads.size() < 2) threads.emplace_back([this] { worker(); });
        t.done = false;
        queue.push_back({&t, std::move(fn)});
        cv.notify_one();
    }
    void wait(ticket& t) {
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [&] { return t.done; });
    }
    ~rknpu_fn_pool() {
        { std::lock_guard<std::mutex> lock(mutex); quit = true; cv.notify_all(); }
        for (auto& th : threads) th.join();
    }
  private:
    void worker() {
        for (;;) {
            std::pair<ticket*, std::function<void()>> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return quit || !queue.empty(); });
                if (quit) return;
                job = std::move(queue.front());
                queue.pop_front();
            }
            job.second();
            { std::lock_guard<std::mutex> lock(mutex); job.first->done = true; }
            cv_done.notify_all();
        }
    }
    std::vector<std::thread> threads;
    std::deque<std::pair<ticket*, std::function<void()>>> queue;
    std::mutex mutex;
    std::condition_variable cv, cv_done;
    bool quit = false;
};

// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache (tensor_fd, offset, M, K, N, core_id, type, domain_id)
    std::unordered_map<std::tuple<uintptr_t, size_t, int, int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // A-matrices cache (M, K, npu_type_a, domain_id)
    std::unordered_map<std::tuple<int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;

    // C-matrices cache (M, N, core_id, npu_type_c, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    // Persistent threads for the per-node segment runs (see struct comment)
    rknpu_dispatch_pool dispatch_pool;
    rknpu_async_runner async_runner;
    rknpu_deferred_gate deferred_gate;
    rknpu_w4a4_job down_job;
    // nodes already computed ahead of their graph position (a KV-cache write
    // done inside its projection's dequant)
    std::unordered_set<const struct ggml_tensor*> done_nodes;
    rknpu_fn_pool fa_pool;
    std::vector<float> fa_stats[3];   // online softmax: per in-flight item (core slot), [row][chunk][m, l]
    std::vector<float> fa_out[3];     // host-side P*V output per slot: the chunk outputs are combined here, never in the DMA buffers
    // whole-FFN schedule (rknpu_ffn_block): gate, up and down jobs, and
    // gate's and up's chunk contexts as one runner batch
    rknpu_w4a4_job ffn_gate, ffn_up, ffn_down;
    std::vector<std::shared_ptr<rknpu_matmul_context>> ffn_gu_ctx, ffn_gud_ctx;

    // (M, K, N, B layout, core, slot) -> attention matmul context; their DMA buffers grow
    // with n_kv, so stale shapes are freed (see get_attn_ctx) or they exhaust the IOMMU domain
    // slot: the P*V chunk index (chunks of one item run back to back on one core, each with its own buffers)
    std::map<std::tuple<int, int, int, int, int, int>, std::unique_ptr<rknpu_attn_context>> attn_ctx_cache;
    std::map<std::tuple<int, int, int, int, int>, uint64_t> attn_shape_use;   // shape -> last attention call
    uint64_t attn_call = 0;   // incremented per attention op; shapes of the current call are never evicted
    rknpu_attn_context* get_attn_ctx(int M, int K, int N, int b_layout, int core_id, int slot = 0, bool fp16_c = false) {
        slot += fp16_c ? 1000 : 0;   // FP16-C contexts are distinct shapes
        auto key = std::make_tuple(M, K, N, b_layout, core_id, slot);
        attn_shape_use[std::make_tuple(M, K, N, b_layout, slot)] = attn_call;
        auto it = attn_ctx_cache.find(key);
        if (it != attn_ctx_cache.end()) return it->second.get();
        // Shapes change with the KV length every ubatch; a shape not used in the last
        // RKNPU_FA_CTX_AGE attention ops will not come back (Gemma-4's global-attention
        // layers recur every 6 ops), so its contexts are freed on a miss, before the new
        // shape's buffers are allocated (two 32k shapes do not fit the IOMMU domain)
        static const uint64_t max_age = []() {
            const char* env = std::getenv("RKNPU_FA_CTX_AGE");
            const int v = env ? std::atoi(env) : 5;
            return (uint64_t)(v >= 1 ? v : 5);
        }();
        for (auto s = attn_shape_use.begin(); s != attn_shape_use.end(); ) {
            if (s->second + max_age < attn_call) {
                for (auto c = attn_ctx_cache.begin(); c != attn_ctx_cache.end(); ) {
                    if (std::make_tuple(std::get<0>(c->first), std::get<1>(c->first), std::get<2>(c->first), std::get<3>(c->first), std::get<5>(c->first)) == s->first) {
                        c = attn_ctx_cache.erase(c);
                    } else {
                        ++c;
                    }
                }
                s = attn_shape_use.erase(s);
            } else {
                ++s;
            }
        }
        auto c = std::make_unique<rknpu_attn_context>(M, K, N, b_layout, core_id, fp16_c);
        if (c->ctx == 0) return nullptr;
        return (attn_ctx_cache[key] = std::move(c)).get();
    }

    // Team size for the per-row OpenMP regions, set by llama through
    // ggml_backend_set_n_threads; 0 means libgomp's default. Matching
    // ggml-cpu's team matters whenever M > 1 at decode time (speculative
    // verification batches): a different size makes libgomp tear down and
    // respawn its team around every node, as in decode research #3.
    int n_threads = 0;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(uintptr_t tensor_id, size_t offset, int M, int K, int N, int core_id, rknn_matmul_type type, rknn_matmul_layout ac_layout, int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(tensor_id, offset, M, K, N, core_id, (int)type + ((int)ac_layout << 8), (int)domain_id);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }

        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, ac_layout, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};


//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    delete ctx;
    delete backend;
}

// Defined with the buffer code below; every path that decides whether a
// tensor is stored NPU-packed must use this same predicate.
static const rknpu2_configuration::Rknpu2HardwarePipeline * resolve_packable_pipeline(const struct ggml_tensor * tensor);

// Function for acquiring a pointer for tensor data
// Packed NPU copies exist only for weights. Compute buffers (marked COMPUTE
// by ggml-alloc at allocation) hold plain tensors whatever their dtype, and
// reuse offsets between tensors, so no packed allocation may be created or
// looked up for them: an offset-keyed lookup would hit another tensor's
// allocation. Weight buffers are still ANY during init_tensor (llama marks
// them WEIGHTS after allocation), hence "not COMPUTE" rather than "WEIGHTS".
static bool rknpu_buffer_may_pack(ggml_backend_buffer_t buffer) {
    return buffer && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
           strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)), "RKNPU") == 0;
}

static void* get_tensor_real_ptr(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nullptr;

    // Packed copies exist only for weights in this backend's own buffers;
    // compute tensors (incl. F16 KV slices for NPU attention) are plain host
    // memory whatever their dtype
    if (!rknpu_buffer_may_pack(tensor->buffer)) {
        return tensor->data;
    }

    const auto* pipeline = resolve_packable_pipeline(tensor);

    if (pipeline) {
        auto* ctx = (ggml_backend_rknpu_buffer_context*)tensor->buffer->context;
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->tensor_allocs.find(offset);
        if (it != ctx->tensor_allocs.end()) {
            return it->second.mem->virt_addr;
        }
    }

    return tensor->data;
}

// Function for getting buffer from cache or creating new one
template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_tensor_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->size >= size) {
            return it->second;
        }
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx, size);
    if (!mem) { return nullptr; }

    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m != 0) {
            rknn_destroy_mem(matmul_ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

// RKNPU_PROFILE=1: wall-time split of the backend's per-node work, printed
// every ~5 s to stderr as cumulative totals (take the slope across a steady
// decode window and divide by the token rate). perf cannot attribute this:
// the main thread is off-CPU while it waits in rknn_matmul_run.
struct rknpu_profile {
    bool on = std::getenv("RKNPU_PROFILE") != nullptr;
    uint64_t graph_ns = 0, node_ns = 0, run_ns = 0, nodes = 0, graphs = 0;
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now(), t_print = t_start;
    static uint64_t ns(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
    void maybe_print() {
        auto now = std::chrono::steady_clock::now();
        if (ns(t_print, now) < 5000000000ull) return;
        t_print = now;
        fprintf(stderr, "RKNPU_PROFILE t=%.1fs graphs=%llu nodes=%llu graph=%.1fms node=%.1fms npu_run=%.1fms npu_wait=%.1fms\n",
                ns(t_start, now) / 1e9, (unsigned long long)graphs, (unsigned long long)nodes,
                graph_ns / 1e6, node_ns / 1e6, run_ns / 1e6, g_rknpu_wait_ns.load() / 1e6);
    }
};
static rknpu_profile g_rknpu_profile;

// RKNPU_PIPELINE=0 disables the pipelined W4A4 prefill path
static bool rknpu_pipeline_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("RKNPU_PIPELINE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

// ===========================================
// ===== Flash attention on the NPU (prefill) =====
// ===========================================
// Per KV head g, the rows of its rk2 query heads are stacked into one matmul:
//   S = Q_g K_g^T   on the NPU (FP16 in, FP32 out; K rows as TP_NORM B)
//   P = softmax(softcap(S * scale) + mask)   on the CPU team
//   O = P V_g       on the NPU (V rows as NORM B)
// Semantics follow ggml_compute_forward_flash_attn_ext_tiled; P is
// normalized before the second matmul instead of after (same math, FP16 P).
// Only the cases supports_op accepts reach here (see rknpu_fa_supported).
static void rknpu_fp32_to_fp16(const float* src, uint16_t* dst, int64_t n) {
    int64_t i = 0;
#ifdef __ARM_NEON
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src + i)), vcvt_f16_f32(vld1q_f32(src + i + 4)));
        vst1q_u16(dst + i, vreinterpretq_u16_f16(h));
    }
#endif
    for (; i < n; ++i) dst[i] = ggml_fp32_to_fp16(src[i]);
}

#ifdef __ARM_NEON
// exp(x), four lanes: the ARM optimized-routines approximation ggml-cpu uses
// (ggml_v_expf; max error 1.45 ulp + 0.5), copied here because ggml-cpu's
// vector helpers are internal to that backend
static inline float32x4_t rknpu_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}
#endif

// Leading and trailing runs of -inf mask (causal / sliding window) give
// exact zeros: the softmax works only on [lo, hi), aligned to the 4-lane
// vectors so every lane sums the same positions as over the whole row.
// An all-masked row gives lo = hi = 0.
static void rknpu_softmax_range(const ggml_fp16_t* mrow, int64_t n_all, int64_t* lo_out, int64_t* hi_out) {
    int64_t lo = 0, hi = n_all;
    if (mrow) {
        const uint16_t* m16 = (const uint16_t*)mrow;
        while (hi > 0 && m16[hi - 1] == 0xFC00) --hi;
        while (lo < hi && m16[lo] == 0xFC00) ++lo;
        if (hi == 0) lo = 0;
        lo &= ~(int64_t)3;
        hi = hi == 0 ? 0 : std::min(n_all, (hi + 3) & ~(int64_t)3);
    }
    *lo_out = lo; *hi_out = hi;
}

// One attention row: P = softmax(softcap(s*scale) + mask), written as FP16,
// over [lo, hi) from rknpu_softmax_range; only s_row[lo, hi) is read.
// Masked (-inf) positions give exactly 0; an all-masked row gives zeros.
static void rknpu_softmax_row(const float* s_row, const ggml_fp16_t* mrow, int64_t n_all, int64_t lo, int64_t hi,
                              float scale, float softcap, float* tmp, uint16_t* out_all) {
    memset(out_all, 0, lo * sizeof(uint16_t));
    memset(out_all + hi, 0, (n_all - hi) * sizeof(uint16_t));
    s_row += lo;
    if (mrow) mrow += lo;
    uint16_t* out = out_all + lo;
    const int64_t n = hi - lo;
    int64_t j = 0;
    float mx = -INFINITY;
#ifdef __ARM_NEON
    float32x4_t vmx = vdupq_n_f32(-INFINITY);
    const float32x4_t vs = vdupq_n_f32(scale), vcap = vdupq_n_f32(softcap);
    const float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f);
    for (; j + 4 <= n; j += 4) {
        float32x4_t x = vmulq_f32(vld1q_f32(s_row + j), vs);
        if (softcap != 0.0f) {   // softcap * tanh(x), tanh(x) = 1 - 2/(e^{2x}+1)
            x = vmulq_f32(vcap, vsubq_f32(one, vdivq_f32(two, vaddq_f32(rknpu_v_expf(vmulq_f32(two, x)), one))));
        }
        if (mrow) x = vaddq_f32(x, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t*)mrow + j))));
        vst1q_f32(tmp + j, x);
        vmx = vmaxq_f32(vmx, x);
    }
    mx = vmaxvq_f32(vmx);
#endif
    for (; j < n; ++j) {
        float x = s_row[j] * scale;
        if (softcap != 0.0f) x = softcap * tanhf(x);
        if (mrow) x += ggml_fp16_to_fp32(mrow[j]);
        tmp[j] = x;
        mx = std::max(mx, x);
    }
    if (mx == -INFINITY) {
        memset(out, 0, n * sizeof(uint16_t));
        return;
    }
    float sum = 0.0f;
    j = 0;
#ifdef __ARM_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    const float32x4_t vm = vdupq_n_f32(mx);
    for (; j + 4 <= n; j += 4) {
        const float32x4_t e = rknpu_v_expf(vsubq_f32(vld1q_f32(tmp + j), vm));   // -inf -> 0
        vst1q_f32(tmp + j, e);
        vsum = vaddq_f32(vsum, e);
    }
    sum = vaddvq_f32(vsum);
#endif
    for (; j < n; ++j) {
        const float e = tmp[j] == -INFINITY ? 0.0f : expf(tmp[j] - mx);
        tmp[j] = e;
        sum += e;
    }
    const float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    j = 0;
#ifdef __ARM_NEON
    const float32x4_t vinv = vdupq_n_f32(inv);
    for (; j + 4 <= n; j += 4) vst1q_f32(tmp + j, vmulq_f32(vld1q_f32(tmp + j), vinv));
#endif
    for (; j < n; ++j) tmp[j] *= inv;
    rknpu_fp32_to_fp16(tmp, out, n);
}

// One chunk of an attention row from FP16 scores: x = softcap(s*scale) + mask over n
// elements, out = exp(x - m) as FP16 with the chunk's own max m and sum l (flash-attention
// style; the P*V accumulation rescales the chunks). An all-masked chunk gives m = -inf, l = 0.
static void rknpu_softmax_chunk16(const uint16_t* s16, const ggml_fp16_t* mrow, int64_t n, float scale, float softcap,
                                  float* tmp, uint16_t* out, float* m_out, float* l_out) {
    int64_t j = 0;
    float mx = -INFINITY;
#ifdef __ARM_NEON
    float32x4_t vmx = vdupq_n_f32(-INFINITY);
    const float32x4_t vs = vdupq_n_f32(scale), vcap = vdupq_n_f32(softcap);
    const float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f);
    for (; j + 4 <= n; j += 4) {
        float32x4_t x = vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s16 + j))), vs);
        if (softcap != 0.0f) x = vmulq_f32(vcap, vsubq_f32(one, vdivq_f32(two, vaddq_f32(rknpu_v_expf(vmulq_f32(two, x)), one))));
        if (mrow) x = vaddq_f32(x, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t*)mrow + j))));
        vst1q_f32(tmp + j, x);
        vmx = vmaxq_f32(vmx, x);
    }
    mx = vmaxvq_f32(vmx);
#endif
    for (; j < n; ++j) {
        float x = ggml_fp16_to_fp32(s16[j]) * scale;
        if (softcap != 0.0f) x = softcap * tanhf(x);
        if (mrow) x += ggml_fp16_to_fp32(mrow[j]);
        tmp[j] = x;
        mx = std::max(mx, x);
    }
    if (mx == -INFINITY) {
        memset(out, 0, n * sizeof(uint16_t));
        *m_out = -INFINITY; *l_out = 0.0f;
        return;
    }
    float sum = 0.0f;
    j = 0;
#ifdef __ARM_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    const float32x4_t vm = vdupq_n_f32(mx);
    for (; j + 8 <= n; j += 8) {
        const float32x4_t e0 = rknpu_v_expf(vsubq_f32(vld1q_f32(tmp + j), vm));
        const float32x4_t e1 = rknpu_v_expf(vsubq_f32(vld1q_f32(tmp + j + 4), vm));
        vsum = vaddq_f32(vsum, vaddq_f32(e0, e1));
        vst1q_u16(out + j, vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(e0), vcvt_f16_f32(e1))));
    }
    sum = vaddvq_f32(vsum);
#endif
    for (; j < n; ++j) {
        const float e = tmp[j] == -INFINITY ? 0.0f : expf(tmp[j] - mx);
        sum += e;
        out[j] = ggml_fp32_to_fp16(e);
    }
    *m_out = mx; *l_out = sum;
}

// GEGLU (gate, up) on the backend, so [up, gate, GLU, down] stay in one NPU
// split. Same NEON tanh-GELU as ggml-cpu's ggml_vec_geglu_f32 on ARM
// (tanh(z) = 1 - 2/(e^{2z}+1) via the same exp, +-10 cut-offs), so the
// output is identical for widths that are multiples of 4.
// Per-head RMS norms of attention projections (Gemma-4's q/k/v norms):
// RMS_NORM of a reshaped matmul output, optionally times a per-head
// weight. Computed exactly as ggml-cpu does (double-precision sum of
// squares over four 2-lane accumulators, (x*scale)*w), so the backend can
// take these ops and fuse them into the matmul's dequant (RKNPU_HEAD_NORM).
static bool rknpu_head_norm_enabled() {
    static const bool v = []() {
        const char* env = std::getenv("RKNPU_HEAD_NORM");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return v;
}
static bool rknpu_head_norm_supported(const struct ggml_tensor* op) {
    const struct ggml_tensor* a = op->src[0];
    return op->op == GGML_OP_RMS_NORM && a && (a->op == GGML_OP_RESHAPE || a->op == GGML_OP_VIEW) &&
           a->view_src && a->view_src->op == GGML_OP_MUL_MAT && a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(a) && ggml_is_contiguous(op) && a->ne[0] % 8 == 0;
}
// the weight operand of a supported per-head norm's MUL, or nullptr
static const struct ggml_tensor* rknpu_head_norm_weight(const struct ggml_tensor* op) {
    if (op->op != GGML_OP_MUL || !op->src[0] || !op->src[1]) return nullptr;
    const struct ggml_tensor* n = rknpu_head_norm_supported(op->src[0]) ? op->src[0] : rknpu_head_norm_supported(op->src[1]) ? op->src[1] : nullptr;
    if (!n) return nullptr;
    const struct ggml_tensor* w = op->src[0] == n ? op->src[1] : op->src[0];
    if (w->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || w->ne[0] != n->ne[0] || ggml_nrows(w) != 1 ||
        w->nb[0] != sizeof(float) || !ggml_is_contiguous(op) || !ggml_are_same_shape(op, n)) return nullptr;
    return w;
}
static inline double rknpu_rms_sumsq(const float* x, int64_t n) {
    double sum = 0.0;
    int64_t i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    float64x2_t s0 = vdupq_n_f64(0.0), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t a = vld1q_f32(x + i), b = vld1q_f32(x + i + 4);
        const float32x4_t pa = vmulq_f32(a, a), pb = vmulq_f32(b, b);
        s0 = vaddq_f64(s0, vcvt_f64_f32(vget_low_f32(pa)));
        s1 = vaddq_f64(s1, vcvt_high_f64_f32(pa));
        s2 = vaddq_f64(s2, vcvt_f64_f32(vget_low_f32(pb)));
        s3 = vaddq_f64(s3, vcvt_high_f64_f32(pb));
    }
    sum = vaddvq_f64(vaddq_f64(vaddq_f64(s0, s1), vaddq_f64(s2, s3)));
#endif
    for (; i < n; i++) sum += (double)(x[i] * x[i]);
    return sum;
}
// one head: y = rms_norm(x) [* w]
static inline void rknpu_head_norm(float* y, const float* x, int64_t n, float eps, const float* w) {
    const float mean  = rknpu_rms_sumsq(x, n) / n;
    const float scale = 1.0f / sqrtf(mean + eps);
    int64_t j = 0;
#ifdef __ARM_NEON
    const float32x4_t vs = vdupq_n_f32(scale);
    if (w) {
        for (; j + 4 <= n; j += 4) vst1q_f32(y + j, vmulq_f32(vmulq_f32(vld1q_f32(x + j), vs), vld1q_f32(w + j)));
    } else {
        for (; j + 4 <= n; j += 4) vst1q_f32(y + j, vmulq_f32(vld1q_f32(x + j), vs));
    }
#endif
    for (; j < n; j++) y[j] = w ? x[j] * scale * w[j] : x[j] * scale;
}
// standalone: the norm node, fused with its weight MUL when that is `mul`
static void rknpu_head_norm_op(struct ggml_tensor* norm, struct ggml_tensor* mul, int n_omp) {
    const float* x = (const float*)get_tensor_real_ptr(norm->src[0]);
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const struct ggml_tensor* wt = mul ? rknpu_head_norm_weight(mul) : nullptr;
    const float* w = wt ? (const float*)get_tensor_real_ptr(wt) : nullptr;
    float* y = (float*)get_tensor_real_ptr(mul ? mul : norm);
    const int64_t hd = norm->ne[0], nr = ggml_nrows(norm);
    #pragma omp parallel for num_threads(n_omp)
    for (int64_t r = 0; r < nr; ++r) rknpu_head_norm(y + r * hd, x + r * hd, hd, eps, w);
}

// RoPE of the per-head-normed Q/K (Gemma-4), reproducing ggml-cpu
// exactly: the same cos/sin cache (no YaRN extrapolation, ext_factor == 0)
// and the rotation as ggml-cpu's compiled code does it, one rounded product
// then a fused multiply-add. GCC contracted the two modes differently:
// NEOX y0 = fma(x0, c, -(x1*s)), y1 = fma(x0, s, x1*c); NORMAL
// y0 = fma(-x1, s, x0*c), y1 = fma(x0, s, x1*c). Found by testing all four
// orders against ggml-cpu; test-rknpu2-rope.cpp checks it bit for bit (the
// guard runs it), since a different compiler could contract differently.
// Supported only after a head norm this backend runs (RKNPU_ROPE), or for
// any F32 RoPE with RKNPU_ROPE_ANY=1 (exactness tests).
struct rknpu_rope_params {
    int n_dims = 0, mode = 0, n_offs = 0;
    float freq_base = 0, freq_scale = 0, ext_factor = 0, attn_factor = 0;
};
static rknpu_rope_params rknpu_rope_get(const struct ggml_tensor* op) {
    rknpu_rope_params p;
    p.n_dims = ((const int32_t*)op->op_params)[1];
    p.mode   = ((const int32_t*)op->op_params)[2];
    memcpy(&p.freq_base,   (const int32_t*)op->op_params + 5, sizeof(float));
    memcpy(&p.freq_scale,  (const int32_t*)op->op_params + 6, sizeof(float));
    memcpy(&p.ext_factor,  (const int32_t*)op->op_params + 7, sizeof(float));
    memcpy(&p.attn_factor, (const int32_t*)op->op_params + 8, sizeof(float));
    p.n_offs = ((const int32_t*)op->op_params)[15];
    return p;
}
static bool rknpu_rope_enabled() {
    static const bool v = []() {
        const char* env = std::getenv("RKNPU_ROPE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return v;
}
static bool rknpu_rope_any() {
    static const bool v = std::getenv("RKNPU_ROPE_ANY") != nullptr && std::atoi(std::getenv("RKNPU_ROPE_ANY")) != 0;
    return v;
}
static bool rknpu_rope_supported(const struct ggml_tensor* op) {
    if (op->op != GGML_OP_ROPE) return false;
    const struct ggml_tensor *a = op->src[0], *pos = op->src[1], *ff = op->src[2];
    if (!a || !pos || a->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) return false;
    if (ff && ff->type != GGML_TYPE_F32) return false;
    const rknpu_rope_params p = rknpu_rope_get(op);
    if ((p.mode != GGML_ROPE_TYPE_NORMAL && p.mode != GGML_ROPE_TYPE_NEOX) || p.ext_factor != 0.0f) return false;
    if (p.n_dims <= 0 || p.n_dims % 8 != 0 || p.n_offs < 0 || p.n_offs % 2 != 0 || p.n_offs + p.n_dims > a->ne[0]) return false;
    if (ff && ff->ne[0] < p.n_dims / 2) return false;
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(op) || a->ne[3] != 1 || pos->ne[0] != a->ne[2]) return false;
    if (rknpu_rope_any()) return true;
    // after a per-head norm (RMS_NORM, or its weight MUL) of a projection
    return a->op == GGML_OP_RMS_NORM ? rknpu_head_norm_supported(a) : rknpu_head_norm_weight(a) != nullptr;
}
// cos/sin for one position, as ggml_rope_cache_init (ext_factor == 0, forward)
static void rknpu_rope_cache(float* cache, int64_t p, const rknpu_rope_params& rp, const float* ff, int64_t ne0) {
    const float theta_scale = powf(rp.freq_base, -2.0f / rp.n_dims);
    float theta = (float)p;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float f = ff ? ff[i0 / 2] : 1.0f;
        const float theta_interp = rp.freq_scale * (theta / f);
        cache[i0 + 0] = cosf(theta_interp) * rp.attn_factor;
        cache[i0 + 1] = sinf(theta_interp) * rp.attn_factor;
        theta *= theta_scale;
    }
}
// one head row (in place is fine: every pair is read before it is written)
static inline void rknpu_rope_head(float* y, const float* x, const float* cache, int64_t ne0, const rknpu_rope_params& rp) {
    const int n_dims = rp.n_dims, n_offs = rp.n_offs;
    if (rp.mode == GGML_ROPE_TYPE_NEOX) {
        const int h = n_dims / 2;
        const float* xa = x + n_offs;
        float* ya = y + n_offs;
        int i = 0;
#ifdef __ARM_NEON
        for (; i + 4 <= h; i += 4) {
            const float32x4x2_t cs = vld2q_f32(cache + 2 * i);   // cos, sin
            const float32x4_t x0 = vld1q_f32(xa + i), x1 = vld1q_f32(xa + i + h);
            const float32x4_t y0 = vfmaq_f32(vnegq_f32(vmulq_f32(x1, cs.val[1])), x0, cs.val[0]);
            const float32x4_t y1 = vfmaq_f32(vmulq_f32(x1, cs.val[0]), x0, cs.val[1]);
            vst1q_f32(ya + i, y0);
            vst1q_f32(ya + i + h, y1);
        }
#endif
        for (; i < h; ++i) {
            const float c = cache[2 * i], sn = cache[2 * i + 1], x0 = xa[i], x1 = xa[i + h];
            ya[i]     = fmaf(x0, c, -(x1 * sn));
            ya[i + h] = fmaf(x0, sn, x1 * c);
        }
    } else {   // NORMAL: adjacent pairs
        for (int i0 = 0; i0 < n_dims; i0 += 2) {
            const float c = cache[i0], sn = cache[i0 + 1], x0 = x[n_offs + i0], x1 = x[n_offs + i0 + 1];
            y[n_offs + i0]     = fmaf(-x1, sn, x0 * c);
            y[n_offs + i0 + 1] = fmaf(x0, sn, x1 * c);
        }
    }
    if (y != x) {   // channels outside the rotated range are copied
        for (int64_t i0 = 0; i0 < n_offs; ++i0) y[i0] = x[i0];
        for (int64_t i0 = n_offs + n_dims; i0 < ne0; ++i0) y[i0] = x[i0];
    }
}
static void rknpu_rope_op(struct ggml_tensor* dst, int n_omp) {
    const struct ggml_tensor *a = dst->src[0], *pos = dst->src[1], *fft = dst->src[2];
    const rknpu_rope_params rp = rknpu_rope_get(dst);
    const float* x = (const float*)get_tensor_real_ptr(a);
    float* y = (float*)get_tensor_real_ptr(dst);
    const int32_t* pp = (const int32_t*)get_tensor_real_ptr(pos);
    const float* ff = fft ? (const float*)get_tensor_real_ptr(fft) : nullptr;
    const int64_t ne0 = a->ne[0], nh = a->ne[1], nt = a->ne[2];
    #pragma omp parallel for num_threads(n_omp)
    for (int64_t t = 0; t < nt; ++t) {
        static thread_local std::vector<float> cache;
        if ((int64_t)cache.size() < ne0) cache.resize(ne0);
        rknpu_rope_cache(cache.data(), pp[t], rp, ff, ne0);
        for (int64_t h = 0; h < nh; ++h) rknpu_rope_head(y + (t * nh + h) * ne0, x + (t * nh + h) * ne0, cache.data(), ne0, rp);
    }
}

// KV-cache writes of the fused K (normed + roped) and V (normed) rows:
// SET_ROWS from a view of those outputs into an F16 (or F32) cache, with
// ggml-cpu's conversion (round to nearest even) (RKNPU_KV_WRITE)
static bool rknpu_kv_write_enabled() {
    static const bool v = []() {
        const char* env = std::getenv("RKNPU_KV_WRITE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return v;
}
static bool rknpu_set_rows_supported(const struct ggml_tensor* op) {
    const struct ggml_tensor *b = op->src[0], *idx = op->src[1];
    if (op->op != GGML_OP_SET_ROWS || !b || !idx || b->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32) return false;
    if (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) return false;
    if (b->ne[2] != 1 || b->ne[3] != 1 || op->ne[2] != 1 || op->ne[3] != 1 || idx->ne[0] != b->ne[1] || ggml_nrows(idx) != 1) return false;
    if (b->nb[0] != sizeof(float) || op->nb[0] != ggml_type_size(op->type) || b->ne[0] != op->ne[0]) return false;
    const struct ggml_tensor* src = b->view_src ? b->view_src : b;
    return (src->op == GGML_OP_ROPE && rknpu_rope_supported(src)) ||
           (src->op == GGML_OP_RMS_NORM && rknpu_head_norm_supported(src));
}
static inline int64_t rknpu_row_index(const struct ggml_tensor* idx, const void* base, int64_t i) {
    return idx->type == GGML_TYPE_I64 ? ((const int64_t*)base)[i] : (int64_t)((const int32_t*)base)[i];
}
// one row into the cache
static inline void rknpu_kv_store_row(const struct ggml_tensor* sr, char* cache, const void* idxp, int64_t i, const float* row) {
    const int64_t c = rknpu_row_index(sr->src[1], idxp, i);
    GGML_ASSERT(c >= 0 && c < sr->ne[1]);
    char* d = cache + c * sr->nb[1];
    if (sr->type == GGML_TYPE_F16) rknpu_fp32_to_fp16(row, (uint16_t*)d, sr->ne[0]);
    else memcpy(d, row, sr->ne[0] * sizeof(float));
}
static void rknpu_set_rows_op(struct ggml_tensor* sr, int n_omp) {
    const struct ggml_tensor* b = sr->src[0];
    const char* bp = (const char*)get_tensor_real_ptr(b);
    const void* ip = get_tensor_real_ptr(sr->src[1]);
    char* cache = (char*)get_tensor_real_ptr(sr);
    #pragma omp parallel for num_threads(n_omp)
    for (int64_t i = 0; i < b->ne[1]; ++i) rknpu_kv_store_row(sr, cache, ip, i, (const float*)(bp + i * b->nb[1]));
}

static bool rknpu_glu_supported(const struct ggml_tensor* op) {
    if (ggml_get_glu_op(op) != GGML_GLU_OP_GEGLU) return false;
    const struct ggml_tensor *a = op->src[0], *b = op->src[1];
    if (!a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous_1(a) || !ggml_is_contiguous_1(b)) return false;
    if (a->ne[0] % 4 != 0 || a->ne[0] != b->ne[0] || a->nb[0] != 4 || b->nb[0] != 4 || op->nb[0] != 4) return false;
    return ggml_nrows(a) == ggml_nrows(op) && ggml_nrows(b) == ggml_nrows(op);
}

static inline void rknpu_geglu_row(int64_t n, float* y, const float* x, const float* g) {
    int64_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t c0 = vdupq_n_f32(0.79788456080286535587989211986876f), c1 = vdupq_n_f32(0.044715f);
    const float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f), half = vdupq_n_f32(0.5f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t xv = vld1q_f32(x + i);
        const float32x4_t z  = vmulq_f32(vmulq_f32(c0, xv), vfmaq_f32(one, vmulq_f32(c1, xv), xv));
        const float32x4_t th = vsubq_f32(one, vdivq_f32(two, vaddq_f32(rknpu_v_expf(vmulq_f32(two, z)), one)));
        float32x4_t gel = vmulq_f32(vmulq_f32(half, xv), vaddq_f32(one, th));
        gel = vbslq_f32(vcleq_f32(xv, vdupq_n_f32(-10.0f)), vdupq_n_f32(0.0f), gel);
        gel = vbslq_f32(vcgeq_f32(xv, vdupq_n_f32(10.0f)), xv, gel);
        vst1q_f32(y + i, vmulq_f32(gel, vld1q_f32(g + i)));
    }
#endif
    for (; i < n; ++i) {   // not reached for the supported widths (multiple of 4)
        const float v = x[i];
        const float gl = v <= -10.0f ? 0.0f : v >= 10.0f ? v
                       : 0.5f * v * (1.0f + tanhf(0.79788456080286535587989211986876f * v * (1.0f + 0.044715f * v * v)));
        y[i] = gl * g[i];
    }
}

static void rknpu_geglu(struct ggml_tensor* dst, int n_omp) {
    const struct ggml_tensor *a = dst->src[0], *b = dst->src[1];
    const int64_t nc = a->ne[0], nr = ggml_nrows(dst);
    const char* a_base = (const char*)get_tensor_real_ptr(a);
    const char* b_base = (const char*)get_tensor_real_ptr(b);
    char* d_base = (char*)get_tensor_real_ptr(dst);
    #pragma omp parallel for num_threads(n_omp) if(nr > 1)
    for (int64_t r = 0; r < nr; ++r) {
        rknpu_geglu_row(nc, (float*)(d_base + r * dst->nb[1]), (const float*)(a_base + r * a->nb[1]), (const float*)(b_base + r * b->nb[1]));
    }
}

// Fused up-dequant + GEGLU for one row block of a pipelined FFN (llama order
// [gate, up, GLU]): up rows go to a per-thread contiguous buffer, then
// GEGLU(gate from memory, up from the buffer) is written to the GLU output.
// Out of line on purpose: inlined into the pipelined collect's OpenMP region
// it slowed the common path by ~10% even when not taken.
static void __attribute__((noinline)) rknpu_fused_up_geglu_block(
        const struct ggml_tensor* glu, int m_abs, int r0, int nr, int N, const float* common,
        const std::vector<std::shared_ptr<rknn_tensor_mem>>& c_mem, const std::vector<rknpu2_native_geom>& c_geom,
        const std::vector<MatrixSegmentN>& segs, const float* chan) {
    const struct ggml_tensor* gate_t = glu->src[0];
    float* y_base = (float*)get_tensor_real_ptr(glu);
    const float* gate_base = (const float*)get_tensor_real_ptr(gate_t);
    const size_t ys = glu->nb[1] / sizeof(float), gs = gate_t->nb[1] / sizeof(float);
    static thread_local std::vector<float> ubuf;
    if (ubuf.size() < (size_t)4 * N) ubuf.resize((size_t)4 * N);
    for (size_t idx = 0; idx < segs.size(); ++idx) {
        const int N_offset = segs[idx].offset_n;
        rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(
            ubuf.data() + N_offset, (size_t)N, (const int16_t*)c_mem[idx]->virt_addr, r0, nr,
            c_geom[idx].m_stride, c_geom[idx].outer, c_geom[idx].sub, segs[idx].size_n, common,
            chan + N_offset, /*store=*/ true);
    }
    for (int r = 0; r < nr; ++r) {
        rknpu_geglu_row(N, y_base + (size_t)(m_abs + r) * ys, gate_base + (size_t)(m_abs + r) * gs, ubuf.data() + (size_t)r * N);
    }
}

// Rows r0..r0+nr of chunk `chunk` of a deferred gate, exactly as its own
// collect would have stored them
static void rknpu_deferred_gate_rows(const rknpu_deferred_gate& dg, float* dst, size_t dst_stride, int chunk, int r0, int nr) {
    const int m_abs = chunk * dg.MC + r0;
    for (size_t idx = 0; idx < dg.segs.size(); ++idx) {
        const int N_offset = dg.segs[idx].offset_n;
        rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(
            dst + N_offset, dst_stride, (const int16_t*)dg.c[chunk][idx]->virt_addr, r0, nr,
            dg.c_geom[idx].m_stride, dg.c_geom[idx].outer, dg.c_geom[idx].sub, dg.segs[idx].size_n,
            dg.common.data() + m_abs, dg.chan + N_offset, /*store=*/ true);
    }
}

// Fallback when the fused up does not follow: write the gate's output after all
static void rknpu_materialize_deferred_gate(rknpu_deferred_gate& dg, int n_omp) {
    if (!dg.gate) return;
    float* d = (float*)get_tensor_real_ptr(dg.gate);
    const int n_chunks = (dg.M + dg.MC - 1) / dg.MC;
    for (int c = 0; c < n_chunks; ++c) {
        const int rows = std::min(dg.MC, dg.M - c * dg.MC);
        const int n_blocks = (rows + 3) / 4;
        #pragma omp parallel for num_threads(n_omp)
        for (int blk = 0; blk < n_blocks; ++blk) {
            const int r0 = blk * 4;
            rknpu_deferred_gate_rows(dg, d + (size_t)(c * dg.MC + r0) * dg.N, (size_t)dg.N, c, r0, std::min(4, rows - r0));
        }
    }
    dg.gate = nullptr;
    dg.up = nullptr;
}

// rknpu_fused_up_geglu_block with the gate rows from a deferred gate
static void __attribute__((noinline)) rknpu_fused_gate_up_geglu_block(
        const struct ggml_tensor* glu, const rknpu_deferred_gate& dg, int chunk, int m_abs, int r0, int nr, int N,
        const float* common, const std::vector<std::shared_ptr<rknn_tensor_mem>>& c_mem,
        const std::vector<rknpu2_native_geom>& c_geom, const std::vector<MatrixSegmentN>& segs, const float* chan) {
    float* y_base = (float*)get_tensor_real_ptr(glu);
    const size_t ys = glu->nb[1] / sizeof(float);
    static thread_local std::vector<float> ubuf, gbuf;
    if (ubuf.size() < (size_t)4 * N) ubuf.resize((size_t)4 * N);
    if (gbuf.size() < (size_t)4 * N) gbuf.resize((size_t)4 * N);
    for (size_t idx = 0; idx < segs.size(); ++idx) {
        const int N_offset = segs[idx].offset_n;
        rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(
            ubuf.data() + N_offset, (size_t)N, (const int16_t*)c_mem[idx]->virt_addr, r0, nr,
            c_geom[idx].m_stride, c_geom[idx].outer, c_geom[idx].sub, segs[idx].size_n, common,
            chan + N_offset, /*store=*/ true);
    }
    rknpu_deferred_gate_rows(dg, gbuf.data(), (size_t)N, chunk, r0, nr);
    for (int r = 0; r < nr; ++r) {
        rknpu_geglu_row(N, y_base + (size_t)(m_abs + r) * ys, gbuf.data() + (size_t)r * N, ubuf.data() + (size_t)r * N);
    }
}

// Sets up `down` (the matmul after [up, GLU]) as a rknpu_w4a4_job; false
// when it would not take the pipelined W4A4 path with K-segments on
// Hadamard block bounds (then nothing is staged and it runs normally)
static bool rknpu_w4a4_job_setup(ggml_backend_rknpu_context* bctx, rknpu_w4a4_job& j,
                                 const struct ggml_tensor* down, int M, int MC, int role = 0) {
    // role separates the A/C buffer cache keys of jobs that are live at once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const struct ggml_tensor* w = down->src[0];
    if (!w->buffer || strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer)), "RKNPU") != 0) return false;
    const auto* p = config.resolve_op_support(w);
    if (!p || !p->use_hadamard || !rknpu2_calibration::per_channel_b_scales()) return false;
    if (p->npu_type_a != rknpu2_configuration::NPU_TYPE_INT4 || p->npu_type_c != rknpu2_configuration::NPU_TYPE_INT16 ||
        p->ac_layout != RKNN_MM_LAYOUT_NATIVE ||
        (p->npu_type_b != rknpu2_configuration::NPU_TYPE_INT4 && p->npu_type_b != rknpu2_configuration::NPU_TYPE_INT8)) return false;
    const int K = (int)w->ne[0], N = (int)w->ne[1];
    if (rknpu2_calibration::hadamard_k_op(K) != K) return false;
    int kl = config.max_k_limit;
    if (p->effective_k > 0) kl = (kl > 0) ? std::min(kl, p->effective_k) : p->effective_k;
    const auto ksegs = compute_k_segments(K, kl, p->k_align);
    const int hb = rknpu2_calibration::hadamard_block_len(K);
    if (ksegs.empty() || ksegs.size() > 2) return false;
    for (const auto& ks : ksegs) if (ks.offset_k % hb != 0 || ks.size_k % hb != 0) return false;
    const auto all_n = compute_n_segments(N, config.active_cores, p->n_align);
    std::vector<MatrixSegmentN> nsegs;
    for (const auto& sg : all_n) if (sg.size_n > 0) nsegs.push_back(sg);
    if (nsegs.empty()) return false;
    const size_t nas = nsegs.size();

    auto* bc = (ggml_backend_rknpu_buffer_context*)w->buffer->context;
    int fd = -1; void* virt = nullptr; int32_t dom = 0;
    const float* sv = nullptr; const float* chan = nullptr;
    {
        std::lock_guard<std::mutex> lock(bc->mutex);
        auto it = bc->tensor_allocs.find((uintptr_t)w->data - (uintptr_t)bc->virtual_base);
        if (it == bc->tensor_allocs.end()) return false;
        fd = it->second.mem->fd; virt = it->second.mem->virt_addr; dom = it->second.iommu_domain_id;
        auto is = bc->hadamard_s_vectors.find(w);
        auto ic = bc->quantized_tensor_scales.find(w);
        if (is == bc->hadamard_s_vectors.end() || ic == bc->quantized_tensor_scales.end()) return false;
        sv = is->second.data(); chan = ic->second.data();
    }
    const size_t tsp = p->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 ? 1 : 0;
    std::vector<std::vector<size_t>> boff(ksegs.size(), std::vector<size_t>(nas, 0));
    {
        size_t off = 0;
        for (size_t k = 0; k < ksegs.size(); ++k) {
            for (const auto& sg : all_n) {
                for (size_t idx = 0; idx < nas; ++idx) {
                    if (nsegs[idx].offset_n == sg.offset_n) { boff[k][idx] = off; break; }
                }
                if (sg.size_n > 0) off += tsp > 0 ? (size_t)sg.size_n * ksegs[k].size_k * tsp : (size_t)sg.size_n * ksegs[k].size_k / 2;
            }
        }
    }
    const int ns = (int)ksegs.size(), n_chunks = (M + MC - 1) / MC;
    j.cctx.assign(ns * nas, nullptr);
    for (int sg = 0; sg < ns; ++sg) {
        for (size_t idx = 0; idx < nas; ++idx) {
            auto& mc = j.cctx[sg * nas + idx];
            mc = bctx->get_matmul_ctx((uintptr_t)virt, boff[sg][idx], MC, ksegs[sg].size_k, nsegs[idx].size_n,
                                      nsegs[idx].core_id, p->mm_type, p->ac_layout, dom);
            if (!mc || mc->ctx == 0) return false;
            if (!mc->b_bound) {
                rknn_tensor_mem* mem = rknn_create_mem_from_fd(mc->ctx, fd, virt, mc->io_attr.B.size, boff[sg][idx]);
                if (!mem) return false;
                auto deleter = [ctx = mc->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                mc->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);
                if (rknn_matmul_set_io_mem(mc->ctx, mc->mem_B.get(), &mc->io_attr.B) != 0) return false;
                mc->b_bound = true;
            }
        }
    }
    j.a.assign(n_chunks, std::vector<std::shared_ptr<rknn_tensor_mem>>(ns));
    for (int c = 0; c < n_chunks; ++c) {
        for (int sg = 0; sg < ns; ++sg) {
            auto& c0 = j.cctx[sg * nas];
            j.a[c][sg] = get_tensor_buffer(bctx, c0->ctx, c0->io_attr.A.size,
                std::make_tuple(MC, ksegs[sg].size_k, (int)p->npu_type_a + 16 * (c + 1) + 8192 + 64 * sg + 1024 * role, dom), bctx->a_buffer_cache);
            if (!j.a[c][sg]) return false;
        }
    }
    j.c.assign(2, std::vector<std::shared_ptr<rknn_tensor_mem>>(ns * nas));
    for (int sl = 0; sl < 2; ++sl) {
        for (int sg = 0; sg < ns; ++sg) {
            for (size_t idx = 0; idx < nas; ++idx) {
                auto& mc = j.cctx[sg * nas + idx];
                j.c[sl][sg * nas + idx] = get_tensor_buffer(bctx, mc->ctx, mc->io_attr.C.size,
                    std::make_tuple(MC, nsegs[idx].size_n, nsegs[idx].core_id, (int)p->npu_type_c + 16 * (sl + 1) + 8192 + 64 * sg + 1024 * role, dom),
                    bctx->c_buffer_cache);
                if (!j.c[sl][sg * nas + idx]) return false;
            }
        }
    }
    j.a_geom.resize(ns);
    for (int sg = 0; sg < ns; ++sg) {
        if (rknpu2_native_geom_from_dims(j.cctx[sg * nas]->io_attr.A.dims, j.cctx[sg * nas]->io_attr.A.n_dims, &j.a_geom[sg]) != 0) return false;
    }
    j.c_geom.resize(ns * nas);
    for (size_t q = 0; q < j.c_geom.size(); ++q) {
        if (rknpu2_native_geom_from_dims(j.cctx[q]->io_attr.C.dims, j.cctx[q]->io_attr.C.n_dims, &j.c_geom[q]) != 0) return false;
    }
    for (int sg = 1; sg < ns; ++sg) {   // dequant2 reads both C's with one geometry
        for (size_t idx = 0; idx < nas; ++idx) {
            const auto &g0 = j.c_geom[idx], &g1 = j.c_geom[sg * nas + idx];
            if (g0.m_stride != g1.m_stride || g0.outer != g1.outer || g0.sub != g1.sub) return false;
        }
    }
    j.seg_off.clear(); j.seg_len.clear();
    for (const auto& ks : ksegs) { j.seg_off.push_back(ks.offset_k); j.seg_len.push_back(ks.size_k); }
    j.scales_A.assign(ns, std::vector<float>(M, 1.0f));
    j.nsegs = nsegs; j.nas = nas;
    j.M = M; j.N = N; j.K = K; j.MC = MC; j.n_chunks = n_chunks;
    j.s_vec = sv; j.chan = chan;
    j.a_clip = rknpu2_calibration::a_clip_factor();
    j.hdiv = (float)hb;
    j.started = -1;
    j.node = down;
    return true;
}

static void rknpu_w4a4_job_bind(rknpu_w4a4_job& j, int c) {
    const int ns = (int)j.seg_off.size();
    for (int sg = 0; sg < ns; ++sg) {
        RKNN_CHECK(rknn_mem_sync(j.cctx[sg * j.nas]->ctx, j.a[c][sg].get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A down chunk");
        for (size_t idx = 0; idx < j.nas; ++idx) {
            auto& mc = j.cctx[sg * j.nas + idx];
            if (mc->bound_A != j.a[c][sg].get()) {
                RKNN_CHECK(rknn_matmul_set_io_mem(mc->ctx, j.a[c][sg].get(), &mc->io_attr.A), "set_io_mem A down chunk");
                mc->bound_A = j.a[c][sg].get();
            }
            auto& cm = j.c[c & 1][sg * j.nas + idx];
            if (mc->bound_C != cm.get()) {
                RKNN_CHECK(rknn_matmul_set_io_mem(mc->ctx, cm.get(), &mc->io_attr.C), "set_io_mem C down chunk");
                mc->bound_C = cm.get();
            }
        }
    }
}

static void rknpu_w4a4_job_start(ggml_backend_rknpu_context* bctx, rknpu_w4a4_job& j, int c) {
    rknpu_w4a4_job_bind(j, c);
    bctx->async_runner.start(j.cctx);
    j.started = c;
}

static void rknpu_w4a4_job_collect(rknpu_w4a4_job& j, int c, float* dst, int n_omp) {
    const int ns = (int)j.seg_off.size();
    for (size_t q = 0; q < j.cctx.size(); ++q) {
        RKNN_CHECK(rknn_mem_sync(j.cctx[q]->ctx, j.c[c & 1][q].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C down chunk");
    }
    const int m0 = c * j.MC, rows = std::min(j.MC, j.M - m0);
    const int n_blocks = (rows + 3) / 4;
    #pragma omp parallel for num_threads(n_omp)
    for (int blk = 0; blk < n_blocks; ++blk) {
        const int r0 = blk * 4;
        const int nr = std::min(4, rows - r0);
        float common0[4], common1[4];
        for (int r = 0; r < nr; ++r) {
            common0[r] = j.scales_A[0][m0 + r0 + r] / j.hdiv;
            if (ns > 1) common1[r] = j.scales_A[1][m0 + r0 + r] / j.hdiv;
        }
        for (size_t idx = 0; idx < j.nas; ++idx) {
            const int N_offset = j.nsegs[idx].offset_n;
            const auto& g = j.c_geom[idx];
            float* d = dst + (size_t)(m0 + r0) * j.N + N_offset;
            if (ns > 1) {
                rknpu2_quantization::dequant2_int16_tiled_perchan_rows(d, (size_t)j.N,
                    (const int16_t*)j.c[c & 1][idx]->virt_addr, (const int16_t*)j.c[c & 1][j.nas + idx]->virt_addr,
                    r0, nr, g.m_stride, g.outer, g.sub, j.nsegs[idx].size_n, common0, common1,
                    j.chan + N_offset, j.chan + (size_t)j.N + N_offset);
            } else {
                rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(d, (size_t)j.N,
                    (const int16_t*)j.c[c & 1][idx]->virt_addr, r0, nr, g.m_stride, g.outer, g.sub,
                    j.nsegs[idx].size_n, common0, j.chan + N_offset, /*store=*/ true);
            }
        }
    }
}

// rknpu_fused_gate_up_geglu_block, but each GEGLU row goes through the down
// job's A-prep (per K-segment: Hadamard range, amax, INT4 pack) straight
// into its A buffer for this chunk instead of being written out
static void __attribute__((noinline)) rknpu_fused_gate_up_geglu_prep_block(
        rknpu_w4a4_job& j, const rknpu_deferred_gate& dg, int chunk, int m_abs, int r0, int nr, int N,
        const float* common, const std::vector<std::shared_ptr<rknn_tensor_mem>>& c_mem,
        const std::vector<rknpu2_native_geom>& c_geom, const std::vector<MatrixSegmentN>& segs, const float* chan) {
    static thread_local std::vector<float> ubuf, gbuf, yrow, full_row;
    static thread_local std::vector<uint8_t> packed_row;
    auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(ubuf, (size_t)4 * N);
    grow(gbuf, (size_t)4 * N);
    grow(yrow, (size_t)N);
    for (size_t idx = 0; idx < segs.size(); ++idx) {
        const int N_offset = segs[idx].offset_n;
        rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(
            ubuf.data() + N_offset, (size_t)N, (const int16_t*)c_mem[idx]->virt_addr, r0, nr,
            c_geom[idx].m_stride, c_geom[idx].outer, c_geom[idx].sub, segs[idx].size_n, common,
            chan + N_offset, /*store=*/ true);
    }
    rknpu_deferred_gate_rows(dg, gbuf.data(), (size_t)N, chunk, r0, nr);
    for (int r = 0; r < nr; ++r) {
        const int m = m_abs + r;
        rknpu_geglu_row(N, yrow.data(), gbuf.data() + (size_t)r * N, ubuf.data() + (size_t)r * N);
        for (size_t sg = 0; sg < j.seg_off.size(); ++sg) {
            const int len = j.seg_len[sg];
            grow(full_row, (size_t)len);
            grow(packed_row, (size_t)len / 2);
            rknpu2_calibration::hadamard_transform_signed_range(full_row.data(), yrow.data(), j.s_vec, j.K, j.seg_off[sg], len);
            const float sc = j.a_clip * rknpu2_quantization::amax_fp32(full_row.data(), len) / 7.0f;
            j.scales_A[sg][m] = sc;
            rknpu2_quantization::quantize_fp32_to_int4_packed(full_row.data(), packed_row.data(), len, sc);
            const auto& ag = j.a_geom[sg];
            rknpu2_native_scatter_row((uint8_t*)j.a[chunk][sg]->virt_addr, packed_row.data(), r0 + r, ag.m_stride, ag.outer, ag.sub / 2);
        }
    }
}

// A-prep of chunk c for gate and up together (same activation rows, each
// with its own Hadamard signs): every row is read once. Same steps as the
// pipelined prep for a single K-segment.
static void rknpu_ffn_prep(rknpu_w4a4_job& g, rknpu_w4a4_job& u, const float* src1, int row_stride, int c, int n_omp) {
    const int m0 = c * g.MC, rows = std::min(g.MC, g.M - m0);
    #pragma omp parallel for num_threads(n_omp)
    for (int r = 0; r < rows; ++r) {
        const float* src_row = src1 + (size_t)(m0 + r) * row_stride;
        static thread_local std::vector<float> full_row;
        static thread_local std::vector<uint8_t> packed_row;
        if (full_row.size() < (size_t)g.K) full_row.resize(g.K);
        if (packed_row.size() < (size_t)g.K / 2) packed_row.resize(g.K / 2);
        for (rknpu_w4a4_job* j : {&g, &u}) {
            rknpu2_calibration::hadamard_transform_signed(full_row.data(), src_row, j->s_vec, j->K, j->K);
            const float sc = j->a_clip * rknpu2_quantization::amax_fp32(full_row.data(), j->K) / 7.0f;
            j->scales_A[0][m0 + r] = sc;
            rknpu2_quantization::quantize_fp32_to_int4_packed(full_row.data(), packed_row.data(), j->K, sc);
            const auto& ag = j->a_geom[0];
            rknpu2_native_scatter_row((uint8_t*)j->a[c][0]->virt_addr, packed_row.data(), r, ag.m_stride, ag.outer, ag.sub / 2);
        }
    }
    for (rknpu_w4a4_job* j : {&g, &u}) {
        RKNN_CHECK(rknn_mem_sync(j->cctx[0]->ctx, j->a[c][0].get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A ffn chunk");
    }
}

// Rows r0..r0+nr of chunk c of a single-K-segment job, stored into dst
static void rknpu_job_rows(const rknpu_w4a4_job& j, float* dst, size_t dst_stride, int c, int r0, int nr) {
    float common[4];
    for (int r = 0; r < nr; ++r) common[r] = j.scales_A[0][c * j.MC + r0 + r] / j.hdiv;
    for (size_t idx = 0; idx < j.nas; ++idx) {
        const int N_offset = j.nsegs[idx].offset_n;
        const auto& g = j.c_geom[idx];
        rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(dst + N_offset, dst_stride,
            (const int16_t*)j.c[c & 1][idx]->virt_addr, r0, nr, g.m_stride, g.outer, g.sub,
            j.nsegs[idx].size_n, common, j.chan + N_offset, /*store=*/ true);
    }
}

// Gate and up of rows r0..r0+nr of one N-segment straight from their INT16
// C cells (one shared native geometry, 8-wide cells) to the GEGLU output,
// tile by tile: the same operations as dequantizing both (store path) and
// then rknpu_geglu_row, so element-exact, without the row buffers.
static void __attribute__((noinline)) rknpu_gate_up_geglu_tiles(
        float* y, size_t ys, const int16_t* cg, const int16_t* cu, int r0, int nr, const rknpu2_native_geom& geom,
        int n_limit, const float* common_g, const float* common_u, const float* chan_g, const float* chan_u) {
#ifdef __ARM_NEON
    const float32x4_t c0 = vdupq_n_f32(0.79788456080286535587989211986876f), c1 = vdupq_n_f32(0.044715f);
    const float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f), half = vdupq_n_f32(0.5f);
    auto gelu_mul = [&](float32x4_t xv, float32x4_t g) {
        const float32x4_t z  = vmulq_f32(vmulq_f32(c0, xv), vfmaq_f32(one, vmulq_f32(c1, xv), xv));
        const float32x4_t th = vsubq_f32(one, vdivq_f32(two, vaddq_f32(rknpu_v_expf(vmulq_f32(two, z)), one)));
        float32x4_t gel = vmulq_f32(vmulq_f32(half, xv), vaddq_f32(one, th));
        gel = vbslq_f32(vcleq_f32(xv, vdupq_n_f32(-10.0f)), vdupq_n_f32(0.0f), gel);
        gel = vbslq_f32(vcgeq_f32(xv, vdupq_n_f32(10.0f)), xv, gel);
        return vmulq_f32(gel, g);
    };
    const int outer = n_limit / 8;
    // prefetch both C streams pf tiles ahead: each tile is one line of this
    // row block, a page-sized stride apart, which the hardware prefetcher
    // does not follow (RKNPU_TILE_PF, 0 = off)
    static const int pf = []() {
        const char* env = std::getenv("RKNPU_TILE_PF");
        return env ? std::atoi(env) : 8;
    }();
    for (int t = 0; t < outer; ++t) {
        const int n0 = t * 8;
        if (pf > 0 && t + pf < outer) {
            const size_t cell_pf = ((size_t)(t + pf) * geom.m_stride + r0) * 8;
            __builtin_prefetch(cg + cell_pf);
            __builtin_prefetch(cu + cell_pf);
        }
        const float32x4_t sg0 = vld1q_f32(chan_g + n0), sg1 = vld1q_f32(chan_g + n0 + 4);
        const float32x4_t su0 = vld1q_f32(chan_u + n0), su1 = vld1q_f32(chan_u + n0 + 4);
        for (int r = 0; r < nr; ++r) {
            const size_t cell = ((size_t)t * geom.m_stride + r0 + r) * 8;
            const int16x8_t g16 = vld1q_s16(cg + cell), u16 = vld1q_s16(cu + cell);
            const float32x4_t vcg = vdupq_n_f32(common_g[r]), vcu = vdupq_n_f32(common_u[r]);
            const float32x4_t x0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16))),  vmulq_f32(sg0, vcg));
            const float32x4_t x1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(g16))), vmulq_f32(sg1, vcg));
            const float32x4_t u0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(u16))),  vmulq_f32(su0, vcu));
            const float32x4_t u1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(u16))), vmulq_f32(su1, vcu));
            float* yr = y + (size_t)r * ys + n0;
            vst1q_f32(yr,     gelu_mul(x0, u0));
            vst1q_f32(yr + 4, gelu_mul(x1, u1));
        }
    }
#else
    GGML_ABORT("rknpu_gate_up_geglu_tiles needs NEON");
#endif
}

// Gate and up rows of chunk c -> GEGLU -> down's A-prep into its chunk-c
// A buffer (as rknpu_fused_gate_up_geglu_prep_block)
static void __attribute__((noinline)) rknpu_ffn_fused_block(rknpu_w4a4_job& g, rknpu_w4a4_job& u, rknpu_w4a4_job& d,
                                                           int c, int r0, int nr) {
    const int N = g.N;
    static thread_local std::vector<float> ubuf, gbuf, yrow, full_row;
    static thread_local std::vector<uint8_t> packed_row;
    auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(ubuf, (size_t)4 * N);
    grow(gbuf, (size_t)4 * N);
    grow(yrow, (size_t)N);
    // GEGLU rows straight from the gate and up INT16 tiles when both share
    // 8-wide cell geometries (else via dequantized row buffers)
    static const bool tiles_env = []() {
        const char* env = std::getenv("RKNPU_GEGLU_TILES");
        return env == nullptr || std::atoi(env) != 0;
    }();
    bool tiles = tiles_env && g.nas == u.nas;
    for (size_t idx = 0; tiles && idx < g.nas; ++idx) {
        const auto &a = g.c_geom[idx], &b = u.c_geom[idx];
        tiles = a.sub == 8 && b.sub == 8 && a.m_stride == b.m_stride && a.outer == b.outer &&
                g.nsegs[idx].offset_n == u.nsegs[idx].offset_n && g.nsegs[idx].size_n == u.nsegs[idx].size_n &&
                g.nsegs[idx].size_n % 8 == 0;
    }
#ifndef __ARM_NEON
    tiles = false;
#endif
    grow(ubuf, (size_t)4 * N);   // the GEGLU rows when tiled
    if (tiles) {
        float cg[4], cu[4];
        for (int r = 0; r < nr; ++r) {
            cg[r] = g.scales_A[0][c * g.MC + r0 + r] / g.hdiv;
            cu[r] = u.scales_A[0][c * u.MC + r0 + r] / u.hdiv;
        }
        for (size_t idx = 0; idx < g.nas; ++idx) {
            const int N_offset = g.nsegs[idx].offset_n;
            rknpu_gate_up_geglu_tiles(ubuf.data() + N_offset, (size_t)N,
                (const int16_t*)g.c[c & 1][idx]->virt_addr, (const int16_t*)u.c[c & 1][idx]->virt_addr,
                r0, nr, g.c_geom[idx], g.nsegs[idx].size_n, cg, cu, g.chan + N_offset, u.chan + N_offset);
        }
    } else {
        rknpu_job_rows(u, ubuf.data(), (size_t)N, c, r0, nr);
        rknpu_job_rows(g, gbuf.data(), (size_t)N, c, r0, nr);
    }
    for (int r = 0; r < nr; ++r) {
        const int m = c * g.MC + r0 + r;
        const float* yr = ubuf.data() + (size_t)r * N;
        if (!tiles) {
            rknpu_geglu_row(N, yrow.data(), gbuf.data() + (size_t)r * N, ubuf.data() + (size_t)r * N);
            yr = yrow.data();
        }
        for (size_t sg = 0; sg < d.seg_off.size(); ++sg) {
            const int len = d.seg_len[sg];
            grow(full_row, (size_t)len);
            grow(packed_row, (size_t)len / 2);
            rknpu2_calibration::hadamard_transform_signed_range(full_row.data(), yr, d.s_vec, d.K, d.seg_off[sg], len);
            const float sc = d.a_clip * rknpu2_quantization::amax_fp32(full_row.data(), len) / 7.0f;
            d.scales_A[sg][m] = sc;
            rknpu2_quantization::quantize_fp32_to_int4_packed(full_row.data(), packed_row.data(), len, sc);
            const auto& ag = d.a_geom[sg];
            rknpu2_native_scatter_row((uint8_t*)d.a[c][sg]->virt_addr, packed_row.data(), r0 + r, ag.m_stride, ag.outer, ag.sub / 2);
        }
    }
}

static void rknpu_ffn_fused(rknpu_w4a4_job& g, rknpu_w4a4_job& u, rknpu_w4a4_job& d, int c, int n_omp) {
    for (rknpu_w4a4_job* j : {&g, &u}) {
        for (size_t q = 0; q < j->cctx.size(); ++q) {
            RKNN_CHECK(rknn_mem_sync(j->cctx[q]->ctx, j->c[c & 1][q].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C ffn chunk");
        }
    }
    const int rows = std::min(g.MC, g.M - c * g.MC);
    const int n_blocks = (rows + 3) / 4;
    #pragma omp parallel for num_threads(n_omp)
    for (int blk = 0; blk < n_blocks; ++blk) {
        rknpu_ffn_fused_block(g, u, d, c, blk * 4, std::min(4, rows - blk * 4));
    }
}

// The whole FFN [gate, up, GLU(gate, up), down] scheduled as one block, so
// CPU phases overlap NPU batches across the three matmuls. NPU batch c runs
// gate and up of chunk c together with down of chunk c-2; meanwhile the CPU
// preps chunk c+1, runs the fused gate/up/GEGLU/down-prep of chunk c-1 and
// collects down chunk c-3. Gate, up and the GLU output are never written;
// down's output is element-exact vs the node-by-node path.
static void rknpu_ffn_block(ggml_backend_rknpu_context* bctx, const struct ggml_tensor* src1, struct ggml_tensor* down_dst, int n_omp) {
    auto &g = bctx->ffn_gate, &u = bctx->ffn_up, &d = bctx->ffn_down;
    auto& gu = bctx->ffn_gu_ctx;
    auto& gud = bctx->ffn_gud_ctx;
    gu.clear();
    gu.insert(gu.end(), g.cctx.begin(), g.cctx.end());
    gu.insert(gu.end(), u.cctx.begin(), u.cctx.end());
    gud = gu;
    gud.insert(gud.end(), d.cctx.begin(), d.cctx.end());
    const float* x = (const float*)get_tensor_real_ptr(src1);
    const int row_stride = (int)(src1->nb[1] / sizeof(float));
    float* out = (float*)get_tensor_real_ptr(down_dst);
    const int n = g.n_chunks;
    auto start_batch = [&](int c) {   // gate/up chunk c (if any) + down chunk c - 2 (if any)
        const bool has_gu = c < n, has_d = c >= 2 && c - 2 < n;
        if (has_gu) { rknpu_w4a4_job_bind(g, c); rknpu_w4a4_job_bind(u, c); }
        if (has_d) rknpu_w4a4_job_bind(d, c - 2);
        bctx->async_runner.start(has_gu && has_d ? gud : has_gu ? gu : d.cctx);
    };
    // gate/up preps run ahead: batch 0 (gate/up chunk 0 alone) has no other
    // CPU work to hide it, while every later window has a fused step, so
    // `ahead` chunks are prepped during batch 0 (RKNPU_FFN_PREP_AHEAD; 1/2/3/all
    // = 269.2/272.5/271.9/271.0 t/s pp512)
    static const int ahead = []() {
        const char* env = std::getenv("RKNPU_FFN_PREP_AHEAD");
        return env ? std::max(1, std::atoi(env)) : 2;
    }();
    int prepped = 1;
    rknpu_ffn_prep(g, u, x, row_stride, 0, n_omp);
    start_batch(0);
    for (int c = 1; c < n + 2; ++c) {
        while (prepped < n && prepped < c + (c == 1 ? ahead : 1)) {   // overlaps batch c-1
            rknpu_ffn_prep(g, u, x, row_stride, prepped++, n_omp);
        }
        bctx->async_runner.wait();
        start_batch(c);
        if (c - 1 < n) rknpu_ffn_fused(g, u, d, c - 1, n_omp);       // gate/up chunk c-1 done
        if (c >= 3) rknpu_w4a4_job_collect(d, c - 3, out, n_omp);  // down chunk c-3 done
    }
    bctx->async_runner.wait();
    rknpu_w4a4_job_collect(d, n - 1, out, n_omp);
}

static bool rknpu_fa_supported(const struct ggml_tensor* op) {
    const struct ggml_tensor *q = op->src[0], *k = op->src[1], *v = op->src[2], *mask = op->src[3], *sinks = op->src[4];
    if (!q || !k || !v || sinks) return false;
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32) return false;
    if (mask && mask->type != GGML_TYPE_F16) return false;
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float*)op->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) return false;
    // ne[3] = sequences (llama-perplexity and llama-server batch several);
    // K/V and the mask may broadcast over it
    if (q->ne[3] % k->ne[3] != 0 || q->ne[3] % v->ne[3] != 0) return false;
    if (mask && q->ne[3] % mask->ne[3] != 0) return false;
    const int64_t DK = k->ne[0], DV = v->ne[0], n_q = q->ne[1], n_kv = k->ne[1];
    if (n_q < 32) return false;                       // prefill only; decode stays on the CPU kernel
    if (DK % 32 != 0 || DV % 16 != 0 || n_kv % 32 != 0) return false;
    // Validated envelope: librknnrt aborts the process ("Failed to config layer") on some FP16
    // shapes beyond it (e.g. n_kv 16640, 16896, 17408 at M = 2048); larger runs use ggml-cpu
    static const int64_t max_kv = []() {
        const char* env = std::getenv("RKNPU_FA_MAX_KV");
        return env ? (int64_t)std::atoll(env) : (int64_t)16384;
    }();
    if (n_kv > max_kv || (q->ne[2] / k->ne[2]) * n_q > 2048) return false;
    if (q->ne[2] % k->ne[2] != 0 || k->ne[2] != v->ne[2]) return false;
    if (k->nb[0] != 2 || v->nb[0] != 2 || q->nb[0] != 4 || op->nb[0] != 4) return false;
    return true;
}

// Four consecutive rows r0..r0+3 of a native-layout matrix: in every cell
// column their cells are adjacent (4 * cb contiguous bytes, one cache line
// for 16-byte cells), so they are moved together, line by line
static inline void rknpu_native_gather_rows(uint8_t* const* rows, int R, const uint8_t* src, int r0, const rknpu2_native_geom& g, int cb,
                                            int t0 = 0, int t1 = -1) {
    if (t1 < 0) t1 = g.outer;
    for (int t = t0; t < t1; ++t) {
        const uint8_t* c = src + ((size_t)t * g.m_stride + r0) * cb;
#ifdef __ARM_NEON
        if (cb == 16) {
            for (int k = 0; k < R; ++k) vst1q_u8(rows[k] + (size_t)t * 16, vld1q_u8(c + k * 16));
            continue;
        }
#endif
        for (int k = 0; k < R; ++k) memcpy(rows[k] + (size_t)t * cb, c + k * cb, cb);
    }
}
static inline void rknpu_native_scatter_rows(uint8_t* dst, const uint8_t* const* rows, int R, int r0, const rknpu2_native_geom& g, int cb) {
    for (int t = 0; t < g.outer; ++t) {
        uint8_t* c = dst + ((size_t)t * g.m_stride + r0) * cb;
#ifdef __ARM_NEON
        if (cb == 16) {
            for (int k = 0; k < R; ++k) vst1q_u8(c + k * 16, vld1q_u8(rows[k] + (size_t)t * 16));
            continue;
        }
#endif
        for (int k = 0; k < R; ++k) memcpy(c + k * cb, rows[k] + (size_t)t * cb, cb);
    }
}
// rows per block in the native-layout attention loops (RKNPU_FA_ROWS)
static int rknpu_fa_rows() {
    static const int v = []() {
        const char* env = std::getenv("RKNPU_FA_ROWS");
        const int x = env ? std::atoi(env) : 64;
        return (x >= 4 && x <= 512 && x % 4 == 0) ? x : 64;
    }();
    return v;
}

static void rknpu_flash_attn(ggml_backend_rknpu_context* bctx, struct ggml_tensor* dst, int n_omp) {
    const struct ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *mask = dst->src[3];
    ++bctx->attn_call;
    static const int64_t pv_chunk = []() {
        const char* env = std::getenv("RKNPU_FA_PV_CHUNK");
        const int v = env ? std::atoi(env) : 2048;
        return (int64_t)(v > 0 ? (v + 255) / 256 * 256 : 1 << 30);
    }();
    const int64_t KC = std::min<int64_t>(pv_chunk, dst->src[1]->ne[1]);
    // Q*K^T scores as FP16 (RKNPU_FA_S16, default on): halves the score traffic the softmax
    // reads (+10% pp8192); the scores are rounded to FP16 before the softmax (0 = FP32 scores)
    // Online softmax per P*V chunk (RKNPU_FA_ONLINE, default on): the row is processed in
    // chunk-sized pieces (L1-resident scratch) and P holds exp(s - m_chunk); the chunk sums are
    // rescaled when the P*V outputs are accumulated. Needs FP16 scores and the native path.
    static const bool online_env = []() {
        const char* env = std::getenv("RKNPU_FA_ONLINE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    static const bool s16 = []() {
        const char* env = std::getenv("RKNPU_FA_S16");
        return env == nullptr || std::atoi(env) != 0;
    }();
    const int64_t DK = k->ne[0], DV = v->ne[0];
    const int64_t n_q = q->ne[1], n_head = q->ne[2], n_kv = k->ne[1], n_kvh = k->ne[2];
    const int64_t rk2 = n_head / n_kvh;
    const int64_t M = rk2 * n_q;
    const int64_t n_seq = q->ne[3];

    float scale = 1.0f, softcap = 0.0f;
    memcpy(&scale,   (const float*)dst->op_params + 0, sizeof(float));
    memcpy(&softcap, (const float*)dst->op_params + 2, sizeof(float));
    if (softcap != 0.0f) scale /= softcap;

    const char* q_base = (const char*)get_tensor_real_ptr(q);
    const char* k_base = (const char*)get_tensor_real_ptr(k);
    const char* v_base = (const char*)get_tensor_real_ptr(v);
    const char* m_base = mask ? (const char*)get_tensor_real_ptr(mask) : nullptr;
    char* d_base = (char*)get_tensor_real_ptr(dst);

    // Softmax range [lo, hi) of every mask row, found once and shared by all
    // heads and groups; the softmax gathers only those S cells (RKNPU_FA_RANGE)
    static const bool range_enabled = []() {
        const char* env = std::getenv("RKNPU_FA_RANGE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    std::vector<int32_t> m_lo, m_hi;
    if (mask && range_enabled) {
        const int64_t n_ms = mask->ne[3];
        m_lo.resize(n_ms * n_q);
        m_hi.resize(n_ms * n_q);
        #pragma omp parallel for num_threads(n_omp)
        for (int64_t x = 0; x < n_ms * n_q; ++x) {
            int64_t lo, hi;
            rknpu_softmax_range((const ggml_fp16_t*)(m_base + (x / n_q) * mask->nb[3] + (x % n_q) * mask->nb[1]), n_kv, &lo, &hi);
            m_lo[x] = (int32_t)lo; m_hi[x] = (int32_t)hi;
        }
    }

    // Stage st of the attention of group g of sequence i3: 0 = fill A (Q) and
    // B (K, V), 1 = Q*K^T run, 2 = softmax into P, 3 = P*V run, 4 = output.
    // Groups run on their own NPU core and contexts.
    auto stage = [&](int64_t i3, int64_t g, int st, int core) {
        const int64_t ik3 = i3 / (n_seq / k->ne[3]);
        const int64_t iv3 = i3 / (n_seq / v->ne[3]);
        const char* m_seq = mask ? m_base + (i3 % mask->ne[3]) * mask->nb[3] : nullptr;
        rknpu_attn_context* qk = bctx->get_attn_ctx((int)M, (int)DK, (int)n_kv, RKNN_MM_LAYOUT_TP_NORM, core, 0, s16);
        // P*V in chunks of at most KC positions along K: the NPU's FP16 matmul runs ~8x slower
        // per FLOP once K passes ~4096 (RKNPU_FA_PV_CHUNK; 0 = one run); partial outputs are summed
        rknpu_attn_context* pvs[128];
        const int n_pvc = (int)((n_kv + KC - 1) / KC);
        GGML_ASSERT(n_pvc <= 128);
        for (int c = 0; c < n_pvc; ++c) {
            const int64_t kc = std::min(KC, n_kv - c * KC);
            pvs[c] = bctx->get_attn_ctx((int)M, (int)kc, (int)DV, RKNN_MM_LAYOUT_NORM, core, c);
            GGML_ASSERT(pvs[c] && "RKNPU2: attention matmul context creation failed");
        }
        rknpu_attn_context* pv = pvs[0];
        GGML_ASSERT(qk && "RKNPU2: attention matmul context creation failed");
        bool pv_native = true, pv_b_native = true;
        for (int c = 0; c < n_pvc; ++c) { pv_native = pv_native && pvs[c]->native; pv_b_native = pv_b_native && pvs[c]->b_native && pvs[c]->b_subK == pv->b_subK && pvs[c]->b_subN == pv->b_subN; }
        (void)ik3; (void)iv3; (void)m_seq;

        // A = Q rows of the rk2 heads of this group, FP16, row r = hh*n_q + i
        uint16_t* a = (uint16_t*)qk->A->virt_addr;
        const int FR = rknpu_fa_rows();
        const bool native = qk->native && pv_native && M % FR == 0;
        GGML_ASSERT((native || !qk->c_fp16) && "RKNPU2: FP16 scores need the native attention path");
        const bool online = online_env && native && qk->c_fp16;
        if (online && st == 2 && (int64_t)bctx->fa_stats[core].size() < M * n_pvc * 2) bctx->fa_stats[core].resize(M * n_pvc * 2);
        const bool host_out = online || n_pvc > 1;
        if (host_out && st == 2 && bctx->fa_out[core].size() < pv->io_attr.C.size / 4) bctx->fa_out[core].resize(pv->io_attr.C.size / 4);
        auto q_row = [&](int64_t r) {
            const int64_t hh = r / n_q, i = r % n_q, h = g * rk2 + hh;
            return (const float*)(q_base + i * q->nb[1] + h * q->nb[2] + i3 * q->nb[3]);
        };
        if (st == 0) {
            if (native) {
                #pragma omp parallel for num_threads(n_omp)
                for (int64_t r0 = 0; r0 < M; r0 += FR) {
                    static thread_local std::vector<uint16_t> buf;
                    if ((int64_t)buf.size() < FR * DK) buf.resize(FR * DK);
                    uint8_t* rows[512];
                    for (int k = 0; k < FR; ++k) {
                        rknpu_fp32_to_fp16(q_row(r0 + k), buf.data() + k * DK, DK);
                        rows[k] = (uint8_t*)(buf.data() + k * DK);
                    }
                    rknpu_native_scatter_rows((uint8_t*)a, rows, FR, (int)r0, qk->a_geom, qk->a_geom.sub * 2);
                }
            } else {
                #pragma omp parallel for num_threads(n_omp)
                for (int64_t r = 0; r < M; ++r) rknpu_fp32_to_fp16(q_row(r), a + r * DK, DK);
            }
            // B = K rows (n_kv x DK, TP_NORM) and V rows (n_kv x DV, NORM)
            uint16_t* bk = (uint16_t*)qk->B->virt_addr;
            auto k_row = [&](int64_t j) { return (const uint16_t*)(k_base + j * k->nb[1] + g * k->nb[2] + ik3 * k->nb[3]); };
            auto v_row = [&](int64_t j) { return (const uint16_t*)(v_base + j * v->nb[1] + g * v->nb[2] + iv3 * v->nb[3]); };
            const bool nb = qk->b_native && pv_b_native;
            if (nb) {
                // Q*K^T: B(k = dim, n = position) -> runs of subK dims of one K row
                const int sNq = qk->b_subN, sKq = qk->b_subK, kbq = (int)DK / sKq;
                // P*V: B(k = position, n = dim) -> runs of subK positions of one dim (a transpose of V)
                const int sNv = pv->b_subN, sKv = pv->b_subK, kbv = (int)n_kv / sKv;
                const int kbc = (int)(KC / sKv);   // position blocks per chunk
                #pragma omp parallel num_threads(n_omp)
                {
                    #pragma omp for nowait
                    for (int64_t j = 0; j < n_kv; ++j) {
                        const uint16_t* kr = k_row(j);
                        uint16_t* base = bk + ((size_t)(j / sNq) * kbq * sNq + (j % sNq)) * sKq;
                        for (int kb = 0; kb < kbq; ++kb) memcpy(base + (size_t)kb * sNq * sKq, kr + kb * sKq, sKq * 2);
                    }
                    #pragma omp for
                    for (int64_t pb = 0; pb < kbv; ++pb) {   // a block of subK positions
                        const uint16_t* vr[64];
                        for (int q = 0; q < sKv; ++q) vr[q] = v_row(pb * sKv + q);
                        const int c = (int)(pb / kbc), lpb = (int)(pb % kbc);
                        const int kbv_c = (int)(std::min(KC, n_kv - c * KC) / sKv);
                        uint16_t* bv = (uint16_t*)pvs[c]->B->virt_addr;
                        for (int64_t d = 0; d < DV; ++d) {
                            uint16_t* o = bv + (((size_t)(d / sNv) * kbv_c + lpb) * sNv + (d % sNv)) * sKv;
                            for (int q = 0; q < sKv; ++q) o[q] = vr[q][d];
                        }
                    }
                }
            } else {
                for (int64_t j = 0; j < n_kv; ++j) {
                    memcpy(bk + j * DK, k_row(j), DK * 2);
                    memcpy((uint16_t*)pvs[j / KC]->B->virt_addr + (j % KC) * DV, v_row(j), DV * 2);
                }
            }
            rknn_mem_sync(qk->ctx, qk->A, RKNN_MEMORY_SYNC_TO_DEVICE);
            rknn_mem_sync(qk->ctx, qk->B, RKNN_MEMORY_SYNC_TO_DEVICE);
            for (int c = 0; c < n_pvc; ++c) rknn_mem_sync(pvs[c]->ctx, pvs[c]->B, RKNN_MEMORY_SYNC_TO_DEVICE);
            if (!nb) {
                // Re-bind B after writing it: for a non-native B the driver converts
                // it to its internal layout at set_io_mem time, so data written into
                // an already-bound buffer is never seen (P*V came back all zeros)
                RKNN_CHECK(rknn_matmul_set_io_mem(qk->ctx, qk->B, &qk->io_attr.B), "set_io_mem attn K");
                for (int c = 0; c < n_pvc; ++c) RKNN_CHECK(rknn_matmul_set_io_mem(pvs[c]->ctx, pvs[c]->B, &pvs[c]->io_attr.B), "set_io_mem attn V");
            }
        } else if (st == 1) {
            rknn_matmul_run(qk->ctx);
            rknn_mem_sync(qk->ctx, qk->C, RKNN_MEMORY_SYNC_FROM_DEVICE);
        } else if (st == 2) {

            // softmax rows into P (FP16) = A of the second matmul
            const float* S = (const float*)qk->C->virt_addr;
            static thread_local std::vector<uint16_t> p_full;   // row-major P when the chunks' A are not native
            if (!native && (int64_t)p_full.size() < M * n_kv) p_full.resize(M * n_kv);
            uint16_t* P = native ? nullptr : p_full.data();
            auto m_row = [&](int64_t r) {
                return mask ? (const ggml_fp16_t*)(m_seq + (r % n_q) * mask->nb[1]) : nullptr;
            };
            if (native) {
                #pragma omp parallel for num_threads(n_omp)
                for (int64_t r0 = 0; r0 < M; r0 += FR) {
                    static thread_local std::vector<float> row, s_buf;
                    static thread_local std::vector<uint16_t> p_buf, h_buf;
                    if ((int64_t)row.size() < n_kv) row.resize(n_kv);
                    if ((int64_t)s_buf.size() < FR * n_kv) s_buf.resize(FR * n_kv);
                    if ((int64_t)p_buf.size() < FR * n_kv) p_buf.resize(FR * n_kv);
                    uint8_t* srows[512];
                    uint8_t* prows[512];
                    for (int k = 0; k < FR; ++k) {
                        srows[k] = (uint8_t*)(s_buf.data() + k * n_kv);
                        prows[k] = (uint8_t*)(p_buf.data() + k * n_kv);
                    }
                    int64_t lo[512], hi[512];
                    int64_t blo = n_kv, bhi = 0;
                    for (int k = 0; k < FR; ++k) {
                        if (!m_lo.empty()) {
                            const int64_t x = (i3 % mask->ne[3]) * n_q + (r0 + k) % n_q;
                            lo[k] = m_lo[x]; hi[k] = m_hi[x];
                        } else {
                            rknpu_softmax_range(m_row(r0 + k), n_kv, &lo[k], &hi[k]);
                        }
                        if (hi[k] > lo[k]) { blo = std::min(blo, lo[k]); bhi = std::max(bhi, hi[k]); }
                    }
                    const int sub = qk->c_geom.sub;
                    if (!range_enabled) { blo = 0; bhi = n_kv; }
                    if (bhi > blo && !qk->c_fp16) {
                        rknpu_native_gather_rows(srows, FR, (const uint8_t*)S, (int)r0, qk->c_geom, sub * 4, (int)(blo / sub), (int)((bhi + sub - 1) / sub));
                    } else if (bhi > blo) {
                        // FP16 scores: gather the cells; the plain path widens them per row into s_buf
                        if ((int64_t)h_buf.size() < FR * n_kv) h_buf.resize(FR * n_kv);
                        uint8_t* hrows[512];
                        for (int k = 0; k < FR; ++k) hrows[k] = (uint8_t*)(h_buf.data() + k * n_kv);
                        rknpu_native_gather_rows(hrows, FR, (const uint8_t*)S, (int)r0, qk->c_geom, sub * 2, (int)(blo / sub), (int)((bhi + sub - 1) / sub));
                        for (int k = 0; k < FR && !online; ++k) {
                            if (hi[k] <= lo[k]) continue;
                            const uint16_t* h = h_buf.data() + k * n_kv;
                            float* f = s_buf.data() + k * n_kv;
                            int64_t j = lo[k];
#ifdef __ARM_NEON
                            for (; j + 8 <= hi[k]; j += 8) {
                                const float16x8_t v = vreinterpretq_f16_u16(vld1q_u16(h + j));
                                vst1q_f32(f + j, vcvt_f32_f16(vget_low_f16(v)));
                                vst1q_f32(f + j + 4, vcvt_f32_f16(vget_high_f16(v)));
                            }
#endif
                            for (; j < hi[k]; ++j) f[j] = ggml_fp16_to_fp32(h[j]);
                        }
                    }
                    if (online) {
                        float* st = bctx->fa_stats[core].data();
                        for (int k = 0; k < FR; ++k) {
                            const uint16_t* h = h_buf.data() + k * n_kv;
                            uint16_t* pk = p_buf.data() + k * n_kv;
                            const ggml_fp16_t* mr = m_row(r0 + k);
                            for (int c = 0; c < n_pvc; ++c) {
                                const int64_t c0 = c * KC, c1 = std::min(n_kv, c0 + KC);
                                const int64_t a = std::max(lo[k], c0), b = std::min(hi[k], c1);
                                float* mlc = st + ((r0 + k) * n_pvc + c) * 2;
                                if (b <= a) {
                                    memset(pk + c0, 0, (c1 - c0) * sizeof(uint16_t));
                                    mlc[0] = -INFINITY; mlc[1] = 0.0f;
                                    continue;
                                }
                                memset(pk + c0, 0, (a - c0) * sizeof(uint16_t));
                                memset(pk + b, 0, (c1 - b) * sizeof(uint16_t));
                                rknpu_softmax_chunk16(h + a, mr ? mr + a : nullptr, b - a, scale, softcap, row.data(), pk + a, mlc, mlc + 1);
                            }
                        }
                    } else {
                        for (int k = 0; k < FR; ++k) {
                            rknpu_softmax_row(s_buf.data() + k * n_kv, m_row(r0 + k), n_kv, lo[k], hi[k], scale, softcap, row.data(), p_buf.data() + k * n_kv);
                        }
                    }
                    for (int c = 0; c < n_pvc; ++c) {
                        uint8_t* pr[512];
                        for (int k = 0; k < FR; ++k) pr[k] = prows[k] + (size_t)c * KC * 2;
                        rknpu_native_scatter_rows((uint8_t*)pvs[c]->A->virt_addr, pr, FR, (int)r0, pvs[c]->a_geom, pvs[c]->a_geom.sub * 2);
                    }
                }
            } else {
                #pragma omp parallel for num_threads(n_omp)
                for (int64_t r = 0; r < M; ++r) {
                    static thread_local std::vector<float> row;
                    if ((int64_t)row.size() < n_kv) row.resize(n_kv);
                    int64_t lo, hi;
                    rknpu_softmax_range(m_row(r), n_kv, &lo, &hi);
                    rknpu_softmax_row(S + r * n_kv, m_row(r), n_kv, lo, hi, scale, softcap, row.data(), P + r * n_kv);
                    for (int c = 0; c < n_pvc; ++c) {
                        const int64_t kc = std::min(KC, n_kv - c * KC);
                        memcpy((uint16_t*)pvs[c]->A->virt_addr + r * kc, P + r * n_kv + c * KC, kc * 2);
                    }
                }
            }
        } else if (st == 3) {
            // online softmax: per row, O = sum_c a_rc O_c with a_rc = exp(m_rc - M_r) / L_r,
            // L_r = sum_c exp(m_rc - M_r) l_rc (the chunks' P are unnormalized)
            static thread_local std::vector<float> alpha;
            if (online) {
                const float* st = bctx->fa_stats[core].data();
                if ((int64_t)alpha.size() < M * n_pvc) alpha.resize(M * n_pvc);
                for (int64_t r = 0; r < M; ++r) {
                    float Mr = -INFINITY;
                    for (int c = 0; c < n_pvc; ++c) Mr = std::max(Mr, st[(r * n_pvc + c) * 2]);
                    float Lr = 0.0f;
                    for (int c = 0; c < n_pvc; ++c) {
                        const float m = st[(r * n_pvc + c) * 2];
                        alpha[r * n_pvc + c] = m == -INFINITY ? 0.0f : expf(m - Mr);
                        Lr += alpha[r * n_pvc + c] * st[(r * n_pvc + c) * 2 + 1];
                    }
                    const float inv = Lr > 0.0f ? 1.0f / Lr : 0.0f;
                    for (int c = 0; c < n_pvc; ++c) alpha[r * n_pvc + c] *= inv;
                }
            }
            for (int c = 0; c < n_pvc; ++c) {
                rknn_mem_sync(pvs[c]->ctx, pvs[c]->A, RKNN_MEMORY_SYNC_TO_DEVICE);
                rknn_matmul_run(pvs[c]->ctx);
                rknn_mem_sync(pvs[c]->ctx, pvs[c]->C, RKNN_MEMORY_SYNC_FROM_DEVICE);
                if (online) {   // acc = a_r0 * C_0, then acc += a_rc * C_c: cell (t, r) holds sub floats of row r
                    float* acc = bctx->fa_out[core].data();
                    const float* part = (const float*)pvs[c]->C->virt_addr;
                    const int ms = pv->c_geom.m_stride, outer = pv->c_geom.outer, csub = pv->c_geom.sub;
                    for (int t = 0; t < outer; ++t) {
                        for (int64_t r = 0; r < M; ++r) {
                            const float a = alpha[r * n_pvc + c];
                            float* ac = acc + ((size_t)t * ms + r) * csub;
                            const float* pc = part + ((size_t)t * ms + r) * csub;
                            if (c == 0) { for (int j = 0; j < csub; ++j) ac[j] = a * pc[j]; }
                            else        { for (int j = 0; j < csub; ++j) ac[j] += a * pc[j]; }
                        }
                    }
                    continue;
                }
                if (n_pvc > 1) {   // the host buffer accumulates the partial products (same C geometry for every chunk)
                    float* acc = bctx->fa_out[core].data();
                    const float* part = (const float*)pvs[c]->C->virt_addr;
                    const int64_t n = (int64_t)pv->io_attr.C.size / 4;
                    if (c == 0) { memcpy(acc, part, n * sizeof(float)); continue; }
                    int64_t i = 0;
#ifdef __ARM_NEON
                    for (; i + 4 <= n; i += 4) vst1q_f32(acc + i, vaddq_f32(vld1q_f32(acc + i), vld1q_f32(part + i)));
#endif
                    for (; i < n; ++i) acc[i] += part[i];
                }
            }
        } else {

            // O rows -> dst (permuted: row (i*n_head + h))
            const float* O = host_out ? bctx->fa_out[core].data() : (const float*)pv->C->virt_addr;
            // permute(0, 2, 1, 3): row (i3*n_q*n_head + i*n_head + h)
            auto d_row = [&](int64_t r) {
                const int64_t hh = r / n_q, i = r % n_q, h = g * rk2 + hh;
                return d_base + (i3 * n_q * n_head + i * n_head + h) * dst->nb[1];
            };
            if (native) {
                #pragma omp parallel for num_threads(n_omp)
                for (int64_t r0 = 0; r0 < M; r0 += FR) {
                    // gather into a local block, then write whole rows: the
                    // destination rows are interleaved by head, far apart
                    static thread_local std::vector<float> o_buf;
                    if ((int64_t)o_buf.size() < FR * DV) o_buf.resize(FR * DV);
                    uint8_t* rows[512];
                    for (int k = 0; k < FR; ++k) rows[k] = (uint8_t*)(o_buf.data() + k * DV);
                    rknpu_native_gather_rows(rows, FR, (const uint8_t*)O, (int)r0, pv->c_geom, pv->c_geom.sub * 4);
                    for (int k = 0; k < FR; ++k) memcpy(d_row(r0 + k), o_buf.data() + k * DV, DV * sizeof(float));
                }
            } else {
                #pragma omp parallel for num_threads(n_omp)
                for (int64_t r = 0; r < M; ++r) memcpy(d_row(r), O + r * DV, DV * sizeof(float));
            }
        }
    };

    static const bool overlap_enabled = []() {
        const char* env = std::getenv("RKNPU_FA_OVERLAP");
        return env == nullptr || std::atoi(env) != 0;
    }();
    const int n = (int)(n_seq * n_kvh);   // items it = (i3, g), i3-major
    if (!overlap_enabled || n < 2) {
        for (int64_t i3 = 0; i3 < n_seq; ++i3) {
            for (int64_t g = 0; g < n_kvh; ++g) {
                for (int st = 0; st < 5; ++st) stage(i3, g, st, (int)(g % 3));
            }
        }
        return;
    }
    // Software pipeline over the items: item it runs on core (and contexts)
    // it % 3, and at most items it-1, it, it+1 are in flight, so they never
    // share contexts. The NPU runs of one item overlap the CPU stages of
    // its neighbours.
    for (int c = 0; c < std::min(n, 3); ++c) {   // create every context here: the workers only look them up
        GGML_ASSERT(bctx->get_attn_ctx((int)M, (int)DK, (int)n_kv, RKNN_MM_LAYOUT_TP_NORM, c, 0, s16));
        for (int64_t k0 = 0, i = 0; k0 < n_kv; k0 += KC, ++i) {
            GGML_ASSERT(bctx->get_attn_ctx((int)M, (int)std::min(KC, n_kv - k0), (int)DV, RKNN_MM_LAYOUT_NORM, c, (int)i));
        }
    }
    auto st_k = [&](int it, int st) { stage(it / n_kvh, it % n_kvh, st, it % 3); };
    std::vector<rknpu_fn_pool::ticket> t_qk(n), t_pv(n);
    st_k(0, 0);
    bctx->fa_pool.submit(t_qk[0], [&] { st_k(0, 1); });
    for (int it = 0; it < n; ++it) {
        if (it + 1 < n) st_k(it + 1, 0);                      // overlaps QK(it), PV(it-1)
        bctx->fa_pool.wait(t_qk[it]);
        if (it + 1 < n) bctx->fa_pool.submit(t_qk[it + 1], [&, it] { st_k(it + 1, 1); });
        st_k(it, 2);                                          // overlaps QK(it+1), PV(it-1)
        bctx->fa_pool.submit(t_pv[it], [&, it] { st_k(it, 3); });
        if (it >= 1) { bctx->fa_pool.wait(t_pv[it - 1]); st_k(it - 1, 4); }   // overlaps PV(it)
    }
    bctx->fa_pool.wait(t_pv[n - 1]);
    st_k(n - 1, 4);
}

static enum ggml_status ggml_backend_rknpu_graph_compute_impl(ggml_backend_t backend, struct ggml_cgraph* cgraph);

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    if (!g_rknpu_profile.on) return ggml_backend_rknpu_graph_compute_impl(backend, cgraph);
    auto t0 = std::chrono::steady_clock::now();
    auto st = ggml_backend_rknpu_graph_compute_impl(backend, cgraph);
    g_rknpu_profile.graph_ns += rknpu_profile::ns(t0, std::chrono::steady_clock::now());
    g_rknpu_profile.graphs++;
    g_rknpu_profile.maybe_print();
    return st;
}

static enum ggml_status ggml_backend_rknpu_graph_compute_impl(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;
    const int n_omp = backend_ctx->n_threads > 0 ? backend_ctx->n_threads : omp_get_max_threads();

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (backend_ctx->deferred_gate.gate && node != backend_ctx->deferred_gate.up) {
            rknpu_materialize_deferred_gate(backend_ctx->deferred_gate, n_omp);
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            rknpu_flash_attn(backend_ctx, node, n_omp);
            continue;
        }
        if (node->op == GGML_OP_GLU) {
            rknpu_geglu(node, n_omp);
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM) {   // per-head norm not fused into its matmul
            struct ggml_tensor* mul = nullptr;
            for (int j = node_i + 1; j < cgraph->n_nodes; ++j) {
                struct ggml_tensor* t = cgraph->nodes[j];
                if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_NONE) continue;
                if (t->op == GGML_OP_MUL && (t->src[0] == node || t->src[1] == node) && rknpu_head_norm_weight(t)) {
                    // fuse only if the MUL is the norm's sole consumer
                    bool sole = true;
                    for (int q2 = node_i + 1; q2 < cgraph->n_nodes && sole; ++q2) {
                        const struct ggml_tensor* u = cgraph->nodes[q2];
                        if (u == t) continue;
                        if (u->view_src == node) sole = false;
                        for (int q = 0; q < GGML_MAX_SRC; ++q) sole = sole && u->src[q] != node;
                    }
                    if (sole && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) { mul = t; node_i = j; }
                }
                break;
            }
            rknpu_head_norm_op(node, mul, n_omp);
            continue;
        }
        if (!backend_ctx->done_nodes.empty() && backend_ctx->done_nodes.erase(node)) continue;
        if (node->op == GGML_OP_SET_ROWS) {   // KV-cache write not fused into its projection's dequant
            rknpu_set_rows_op(node, n_omp);
            continue;
        }
        if (node->op == GGML_OP_ROPE) {   // RoPE not fused into a projection's dequant
            rknpu_rope_op(node, n_omp);
            continue;
        }
        if (node->op == GGML_OP_MUL) {   // a weight MUL whose norm ran separately
            const struct ggml_tensor* w = rknpu_head_norm_weight(node);
            const struct ggml_tensor* n = node->src[0] == w ? node->src[1] : node->src[0];
            const float* x = (const float*)get_tensor_real_ptr(n);
            const float* wp = (const float*)get_tensor_real_ptr(w);
            float* y = (float*)get_tensor_real_ptr(node);
            const int64_t hd = node->ne[0], nr = ggml_nrows(node);
            #pragma omp parallel for num_threads(n_omp)
            for (int64_t r = 0; r < nr; ++r) {
                for (int64_t i = 0; i < hd; ++i) y[r * hd + i] = x[r * hd + i] * wp[i];
            }
            continue;
        }
        if (node->op != GGML_OP_MUL_MAT) continue;
        const auto t_node = g_rknpu_profile.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (g_rknpu_profile.on) g_rknpu_profile.nodes++;

        // [gate, up, GLU, down] as one block (rknpu_ffn_block)
        {
            static const bool ffn_block_enabled = []() {
                const char* env = std::getenv("RKNPU_FFN_BLOCK");
                return env == nullptr || std::atoi(env) != 0;
            }();
            // chunk rows: smaller chunks shorten the schedule's head and tail
            static const int MCf = []() {
                const char* env = std::getenv("RKNPU_FFN_MC");
                const int v = env ? std::atoi(env) : 128;
                return (v >= 32 && v <= 512 && v % 32 == 0) ? v : 128;
            }();
            if (ffn_block_enabled && rknpu_pipeline_enabled() && n_omp > 1 && node_i + 3 < cgraph->n_nodes) {
                struct ggml_tensor* nu = cgraph->nodes[node_i + 1];
                struct ggml_tensor* ng = cgraph->nodes[node_i + 2];
                struct ggml_tensor* nd = cgraph->nodes[node_i + 3];
                const struct ggml_tensor* x = node->src[1];
                const int Mf = (int)x->ne[1];
                bool ok = nu->op == GGML_OP_MUL_MAT && ng->op == GGML_OP_GLU && nd->op == GGML_OP_MUL_MAT &&
                          nu->src[1] == x && ng->src[0] == node && ng->src[1] == nu && nd->src[1] == ng &&
                          rknpu_glu_supported(ng) && Mf > MCf && x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
                          x->ne[2] * x->ne[3] == 1 && node->src[0]->ne[1] == nu->src[0]->ne[1] &&
                          node->src[0]->ne[0] == nu->src[0]->ne[0] && nd->src[0]->ne[0] == node->src[0]->ne[1] &&
                          ggml_is_contiguous(nd) && (int)nd->ne[1] == Mf &&
                          !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !(nu->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                          !(ng->flags & GGML_TENSOR_FLAG_OUTPUT);
                for (int j = node_i + 1; j < cgraph->n_nodes && ok; ++j) {   // nothing else reads gate, up or the GLU
                    const struct ggml_tensor* t = cgraph->nodes[j];
                    for (const struct ggml_tensor* v : {(const struct ggml_tensor*)node, (const struct ggml_tensor*)nu, (const struct ggml_tensor*)ng}) {
                        if (t->view_src == v) ok = false;
                        for (int q = 0; q < GGML_MAX_SRC; ++q) {
                            if (t->src[q] == v && !((t == nu && v == node) || (t == ng && (v == node || v == nu)) || (t == nd && v == ng))) ok = false;
                        }
                    }
                }
                ok = ok && rknpu_w4a4_job_setup(backend_ctx, backend_ctx->ffn_gate, node, Mf, MCf, 1) &&
                     rknpu_w4a4_job_setup(backend_ctx, backend_ctx->ffn_up, nu, Mf, MCf, 2) &&
                     rknpu_w4a4_job_setup(backend_ctx, backend_ctx->ffn_down, nd, Mf, MCf, 3) &&
                     backend_ctx->ffn_gate.seg_off.size() == 1 && backend_ctx->ffn_up.seg_off.size() == 1 &&
                     backend_ctx->ffn_gate.seg_len[0] == backend_ctx->ffn_gate.K && backend_ctx->ffn_up.seg_len[0] == backend_ctx->ffn_up.K;
                if (ok) {
                    rknpu_ffn_block(backend_ctx, x, nd, n_omp);
                    backend_ctx->ffn_gate.node = backend_ctx->ffn_up.node = backend_ctx->ffn_down.node = nullptr;
                    node_i += 3;
                    if (g_rknpu_profile.on) g_rknpu_profile.node_ns += rknpu_profile::ns(t_node, std::chrono::steady_clock::now());
                    continue;
                }
                backend_ctx->ffn_gate.node = backend_ctx->ffn_up.node = backend_ctx->ffn_down.node = nullptr;
            }
        }

        if (backend_ctx->down_job.node == node) {   // prepared and started by the up node
            auto& j = backend_ctx->down_job;
            float* d = (float*)get_tensor_real_ptr(node);
            if (j.started < 0) rknpu_w4a4_job_start(backend_ctx, j, 0);
            for (int c = 1; c < j.n_chunks; ++c) {
                backend_ctx->async_runner.wait();
                rknpu_w4a4_job_start(backend_ctx, j, c);
                rknpu_w4a4_job_collect(j, c - 1, d, n_omp);   // overlaps NPU chunk c
            }
            backend_ctx->async_runner.wait();
            rknpu_w4a4_job_collect(j, j.n_chunks - 1, d, n_omp);
            j.node = nullptr;
            if (g_rknpu_profile.on) g_rknpu_profile.node_ns += rknpu_profile::ns(t_node, std::chrono::steady_clock::now());
            continue;
        }

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        // Skipping zero-dimension matmuls
        if (M == 0 || K == 0 || N == 0) {
            continue;
        }

        // Using next power of two for M for efficient caching
        int M_op = M;
        if (M > 1) {
            M_op = rknpu2_calibration::next_power_of_two(M);
        }

        const auto* pipeline = config.resolve_op_support(src0);
        if (!pipeline) continue;

        // Initializing Hadamard Transform Logic. Block-diagonal FWHT keeps
        // K_op == K (no zero-padding: on non-power-of-two models the legacy
        // padding inflated every NPU weight read ~1.5-1.6x — see
        // RKNPU2-decode-research.md #3 re-profile).
        const bool is_hadamard = (pipeline->use_hadamard);
        const int K_op = is_hadamard ? rknpu2_calibration::hadamard_k_op(K) : K;

        const rknn_matmul_type matmul_type = pipeline->mm_type;
        const int alignment = pipeline->n_align;

        // W8A8 at small M > 1 (speculative verify batches): NATIVE A/C. With
        // NORM the runtime converts C on one thread inside every run, which made
        // E4B's Q8_0 output projection 4x slower at M=4 than at M=1 (RKNPU_W8A8_NATIVE)
        static const bool w8a8_native = []() {
            const char* env = std::getenv("RKNPU_W8A8_NATIVE");
            return env == nullptr || std::atoi(env) != 0;
        }();
        const rknn_matmul_layout ac_layout =
            (w8a8_native && M > 1 && M <= 32 && pipeline->ac_layout == RKNN_MM_LAYOUT_NORM &&
             pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 &&
             pipeline->npu_type_c == rknpu2_configuration::NPU_TYPE_INT32 &&
             rknpu2_calibration::per_channel_b_scales())
                ? RKNN_MM_LAYOUT_NATIVE : pipeline->ac_layout;

        // Computing specific hardware segments
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto all_k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto all_n_segments = compute_n_segments(N, config.active_cores, alignment);

        std::vector<MatrixSegmentN> active_n_segments;
        for (const auto& seg : all_n_segments) {
            if (seg.size_n > 0) active_n_segments.push_back(seg);
        }

        if (active_n_segments.empty()) continue;

        // Initializing variables
        const size_t num_active_segments = active_n_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        // Acquiring the B-matrix buffer
        ggml_backend_buffer_t src0_buffer = src0->buffer;
        auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
        size_t tensor_offset_in_virtual = (uintptr_t)src0->data - (uintptr_t)src0_buf_ctx->virtual_base;

        int32_t b_domain_id = 0;
        int tensor_fd = -1;
        void* tensor_virt_addr = nullptr;
        {
            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->tensor_allocs.find(tensor_offset_in_virtual);
            GGML_ASSERT(it != src0_buf_ctx->tensor_allocs.end() && "B-matrix RKNN buffer not found");

            tensor_fd = it->second.mem->fd;
            tensor_virt_addr = it->second.mem->virt_addr;
            b_domain_id = it->second.iommu_domain_id;
        }

        // Batch geometry. src0 is 2D (checked in supports_op); src1 and dst
        // may carry ne[2]/ne[3], in which case the same weights are applied
        // to each slice. Strides come from nb[] rather than being assumed
        // contiguous.
        const int64_t nbatch = src1->ne[2] * src1->ne[3];
        const size_t  src1_batch_nb = (src1->ne[2] > 1) ? src1->nb[2] : src1->nb[3];
        const size_t  dst_batch_nb  = (dst->ne[2]  > 1) ? dst->nb[2]  : dst->nb[3];
        const char* const src1_base = (const char*)get_tensor_real_ptr(src1);

        // Cleaning the C-matrix buffer, every slice of it
        float* dst_data = (float*)get_tensor_real_ptr(dst);
        // The pipelined prefill path (below) stores on the first K-segment
        // instead of accumulating, so its M x N output is not zeroed here
        // (a serial memset on the main thread, ~8% of pp512). Same
        // conditions as `pipelined` there.
        const bool pipelined_node = rknpu_pipeline_enabled() && nbatch == 1 && M > 256 &&
            pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4 &&
            pipeline->npu_type_c == rknpu2_configuration::NPU_TYPE_INT16 &&
            pipeline->ac_layout == RKNN_MM_LAYOUT_NATIVE &&
            (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) &&
            rknpu2_calibration::per_channel_b_scales() && n_omp > 1;
        // Fuse the following GEGLU into this (up) matmul's dequant: llama
        // emits [gate, up, GLU], so when up is dequantized gate (the GLU's
        // src0) is already in memory and up need never be written. Only if
        // nothing else can need up (single K-segment, not a graph output).
        struct ggml_tensor* fuse_glu = nullptr;
        static const bool fuse_enabled = []() {
            const char* env = std::getenv("RKNPU_FUSE_GLU");
            return env == nullptr || std::atoi(env) != 0;
        }();
        if (fuse_enabled && pipelined_node && node_i + 1 < cgraph->n_nodes) {
            struct ggml_tensor* nx = cgraph->nodes[node_i + 1];
            if (nx->op == GGML_OP_GLU && nx->src[1] == node && nx->src[0] != node && rknpu_glu_supported(nx) &&
                nx->ne[0] == N && ggml_nrows(nx) == M && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                fuse_glu = nx;
            }
        }
        // Defer this matmul's dequant into the next one's fused GEGLU when it
        // is the gate of [gate, up, GLU] and nothing in the graph but the GLU
        // reads it (see rknpu_deferred_gate)
        static const bool defer_enabled = []() {
            const char* env = std::getenv("RKNPU_DEFER_GATE");
            return env == nullptr || std::atoi(env) != 0;
        }();
        bool defer_gate = false;
        if (defer_enabled && fuse_enabled && pipelined_node && all_k_segments.size() == 1 &&
            node_i + 2 < cgraph->n_nodes && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            const struct ggml_tensor* nu = cgraph->nodes[node_i + 1];
            const struct ggml_tensor* ng = cgraph->nodes[node_i + 2];
            if (nu->op == GGML_OP_MUL_MAT && ng->op == GGML_OP_GLU && ng->src[0] == node && ng->src[1] == nu &&
                rknpu_glu_supported(ng) && nu->src[1]->ne[1] == M && nu->src[0]->ne[1] == N &&
                ng->ne[0] == N && ggml_nrows(ng) == M && !(nu->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                defer_gate = true;
                for (int j = node_i + 1; j < cgraph->n_nodes && defer_gate; ++j) {
                    const struct ggml_tensor* t = cgraph->nodes[j];
                    if (t == ng) continue;
                    if (t->view_src == node) defer_gate = false;
                    for (int q = 0; q < GGML_MAX_SRC; ++q) defer_gate = defer_gate && t->src[q] != node;
                }
            }
        }
        for (int64_t ib = 0; ib < nbatch && !pipelined_node; ++ib) {
            memset((char*)dst_data + ib * dst_batch_nb, 0, (size_t)M * N * sizeof(float));
        }

        // Acquiring the Hadamard vector
        std::vector<float> s_vec;
        if (is_hadamard) {
            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->hadamard_s_vectors.find(src0);
            GGML_ASSERT(it != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
            s_vec = it->second;
        }

        // Acquiring the B-matrix scale grid. By pointer: with per-channel
        // scales the grid is N-sized per k-segment and must not be copied
        // per node (map values are pointer-stable; entries are never erased).
        const std::vector<float>* scales_B_grid = nullptr;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->quantized_tensor_scales.find(src0);
            GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scales grid not found");
            scales_B_grid = &it->second;
        }
        const bool b_per_channel =
            (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4 ||
             pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) &&
            rknpu2_calibration::per_channel_b_scales();

        // Calculating tensor packed size
        size_t type_size_packed = 0;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
        else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;

        // Computing K dimensions segments
        size_t current_offset_in_tensor = 0;
        std::vector<size_t> seg_b_offset(num_active_segments, 0);
        int hn_last_node = -1;   // last node fused into a per-head-norm dequant (see below)
        const struct ggml_tensor* hn_kv_node = nullptr;   // the cache write it did ahead of its position
        for (size_t k_idx = 0; k_idx < all_k_segments.size(); ++k_idx) {
            const auto& k_seg = all_k_segments[k_idx];
            const int K_seg_op = k_seg.size_k;

            // ===========================================
            // ========== 1. Preparing Contexts ==========
            // ===========================================
            for (const auto& n_seg : all_n_segments) {
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    if (active_n_segments[idx].offset_n == n_seg.offset_n) {
                        size_t offset_in_dma = current_offset_in_tensor;
                        seg_b_offset[idx] = offset_in_dma;

                        // Getting matmul context from cache
                        matmul_ctxs[idx] = backend_ctx->get_matmul_ctx(
                            (uintptr_t)tensor_virt_addr, offset_in_dma, M_op, K_seg_op, n_seg.size_n,
                            n_seg.core_id, matmul_type, ac_layout, b_domain_id
                        );
                        if (!matmul_ctxs[idx] || matmul_ctxs[idx]->ctx == 0) return GGML_STATUS_FAILED;

                        auto& matmul_ctx = matmul_ctxs[idx];

                        // Assigning B-matrix only once to reduce computation overhead
                        if (!matmul_ctx->b_bound) {
                            size_t segment_size_bytes = matmul_ctx->io_attr.B.size;

                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(
                                matmul_ctx->ctx,
                                tensor_fd,
                                tensor_virt_addr,
                                segment_size_bytes,
                                offset_in_dma
                            );
                            if (!mem) return GGML_STATUS_FAILED;

                            auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                            matmul_ctx->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);

                            RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->mem_B.get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");

                            matmul_ctx->b_bound = true;
                        }
                        break;
                    }
                }

                if (n_seg.size_n > 0) {
                    current_offset_in_tensor += type_size_packed > 0 ? (size_t)n_seg.size_n * K_seg_op * type_size_packed : (size_t)n_seg.size_n * K_seg_op / 2;
                }
            }

            // Batched mul_mat: src0 (the weights) is always 2D, but src1
            // and dst may carry higher dimensions — LFM2's short-convolution
            // projections arrive as src1[2048,128,4]. Everything below
            // computes one 2D slice, so each slice is run in turn with the
            // activation and destination pointers advanced. Dense models
            // have nbatch == 1, where this loop runs once at zero offset and
            // is bit-identical to the unbatched path.
            for (int64_t ib = 0; ib < nbatch; ++ib) {
                const float * const src1_batch =
                    (const float *)((const char *)src1_base + ib * src1_batch_nb);
                float * const dst_batch =
                    (float *)((char *)dst_data + ib * dst_batch_nb);

                // ===========================================
                // ===== Pipelined prefill (W4A4, M > 256) =====
                // ===========================================
                // Rows are processed in 256-row chunks so the CPU work of one
                // chunk overlaps the NPU work of another: A-prep of chunk c+1
                // and dequant of chunk c-1 run on the OpenMP team while chunk
                // c runs on the async runner's threads. Rows are quantized
                // independently and B scales are per channel, so every output
                // element is computed exactly as on the path below. Costs one
                // extra B read per chunk (prefill NPU runs are compute-bound:
                // 1.55 vs 1.62 ms/token at M=512 vs 256). RKNPU_PIPELINE=0
                // disables it.
                const int MC = 256;
                static_assert(MC == 256, "pipelined_node uses M > 256");
                const bool fused_now = fuse_glu != nullptr && all_k_segments.size() == 1;
                // Per-head norm (+ weight) of this matmul's output, fused into
                // its dequant: [matmul, reshape, RMS_NORM, (MUL)] with nothing
                // else reading the matmul or the norm. The output is the norm
                // (or MUL) tensor; the raw projection is never written.
                struct ggml_tensor* hn_norm = nullptr;
                struct ggml_tensor* hn_out = nullptr;
                struct ggml_tensor* hn_rope = nullptr;   // RoPE fused after the norm, if any
                struct ggml_tensor* hn_kv = nullptr;     // KV-cache write fused at the end, if any
                bool hn_kv_only = false;                 // ... and nothing else reads the FP32 rows
                const float* hn_w = nullptr;
                float hn_eps = 0.0f;
                int hn_last = -1;
                if (rknpu_head_norm_enabled() && pipelined_node && !fuse_glu && !defer_gate && nbatch == 1 &&
                    all_k_segments.size() == 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    auto next_op = [&](int j) {
                        for (++j; j < cgraph->n_nodes; ++j) {
                            const enum ggml_op o = cgraph->nodes[j]->op;
                            if (o != GGML_OP_VIEW && o != GGML_OP_RESHAPE && o != GGML_OP_PERMUTE && o != GGML_OP_TRANSPOSE && o != GGML_OP_NONE) return j;
                        }
                        return -1;
                    };
                    const int jn = next_op(node_i);
                    struct ggml_tensor* nn = jn >= 0 ? cgraph->nodes[jn] : nullptr;
                    if (nn && rknpu_head_norm_supported(nn) && nn->src[0]->view_src == node && N % nn->ne[0] == 0 &&
                        ggml_nelements(nn) == (int64_t)M * N && !(nn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                        const int jm = next_op(jn);
                        struct ggml_tensor* mm = jm >= 0 ? cgraph->nodes[jm] : nullptr;
                        const struct ggml_tensor* w = (mm && (mm->src[0] == nn || mm->src[1] == nn)) ? rknpu_head_norm_weight(mm) : nullptr;
                        const struct ggml_tensor* view = nn->src[0];
                        bool ok = true;
                        for (int j = node_i + 1; j < cgraph->n_nodes && ok; ++j) {
                            const struct ggml_tensor* t = cgraph->nodes[j];
                            if (t != view && (t->view_src == node || t->view_src == view)) ok = false;
                            for (int q = 0; q < GGML_MAX_SRC; ++q) {
                                const struct ggml_tensor* sq = t->src[q];
                                if (sq == node && t != view) ok = false;
                                if (sq == view && t != nn) ok = false;
                                if (w && sq == nn && t != mm) ok = false;
                            }
                            if (w && t->view_src == nn) ok = false;
                        }
                        if (ok) {
                            hn_norm = nn;
                            hn_out = w ? mm : nn;
                            hn_w = w ? (const float*)get_tensor_real_ptr(w) : nullptr;
                            memcpy(&hn_eps, nn->op_params, sizeof(float));
                            hn_last = w ? jm : jn;
                            // and the RoPE of the normed heads, when it follows and
                            // is the only reader of the norm output
                            const int jr = next_op(hn_last);
                            struct ggml_tensor* rr = jr >= 0 ? cgraph->nodes[jr] : nullptr;
                            static const bool rope_fuse = []() {
                                const char* env = std::getenv("RKNPU_ROPE_FUSE");
                                return env == nullptr || std::atoi(env) != 0;
                            }();
                            if (rope_fuse && rknpu_rope_enabled() && rr && rr->src[0] == hn_out && rknpu_rope_supported(rr) &&
                                rr->ne[2] == M && !(hn_out->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                                bool sole = true;
                                for (int j = node_i + 1; j < cgraph->n_nodes && sole; ++j) {
                                    const struct ggml_tensor* t = cgraph->nodes[j];
                                    if (t->view_src == hn_out) sole = false;
                                    for (int q = 0; q < GGML_MAX_SRC; ++q) sole = sole && (t->src[q] != hn_out || t == rr);
                                }
                                if (sole) {
                                    hn_rope = rr;
                                    hn_out = rr;
                                    hn_last = jr;
                                }
                            }
                            hn_last_node = hn_last;
                            // and the KV-cache write of the result, found further on
                            // (K's comes after V's projection): done here if no
                            // computing node in between touches that cache
                            if (rknpu_kv_write_enabled()) {
                                for (int j = hn_last + 1; j < cgraph->n_nodes && j < hn_last + 64; ++j) {
                                    struct ggml_tensor* t = cgraph->nodes[j];
                                    if (t->op == GGML_OP_SET_ROWS && t->src[0] && t->src[0]->view_src == hn_out &&
                                        rknpu_set_rows_supported(t) && t->src[0]->ne[0] == N && t->src[0]->ne[1] == M &&
                                        t->src[0]->nb[1] == (size_t)N * sizeof(float)) {
                                        const struct ggml_tensor* cache = t->view_src;
                                        bool clear = cache != nullptr;
                                        for (int q2 = hn_last + 1; q2 < j && clear; ++q2) {
                                            const struct ggml_tensor* u = cgraph->nodes[q2];
                                            if (u->op == GGML_OP_VIEW || u->op == GGML_OP_RESHAPE || u->op == GGML_OP_PERMUTE || u->op == GGML_OP_TRANSPOSE || u->op == GGML_OP_NONE) continue;
                                            if (u->view_src == cache) clear = false;
                                            for (int q = 0; q < GGML_MAX_SRC; ++q) {
                                                const struct ggml_tensor* sq = u->src[q];
                                                clear = clear && !(sq && (sq == cache || sq->view_src == cache));
                                            }
                                        }
                                        if (clear) {
                                            hn_kv = t;
                                            hn_kv_node = t;
                                            // the FP32 rows need not be written if only the cache write reads them
                                            bool only = true;
                                            for (int q2 = node_i + 1; q2 < cgraph->n_nodes && only; ++q2) {
                                                const struct ggml_tensor* u = cgraph->nodes[q2];
                                                if (u == t || u == t->src[0]) continue;
                                                if (u->view_src == hn_out) only = false;
                                                for (int q = 0; q < GGML_MAX_SRC; ++q) only = only && u->src[q] != hn_out && u->src[q] != t->src[0];
                                            }
                                            hn_kv_only = only && !(hn_out->flags & GGML_TENSOR_FLAG_OUTPUT);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                auto& dg = backend_ctx->deferred_gate;
                const bool use_dg = dg.gate != nullptr && pipelined_node && fused_now && fuse_glu->src[0] == dg.gate &&
                                    dg.up == node && dg.M == M && dg.N == N && dg.MC == MC;
                if (dg.gate && !use_dg) rknpu_materialize_deferred_gate(dg, n_omp);
                // [.., up, GLU, down] with the GLU read by nothing else: run
                // down as a rknpu_w4a4_job prepared from this collect
                bool stage_down = false;
                {
                    static const bool prep_down_enabled = []() {
                        const char* env = std::getenv("RKNPU_DOWN_JOB");
                        return env == nullptr || std::atoi(env) != 0;
                    }();
                    const struct ggml_tensor* down = use_dg && node_i + 2 < cgraph->n_nodes ? cgraph->nodes[node_i + 2] : nullptr;
                    if (prep_down_enabled && down && down->op == GGML_OP_MUL_MAT && down->src[1] == fuse_glu &&
                        !(fuse_glu->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_is_contiguous(fuse_glu) && ggml_is_contiguous(down) &&
                        down->src[1]->ne[2] * down->src[1]->ne[3] == 1 && (int)down->src[0]->ne[0] == N && (int)down->ne[1] == M) {
                        stage_down = true;
                        for (int j = node_i + 3; j < cgraph->n_nodes && stage_down; ++j) {
                            const struct ggml_tensor* t = cgraph->nodes[j];
                            if (t->view_src == fuse_glu) stage_down = false;
                            for (int q = 0; q < GGML_MAX_SRC; ++q) stage_down = stage_down && t->src[q] != fuse_glu;
                        }
                        stage_down = stage_down && rknpu_w4a4_job_setup(backend_ctx, backend_ctx->down_job, down, M, MC);
                    }
                }
                if (pipelined_node) {
                    const int n_chunks = (M + MC - 1) / MC;
                    // chunk contexts (M_op = MC), B bound once per context
                    std::vector<std::shared_ptr<rknpu_matmul_context>> cctx(num_active_segments);
                    for (size_t idx = 0; idx < num_active_segments; ++idx) {
                        cctx[idx] = backend_ctx->get_matmul_ctx(
                            (uintptr_t)tensor_virt_addr, seg_b_offset[idx], MC, K_seg_op,
                            active_n_segments[idx].size_n, active_n_segments[idx].core_id,
                            matmul_type, pipeline->ac_layout, b_domain_id);
                        if (!cctx[idx] || cctx[idx]->ctx == 0) return GGML_STATUS_FAILED;
                        auto& mc = cctx[idx];
                        if (!mc->b_bound) {
                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(mc->ctx, tensor_fd, tensor_virt_addr,
                                                                          mc->io_attr.B.size, seg_b_offset[idx]);
                            if (!mem) return GGML_STATUS_FAILED;
                            auto deleter = [ctx = mc->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                            mc->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);
                            RKNN_CHECK(rknn_matmul_set_io_mem(mc->ctx, mc->mem_B.get(), &mc->io_attr.B), "set_io_mem B chunk");
                            mc->b_bound = true;
                        }
                    }
                    // double-buffered A and C (slot encoded in the cache key's type field)
                    // (a deferred gate keeps one C per chunk, keyed apart from
                    // the up matmul's slots)
                    std::shared_ptr<rknn_tensor_mem> a_slot[2];
                    const int n_cslots = defer_gate ? n_chunks : 2;
                    std::vector<std::vector<std::shared_ptr<rknn_tensor_mem>>> c_slot(n_cslots);
                    for (int sl = 0; sl < 2; ++sl) {
                        a_slot[sl] = get_tensor_buffer(backend_ctx, cctx[0]->ctx, cctx[0]->io_attr.A.size,
                            std::make_tuple(MC, K_seg_op, (int)pipeline->npu_type_a + 16 * (sl + 1), b_domain_id),
                            backend_ctx->a_buffer_cache);
                        if (!a_slot[sl]) return GGML_STATUS_FAILED;
                    }
                    for (int sl = 0; sl < n_cslots; ++sl) {
                        c_slot[sl].resize(num_active_segments);
                        for (size_t idx = 0; idx < num_active_segments; ++idx) {
                            c_slot[sl][idx] = get_tensor_buffer(backend_ctx, cctx[idx]->ctx, cctx[idx]->io_attr.C.size,
                                std::make_tuple(MC, active_n_segments[idx].size_n, active_n_segments[idx].core_id,
                                                (int)pipeline->npu_type_c + 16 * (sl + 1) + (defer_gate ? 4096 : 0), b_domain_id),
                                backend_ctx->c_buffer_cache);
                            if (!c_slot[sl][idx]) return GGML_STATUS_FAILED;
                        }
                    }
                    auto cslot = [&](int c) -> std::vector<std::shared_ptr<rknn_tensor_mem>>& { return c_slot[defer_gate ? c : (c & 1)]; };
                    rknpu2_native_geom a_geom = {0, 0, 0};
                    GGML_ASSERT(rknpu2_native_geom_from_dims(cctx[0]->io_attr.A.dims, cctx[0]->io_attr.A.n_dims, &a_geom) == 0);
                    std::vector<rknpu2_native_geom> c_geom(num_active_segments);
                    for (size_t idx = 0; idx < num_active_segments; ++idx) {
                        GGML_ASSERT(rknpu2_native_geom_from_dims(cctx[idx]->io_attr.C.dims, cctx[idx]->io_attr.C.n_dims, &c_geom[idx]) == 0);
                    }
                    const float a_clip = rknpu2_calibration::a_clip_factor();
                    const float hadamard_divisor = is_hadamard ? (float)rknpu2_calibration::hadamard_block_len(K) : 1.0f;
                    const int row_stride = (int)(src1->nb[1] / sizeof(float));
                    std::vector<float> scales_A(M, 1.0f);
                    const int h_block = is_hadamard ? rknpu2_calibration::hadamard_block_len(K) : 1;
                    const bool seg_on_blocks = is_hadamard && all_k_segments.size() > 1 && K_op == K &&
                                               k_seg.offset_k % h_block == 0 && K_seg_op % h_block == 0;

                    auto prep = [&](int c) {
                        const int m0 = c * MC, rows = std::min(MC, M - m0);
                        uint8_t* dst_a = (uint8_t*)a_slot[c & 1]->virt_addr;
                        #pragma omp parallel for num_threads(n_omp)
                        for (int r = 0; r < rows; ++r) {
                            const int m = m0 + r;
                            const float* src_row = src1_batch + (size_t)m * row_stride;
                            static thread_local std::vector<float> full_row;
                            static thread_local std::vector<uint8_t> packed_row;
                            auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
                            const float* ready_row;
                            if (is_hadamard && seg_on_blocks) {
                                // K-segmented weight: transform only this segment's blocks
                                grow(full_row, (size_t)K_seg_op);
                                rknpu2_calibration::hadamard_transform_signed_range(full_row.data(), src_row, s_vec.data(), K, k_seg.offset_k, K_seg_op);
                                ready_row = full_row.data();
                            } else if (is_hadamard) {
                                grow(full_row, (size_t)K_op);
                                rknpu2_calibration::hadamard_transform_signed(full_row.data(), src_row, s_vec.data(), K, K_op);
                                ready_row = full_row.data() + k_seg.offset_k;
                            } else {
                                ready_row = src_row + k_seg.offset_k;
                            }
                            scales_A[m] = a_clip * rknpu2_quantization::amax_fp32(ready_row, K_seg_op) / 7.0f;
                            grow(packed_row, (size_t)K_seg_op / 2);
                            rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row, packed_row.data(), K_seg_op, scales_A[m]);
                            rknpu2_native_scatter_row(dst_a, packed_row.data(), r, a_geom.m_stride, a_geom.outer, a_geom.sub / 2);
                        }
                        RKNN_CHECK(rknn_mem_sync(cctx[0]->ctx, a_slot[c & 1].get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A chunk");
                    };
                    auto start = [&](int c) {
                        for (size_t idx = 0; idx < num_active_segments; ++idx) {
                            auto& mc = cctx[idx];
                            if (mc->bound_A != a_slot[c & 1].get()) {
                                RKNN_CHECK(rknn_matmul_set_io_mem(mc->ctx, a_slot[c & 1].get(), &mc->io_attr.A), "set_io_mem A chunk");
                                mc->bound_A = a_slot[c & 1].get();
                            }
                            if (mc->bound_C != cslot(c)[idx].get()) {
                                RKNN_CHECK(rknn_matmul_set_io_mem(mc->ctx, cslot(c)[idx].get(), &mc->io_attr.C), "set_io_mem C chunk");
                                mc->bound_C = cslot(c)[idx].get();
                            }
                        }
                        backend_ctx->async_runner.start(cctx);
                    };
                    auto collect = [&](int c) {
                        const int m0 = c * MC, rows = std::min(MC, M - m0);
                        for (size_t idx = 0; idx < num_active_segments; ++idx) {
                            RKNN_CHECK(rknn_mem_sync(cctx[idx]->ctx, cslot(c)[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C chunk");
                        }
                        const int n_blocks = (rows + 3) / 4;
                        #pragma omp parallel for num_threads(n_omp)
                        for (int blk = 0; blk < n_blocks; ++blk) {
                            const int r0 = blk * 4;
                            const int nr = std::min(4, rows - r0);
                            float common[4];
                            for (int r = 0; r < nr; ++r) common[r] = scales_A[m0 + r0 + r] / hadamard_divisor;
                            for (size_t idx = 0; idx < num_active_segments; ++idx) {
                                const int N_offset = active_n_segments[idx].offset_n;
                                rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(
                                    dst_batch + (size_t)(m0 + r0) * N + N_offset, (size_t)N,
                                    (const int16_t*)cslot(c)[idx]->virt_addr, r0, nr,
                                    c_geom[idx].m_stride, c_geom[idx].outer, c_geom[idx].sub,
                                    active_n_segments[idx].size_n, common,
                                    scales_B_grid->data() + k_idx * (size_t)N + N_offset,
                                    /*store=*/ k_idx == 0);   // dst not zeroed (pipelined_node)
                            }
                        }
                    };
                    // GEGLU-fused variant of collect for the up matmul (a
                    // separate lambda: a branch inside collect's OpenMP region
                    // cost ~10% even when not taken)
                    auto collect_fused = [&](int c) {
                        const int m0 = c * MC, rows = std::min(MC, M - m0);
                        for (size_t idx = 0; idx < num_active_segments; ++idx) {
                            RKNN_CHECK(rknn_mem_sync(cctx[idx]->ctx, cslot(c)[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C chunk");
                        }
                        const int n_blocks = (rows + 3) / 4;
                        #pragma omp parallel for num_threads(n_omp)
                        for (int blk = 0; blk < n_blocks; ++blk) {
                            const int r0 = blk * 4;
                            const int nr = std::min(4, rows - r0);
                            float common[4];
                            for (int r = 0; r < nr; ++r) common[r] = scales_A[m0 + r0 + r] / hadamard_divisor;
                            rknpu_fused_up_geglu_block(fuse_glu, m0 + r0, r0, nr, N, common,
                                cslot(c), c_geom, active_n_segments,
                                scales_B_grid->data() + k_idx * (size_t)N);
                        }
                    };


                    auto collect_fused_dg = [&](int c) {
                        const int m0 = c * MC, rows = std::min(MC, M - m0);
                        for (size_t idx = 0; idx < num_active_segments; ++idx) {
                            RKNN_CHECK(rknn_mem_sync(cctx[idx]->ctx, cslot(c)[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C chunk");
                        }
                        const int n_blocks = (rows + 3) / 4;
                        #pragma omp parallel for num_threads(n_omp)
                        for (int blk = 0; blk < n_blocks; ++blk) {
                            const int r0 = blk * 4;
                            const int nr = std::min(4, rows - r0);
                            float common[4];
                            for (int r = 0; r < nr; ++r) common[r] = scales_A[m0 + r0 + r] / hadamard_divisor;
                            if (stage_down) {
                                rknpu_fused_gate_up_geglu_prep_block(backend_ctx->down_job, dg, c, m0 + r0, r0, nr, N, common,
                                    cslot(c), c_geom, active_n_segments, scales_B_grid->data() + k_idx * (size_t)N);
                            } else {
                                rknpu_fused_gate_up_geglu_block(fuse_glu, dg, c, m0 + r0, r0, nr, N, common,
                                    cslot(c), c_geom, active_n_segments, scales_B_grid->data() + k_idx * (size_t)N);
                            }
                        }
                    };
                    // deferred gate: C synced for the CPU, kept for the up's collect
                    auto keep = [&](int c) {
                        for (size_t idx = 0; idx < num_active_segments; ++idx) {
                            RKNN_CHECK(rknn_mem_sync(cctx[idx]->ctx, cslot(c)[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C chunk");
                        }
                    };
                    // per-head norm fused into the dequant (a separate lambda, as above)
                    auto collect_norm = [&](int c) {
                        const int m0 = c * MC, rows = std::min(MC, M - m0);
                        for (size_t idx = 0; idx < num_active_segments; ++idx) {
                            RKNN_CHECK(rknn_mem_sync(cctx[idx]->ctx, cslot(c)[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C chunk");
                        }
                        float* out = (float*)get_tensor_real_ptr(hn_out);
                        const int64_t hd = hn_norm->ne[0];
                        rknpu_rope_params rp;
                        const int32_t* rpos = nullptr;
                        const float* rff = nullptr;
                        char* kv_cache = hn_kv ? (char*)get_tensor_real_ptr(hn_kv) : nullptr;
                        const void* kv_idx = hn_kv ? get_tensor_real_ptr(hn_kv->src[1]) : nullptr;
                        if (hn_rope) {
                            rp = rknpu_rope_get(hn_rope);
                            rpos = (const int32_t*)get_tensor_real_ptr(hn_rope->src[1]);
                            rff = hn_rope->src[2] ? (const float*)get_tensor_real_ptr(hn_rope->src[2]) : nullptr;
                        }
                        const int n_blocks = (rows + 3) / 4;
                        #pragma omp parallel for num_threads(n_omp)
                        for (int blk = 0; blk < n_blocks; ++blk) {
                            const int r0 = blk * 4;
                            const int nr = std::min(4, rows - r0);
                            static thread_local std::vector<float> rbuf;
                            if (rbuf.size() < (size_t)4 * N) rbuf.resize((size_t)4 * N);
                            float common[4];
                            for (int r = 0; r < nr; ++r) common[r] = scales_A[m0 + r0 + r] / hadamard_divisor;
                            for (size_t idx = 0; idx < num_active_segments; ++idx) {
                                const int N_offset = active_n_segments[idx].offset_n;
                                rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(
                                    rbuf.data() + N_offset, (size_t)N,
                                    (const int16_t*)cslot(c)[idx]->virt_addr, r0, nr,
                                    c_geom[idx].m_stride, c_geom[idx].outer, c_geom[idx].sub,
                                    active_n_segments[idx].size_n, common,
                                    scales_B_grid->data() + N_offset, /*store=*/ true);
                            }
                            static thread_local std::vector<float> rcache;
                            if (hn_rope && (int64_t)rcache.size() < hd) rcache.resize(hd);
                            for (int r = 0; r < nr; ++r) {
                                float* x = rbuf.data() + (size_t)r * N;
                                float* y = hn_kv_only ? x : out + (size_t)(m0 + r0 + r) * N;   // in place when only the cache needs it
                                if (hn_rope) rknpu_rope_cache(rcache.data(), rpos[m0 + r0 + r], rp, rff, hd);
                                for (int64_t h = 0; h < N; h += hd) {
                                    rknpu_head_norm(y + h, x + h, hd, hn_eps, hn_w);
                                    if (hn_rope) rknpu_rope_head(y + h, y + h, rcache.data(), hd, rp);
                                }
                                if (hn_kv) rknpu_kv_store_row(hn_kv, kv_cache, kv_idx, m0 + r0 + r, y);
                            }
                        }
                    };
                    auto collect_any = [&](int c) {
                        if (defer_gate) keep(c);
                        else if (use_dg) collect_fused_dg(c);
                        else if (fused_now) collect_fused(c);
                        else if (hn_out) collect_norm(c);
                        else collect(c);
                    };

                    const auto t_run = g_rknpu_profile.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    prep(0);
                    start(0);
                    for (int c = 1; c < n_chunks; ++c) {
                        prep(c);                         // overlaps NPU chunk c-1
                        backend_ctx->async_runner.wait();
                        start(c);
                        collect_any(c - 1);   // overlaps NPU chunk c
                    }
                    backend_ctx->async_runner.wait();
                    // down's first chunk (its rows were staged by collect(0))
                    // overlaps this node's last collect
                    if (stage_down && n_chunks > 1) rknpu_w4a4_job_start(backend_ctx, backend_ctx->down_job, 0);
                    collect_any(n_chunks - 1);
                    if (use_dg) {
                        dg.gate = nullptr;
                        dg.up = nullptr;
                    }
                    if (defer_gate) {
                        dg.gate = node;
                        dg.up = cgraph->nodes[node_i + 1];
                        dg.M = M; dg.N = N; dg.MC = MC;
                        dg.c = c_slot;
                        dg.c_geom = c_geom;
                        dg.segs = active_n_segments;
                        dg.common.resize(M);
                        for (int m = 0; m < M; ++m) dg.common[m] = scales_A[m] / hadamard_divisor;
                        dg.chan = scales_B_grid->data() + k_idx * (size_t)N;
                    }
                    if (g_rknpu_profile.on) g_rknpu_profile.run_ns += rknpu_profile::ns(t_run, std::chrono::steady_clock::now());
                    continue;
                }

                // ===========================================
                // ========== 2. Preparing A-matrix ==========
                // ===========================================
                std::vector<float> scales_A(M, 1.0f);
                {
                    auto cache_key = std::make_tuple(M_op, K_seg_op, (int)pipeline->npu_type_a, b_domain_id);
                    auto& matmul_ctx_0 = matmul_ctxs[0];

                    // Getting A-buffer from cache
                    mem_A_shared = get_tensor_buffer(backend_ctx, matmul_ctx_0->ctx, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
                    if (!mem_A_shared) return GGML_STATUS_FAILED;

                    const float* x = src1_batch;
                    const int row_stride = (int)(src1->nb[1] / sizeof(float));
                    void* dst_base = mem_A_shared->virt_addr;

                    // Native A layout: produce rows directly in the NPU tiling
                    // ([K/sub, M, sub] cells) instead of row-major, skipping the
                    // runtime's serial per-run repack
                    rknpu2_native_geom a_geom = {0, 0, 0};
                    const bool a_native = ac_layout == RKNN_MM_LAYOUT_NATIVE &&
                        rknpu2_native_geom_from_dims(matmul_ctx_0->io_attr.A.dims,
                                                     matmul_ctx_0->io_attr.A.n_dims, &a_geom) == 0;
                    // Not a fallback: the context was created with
                    // AC_layout = NATIVE, so if the geometry does not parse we
                    // would fill a row-major buffer the NPU reads as
                    // [K/sub, M, sub] tiles — silently wrong output, no error.
                    GGML_ASSERT((ac_layout != RKNN_MM_LAYOUT_NATIVE || a_native) &&
                                "RKNPU2: native A layout requested but io_attr.A geometry did not parse");

                    // hoisted: the getter guards a function-local static, and
                    // the loop below runs per row per node
                    const float a_clip = rknpu2_calibration::a_clip_factor();

                    // Decode (M == 1) on the INT4 native path: split the one
                    // row across ggml's (hot, size-matched) OpenMP team —
                    // Hadamard blocks per thread, then amax and quantization
                    // in 64-element chunks. Element-exact vs the per-row loop
                    // below: blocks are independent, max is order-free, and
                    // chunk edges sit on the kernels' vector boundaries
                    // (test_prep_split in test-rknpu2-prep-kernels.cpp).
                    static const bool prep_serial = std::getenv("RKNPU_DISPATCH_POOL") != nullptr;
                    const bool prep_on_team = M == 1 && n_omp > 1 && n_omp <= 64 && !prep_serial &&
                                              pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4 && a_native;
                    if (prep_on_team) {
                        const float* src_row = x;
                        std::vector<float> signed_row(is_hadamard ? K : 0);
                        std::vector<float> full_row(is_hadamard ? K_op : 0);
                        std::vector<uint8_t> packed_row(K_seg_op / 2);
                        float amax_part[64];
                        float scale = 0.0f;
                        const int block = is_hadamard ? rknpu2_calibration::hadamard_block_len(K) : 1;
                        const int n_blocks = is_hadamard ? K_op / block : 0;
                        const int CH = 64;
                        const int n_chunks = (K_seg_op + CH - 1) / CH;
                        #pragma omp parallel num_threads(n_omp)
                        {
                            const int t = omp_get_thread_num();
                            const int T = omp_get_num_threads();
                            if (is_hadamard) {
                                for (int b = t; b < n_blocks; b += T) {
                                    const int off = b * block;
                                    const int lim = std::min(off + block, K);
                                    if (lim > off) {
                                        rknpu2_quantization::mul_fp32(signed_row.data() + off, src_row + off, s_vec.data() + off, lim - off);
                                    }
                                    rknpu2_calibration::hadamard_transform_block(full_row.data(), signed_row.data(), K, K_op, b);
                                }
                                #pragma omp barrier
                            }
                            const float* ready = (is_hadamard ? full_row.data() : src_row) + k_seg.offset_k;
                            const int e0 = (n_chunks * t / T) * CH;
                            const int e1 = std::min((n_chunks * (t + 1) / T) * CH, K_seg_op);
                            amax_part[t] = e1 > e0 ? rknpu2_quantization::amax_fp32(ready + e0, e1 - e0) : 0.0f;
                            #pragma omp barrier
                            #pragma omp single
                            {
                                float mx = 0.0f;
                                for (int i = 0; i < T; ++i) mx = std::max(mx, amax_part[i]);
                                // clip < 1 saturates the far tail for finer steps
                                // on the mass (RKNPU_A_CLIP, decode research #3d)
                                scale = a_clip * mx / 7.0f;
                            }
                            if (e1 > e0) {
                                rknpu2_quantization::quantize_fp32_to_int4_packed(ready + e0, packed_row.data() + e0 / 2, e1 - e0, scale);
                            }
                        }
                        scales_A[0] = scale;
                        rknpu2_native_scatter_row((uint8_t*)dst_base, packed_row.data(), 0,
                                                  a_geom.m_stride, a_geom.outer, a_geom.sub / 2);
                    }

                    // if(M > 1): at decode a single row costs a few us of NEON;
                    // forming a team per node is what libgomp punishes (#3)
                    #pragma omp parallel for if(M > 1) num_threads(n_omp)
                    for (int m = 0; m < (prep_on_team ? 0 : M); ++m) {
                        const float* src_row = x + (size_t)m * row_stride;
                        // Per-thread scratch that only grows: a fresh vector
                        // per row per node was a heap allocation plus a zero
                        // fill (std::vector value-initializes) that the code
                        // below overwrites anyway — ~0.6 M of them per 512
                        // token prefill. Every buffer is fully written before
                        // it is read, so contents carried between rows are
                        // never observed.
                        static thread_local std::vector<float> full_hadamard_row;
                        static thread_local std::vector<uint8_t> packed_row;
                        auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };

                        // Applying Hadamard Transform (the A row is read in
                        // place from the transform output or the source row)
                        const float* ready_row;
                        if (is_hadamard) {
                            grow(full_hadamard_row, (size_t)K_op);
                            rknpu2_calibration::hadamard_transform_signed(full_hadamard_row.data(), src_row, s_vec.data(), K, K_op);
                            ready_row = full_hadamard_row.data() + k_seg.offset_k;
                        } else {
                            ready_row = src_row + k_seg.offset_k;
                        }

                        // Handling types and quantizations
                        if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                            uint16_t* dst_ptr = (uint16_t*)dst_base;
                            uint16_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                            rknpu2_quantization::convert_fp32_to_fp16(ready_row, dst_row, K_seg_op);
                        }
                        else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                            scales_A[m] = rknpu2_quantization::amax_fp32(ready_row, K_seg_op) / 127.0f;

                            int8_t* dst_ptr = (int8_t*)dst_base;
                            if (a_native) {
                                grow(packed_row, (size_t)K_seg_op);
                                rknpu2_quantization::quantize_fp32_to_int8(ready_row, (int8_t*)packed_row.data(), K_seg_op, scales_A[m]);
                                rknpu2_native_scatter_row((uint8_t*)dst_ptr, packed_row.data(), m,
                                                          a_geom.m_stride, a_geom.outer, a_geom.sub);
                            } else {
                                int8_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                                rknpu2_quantization::quantize_fp32_to_int8(ready_row, dst_row, K_seg_op, scales_A[m]);
                            }
                        }
                        else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                            // clip < 1 saturates the far tail for finer steps
                            // on the mass (RKNPU_A_CLIP, decode research #3d)
                            scales_A[m] = a_clip *
                                          rknpu2_quantization::amax_fp32(ready_row, K_seg_op) / 7.0f;

                            uint8_t* dst_ptr = (uint8_t*)dst_base;
                            if (a_native) {
                                grow(packed_row, (size_t)K_seg_op / 2);
                                rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row, packed_row.data(), K_seg_op, scales_A[m]);
                                rknpu2_native_scatter_row(dst_ptr, packed_row.data(), m,
                                                          a_geom.m_stride, a_geom.outer, a_geom.sub / 2);
                            } else {
                                uint8_t* dst_row = dst_ptr + (size_t)m * (K_seg_op / 2);
                                rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row, dst_row, K_seg_op, scales_A[m]);
                            }
                        }
                    }

                    // Assigning A-matrix to all contexts for the parallel execution
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        if (matmul_ctxs[idx]->bound_A != mem_A_shared.get()) {
                            RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[idx]->ctx, mem_A_shared.get(), &matmul_ctxs[idx]->io_attr.A), "set_io_mem A for core");
                            matmul_ctxs[idx]->bound_A = mem_A_shared.get();
                        }
                    }

                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[0]->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");
                }

                // ===========================================
                // ========== 3. Preparing C-matrix ==========
                // ===========================================
                {
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        auto& matmul_ctx = matmul_ctxs[idx];
                        auto cache_key = std::make_tuple(M_op, active_n_segments[idx].size_n, active_n_segments[idx].core_id, (int)pipeline->npu_type_c, b_domain_id);

                        // Getting C-buffer from cache
                        mem_C_segments[idx] = get_tensor_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                        if (!mem_C_segments[idx]) return GGML_STATUS_FAILED;

                        // Assigning C-matrix to current context for the parallel execution
                        if (matmul_ctx->bound_C != mem_C_segments[idx].get()) {
                            RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[idx].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
                            matmul_ctx->bound_C = mem_C_segments[idx].get();
                        }
                    }
                }

                // ==========================================
                // ========== 4. Running operation ==========
                // ==========================================

                // H*H^T = block_len * I per FWHT block (legacy: block = K_op)
                const float hadamard_divisor = pipeline->use_hadamard ? (float)rknpu2_calibration::hadamard_block_len(K) : 1.0f;

                // Native C layout: each segment's C comes back as
                // [N_seg/sub, M, sub] cells and is untiled inside the
                // dequantization pass below (address arithmetic only)
                std::vector<rknpu2_native_geom> c_geoms(num_active_segments);
                std::vector<uint8_t> c_native(num_active_segments, 0);
                if (ac_layout == RKNN_MM_LAYOUT_NATIVE) {
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        c_native[idx] = rknpu2_native_geom_from_dims(
                            matmul_ctxs[idx]->io_attr.C.dims,
                            matmul_ctxs[idx]->io_attr.C.n_dims, &c_geoms[idx]) == 0;
                        // As for A above: reading a NATIVE-layout C as
                        // row-major is silent corruption, not a fallback.
                        GGML_ASSERT(c_native[idx] &&
                            "RKNPU2: native C layout requested but io_attr.C geometry did not parse");
                    }
                }

                // Dequantizes segment idx of row m into dst. Segments own
                // disjoint dst columns, so segments can be done in any order
                // or concurrently without changing any result.
                auto dequant_seg = [&](int m, size_t idx) {
                    const int N_offset = active_n_segments[idx].offset_n;
                    const int N_segment = active_n_segments[idx].size_n;
                    float* dst_ptr = dst_batch + (size_t)m * N + N_offset;
                    switch (pipeline->npu_type_c) {
                        case rknpu2_configuration::NPU_TYPE_FP32: {
                            float scale_B = scales_B_grid == nullptr ? 1.0f : (*scales_B_grid)[k_idx * num_active_segments + idx];
                            float dequant_scale = (scales_A[m] * scale_B) / hadamard_divisor;
                            float* src_ptr = (float*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                            for (int n = 0; n < N_segment; ++n) {
                                dst_ptr[n] += src_ptr[n] * dequant_scale;
                            }
                            break;
                        }
                        case rknpu2_configuration::NPU_TYPE_INT32: {
                            int32_t* src_ptr = (int32_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                            if (b_per_channel && c_native[idx]) {
                                rknpu2_quantization::dequant_acc_int32_tiled_perchan(
                                    dst_ptr, (const int32_t*)mem_C_segments[idx]->virt_addr,
                                    m, c_geoms[idx].m_stride, c_geoms[idx].outer, c_geoms[idx].sub,
                                    N_segment, scales_A[m] / hadamard_divisor,
                                    scales_B_grid->data() + k_idx * (size_t)N + N_offset);
                                break;
                            }
                            if (b_per_channel) {
                                // grid layout [k_idx * N + global_n]
                                rknpu2_quantization::dequant_acc_int32_to_fp32_perchan(
                                    dst_ptr, src_ptr, N_segment,
                                    scales_A[m] / hadamard_divisor,
                                    scales_B_grid->data() + k_idx * (size_t)N + N_offset);
                                break;
                            }
                            float scale_B = scales_B_grid == nullptr ? 1.0f : (*scales_B_grid)[k_idx * num_active_segments + idx];
                            float dequant_scale = (scales_A[m] * scale_B) / hadamard_divisor;
                            for (int n = 0; n < N_segment; ++n) {
                                dst_ptr[n] += (float)src_ptr[n] * dequant_scale;
                            }
                            break;
                        }
                        case rknpu2_configuration::NPU_TYPE_INT16: {
                            if (b_per_channel) {
                                // grid layout [k_idx * N + global_n]
                                const float common = scales_A[m] / hadamard_divisor;
                                const float* chan = scales_B_grid->data() + k_idx * (size_t)N + N_offset;
                                if (c_native[idx]) {
                                    rknpu2_quantization::dequant_acc_int16_tiled_perchan(
                                        dst_ptr, (const int16_t*)mem_C_segments[idx]->virt_addr,
                                        m, c_geoms[idx].m_stride, c_geoms[idx].outer, c_geoms[idx].sub,
                                        N_segment, common, chan);
                                } else {
                                    const int16_t* src_ptr = (const int16_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                                    rknpu2_quantization::dequant_acc_int16_to_fp32_perchan(dst_ptr, src_ptr, N_segment, common, chan);
                                }
                                break;
                            }
                            float scale_B = scales_B_grid == nullptr ? 1.0f : (*scales_B_grid)[k_idx * num_active_segments + idx];
                            float dequant_scale = (scales_A[m] * scale_B) / hadamard_divisor;
                            if (c_native[idx]) {
                                rknpu2_quantization::dequant_acc_int16_tiled(
                                    dst_ptr, (const int16_t*)mem_C_segments[idx]->virt_addr,
                                    m, c_geoms[idx].m_stride, c_geoms[idx].outer, c_geoms[idx].sub,
                                    N_segment, dequant_scale);
                            } else {
                                const int16_t* src_ptr = (const int16_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;
                                rknpu2_quantization::dequant_acc_int16_to_fp32(dst_ptr, src_ptr, N_segment, dequant_scale);
                            }
                            break;
                        }
                        default:
                            // This should not be reached if config is correct
                            break;
                    }
                };

                const auto t_run = g_rknpu_profile.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                // On ggml's own OpenMP team when it is wide enough: with the
                // team size matched to ggml's (set_n_threads), libgomp keeps
                // the team hot, so the segments start on workers that are
                // already spinning instead of waking dispatch-pool threads
                // that compete with those spinners for the same cores.
                static const bool use_pool = std::getenv("RKNPU_DISPATCH_POOL") != nullptr;
                const int n_seg = (int)matmul_ctxs.size();
                const bool on_team = !use_pool && n_seg > 1 && n_omp >= n_seg;
                // At decode (M == 1) each thread also collects its own segment
                // right after its run returns, overlapping the dequant with
                // the other segments' NPU time
                const bool collect_on_team = on_team && M == 1;
                if (collect_on_team) {
                    #pragma omp parallel for num_threads(n_omp) schedule(static, 1)
                    for (int i = 0; i < n_seg; ++i) {
                        matmul_ctxs[i]->run();
                        RKNN_CHECK(rknn_mem_sync(matmul_ctxs[i]->ctx, mem_C_segments[i].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
                        dequant_seg(0, (size_t)i);
                    }
                } else if (on_team) {
                    #pragma omp parallel for num_threads(n_omp) schedule(static, 1)
                    for (int i = 0; i < n_seg; ++i) {
                        matmul_ctxs[i]->run();
                    }
                } else {
                    backend_ctx->dispatch_pool.run_all(matmul_ctxs);
                }
                if (g_rknpu_profile.on) g_rknpu_profile.run_ns += rknpu_profile::ns(t_run, std::chrono::steady_clock::now());

                // ===========================================
                // ========== 5. Collecting results ==========
                // ===========================================
                if (!collect_on_team) {
                    for (size_t idx = 0; idx < num_active_segments; idx++) {
                        RKNN_CHECK(rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_C_segments[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
                    }

                    // Native INT16 C with per-channel scales (W4A4 prefill):
                    // four rows per tile, so the tiled C is read sequentially
                    // (dequant_acc_int16_tiled_perchan_rows; element-exact vs
                    // dequant_seg, test_dequant_tiled_rows)
                    bool rows_path = M > 1 && pipeline->npu_type_c == rknpu2_configuration::NPU_TYPE_INT16 && b_per_channel;
                    for (size_t idx = 0; idx < num_active_segments && rows_path; idx++) {
                        rows_path = c_native[idx] != 0;
                    }
                    if (rows_path) {
                        const int n_blocks = (M + 3) / 4;
                        #pragma omp parallel for num_threads(n_omp)
                        for (int blk = 0; blk < n_blocks; ++blk) {
                            const int m0 = blk * 4;
                            const int nr = std::min(4, M - m0);
                            float common[4];
                            for (int r = 0; r < nr; ++r) {
                                common[r] = scales_A[m0 + r] / hadamard_divisor;
                            }
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                const int N_offset = active_n_segments[idx].offset_n;
                                rknpu2_quantization::dequant_acc_int16_tiled_perchan_rows(
                                    dst_batch + (size_t)m0 * N + N_offset, (size_t)N,
                                    (const int16_t*)mem_C_segments[idx]->virt_addr, m0, nr,
                                    c_geoms[idx].m_stride, c_geoms[idx].outer, c_geoms[idx].sub,
                                    active_n_segments[idx].size_n, common,
                                    scales_B_grid->data() + k_idx * (size_t)N + N_offset);
                            }
                        }
                    } else {
                        #pragma omp parallel for if(M > 1) num_threads(n_omp)
                        for (int m = 0; m < M; m++) {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                dequant_seg(m, idx);
                            }
                        }
                    }
                }
            }
        }
        if (g_rknpu_profile.on) g_rknpu_profile.node_ns += rknpu_profile::ns(t_node, std::chrono::steady_clock::now());
        if (fuse_glu && all_k_segments.size() == 1) {
            ++node_i;   // the GLU was computed inside this node's dequant
        }
        if (hn_last_node >= 0) node_i = hn_last_node;   // the norm (and MUL) were computed in its dequant
        if (hn_kv_node) backend_ctx->done_nodes.insert(hn_kv_node);   // its cache write too
    }
    rknpu_materialize_deferred_gate(backend_ctx->deferred_gate, n_omp);
    backend_ctx->done_nodes.clear();   // pointers are only meaningful within this graph

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

// A tensor may only be stored in NPU-packed form if pack_native can
// actually tile it: it asserts when a segment is not a multiple of the
// pipeline's alignment. supports_op enforces the same constraints before
// running an op, but the buffer paths did not — a tensor accepted purely
// on its dtype was packed anyway, and aborted at load. Dense LLM
// dimensions happen to be aligned, which is why only the CLIP vision
// tower ever tripped it (its K is not a multiple of k_align).
// Deliberately does NOT filter inside resolve_op_support: that assigns
// the sequence numbers driving cyclic hybrid patterns, and skipping
// tensors there would shift pipeline assignment for every later tensor.
static const rknpu2_configuration::Rknpu2HardwarePipeline * resolve_packable_pipeline(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);
    if (!pipeline) return nullptr;
    if (tensor->ne[0] % pipeline->k_align != 0) return nullptr;
    if (tensor->ne[1] % pipeline->n_align != 0) return nullptr;
    return pipeline;
}

// Function for calculating a real tensor size for the NPU
static size_t get_tensor_packed_size(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = resolve_packable_pipeline(tensor);

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::hadamard_k_op(K) : K;

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        size_t total_size = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& seg : n_segments) {
                if (seg.size_n > 0) {
                    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                        total_size += (size_t)seg.size_n * k_seg.size_k / 2;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
                        total_size += (size_t)seg.size_n * k_seg.size_k;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
                        total_size += (size_t)seg.size_n * k_seg.size_k * 2;
                    }
                }
            }
        }
        return total_size;
    }
    return ggml_nbytes(tensor);
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    // Freeing an every individual RKNN buffer using the allocator context
    for (auto& pair : ctx->tensor_allocs) {
        if (pair.second.mem) {
            rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(pair.second.iommu_domain_id);
            rknn_destroy_mem(alloc_ctx, pair.second.mem);
            g_domain_manager.release_domain_memory(pair.second.iommu_domain_id, pair.second.size);
        }
    }

    // Freeing the virtual memory block
    munmap(ctx->virtual_base, ctx->total_size);

    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->virtual_base;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    const auto* pipeline = rknpu_buffer_may_pack(buffer) ? resolve_packable_pipeline(tensor) : nullptr;

    // Initialize tensor only if it is supported by the pipeline
    if (pipeline) {
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;
        size_t size = get_tensor_packed_size(tensor);
        ctx->get_tensor_allocation(offset, size);
    }

    return GGML_STATUS_SUCCESS;
}

// Function for dequantizing a single row from GGUF format to FP32
static void dequantize_row(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int n, int K,
    float * row_out)
{
    if (tensor->type == GGML_TYPE_F32) {
        const float* src = (const float*)raw_data;
        memcpy(row_out, src + (size_t)n * K, K * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)raw_data;
        const ggml_fp16_t* src_row = src + (size_t)n * K;
        for (int k = 0; k < K; ++k) row_out[k] = ggml_fp16_to_fp32(src_row[k]);
    } else if (tensor->type == GGML_TYPE_BF16) {
        const ggml_bf16_t* src_row = (const ggml_bf16_t*)raw_data + (size_t)n * K;
        for (int k = 0; k < K; ++k) row_out[k] = ggml_bf16_to_fp32(src_row[k]);
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        const block_q8_0* src = (const block_q8_0*)raw_data;
        dequantize_row_q8_0(src + (size_t)n * (K / QK8_0), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        const block_q6_K* src = (const block_q6_K*)raw_data;
        dequantize_row_q6_K(src + (size_t)n * (K / QK_K), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        const block_q4_0* src = (const block_q4_0*)raw_data;
        dequantize_row_q4_0(src + (size_t)n * (K / QK4_0), row_out, K);
    } else {
        GGML_ASSERT(false && "Unsupported weight type for NPU pipeline");
    }
}

// Function for extracting a specific tensor segment and converting it to FP32
static void dequantize_tensor_segment(
    std::vector<float>& out_segment,
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const void * raw_data,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    bool use_hadamard)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
    out_segment.resize(seg_elements);

    std::vector<float> s_vec;
    if (use_hadamard) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        s_vec = ctx->hadamard_s_vectors[tensor];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_seg.size_n; ++i) {
        int global_n = n_seg.offset_n + i;

        if (global_n < N) {
            std::vector<float> row_raw(K);
            std::vector<float> row_processed(K_op, 0.0f);

            dequantize_row(tensor, raw_data, global_n, K, row_raw.data());

            if (use_hadamard) {
                std::vector<float> signed_row(K);
                for (int k = 0; k < K; ++k) signed_row[k] = row_raw[k] * s_vec[k];
                rknpu2_calibration::hadamard_transform(row_processed.data(), signed_row.data(), K, K_op);
            } else {
                memcpy(row_processed.data(), row_raw.data(), K * sizeof(float));
            }

            memcpy(&out_segment[i * k_seg.size_k], &row_processed[k_seg.offset_k], k_seg.size_k * sizeof(float));
        } else {
            memset(&out_segment[i * k_seg.size_k], 0, k_seg.size_k * sizeof(float));
        }
    }
}

// Function for quantizing the FP32 segment to the target NPU format
static void quantize_tensor_segment(
    const std::vector<float>& fp32_segment,
    std::vector<uint8_t>& out_quantized,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    float scale,
    rknpu2_configuration::Rknpu2NpuType npu_type)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;

    if (npu_type == rknpu2_configuration::NPU_TYPE_FP16) {
        out_quantized.resize(seg_elements * 2);
        rknpu2_quantization::convert_fp32_to_fp16(
            fp32_segment.data(),
            (uint16_t*)out_quantized.data(),
            seg_elements);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT8) {
        out_quantized.resize(seg_elements);
        rknpu2_quantization::quantize_fp32_to_int8(
            fp32_segment.data(),
            (int8_t*)out_quantized.data(),
            seg_elements,
            scale);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT4) {
        out_quantized.resize(seg_elements / 2);
        rknpu2_quantization::quantize_fp32_to_int4_packed(
            fp32_segment.data(),
            out_quantized.data(),
            seg_elements,
            scale);
    }
}

// Function for packing
static void pack_native(
    uint8_t* dst, const uint8_t* src,
    int K_total, int k_offset, int k_segment, int k_align,
    int N_total, int n_offset, int n_segment, int n_align,
    int element_bits)
{
    UNUSED(N_total);

    GGML_ASSERT(k_segment % k_align == 0 && "k_segment must be aligned to k_align");
    GGML_ASSERT(n_segment % n_align == 0 && "n_segment must be aligned to n_align");

    const size_t k_sub_bytes     = (size_t)k_align * element_bits / 8;
    const size_t src_row_bytes  = (size_t)K_total * element_bits / 8;
    const size_t n_blocks       = n_segment / n_align;
    const size_t k_blocks       = k_segment / k_align;
    const size_t kblock_stride  = (size_t)n_align * k_sub_bytes;
    const size_t nblock_stride  = k_blocks * kblock_stride;

    for (size_t ni = 0; ni < n_blocks; ++ni) {
        for (size_t ki = 0; ki < k_blocks; ++ki) {
            uint8_t* dst_tile = dst + ni * nblock_stride + ki * kblock_stride;

            for (int nn = 0; nn < n_align; ++nn) {
                const size_t n_global = (size_t)n_offset + ni * n_align + nn;
                const size_t k_start  = (size_t)k_offset + ki * k_align;

                const uint8_t* src_ptr = src + n_global * src_row_bytes
                                             + k_start * element_bits / 8;
                uint8_t* dst_ptr = dst_tile + nn * k_sub_bytes;

                size_t off = 0;
                for (; off + 16 <= k_sub_bytes; off += 16) {
                    vst1q_u8(dst_ptr + off, vld1q_u8(src_ptr + off));
                }
                for (; off < k_sub_bytes; ++off) {
                    dst_ptr[off] = src_ptr[off];
                }
            }
        }
    }
}

// Function for packing the quantized segment into the native NPU layout and writing to DMA
static size_t pack_tensor_segment(
    const std::vector<uint8_t>& quantized_segment,
    uint8_t * dst_dma_ptr,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline)
{
    int element_bits = 0;
    size_t segment_packed_size = 0;

    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
        element_bits = 16;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k * 2;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
        element_bits = 8;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
        element_bits = 4;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k / 2;
    }

    pack_native(dst_dma_ptr, quantized_segment.data(),
                k_seg.size_k, 0, k_seg.size_k, pipeline->k_align,
                n_seg.size_n, 0, n_seg.size_n, pipeline->n_align,
                element_bits);

    return segment_packed_size;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    // Only weights are packed: a compute buffer also receives F16/F32 copies
    // (the KV-cache slices consumed by NPU flash attention), which must stay
    // in their own layout even when their dtype has an NPU pipeline.
    const auto* pipeline = rknpu_buffer_may_pack(buffer) ? resolve_packable_pipeline(tensor) : nullptr;

    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];
        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::hadamard_k_op(K) : K;

        // Initializing Hadamard Transform Logic
        if (pipeline->use_hadamard) {
            std::vector<float> s_vec(K_op, 1.0f);
            // Seed from the tensor NAME (FNV-1a), not its address: pointer
            // seeds change with ASLR, making W4A4_HADAMARD results
            // unreproducible across runs and builds. Fall back to the
            // address only for unnamed tensors.
            uint64_t seed = 1469598103934665603ull;
            // RKNPU_SHARED_SIGNS=1: seed from K instead, so every weight
            // with the same K shares one sign vector. The rotation only has
            // to be consistent between a matmul's A and B, not unique per
            // tensor — and sharing makes the A-side transform reusable
            // across nodes that consume the same activations (Q/K/V, and
            // gate/up, all have K=2560 on E4B). This flag exists to measure
            // whether correlating those tensors' quantization error costs
            // anything before the reuse cache is built. See #3h.
            static const bool shared_signs = []() {
                const char* e = std::getenv("RKNPU_SHARED_SIGNS");
                return e != nullptr && std::atoi(e) == 1;
            }();
            if (shared_signs) {
                seed = (seed ^ (uint64_t)K_op) * 1099511628211ull;
            } else if (tensor->name[0] != '\0') {
                for (const char* c = tensor->name; *c; ++c) {
                    seed = (seed ^ (uint8_t)*c) * 1099511628211ull;
                }
            } else {
                seed = reinterpret_cast<uintptr_t>(tensor);
            }
            std::mt19937 gen((std::mt19937::result_type)seed);
            std::uniform_int_distribution<int> distrib(0, 1);

            for(int k = 0; k < K_op; ++k) {
                s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
            }

            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->hadamard_s_vectors[tensor] = s_vec;
        }

        // Computing global scale
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        // Allocating a new buffer for a tensor
        size_t required_size = get_tensor_packed_size(tensor);
        auto alloc = ctx->get_tensor_allocation(tensor_offset_in_virtual, required_size);
        uint8_t* tensor_dma_ptr = (uint8_t*)alloc.mem->virt_addr;

        // Computing specific hardware segments
        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        std::vector<float> seg_fp32;
        std::vector<uint8_t> seg_npu;
        uint8_t* current_write_ptr = tensor_dma_ptr + offset;

        std::vector<float> tensor_block_scales;

        // INT4: one weight scale per output channel per k-segment (default).
        // The channel scale factors out of the hardware K summation and is
        // applied in the C dequant pass, so the finer granularity is free on
        // the NPU. Plain per-row amax replaces the segment-wide entropy
        // search (which also removes the minutes-long W4A4 calibration at
        // load). Grid layout: [k_idx * N + global_n]; padded rows beyond N
        // hold quantized zeros and need no scale. Legacy per-segment
        // entropy scales: RKNPU_PER_CHANNEL=0.
        const bool b_int4 = pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4;
        const bool b_int8 = pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8;
        const bool per_channel =
            (b_int4 || b_int8) && rknpu2_calibration::per_channel_b_scales();
        if (per_channel) {
            tensor_block_scales.assign(k_segments.size() * (size_t)N, 1.0f);
        }
        // int4 packs 2 values per byte and quantizes to +-7; int8 is one
        // byte per value to +-127
        // int8 is never clipped: its packer has no clamp and relies on
        // scale == amax/127 (see the note in rknpu2-calibration.h)
        const float pc_divisor = b_int4 ? 7.0f : 127.0f;
        const float pc_clip    = b_int4 ? rknpu2_calibration::b_clip_factor() : 1.0f;

        // Processing individual segments block-by-block
        size_t k_idx = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& n_seg : n_segments) {
                if (n_seg.size_n == 0) continue;

                // Dequantizing the block
                dequantize_tensor_segment(seg_fp32, tensor, ctx, data, K, N, K_op, k_seg, n_seg, pipeline->use_hadamard);

                if (per_channel) {
                    const size_t row_bytes = b_int4 ? (size_t)k_seg.size_k / 2
                                                    : (size_t)k_seg.size_k;
                    seg_npu.resize((size_t)n_seg.size_n * row_bytes);
                    #pragma omp parallel for
                    for (int i = 0; i < n_seg.size_n; ++i) {
                        const float* row = seg_fp32.data() + (size_t)i * k_seg.size_k;
                        const float amax = rknpu2_quantization::amax_fp32(row, k_seg.size_k);
                        const float row_scale = (amax == 0.0f) ? 1.0f : pc_clip * amax / pc_divisor;
                        uint8_t* out_row = seg_npu.data() + (size_t)i * row_bytes;
                        if (b_int4) {
                            rknpu2_quantization::quantize_fp32_to_int4_packed(
                                row, out_row, k_seg.size_k, row_scale);
                        } else {
                            rknpu2_quantization::quantize_fp32_to_int8(
                                row, (int8_t*)out_row, k_seg.size_k, row_scale);
                        }
                        const int global_n = n_seg.offset_n + i;
                        if (global_n < N) {
                            tensor_block_scales[k_idx * (size_t)N + global_n] = row_scale;
                        }
                    }
                } else {
                    // Calculating local scale of the block
                    float block_scale = 1.0f;
                    if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
                        float amax = 0.0f;
                        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                            amax = rknpu2_calibration::calculate_entropy_amax(seg_fp32.data(), seg_fp32.size());
                        } else {
                            for (float val : seg_fp32) {
                                amax = std::max(amax, std::abs(val));
                            }
                        }
                        float quant_divisor = (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) ? 7.0f : 127.0f;
                        block_scale = (amax == 0.0f) ? 1.0f : amax / quant_divisor;
                    }
                    tensor_block_scales.push_back(block_scale);

                    // Quantizing
                    quantize_tensor_segment(seg_fp32, seg_npu, k_seg, n_seg, block_scale, pipeline->npu_type_b);
                }

                // Packing into chip native layout
                size_t bytes_written = pack_tensor_segment(seg_npu, current_write_ptr, k_seg, n_seg, pipeline);

                current_write_ptr += bytes_written;
            }
            ++k_idx;
        }

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->quantized_tensor_scales[tensor] = tensor_block_scales;
        }

        rknn_matmul_ctx sync_ctx = g_domain_manager.get_allocator_context(alloc.iommu_domain_id);
        RKNN_CHECK(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE");

        // Dual residency for M-dependent routing: keep the original bytes
        // host-resident so the CPU backend can compute small-M mul_mats
        // from them in place (get_alloc_size reserves the room).
        if (rknpu_cpu_decode_threshold() > 0) {
            memcpy((uint8_t*)tensor->data + offset, data, size);
        }
    } else {
        memcpy((uint8_t*)tensor->data + offset, data, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context*)buffer->context;
    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    // With dual residency the host region holds the original bytes, which is
    // what readers expect (the DMA copy is packed in NPU-native layout).
    if (rknpu_cpu_decode_threshold() > 0) {
        memcpy(data, (uint8_t*)tensor->data + offset, size);
        return;
    }

    if (!rknpu_buffer_may_pack(buffer)) {
        memcpy(data, (uint8_t*)tensor->data + offset, size);
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto it = ctx->tensor_allocs.find(tensor_offset_in_virtual);
    if (it != ctx->tensor_allocs.end()) {
        // A packed weight holds the NPU layout, not the tensor's own bytes,
        // and is usually smaller than ggml_nbytes: a read-back can only
        // return wrong data, and a full-size one runs off the allocation.
        // Fail loudly rather than corrupt or fault later.
        if (offset + size > it->second.size) {
            GGML_ABORT("RKNPU2: get_tensor on NPU-packed weight '%s' (%zu bytes requested at offset %zu, %zu packed) - "
                       "packed weights cannot be read back; an op on this weight was probably rejected by supports_op",
                       tensor->name, size, offset, it->second.size);
        }
        memcpy(data, (uint8_t*)it->second.mem->virt_addr + offset, size);
    } else {
        memcpy(data, (uint8_t*)tensor->data + offset, size);
    }
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);

    for (auto& pair : ctx->tensor_allocs) {
        memset((uint8_t*)pair.second.mem->virt_addr, value, pair.second.size);
    }
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    // Reserving virtual memory block
    void* virtual_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (virtual_base == MAP_FAILED) {
        return NULL;
    }

    // Initializing buffer context
    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context();
    ctx->virtual_base = virtual_base;
    ctx->total_size = size;
    ctx->name = "rknpu_virtual_buffer";

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .set_tensor_2d = */ NULL,
        /* .get_tensor_2d = */ NULL,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    // Never less than the tensor's own bytes: this hook cannot tell a weight
    // from a compute tensor, and a compute tensor is stored unpacked. For
    // weights the extra is only host-region address space (the packed data
    // lives in its own DMA allocation); with dual residency the host region
    // holds the original bytes anyway.
    return std::max(get_tensor_packed_size(tensor), ggml_nbytes(tensor));
}

static bool ggml_backend_rknpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    // Only advertised with dual residency, where the host region holds valid
    // original bytes the CPU backend can read in place. NOTE: this path
    // requires mmap model loading (the default); --no-mmap bypasses
    // set_tensor for host buffers and the NPU copy would never be built.
    // Also always since compute buffers are plain host memory: the CPU
    // backend then reads NPU outputs in place instead of the scheduler
    // copying them across at every split. Packed weights are never read by
    // the CPU because supports_op accepts every op on them (see get_tensor's
    // abort for the one historical exception, empty mul_mats).
    static const bool host_compute = []() {
        const char* env = std::getenv("RKNPU_HOST_BUFFERS");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return host_compute || rknpu_cpu_decode_threshold() > 0;
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_RMS_NORM:
            return rknpu_head_norm_enabled() && rknpu_head_norm_supported(op);

        case GGML_OP_SET_ROWS:
            return rknpu_kv_write_enabled() && rknpu_set_rows_supported(op);

        case GGML_OP_ROPE:
            return (rknpu_rope_any() || (rknpu_rope_enabled() && rknpu_head_norm_enabled())) && rknpu_rope_supported(op);

        case GGML_OP_MUL:
            return rknpu_head_norm_enabled() && rknpu_head_norm_weight(op) != nullptr;

        case GGML_OP_GLU: {
            static const bool glu_enabled = []() {
                const char* env = std::getenv("RKNPU_GLU");
                return env == nullptr || std::atoi(env) != 0;
            }();
            return glu_enabled && rknpu_glu_supported(op);
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            static const bool fa_enabled = []() {
                const char* env = std::getenv("RKNPU_FLASH_ATTN");
                return env == nullptr || std::atoi(env) != 0;
            }();
            return fa_enabled && rknpu_fa_supported(op);
        }

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            // The weight must be one of ours: only RKNPU weight buffers hold
            // the packed NPU copy graph_compute runs from (a null buffer is
            // llama's load-time placement probe). Needed since supports_buft
            // also accepts host buffers.
            if (src0->buffer && strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(src0->buffer)), "RKNPU") != 0) {
                return false;
            }

            // Searching for available hardware pipeline for this tensor
            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) {
                return false;
            }

            // Empty activations (llama-server emits output mul_mats with
            // zero rows when a ubatch produces no logits) are a no-op that
            // graph_compute skips. Accept them: rejecting sends the op to
            // the CPU, and the scheduler then copies the whole weight out of
            // this buffer through get_tensor — for E4B's tied output that is
            // 713 MB of NPU-packed bytes, which overran the smaller packed
            // allocation and segfaulted the server intermittently.
            if (src0->ne[0] > 0 && src0->ne[1] > 0 && ggml_nelements(src1) == 0) {
                return true;
            }

            // M-dependent routing: reject small-M (token generation) mul_mats
            // so the scheduler runs them on the CPU from the host-resident
            // original bytes, while large-M (prefill) stays on the NPU
            if (src1->ne[1] < rknpu_cpu_decode_threshold()) {
                return false;
            }

            // Rejecting zero-dimension ops
            if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
                src1->ne[0] == 0 || src1->ne[1] == 0) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % pipeline->k_align != 0) {
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % pipeline->n_align != 0) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Checking contiguous memory. dst is included: graph_compute
            // addresses it as `dst_batch + m*N + N_offset` with only nb[2]/
            // nb[3] honoured for the batch stride, so a permuted or
            // row-strided dst would be memset and written outside its rows.
            // RKNPU_DEBUG_OPS used to only *print* "STRIDE MISMATCH" here.
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
                !ggml_is_contiguous(op)) {
                return false;
            }

            // The batch loop in graph_compute applies one weight matrix to
            // every src1 slice, so src0 must be 2D — a 3D weight tensor is a
            // multi-expert one and belongs to MUL_MAT_ID, which this backend
            // does not implement — and dst must have src1's batch shape, so
            // no broadcasting.
            if (src0->ne[2] > 1 || src0->ne[3] > 1) {
                return false;
            }
            if (op->ne[2] != src1->ne[2] || op->ne[3] != src1->ne[3]) {
                return false;
            }

            // RKNPU_DEBUG_OPS=1: report the geometry of every accepted
            // mul_mat, so a wrong-output bug can be traced to the shapes
            // the backend actually agreed to compute.
            static const bool debug_ops = std::getenv("RKNPU_DEBUG_OPS") != nullptr;
            if (debug_ops) {
                const bool dst_contig = ggml_is_contiguous(op);
                const bool dst_rowstride_ok = (op->nb[1] == op->ne[0] * sizeof(float));
                GGML_LOG_INFO("rknpu-dbg accept %-34s src0[%ld,%ld,%ld] src1[%ld,%ld,%ld] "
                              "dst[%ld,%ld,%ld] dst_contig=%d dst_nb1=%zu expect=%zu%s\n",
                    src0->name,
                    (long)src0->ne[0], (long)src0->ne[1], (long)src0->ne[2],
                    (long)src1->ne[0], (long)src1->ne[1], (long)src1->ne[2],
                    (long)op->ne[0], (long)op->ne[1], (long)op->ne[2],
                    (int)dst_contig, (size_t)op->nb[1], (size_t)(op->ne[0] * sizeof(float)),
                    dst_rowstride_ok ? "" : "   <-- STRIDE MISMATCH");
            }

            return true;
        }
        default:
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // Fetch device from environment variable, default to RK3588 if not set
    const char* env_device = std::getenv("RKNPU_DEVICE");
    std::string target_device = env_device ? env_device : "RK3588";
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device(target_device)) return NULL;

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();

    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .set_tensor_2d_async= */ NULL,
        /* .get_tensor_2d_async= */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return 1;
}

static void ggml_backend_rknpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ((ggml_backend_rknpu_context*)backend->context)->n_threads = n_threads;
}

static void* ggml_backend_rknpu_reg_get_proc_address(ggml_backend_reg_t reg, const char* name) {
    UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void*)ggml_backend_rknpu_set_n_threads;
    }
    return NULL;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }

    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .alloc_buffer_n = */ NULL,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .get_alloc_size_n = */ NULL,
        /* .is_host        = */ ggml_backend_rknpu_buffer_type_is_host,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        // Host buffers too: every op here reads its inputs from host memory
        // (get_tensor_real_ptr returns tensor->data outside RKNPU weight
        // buffers), so the scheduler need not copy CPU-resident activations,
        // KV cache or small weights into an RKNPU buffer at each split.
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type || ggml_backend_buft_is_host(buft); },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
#if defined(__GLIBC__)
    // librknnrt manages device memory mappings from its own worker threads and
    // can unmap address ranges that alias glibc's mmap-served large malloc
    // chunks, pulling pages out from under live allocations (observed as
    // intermittent SIGSEGV in the calibration buffers during weight upload).
    // Keeping large allocations on the brk heap moves them out of the
    // contested mmap address range.
    //
    // Re-tested 2026-08-29 after review flagged it as a process-wide side
    // effect: the crash no longer reproduces without it (26/26 runs pass,
    // including the full pre-2026-08-16 legacy anchor that recreates the
    // original large calibration buffers — against an original 0/6 survival
    // rate), and peak RSS is bit-identical with and without it in every
    // configuration tried. Kept anyway: the allocation pattern that used to
    // get hit is gone, not the hazard — librknnrt is closed source and still
    // unmaps in that range. Zero measured cost, non-zero latent protection.
    // See RKNPU2-W4A4-investigation.md, "Re-tested 2026-08-29".
    static const int rknpu_malloc_no_mmap = mallopt(M_MMAP_MAX, 0);
    (void) rknpu_malloc_no_mmap;
#endif

    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ ggml_backend_rknpu_reg_get_proc_address,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif