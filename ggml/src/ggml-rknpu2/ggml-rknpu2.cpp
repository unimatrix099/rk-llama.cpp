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
    rknpu_attn_context(int M, int K, int N, int b_layout, int core_id) {
        memset(&info, 0, sizeof(info));
        info.M = M; info.K = K; info.N = N;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = (int16_t)b_layout;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        if (rknn_matmul_create(&ctx, &info, &io_attr) < 0) { ctx = 0; return; }
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
        job = &ctxs;
        pending = (int)ctxs.size();
        ++generation;
        cv_start.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [this] { return pending == 0; });
        job = nullptr;
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
            const std::vector<std::shared_ptr<rknpu_matmul_context>>* j;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_start.wait(lock, [&] { return quit || generation != seen; });
                if (quit) return;
                seen = generation;
                j = job;
            }
            if (j && idx < (int)j->size()) {
                (*j)[idx]->run();
                std::lock_guard<std::mutex> lock(mutex);
                if (--pending == 0) cv_done.notify_all();
            }
        }
    }
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv_start, cv_done;
    const std::vector<std::shared_ptr<rknpu_matmul_context>>* job = nullptr;
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

    // (M, K, N, B layout, core) -> attention matmul context
    std::map<std::tuple<int, int, int, int, int>, std::unique_ptr<rknpu_attn_context>> attn_ctx_cache;
    rknpu_attn_context* get_attn_ctx(int M, int K, int N, int b_layout, int core_id) {
        auto key = std::make_tuple(M, K, N, b_layout, core_id);
        auto it = attn_ctx_cache.find(key);
        if (it != attn_ctx_cache.end()) return it->second.get();
        auto c = std::make_unique<rknpu_attn_context>(M, K, N, b_layout, core_id);
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
        fprintf(stderr, "RKNPU_PROFILE t=%.1fs graphs=%llu nodes=%llu graph=%.1fms node=%.1fms npu_run=%.1fms\n",
                ns(t_start, now) / 1e9, (unsigned long long)graphs, (unsigned long long)nodes,
                graph_ns / 1e6, node_ns / 1e6, run_ns / 1e6);
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

// One attention row: P = softmax(softcap(s*scale) + mask), written as FP16.
// Masked (-inf) positions give exactly 0; an all-masked row gives zeros.
static void rknpu_softmax_row(const float* s_row, const ggml_fp16_t* mrow, int64_t n, float scale, float softcap,
                              float* tmp, uint16_t* out) {
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

// GEGLU (gate, up) on the backend, so [up, gate, GLU, down] stay in one NPU
// split. Same NEON tanh-GELU as ggml-cpu's ggml_vec_geglu_f32 on ARM
// (tanh(z) = 1 - 2/(e^{2z}+1) via the same exp, +-10 cut-offs), so the
// output is identical for widths that are multiples of 4.
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
    for (int t = 0; t < outer; ++t) {
        const int n0 = t * 8;
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
    if (q->ne[2] % k->ne[2] != 0 || k->ne[2] != v->ne[2]) return false;
    if (k->nb[0] != 2 || v->nb[0] != 2 || q->nb[0] != 4 || op->nb[0] != 4) return false;
    return true;
}

static void rknpu_flash_attn(ggml_backend_rknpu_context* bctx, struct ggml_tensor* dst, int n_omp) {
    const struct ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *mask = dst->src[3];
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

    for (int64_t i3 = 0; i3 < n_seq; ++i3) {
    const int64_t ik3 = i3 / (n_seq / k->ne[3]);
    const int64_t iv3 = i3 / (n_seq / v->ne[3]);
    const char* m_seq = mask ? m_base + (i3 % mask->ne[3]) * mask->nb[3] : nullptr;
    for (int64_t g = 0; g < n_kvh; ++g) {
        const int core = (int)(g % 3);
        rknpu_attn_context* qk = bctx->get_attn_ctx((int)M, (int)DK, (int)n_kv, RKNN_MM_LAYOUT_TP_NORM, core);
        rknpu_attn_context* pv = bctx->get_attn_ctx((int)M, (int)n_kv, (int)DV, RKNN_MM_LAYOUT_NORM, core);
        GGML_ASSERT(qk && pv && "RKNPU2: attention matmul context creation failed");

        // A = Q rows of the rk2 heads of this group, FP16, row r = hh*n_q + i
        uint16_t* a = (uint16_t*)qk->A->virt_addr;
        #pragma omp parallel for num_threads(n_omp)
        for (int64_t r = 0; r < M; ++r) {
            const int64_t hh = r / n_q, i = r % n_q, h = g * rk2 + hh;
            rknpu_fp32_to_fp16((const float*)(q_base + i * q->nb[1] + h * q->nb[2] + i3 * q->nb[3]), a + r * DK, DK);
        }
        // B = K rows (n_kv x DK, TP_NORM) and V rows (n_kv x DV, NORM)
        uint16_t* bk = (uint16_t*)qk->B->virt_addr;
        uint16_t* bv = (uint16_t*)pv->B->virt_addr;
        for (int64_t j = 0; j < n_kv; ++j) {
            memcpy(bk + j * DK, k_base + j * k->nb[1] + g * k->nb[2] + ik3 * k->nb[3], DK * 2);
            memcpy(bv + j * DV, v_base + j * v->nb[1] + g * v->nb[2] + iv3 * v->nb[3], DV * 2);
        }
        rknn_mem_sync(qk->ctx, qk->A, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_mem_sync(qk->ctx, qk->B, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_mem_sync(pv->ctx, pv->B, RKNN_MEMORY_SYNC_TO_DEVICE);
        // Re-bind B after writing it: for a non-native B the driver converts
        // it to its internal layout at set_io_mem time, so data written into
        // an already-bound buffer is never seen (P*V came back all zeros)
        RKNN_CHECK(rknn_matmul_set_io_mem(qk->ctx, qk->B, &qk->io_attr.B), "set_io_mem attn K");
        RKNN_CHECK(rknn_matmul_set_io_mem(pv->ctx, pv->B, &pv->io_attr.B), "set_io_mem attn V");
        rknn_matmul_run(qk->ctx);
        rknn_mem_sync(qk->ctx, qk->C, RKNN_MEMORY_SYNC_FROM_DEVICE);

        // softmax rows into P (FP16) = A of the second matmul
        const float* S = (const float*)qk->C->virt_addr;
        uint16_t* P = (uint16_t*)pv->A->virt_addr;
        #pragma omp parallel for num_threads(n_omp)
        for (int64_t r = 0; r < M; ++r) {
            const int64_t i = r % n_q;
            static thread_local std::vector<float> row;
            if ((int64_t)row.size() < n_kv) row.resize(n_kv);
            const float* s_row = S + r * n_kv;
            const ggml_fp16_t* mrow = mask ? (const ggml_fp16_t*)(m_seq + i * mask->nb[1]) : nullptr;
            rknpu_softmax_row(s_row, mrow, n_kv, scale, softcap, row.data(), P + r * n_kv);
        }
        rknn_mem_sync(pv->ctx, pv->A, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_matmul_run(pv->ctx);
        rknn_mem_sync(pv->ctx, pv->C, RKNN_MEMORY_SYNC_FROM_DEVICE);

        // O rows -> dst (permuted: row (i*n_head + h))
        const float* O = (const float*)pv->C->virt_addr;
        #pragma omp parallel for num_threads(n_omp)
        for (int64_t r = 0; r < M; ++r) {
            const int64_t hh = r / n_q, i = r % n_q, h = g * rk2 + hh;
            // permute(0, 2, 1, 3): row (i3*n_q*n_head + i*n_head + h)
            memcpy(d_base + (i3 * n_q * n_head + i * n_head + h) * dst->nb[1], O + r * DV, DV * sizeof(float));
        }
    }
    }
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
        if (node->op != GGML_OP_MUL_MAT) continue;
        const auto t_node = g_rknpu_profile.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (g_rknpu_profile.on) g_rknpu_profile.nodes++;

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
                            n_seg.core_id, matmul_type, pipeline->ac_layout, b_domain_id
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
                auto& dg = backend_ctx->deferred_gate;
                const bool use_dg = dg.gate != nullptr && pipelined_node && fused_now && fuse_glu->src[0] == dg.gate &&
                                    dg.up == node && dg.M == M && dg.N == N && dg.MC == MC;
                if (dg.gate && !use_dg) rknpu_materialize_deferred_gate(dg, n_omp);
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


                    // single-pass tiles when gate and up share 8-wide cell geometries
                    bool dg_tiles = false;
#ifdef __ARM_NEON
                    if (use_dg) {
                        static const bool tiles_env = []() {
                            const char* env = std::getenv("RKNPU_GEGLU_TILES");
                            return env == nullptr || std::atoi(env) != 0;
                        }();
                        dg_tiles = tiles_env && dg.c_geom.size() == num_active_segments;
                        for (size_t idx = 0; dg_tiles && idx < num_active_segments; ++idx) {
                            const auto &a = c_geom[idx], &b = dg.c_geom[idx];
                            dg_tiles = a.sub == 8 && b.sub == 8 && a.m_stride == b.m_stride && a.outer == b.outer &&
                                       active_n_segments[idx].size_n % 8 == 0 &&
                                       dg.segs[idx].offset_n == active_n_segments[idx].offset_n &&
                                       dg.segs[idx].size_n == active_n_segments[idx].size_n;
                        }
                    }
#endif
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
                            if (dg_tiles) {
                                float* y = (float*)get_tensor_real_ptr(fuse_glu);
                                const size_t ys = fuse_glu->nb[1] / sizeof(float);
                                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                                    const int N_offset = active_n_segments[idx].offset_n;
                                    rknpu_gate_up_geglu_tiles(y + (size_t)(m0 + r0) * ys + N_offset, ys,
                                        (const int16_t*)dg.c[c][idx]->virt_addr, (const int16_t*)cslot(c)[idx]->virt_addr,
                                        r0, nr, c_geom[idx], active_n_segments[idx].size_n,
                                        dg.common.data() + m0 + r0, common, dg.chan + N_offset,
                                        scales_B_grid->data() + k_idx * (size_t)N + N_offset);
                                }
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
                    auto collect_any = [&](int c) {
                        if (defer_gate) keep(c);
                        else if (use_dg) collect_fused_dg(c);
                        else if (fused_now) collect_fused(c);
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
                    const bool a_native = pipeline->ac_layout == RKNN_MM_LAYOUT_NATIVE &&
                        rknpu2_native_geom_from_dims(matmul_ctx_0->io_attr.A.dims,
                                                     matmul_ctx_0->io_attr.A.n_dims, &a_geom) == 0;
                    // Not a fallback: the context was created with
                    // AC_layout = NATIVE, so if the geometry does not parse we
                    // would fill a row-major buffer the NPU reads as
                    // [K/sub, M, sub] tiles — silently wrong output, no error.
                    GGML_ASSERT((pipeline->ac_layout != RKNN_MM_LAYOUT_NATIVE || a_native) &&
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
                            int8_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                            rknpu2_quantization::quantize_fp32_to_int8(ready_row, dst_row, K_seg_op, scales_A[m]);
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
                if (pipeline->ac_layout == RKNN_MM_LAYOUT_NATIVE) {
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
    }
    rknpu_materialize_deferred_gate(backend_ctx->deferred_gate, n_omp);

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