#pragma once

// Persistent dispatch threads for the per-node NPU segment runs.
//
// Dispatching the segment runs with an OpenMP team of a different size than
// the surrounding regions makes libgomp tear down and respawn its workers
// around every region: measured ~550 pthread_create per generated token at
// M=1 (one per graph split), costing ~30% of NPU-decode throughput and
// requiring OMP_NUM_THREADS workarounds (profiling chain in
// docs/backend/RKNPU2-decode-research.md, avenue #3). This pool keeps the
// dispatch outside OpenMP entirely: the calling thread runs segment 0
// itself, persistent workers run the rest. Workers are lazily spawned up to
// max-segments-1 (= NPU cores - 1) and live until the backend is freed.
//
// Templated on the segment type purely so the synchronisation protocol can
// be unit-tested against a mock segment (docs/backend/test-rknpu2-dispatch-pool.cpp)
// without linking librknnrt. `Ctx` must expose `void run()`. The production
// instantiation monomorphises to the same code the hand-written version
// emitted — this path is latency-sensitive and amplified ~2.5x by spin-wait
// contention (decode research #4c), so it must not grow indirection.

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

template <typename Ctx>
struct rknpu_dispatch_pool_t {
    void run_all(const std::vector<std::shared_ptr<Ctx>>& ctxs) {
        const int n = (int)ctxs.size();
        if (n <= 1) {
            if (n == 1) ctxs[0]->run();
            return;
        }
        ensure_workers(n - 1);

        // Publish the job: job_ctxs/done written before the release bump of
        // generation, read by workers after their acquire load of it
        job_ctxs.store(&ctxs, std::memory_order_relaxed);
        done.store(0, std::memory_order_relaxed);
#ifdef RKNPU2_DISPATCH_POOL_TESTING
        // Test-only: widen the publish→announce window. The stale-seed defect
        // needs a freshly spawned worker to execute between these two points,
        // which is a few nanoseconds in a real build — narrower than thread
        // creation, so it cannot be provoked by timing alone. Compiled out
        // entirely unless the test defines this.
        if (auto h = publish_hook.load(std::memory_order_relaxed)) h();
#endif
        // seq_cst for the same Dekker reason as done/master_sleeping below:
        // we store generation then load sleepers; a worker entering the
        // sleep path stores sleepers then loads generation
        generation.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lock(mutex);
            cv_start.notify_all();
        }

        ctxs[0]->run();   // the caller takes segment 0

        // Segments finish together (equal-size N split), so the workers are
        // usually done by the time our own run returns: spin briefly, then
        // sleep — the same latency/burn trade the workers make below
        for (int s = 0; s < SPIN_ITERS; ++s) {
            if (done.load(std::memory_order_acquire) == n - 1) {
                job_ctxs.store(nullptr, std::memory_order_relaxed);
                return;
            }
            cpu_relax();
        }
        std::unique_lock<std::mutex> lock(mutex);
        // seq_cst pairing with the worker's done/master_sleeping accesses:
        // both sides store one variable then load the other (Dekker), so at
        // least one side must observe the other's store
        master_sleeping.store(true, std::memory_order_seq_cst);
        cv_done.wait(lock, [&] { return done.load(std::memory_order_seq_cst) == n - 1; });
        master_sleeping.store(false, std::memory_order_relaxed);
        job_ctxs.store(nullptr, std::memory_order_relaxed);
    }

    ~rknpu_dispatch_pool_t() {
        quit.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex);
            cv_start.notify_all();
        }
        for (auto& w : workers) w.join();
    }

  private:
    // ~0.1 ms of `yield` on an A76, then sleep. Measured trade-off: longer
    // spins and pre-waking the workers before the job is ready both LOSE
    // throughput on this 8-core part — the spinning steals exactly the
    // cores the A-prep/dequant OMP regions and the CPU graph ops need
    // (prewake variant: routed pp 42.5 -> 37.9, pinned NPU tg 3.9 -> 1.9).
    // The remaining ~0.1 ms/node of wake latency on the NPU-decode path
    // could be removed structurally (workers doing their own segment's
    // set_io) rather than by spinning harder — future work, low priority
    // while routed decode is the recommended mode.
    static constexpr int SPIN_ITERS = 65536;

    static inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
        __asm__ volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
        __asm__ volatile("pause");
#else
        std::atomic_thread_fence(std::memory_order_relaxed);
#endif
    }

    void ensure_workers(int needed) {
        while ((int)workers.size() < needed) {
            const int idx = (int)workers.size();
            // Seed the worker with the CURRENT generation. Starting it at 0
            // meant a worker spawned once generation > 0 woke immediately on
            // the stale counter, latched the old value, and could read a
            // job_ctxs the master had already published but not yet
            // announced — running its segment, bumping `done`, then waking
            // again on the real bump and running the SAME segment twice. If
            // the first increment landed after done.store(0), `done` reached
            // n-1 from one worker alone and the master proceeded to read C
            // segments the other worker had not computed yet. Reachable
            // whenever the pool grows mid-graph, i.e. a narrow node (fewer
            // active N-segments) before a full-width one.
            const uint64_t start_gen = generation.load(std::memory_order_acquire);
            workers.emplace_back([this, idx, start_gen] { worker_loop(idx, start_gen); });
        }
    }

    void worker_loop(int idx, uint64_t seen) {
        for (;;) {
            // hot path: spin on the generation counter
            bool woke = false;
            for (int s = 0; s < SPIN_ITERS; ++s) {
                if (generation.load(std::memory_order_acquire) != seen ||
                    quit.load(std::memory_order_relaxed)) { woke = true; break; }
                cpu_relax();
            }
            if (!woke) {
                std::unique_lock<std::mutex> lock(mutex);
                sleepers.fetch_add(1, std::memory_order_seq_cst);
                cv_start.wait(lock, [&] {
                    return quit.load(std::memory_order_relaxed) ||
                           generation.load(std::memory_order_seq_cst) != seen;
                });
                sleepers.fetch_sub(1, std::memory_order_relaxed);
            }
            if (quit.load(std::memory_order_relaxed)) return;
            seen = generation.load(std::memory_order_acquire);

            // worker idx handles segment idx+1; a pool grown by an earlier
            // wider node may have more workers than this node has segments.
            // Loaded atomically: a worker that finds no segment for itself
            // never touches `done`, so run_all can null this out and return
            // while such a worker is still between the two reads.
            const auto* job = job_ctxs.load(std::memory_order_acquire);
            if (job && idx + 1 < (int)job->size()) {
                (*job)[idx + 1]->run();
                done.fetch_add(1, std::memory_order_seq_cst);
                if (master_sleeping.load(std::memory_order_seq_cst)) {
                    std::lock_guard<std::mutex> lock(mutex);
                    cv_done.notify_one();
                }
            }
        }
    }

#ifdef RKNPU2_DISPATCH_POOL_TESTING
  public:
    // Invoked between publishing the job and announcing it. See run_all().
    std::atomic<void (*)()> publish_hook{nullptr};
  private:
#endif

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable cv_start, cv_done;
    std::atomic<const std::vector<std::shared_ptr<Ctx>>*> job_ctxs{nullptr};
    std::atomic<uint64_t> generation{0};
    std::atomic<int> done{0};
    std::atomic<int> sleepers{0};
    std::atomic<bool> quit{false};
    std::atomic<bool> master_sleeping{false};
};
