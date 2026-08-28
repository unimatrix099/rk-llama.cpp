// Protocol tests for the RKNPU2 persistent dispatch pool.
//
// The pool hands segment 0 to the calling thread and segments 1..n-1 to
// persistent workers, synchronising on a generation counter. Two defects
// found in review (decode research, dispatch-pool findings) are pinned here:
//
//   1. Workers were seeded with `seen = 0`. A worker spawned once generation
//      was already > 0 woke immediately on the stale counter and could run
//      its segment twice — once on the published-but-unannounced job, once
//      on the real bump. When the first increment landed after done.store(0),
//      `done` reached n-1 from a single worker and run_all returned while the
//      other segment was still uncomputed.
//
//   2. `job_ctxs` was a plain pointer. A worker with no segment for itself
//      (left over from an earlier, wider node) never touches `done`, so
//      run_all could null the pointer and return while that worker sat
//      between loading it and dereferencing it.
//
// Both need the pool to GROW mid-run to reproduce, which happens whenever a
// narrow node (fewer active N-segments) precedes a full-width one — the
// segment count comes from compute_n_segments(N, cores, alignment).
//
// Build & run (from docs/backend):
//   make -f Makefile.rknpu2-tools check-dispatch

#define RKNPU2_DISPATCH_POOL_TESTING 1
#include "../../ggml/src/ggml-rknpu2/rknpu2-dispatch-pool.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        printf("FAIL %s:%d  ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

// Mock segment: counts how many times the pool ran it, and records the
// generation it belonged to so a stale re-run is distinguishable from a
// legitimate one.
struct MockSeg {
    std::atomic<int> runs{0};
    std::atomic<int> concurrent{0};
    std::atomic<int> max_concurrent{0};
    void run() {
        const int c = concurrent.fetch_add(1) + 1;
        int prev = max_concurrent.load();
        while (c > prev && !max_concurrent.compare_exchange_weak(prev, c)) {}
        runs.fetch_add(1, std::memory_order_seq_cst);
        concurrent.fetch_sub(1);
    }
};

using Pool = rknpu_dispatch_pool_t<MockSeg>;
using Job  = std::vector<std::shared_ptr<MockSeg>>;

static Job make_job(int n) {
    Job j;
    for (int i = 0; i < n; ++i) j.push_back(std::make_shared<MockSeg>());
    return j;
}

// Every segment of a job must run exactly once, whatever the width.
static void test_each_segment_runs_once() {
    for (int n : {1, 2, 3}) {
        Pool pool;
        Job j = make_job(n);
        pool.run_all(j);
        for (int i = 0; i < n; ++i) {
            CHECK(j[i]->runs.load() == 1, "n=%d seg %d ran %d times, expected 1",
                  n, i, j[i]->runs.load());
        }
    }
}

// Repeated jobs of the same width: the pool is reused, not regrown.
static void test_repeated_jobs() {
    Pool pool;
    const int iters = 200;
    for (int it = 0; it < iters; ++it) {
        Job j = make_job(3);
        pool.run_all(j);
        for (int i = 0; i < 3; ++i) {
            if (j[i]->runs.load() != 1) {
                CHECK(false, "iter %d seg %d ran %d times", it, i, j[i]->runs.load());
                return;
            }
        }
    }
    CHECK(true, "200 repeated 3-way jobs");
}

// Pool growth under natural timing. Narrow jobs first so the pool holds fewer
// workers than the wide job needs, forcing ensure_workers() to spawn while
// generation is already > 0. Without the publish hook the window is too narrow
// to provoke the defect (thread creation outlasts it), so this is a soak test
// for the growth path rather than the regression test — that one is
// test_no_segment_runs_before_announce below.
static void test_pool_growth_midrun() {
    for (int trial = 0; trial < 60; ++trial) {
        Pool pool;
        // several narrow jobs: pool stays at 0 then 1 worker
        for (int i = 0; i < 5; ++i) {
            Job j1 = make_job(1);
            pool.run_all(j1);
            CHECK(j1[0]->runs.load() == 1, "trial %d narrow seg ran %d times",
                  trial, j1[0]->runs.load());
        }
        Job j2 = make_job(2);          // grows pool to 1 worker at generation > 0
        pool.run_all(j2);
        for (int i = 0; i < 2; ++i) {
            CHECK(j2[i]->runs.load() == 1, "trial %d width-2 seg %d ran %d times",
                  trial, i, j2[i]->runs.load());
        }
        Job j3 = make_job(3);          // grows pool to 2 workers, generation higher still
        pool.run_all(j3);
        for (int i = 0; i < 3; ++i) {
            CHECK(j3[i]->runs.load() == 1, "trial %d width-3 seg %d ran %d times",
                  trial, i, j3[i]->runs.load());
        }
    }
}

// run_all must not return before every segment has actually finished: that
// is the corruption mode (master dequantises C for a segment nobody ran).
static void test_no_early_return() {
    for (int trial = 0; trial < 60; ++trial) {
        Pool pool;
        Job narrow = make_job(2);
        pool.run_all(narrow);          // pool at 1 worker
        Job wide = make_job(3);        // grows to 2 mid-run
        pool.run_all(wide);
        // Checked immediately after return, with no synchronisation of our
        // own: if run_all released early this reads 0 for some segment.
        for (int i = 0; i < 3; ++i) {
            CHECK(wide[i]->runs.load(std::memory_order_seq_cst) == 1,
                  "trial %d: run_all returned with seg %d at %d runs",
                  trial, i, wide[i]->runs.load());
        }
    }
}

// A shrinking job must not let a left-over worker touch the narrower job's
// segments (finding 2: stale job_ctxs / out-of-range segment index).
static void test_shrinking_job() {
    for (int trial = 0; trial < 60; ++trial) {
        Pool pool;
        Job wide = make_job(3);
        pool.run_all(wide);            // pool grown to 2 workers
        for (int i = 0; i < 200; ++i) {
            Job narrow = make_job(2);  // worker idx 1 has no segment now
            pool.run_all(narrow);
            if (narrow[0]->runs.load() != 1 || narrow[1]->runs.load() != 1) {
                CHECK(false, "trial %d iter %d: narrow runs %d/%d",
                      trial, i, narrow[0]->runs.load(), narrow[1]->runs.load());
                return;
            }
        }
    }
    CHECK(true, "shrinking jobs after growth");
}

// Segment 0 belongs to the calling thread; no segment should ever be entered
// by two threads at once.
static void test_no_concurrent_same_segment() {
    Pool pool;
    for (int i = 0; i < 200; ++i) {
        Job j = make_job(3);
        pool.run_all(j);
        for (int k = 0; k < 3; ++k) {
            CHECK(j[k]->max_concurrent.load() <= 1,
                  "seg %d had %d concurrent runners", k, j[k]->max_concurrent.load());
        }
    }
}

// THE REGRESSION TEST for the stale-seed defect.
//
// What the defect really is: a worker seeded with `seen = 0` and spawned once
// generation is already > 0 wakes on the *stale* counter, and if the master
// has published job_ctxs but not yet announced it, runs its segment for a job
// it was never told about. `done` then counts an increment that does not
// correspond to an announced run, so `done == n-1` stops proving "every
// segment ran" — it can be satisfied by one worker running its own segment
// twice while a peer has not run at all. That is the corruption mode: the
// master dequantises a C segment nobody computed.
//
// Reproducing the *corruption* needs two coincidences (the early run AND a
// lagging peer). Reproducing the *cause* needs only one, and it is the cause
// the fix removes, so that is what this pins: between publishing and
// announcing a job, no segment of it may have run. The hook holds that window
// open — in a real build it is a couple of stores wide, far narrower than
// thread creation, so timing alone can never provoke it.
static std::atomic<const Job*> g_window_job{nullptr};
static std::atomic<int>        g_ran_early{0};

static void hold_publish_window() {
    // Give a mis-seeded worker every chance to wake and act inside the window.
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    const Job* j = g_window_job.load();
    if (!j) return;
    int early = 0;
    for (const auto& seg : *j) early += seg->runs.load(std::memory_order_seq_cst);
    g_ran_early.store(early);
}

static void test_no_segment_runs_before_announce() {
    for (int trial = 0; trial < 8; ++trial) {
        Pool pool;
        // Width-1 jobs take the n<=1 early return and never publish, so they
        // do NOT advance generation. Only width >= 2 does. Run one first so
        // generation becomes 1 and worker 0 exists...
        { Job warm = make_job(2); pool.run_all(warm); }

        // ...then grow the pool: worker 1 is spawned while generation == 1,
        // so a `seen = 0` seed mismatches immediately and it wakes early.
        Job wide = make_job(3);
        g_window_job.store(&wide);
        g_ran_early.store(0);
        pool.publish_hook.store(&hold_publish_window);
        pool.run_all(wide);
        pool.publish_hook.store(nullptr);
        g_window_job.store(nullptr);

        CHECK(g_ran_early.load() == 0,
              "trial %d: %d segment run(s) happened between publish and "
              "announce — a worker woke on a stale generation",
              trial, g_ran_early.load());

        for (int i = 0; i < 3; ++i) {
            CHECK(wide[i]->runs.load() == 1,
                  "trial %d: segment %d ran %d times, expected 1",
                  trial, i, wide[i]->runs.load());
        }
    }
}

int main(void) {
    test_no_segment_runs_before_announce();
    test_each_segment_runs_once();
    test_repeated_jobs();
    test_pool_growth_midrun();
    test_no_early_return();
    test_shrinking_job();
    test_no_concurrent_same_segment();
    printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
