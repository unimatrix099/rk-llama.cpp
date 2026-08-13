// LD_PRELOAD shim: wall-time accounting of librknnrt entry points.
//
// The rknpu2 backend calls librknnrt through the PLT (cross-DSO), so
// interposition sees every driver call. A monitor thread prints cumulative
// {calls, ns} every 5 s to stderr; the slope across a steady-state decode
// window divided by the token rate gives ms/token per driver entry point —
// wall-time attribution that cycle sampling cannot provide (blocked time
// is invisible to perf).
//
//   gcc -O2 -shared -fPIC rknnshim.c -o rknnshim.so -ldl -lpthread
//   LD_PRELOAD=./rknnshim.so <llama-bench ...> 2>shim.log
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

typedef struct { _Atomic uint64_t calls, ns; } acc_t;
static acc_t a_run, a_setio, a_sync, a_mmcreate, a_creatmem, a_creatfd;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

#define WRAP(name, ACC, argdecl, argcall)                                   \
    static int (*real_##name) argdecl = NULL;                               \
    int name argdecl {                                                      \
        if (!real_##name)                                                   \
            real_##name = (int (*) argdecl)dlsym(RTLD_NEXT, #name);         \
        uint64_t t0 = now_ns();                                             \
        int r = real_##name argcall;                                        \
        atomic_fetch_add_explicit(&ACC.ns, now_ns() - t0,                   \
                                  memory_order_relaxed);                    \
        atomic_fetch_add_explicit(&ACC.calls, 1, memory_order_relaxed);     \
        return r;                                                           \
    }

// rknn_context / rknn_matmul_ctx are uint64_t; pointer args as void*
WRAP(rknn_matmul_run,        a_run,      (unsigned long long c), (c))
WRAP(rknn_matmul_set_io_mem, a_setio,    (unsigned long long c, void* m, void* a), (c, m, a))
WRAP(rknn_mem_sync,          a_sync,     (unsigned long long c, void* m, int mode), (c, m, mode))
WRAP(rknn_matmul_create,     a_mmcreate, (void* c, void* i, void* io), (c, i, io))

static void* (*real_rknn_create_mem)(unsigned long long, unsigned int) = NULL;
void* rknn_create_mem(unsigned long long c, unsigned int size) {
    if (!real_rknn_create_mem)
        real_rknn_create_mem = (void* (*)(unsigned long long, unsigned int))dlsym(RTLD_NEXT, "rknn_create_mem");
    uint64_t t0 = now_ns();
    void* r = real_rknn_create_mem(c, size);
    atomic_fetch_add_explicit(&a_creatmem.ns, now_ns() - t0, memory_order_relaxed);
    atomic_fetch_add_explicit(&a_creatmem.calls, 1, memory_order_relaxed);
    return r;
}

static void* (*real_rknn_create_mem_from_fd)(unsigned long long, int, void*, unsigned int, int) = NULL;
void* rknn_create_mem_from_fd(unsigned long long c, int fd, void* v, unsigned int size, int off) {
    if (!real_rknn_create_mem_from_fd)
        real_rknn_create_mem_from_fd = (void* (*)(unsigned long long, int, void*, unsigned int, int))dlsym(RTLD_NEXT, "rknn_create_mem_from_fd");
    uint64_t t0 = now_ns();
    void* r = real_rknn_create_mem_from_fd(c, fd, v, size, off);
    atomic_fetch_add_explicit(&a_creatfd.ns, now_ns() - t0, memory_order_relaxed);
    atomic_fetch_add_explicit(&a_creatfd.calls, 1, memory_order_relaxed);
    return r;
}

static void print_row(const char* name, acc_t* a) {
    fprintf(stderr, " %s=%llu/%.1fms", name,
            (unsigned long long)atomic_load_explicit(&a->calls, memory_order_relaxed),
            atomic_load_explicit(&a->ns, memory_order_relaxed) / 1e6);
}

static void* monitor(void* arg) {
    (void)arg;
    uint64_t t0 = now_ns();
    for (;;) {
        struct timespec ts = {5, 0};
        nanosleep(&ts, NULL);
        fprintf(stderr, "RKNNSHIM t=%.1fs", (now_ns() - t0) / 1e9);
        print_row("run",    &a_run);
        print_row("setio",  &a_setio);
        print_row("sync",   &a_sync);
        print_row("mmcreate", &a_mmcreate);
        print_row("creatmem", &a_creatmem);
        print_row("creatfd",  &a_creatfd);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    return NULL;
}

__attribute__((constructor)) static void shim_init(void) {
    pthread_t t;
    pthread_create(&t, NULL, monitor, NULL);
    pthread_detach(t);
}
