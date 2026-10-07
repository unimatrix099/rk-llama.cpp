import re
f='ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'; s=open(f).read()
hdr = '''
static std::atomic<uint64_t> g_sync_ns[2], g_sync_n[2], g_sync_bytes[2];
static int dbg_mem_sync(rknn_matmul_ctx ctx, rknn_tensor_mem* m, int mode) {
    const auto t0 = std::chrono::steady_clock::now();
    const int r = rknn_mem_sync(ctx, m, (rknn_mem_sync_mode)mode);
    const int i = mode == RKNN_MEMORY_SYNC_TO_DEVICE ? 0 : 1;
    g_sync_ns[i] += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    g_sync_n[i]++; g_sync_bytes[i] += m->size;
    static auto last = std::chrono::steady_clock::now();
    if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) { last = std::chrono::steady_clock::now();
        fprintf(stderr, "DBGSYNC to_dev %.1f ms n=%llu %.0f MB | from_dev %.1f ms n=%llu %.0f MB\\n", g_sync_ns[0] / 1e6, (unsigned long long)g_sync_n[0].load(), g_sync_bytes[0] / 1e6, g_sync_ns[1] / 1e6, (unsigned long long)g_sync_n[1].load(), g_sync_bytes[1] / 1e6); }
    return r;
}
'''
i = s.index("struct rknpu_async_runner {")
s = s[:i] + hdr + s[i:]
body_start = i + len(hdr)
s = s[:body_start] + s[body_start:].replace("rknn_mem_sync(", "dbg_mem_sync(")
open(f,'w').write(s)
