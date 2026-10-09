f='ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'; s=open(f).read()
def rep(o,n):
    global s
    assert s.count(o)==1, o[:60]; s=s.replace(o,n)
rep("""struct rknpu_async_runner {
    void start(const std::vector<std::shared_ptr<rknpu_matmul_context>>& ctxs) {
        std::unique_lock<std::mutex> lock(mutex);""","""static std::atomic<uint64_t> g_span_ns{0}, g_span_n{0}, g_ctx_n{0};
struct rknpu_async_runner {
    std::chrono::steady_clock::time_point t_start;
    void start(const std::vector<std::shared_ptr<rknpu_matmul_context>>& ctxs) {
        std::unique_lock<std::mutex> lock(mutex);
        t_start = std::chrono::steady_clock::now(); g_span_n++; g_ctx_n += ctxs.size();""")
rep("""                if (--pending == 0) cv_done.notify_all();""","""                if (--pending == 0) { g_span_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_start).count(); cv_done.notify_all(); }""")
rep("""static void rknpu_ffn_block(ggml_backend_rknpu_context* bctx, const struct ggml_tensor* src1, struct ggml_tensor* down_dst, int n_omp) {""","""static void rknpu_ffn_block(ggml_backend_rknpu_context* bctx, const struct ggml_tensor* src1, struct ggml_tensor* down_dst, int n_omp) {
    static int calls = 0; if (++calls % 42 == 0) fprintf(stderr, "DBGSPAN calls=%d batches=%llu ctxs=%llu span=%.1f ms\\n", calls, (unsigned long long)g_span_n.load(), (unsigned long long)g_ctx_n.load(), g_span_ns.load() / 1e6);""")
open(f,'w').write(s)
