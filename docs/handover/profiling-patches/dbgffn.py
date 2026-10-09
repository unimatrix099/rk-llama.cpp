f='ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'; s=open(f).read()
def rep(o,n):
    global s
    assert s.count(o)==1, o[:60]; s=s.replace(o,n)
rep("""static void rknpu_ffn_block(ggml_backend_rknpu_context* bctx, const struct ggml_tensor* src1, struct ggml_tensor* down_dst, int n_omp) {""",
"""static double g_ffn_t[6]; static int g_ffn_n = 0;
static inline double ffn_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void rknpu_ffn_block(ggml_backend_rknpu_context* bctx, const struct ggml_tensor* src1, struct ggml_tensor* down_dst, int n_omp) {
    if (++g_ffn_n % 42 == 0) fprintf(stderr, "DBGFFN calls=%d prep %.1f wait %.1f start %.1f fused %.1f collectD %.1f ms\\n", g_ffn_n, g_ffn_t[0], g_ffn_t[1], g_ffn_t[2], g_ffn_t[3], g_ffn_t[4]);
    double t0 = ffn_now(), t1;
#define FT(i) do { t1 = ffn_now(); g_ffn_t[i] += t1 - t0; t0 = t1; } while (0)""")
rep("""    rknpu_ffn_prep(g, u, x, row_stride, 0, n_omp);
    start_batch(0);
    for (int c = 1; c < n + 2; ++c) {
        if (c < n) rknpu_ffn_prep(g, u, x, row_stride, c, n_omp);   // overlaps batch c-1
        bctx->async_runner.wait();
        start_batch(c);
        if (c - 1 < n) rknpu_ffn_fused(g, u, d, c - 1, n_omp);       // gate/up chunk c-1 done
        if (c >= 3) rknpu_w4a4_job_collect(d, c - 3, out, n_omp);  // down chunk c-3 done
    }
    bctx->async_runner.wait();
    rknpu_w4a4_job_collect(d, n - 1, out, n_omp);""","""    FT(5);
    rknpu_ffn_prep(g, u, x, row_stride, 0, n_omp); FT(0);
    start_batch(0); FT(2);
    for (int c = 1; c < n + 2; ++c) {
        if (c < n) { rknpu_ffn_prep(g, u, x, row_stride, c, n_omp); FT(0); }
        bctx->async_runner.wait(); FT(1);
        start_batch(c); FT(2);
        if (c - 1 < n) { rknpu_ffn_fused(g, u, d, c - 1, n_omp); FT(3); }
        if (c >= 3) { rknpu_w4a4_job_collect(d, c - 3, out, n_omp); FT(4); }
    }
    bctx->async_runner.wait(); FT(1);
    rknpu_w4a4_job_collect(d, n - 1, out, n_omp); FT(4);""")
open(f,'w').write(s)
