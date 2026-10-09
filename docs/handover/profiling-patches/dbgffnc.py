f='ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'; s=open(f).read()
def rep(o,n):
    global s
    assert s.count(o)==1, o[:60]; s=s.replace(o,n)
rep("""static void rknpu_ffn_block(ggml_backend_rknpu_context* bctx, const struct ggml_tensor* src1, struct ggml_tensor* down_dst, int n_omp) {""","""static double g_wc[16]; static double g_cpu[16]; static int g_wn = 0;
static inline double wc_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void rknpu_ffn_block(ggml_backend_rknpu_context* bctx, const struct ggml_tensor* src1, struct ggml_tensor* down_dst, int n_omp) {
    if (++g_wn % 42 == 0) { fprintf(stderr, "DBGWC"); for (int i = 0; i < 7; ++i) fprintf(stderr, " c%d: cpu %.1f wait %.1f |", i, g_cpu[i], g_wc[i]); fprintf(stderr, "\\n"); }
    double tc0 = wc_now();""")
rep("""        bctx->async_runner.wait();
        start_batch(c);""","""        g_cpu[c - 1] += wc_now() - tc0; { const double a = wc_now(); bctx->async_runner.wait(); g_wc[c - 1] += wc_now() - a; } tc0 = wc_now();
        start_batch(c);""")
rep("""    bctx->async_runner.wait();
    rknpu_w4a4_job_collect(d, n - 1, out, n_omp);
}""","""    g_cpu[n + 1] += wc_now() - tc0; { const double a = wc_now(); bctx->async_runner.wait(); g_wc[n + 1] += wc_now() - a; }
    rknpu_w4a4_job_collect(d, n - 1, out, n_omp);
}""")
open(f,'w').write(s)
