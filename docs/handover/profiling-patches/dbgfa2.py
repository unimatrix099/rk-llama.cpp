f='ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'; s=open(f).read()
def rep(o,n):
    global s
    assert s.count(o)==1, o[:70]; s=s.replace(o,n)
rep("""static void rknpu_flash_attn(ggml_backend_rknpu_context* bctx, struct ggml_tensor* dst, int n_omp) {""",
"""static double g_fa2[8]; static int g_fa2_n = 0;
static inline double fa2_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void rknpu_flash_attn(ggml_backend_rknpu_context* bctx, struct ggml_tensor* dst, int n_omp) {
    if (++g_fa2_n % 42 == 0) fprintf(stderr, "DBGFA2 calls=%d fill %.1f wqk %.1f soft %.1f wpv %.1f out %.1f total %.1f ms\\n", g_fa2_n, g_fa2[0], g_fa2[1], g_fa2[2], g_fa2[3], g_fa2[4], g_fa2[5]);
    const double t_all = fa2_now();
    struct FaT { double t0; ~FaT() { g_fa2[5] += fa2_now() - t0; } } fat{t_all};""")
rep("""    auto st_k = [&](int it, int st) { stage(it / n_kvh, it % n_kvh, st, it % 3); };""",
"""    auto st_k = [&](int it, int st) { const double a0 = fa2_now(); stage(it / n_kvh, it % n_kvh, st, it % 3); if (st == 0 || st == 2 || st == 4) g_fa2[st == 0 ? 0 : st == 2 ? 2 : 4] += fa2_now() - a0; };
    auto wt = [&](rknpu_fn_pool::ticket& t, int i) { const double a0 = fa2_now(); bctx->fa_pool.wait(t); g_fa2[i] += fa2_now() - a0; };""")
rep("""        bctx->fa_pool.wait(t_qk[it]);""","""        wt(t_qk[it], 1);""")
rep("""        if (it >= 1) { bctx->fa_pool.wait(t_pv[it - 1]); st_k(it - 1, 4); }""","""        if (it >= 1) { wt(t_pv[it - 1], 3); st_k(it - 1, 4); }""")
rep("""    bctx->fa_pool.wait(t_pv[n - 1]);
    st_k(n - 1, 4);""","""    wt(t_pv[n - 1], 3);
    st_k(n - 1, 4);""")
open(f,'w').write(s)
