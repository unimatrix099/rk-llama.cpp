// FLASH_ATTN_EXT: RKNPU backend vs CPU backend at Gemma-4 E4B shapes
// (head dim 256, 8 query heads on 2 KV heads, causal mask, softcap, 1-2
// sequences, two rounds). Hardware test: needs the board's NPU. Caught the P*V-all-zeros
// bug of the first NPU attention version (B re-bind, decode research #1e).
//
// Build on the board from the repo root:
//   g++ -O2 docs/backend/test-rknpu2-flash-attn.cpp -I ggml/include -Lbuild/bin \
//       -lggml -lggml-base -lggml-cpu -lggml-rknpu2 -Wl,-rpath,$PWD/build/bin -o test-rknpu2-flash-attn
//   ulimit -n 65536 && ./test-rknpu2-flash-attn
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
extern "C" ggml_backend_reg_t ggml_backend_rknpu2_reg(void);

static std::vector<float> run(ggml_backend_t be, int DK, int nq, int nh, int nkvh, int nkv, int nseq, bool use_mask, float softcap,
                              const std::vector<float>& qv, const std::vector<ggml_fp16_t>& kv, const std::vector<ggml_fp16_t>& vv, const std::vector<ggml_fp16_t>& mv) {
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, nq, nh, nseq);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, nkv, nkvh, nseq);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, nkv, nkvh, nseq);
    ggml_tensor* m = use_mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, GGML_PAD(nq, 1), 1, nseq) : nullptr;
    ggml_tensor* o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / 16.0f, 0.0f, softcap);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kv.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, vv.data(), 0, ggml_nbytes(v));
    if (m) ggml_backend_tensor_set(m, mv.data(), 0, ggml_nbytes(m));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out(ggml_nelements(o));
    ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return out;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_reg_t reg = ggml_backend_rknpu2_reg();
    ggml_backend_t npu = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), NULL);
    int bad = 0;
    struct { int nq, nkv, nseq; bool mask; float softcap; } cases[] = {
        {64, 256, 1, false, 0.0f}, {64, 256, 1, true, 0.0f}, {64, 256, 1, true, 30.0f}, {64, 256, 2, true, 0.0f}, {512, 512, 1, true, 0.0f},
        {512, 512, 4, true, 0.0f},
    };
    const int DK = 256, nh = 8, nkvh = 2;
    // two rounds with different data: the second reuses the backend's warm
    // attention contexts, so state left in them from earlier calls shows up
    for (int round = 0; round < 2; ++round)
    for (auto& c : cases) {
        std::mt19937 g(11 + 1000 * round); std::normal_distribution<float> d(0, 1);
        std::vector<float> qv((size_t)DK * c.nq * nh * c.nseq);
        std::vector<ggml_fp16_t> kv((size_t)DK * c.nkv * nkvh * c.nseq), vv(kv.size());
        for (auto& x : qv) x = d(g);
        for (auto& x : kv) x = ggml_fp32_to_fp16(d(g));
        for (auto& x : vv) x = ggml_fp32_to_fp16(d(g));
        const int nqp = GGML_PAD(c.nq, 1);
        std::vector<ggml_fp16_t> mv((size_t)c.nkv * nqp * c.nseq);
        for (int s = 0; s < c.nseq; ++s) for (int i = 0; i < nqp; ++i) for (int j = 0; j < c.nkv; ++j)
            mv[((size_t)s * nqp + i) * c.nkv + j] = ggml_fp32_to_fp16(j <= i + (c.nkv - c.nq) ? 0.0f : -INFINITY);
        auto a = run(cpu, DK, c.nq, nh, nkvh, c.nkv, c.nseq, c.mask, c.softcap, qv, kv, vv, mv);
        auto b = run(npu, DK, c.nq, nh, nkvh, c.nkv, c.nseq, c.mask, c.softcap, qv, kv, vv, mv);
        double worst = 0; size_t wi = 0;
        for (size_t i = 0; i < a.size(); ++i) { double e = fabs(a[i] - b[i]); if (e > worst) { worst = e; wi = i; } }
        const bool ok = worst < 2e-2;
        bad += !ok;
        printf("round %d nq=%d nkv=%d nseq=%d mask=%d softcap=%g: max abs err %.3e at %zu (cpu %g npu %g) %s\n",
               round, c.nq, c.nkv, c.nseq, c.mask, c.softcap, worst, wi, a[wi], b[wi], ok ? "OK" : "WRONG");
    }
    return bad;
}
