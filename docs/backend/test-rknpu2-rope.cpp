// ROPE: RKNPU backend vs CPU backend, bit for bit (decode research #1g/#2
// follow-up: RoPE fused into the Q/K dequant). Gemma-4 shapes: head dim
// 256 (sliding-window layers) and 512 with frequency factors (global
// layers), NEOX and NORMAL modes, partial rotation, scaled frequencies.
// Hardware test: needs the board's NPU library. RKNPU_ROPE_ANY=1 lets the
// backend take a plain RoPE (in the model it only takes it after a head
// norm it runs itself).
//
// Build on the board from the repo root:
//   g++ -O2 docs/backend/test-rknpu2-rope.cpp -I ggml/include -Lbuild/bin \
//       -lggml -lggml-base -lggml-cpu -lggml-rknpu2 -Wl,-rpath,$PWD/build/bin -o test-rknpu2-rope
//   ulimit -n 65536 && ./test-rknpu2-rope
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
extern "C" ggml_backend_reg_t ggml_backend_rknpu2_reg(void);

struct rope_case { int hd, n_dims, mode; float base, fscale, attn; bool ff; int max_pos; };

static std::vector<float> run(ggml_backend_t be, const rope_case& c, int nh, int nt, const std::vector<float>& xv,
                              const std::vector<int32_t>& pv, const std::vector<float>& fv) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.hd, nh, nt);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
    ggml_tensor* ff  = c.ff ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_dims / 2) : nullptr;
    ggml_tensor* y = ggml_rope_ext(ctx, x, pos, ff, c.n_dims, c.mode, 8192, c.base, c.fscale, 0.0f, c.attn, 32.0f, 1.0f);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(pos, pv.data(), 0, ggml_nbytes(pos));
    if (ff) ggml_backend_tensor_set(ff, fv.data(), 0, ggml_nbytes(ff));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main() {
    setenv("RKNPU_ROPE_ANY", "1", 1);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t npu = ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_rknpu2_reg(), 0), NULL);
    const rope_case cases[] = {
        {256, 256, GGML_ROPE_TYPE_NEOX,   10000.0f,    1.0f,  1.0f,  false, 600},    // sliding-window layer
        {512, 512, GGML_ROPE_TYPE_NEOX,   1000000.0f,  1.0f,  1.0f,  true,  600},    // global layer, freq factors
        {512, 128, GGML_ROPE_TYPE_NEOX,   1000000.0f,  1.0f,  1.0f,  true,  32000},  // partial rotation, long positions
        {256, 256, GGML_ROPE_TYPE_NORMAL, 10000.0f,    0.5f,  1.3f,  false, 4000},   // NORMAL mode, scaled
        {128, 64,  GGML_ROPE_TYPE_NEOX,   500000.0f,   0.25f, 0.9f,  true,  100000},
    };
    int bad = 0;
    for (const auto& c : cases) {
        const int nh = 8, nt = 96;
        std::mt19937 g(7 + c.hd + c.n_dims);
        std::normal_distribution<float> d(0, 3);
        std::vector<float> xv((size_t)c.hd * nh * nt), fv(c.n_dims / 2);
        std::vector<int32_t> pv(nt);
        for (auto& v : xv) v = d(g);
        for (auto& v : pv) v = (int32_t)(g() % (uint32_t)c.max_pos);
        for (auto& v : fv) v = 1.0f + (float)(g() % 1000) / 37.0f;
        const auto a = run(cpu, c, nh, nt, xv, pv, fv);
        const auto b = run(npu, c, nh, nt, xv, pv, fv);
        size_t diff = 0;
        double worst = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (memcmp(&a[i], &b[i], sizeof(float)) != 0) { ++diff; worst = std::max(worst, (double)fabsf(a[i] - b[i])); }
        }
        const bool ok = diff == 0;
        bad += !ok;
        printf("hd=%d n_dims=%d mode=%d base=%g fscale=%g attn=%g ff=%d: %zu of %zu differ (max %.3e) %s\n",
               c.hd, c.n_dims, c.mode, c.base, c.fscale, c.attn, (int)c.ff, diff, a.size(), worst, ok ? "OK" : "WRONG");
    }
    return bad;
}
