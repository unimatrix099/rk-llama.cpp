// Prefix decoded one token at a time to length P, then N tokens as one batch vs one at a time; compare logits
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
int main(int argc, char ** argv) {
    const int N = 4;
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::string text;
    for (int i = 0; i < 40; ++i) text += "The lighthouse keeper climbed the stairs every night to light the lamp, watching ships pass in the dark. ";
    std::vector<llama_token> toks(4096);
    int nt = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false); toks.resize(nt);
    for (int ai = 2; ai < argc; ++ai) {
        const int P = atoi(argv[ai]);
        std::vector<std::vector<float>> out[3];
        for (int run = 0; run < 3; ++run) {
            auto cp = llama_context_default_params(); cp.n_ctx = 4096; cp.n_batch = 1024; cp.n_ubatch = 1024; cp.n_threads = cp.n_threads_batch = 4;
            llama_context * ctx = llama_init_from_model(model, cp);
            for (int i = 0; i < P + N; ) {
                const int step = run == 2 ? P + N : ((i >= P && run == 1) ? N : 1);
                llama_batch b = llama_batch_init(step, 0, 1);
                for (int j = 0; j < step; ++j) { b.token[j] = toks[i + j]; b.pos[j] = i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = i + j >= P; }
                b.n_tokens = step; llama_decode(ctx, b);
                for (int j = 0; j < step; ++j) if (i + j >= P) { const float * l = llama_get_logits_ith(ctx, j); out[run].emplace_back(l, l + nv); }
                llama_batch_free(b); i += step;
            }
            llama_free(ctx);
        }
        auto kl = [&](int a, int b) { double worst = 0; int top_diff = 0;
            for (int r = 0; r < N; ++r) {
                const auto & x = out[a][r]; const auto & y = out[b][r];
                double mx = -1e30, my = -1e30; int ax = 0, ay = 0;
                for (int k = 0; k < nv; ++k) { if (x[k] > mx) { mx = x[k]; ax = k; } if (y[k] > my) { my = y[k]; ay = k; } }
                double sx = 0, sy = 0; for (int k = 0; k < nv; ++k) { sx += std::exp(x[k] - mx); sy += std::exp(y[k] - my); }
                double d = 0; for (int k = 0; k < nv; ++k) { double lp = x[k] - mx - std::log(sx), lq = y[k] - my - std::log(sy); d += std::exp(lp) * (lp - lq); }
                worst = std::max(worst, d); top_diff += ax != ay;
            }
            char buf[64]; snprintf(buf, sizeof buf, "KL %.2e top1-diff %d/%d", worst, top_diff, N); return std::string(buf); };
        printf("positions %d..%d: seq-vs-full %s | batch4-vs-full %s | seq-vs-batch4 %s\n", P, P + N - 1, kl(2, 0).c_str(), kl(2, 1).c_str(), kl(0, 1).c_str());
    }
    return 0;
}
