// Long prompt, then the same tail tokens decoded one at a time vs as one verify-sized batch:
// per tail position, max |logit diff| and the top-2 of each run.
// usage: longdiff <model> <prompt.txt (already chat-formatted)> <tail text>   (env NB = batch size, default 6)
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s, bool bos) {
    std::vector<llama_token> t(s.size() + 16);
    int n = llama_tokenize(v, s.c_str(), s.size(), t.data(), t.size(), bos, true);
    t.resize(n);
    return t;
}

int main(int argc, char ** argv) {
    const int NB = getenv("NB") ? atoi(getenv("NB")) : 6;
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::stringstream ss; ss << std::ifstream(argv[2]).rdbuf();
    std::vector<llama_token> pre = tok(vocab, ss.str(), true), tail = tok(vocab, argv[3], false);
    const int nv = llama_vocab_n_tokens(vocab);
    printf("prompt %zu tokens, tail %zu tokens\n", pre.size(), tail.size());
    std::vector<std::vector<float>> L[2];
    for (int run = 0; run < 2; ++run) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 16384; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 4;
        llama_context * ctx = llama_init_from_model(model, cp);
        for (size_t i = 0; i < pre.size(); i += 2048) {
            const int n = std::min<size_t>(2048, pre.size() - i);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; ++j) { b.token[j] = pre[i + j]; b.pos[j] = i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = i + j + 1 == pre.size(); }
            b.n_tokens = n;
            if (llama_decode(ctx, b)) { printf("prefill failed\n"); return 1; }
            llama_batch_free(b);
        }
        const int step = run == 0 ? 1 : NB;
        for (size_t i = 0; i < tail.size(); i += step) {
            const int n = std::min<size_t>(step, tail.size() - i);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; ++j) { b.token[j] = tail[i + j]; b.pos[j] = pre.size() + i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = true; }
            b.n_tokens = n;
            if (llama_decode(ctx, b)) { printf("decode failed\n"); return 1; }
            for (int j = 0; j < n; ++j) { const float * l = llama_get_logits_ith(ctx, j); L[run].emplace_back(l, l + nv); }
            llama_batch_free(b);
        }
        llama_free(ctx);
    }
    auto top2 = [&](const std::vector<float> & l, int & a, int & b) { a = b = -1; for (int k = 0; k < (int)l.size(); ++k) { if (a < 0 || l[k] > l[a]) { b = a; a = k; } else if (b < 0 || l[k] > l[b]) b = k; } };
    auto piece = [&](int t) { char buf[64]; int n = llama_token_to_piece(vocab, t, buf, sizeof buf - 1, 0, true); buf[n > 0 ? n : 0] = 0; return std::string(buf); };
    for (size_t p = 0; p < tail.size(); ++p) {
        double md = 0; for (int k = 0; k < nv; ++k) md = std::max(md, std::fabs((double)L[0][p][k] - L[1][p][k]));
        int a0, b0, a1, b1; top2(L[0][p], a0, b0); top2(L[1][p], a1, b1);
        printf("after %-10s max|d|=%-9.3g single: %s %.3f / %s %.3f   batch: %s %.3f / %s %.3f%s\n", ("'" + piece(tail[p]) + "'").c_str(), md,
               piece(a0).c_str(), L[0][p][a0], piece(b0).c_str(), L[0][p][b0], piece(a1).c_str(), L[1][p][a1], piece(b1).c_str(), L[1][p][b1], a0 != a1 ? "  <-- ARGMAX DIFFERS" : "");
    }
    llama_model_free(model);
    return 0;
}
