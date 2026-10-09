// Replays an MTP verify sequence after a long prompt, against one-token-at-a-time decoding:
//   single: decode the tail tokens one by one
//   verify: batch [t0, drafts...], keep the first `keep` tokens, drop the rest (seq_rm), next batch, ...
// Context set up like llama-server (-c 32768, 4 slots, unified KV, slot 3).
// usage: longrej <model> <prompt.txt> <tail> <batch1> <keep1> <batch2> ...   (batches are texts)
// The logits of every verify-batch row are compared with the single run at the same position.
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

static llama_context * make_ctx(llama_model * model, const std::vector<llama_token> & pre, int seq) {
    auto cp = llama_context_default_params();
    cp.n_ctx = getenv("NCTX") ? atoi(getenv("NCTX")) : 32768; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.n_seq_max = getenv("NSEQ") ? atoi(getenv("NSEQ")) : 4; cp.kv_unified = true;
    cp.n_threads = cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    // SPLIT=1,2048,... : prompt batch sizes as llama-server logs them (it checkpoints near the end)
    std::vector<int> split;
    if (getenv("SPLIT")) { std::stringstream sp(getenv("SPLIT")); std::string x; while (std::getline(sp, x, ',')) split.push_back(atoi(x.c_str())); }
    for (size_t i = 0, k = 0; i < pre.size(); ) {
        const int n = std::min<size_t>(k < split.size() ? split[k++] : 2048, pre.size() - i);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int j = 0; j < n; ++j) { b.token[j] = pre[i + j]; b.pos[j] = i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = seq; b.logits[j] = i + j + 1 == pre.size(); }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) { printf("prefill failed\n"); exit(1); }
        llama_batch_free(b);
        i += n;
    }
    return ctx;
}

static std::vector<std::vector<float>> run(llama_context * ctx, const std::vector<llama_token> & t, int pos0, int seq, int nv) {
    llama_batch b = llama_batch_init(t.size(), 0, 1);
    for (size_t j = 0; j < t.size(); ++j) { b.token[j] = t[j]; b.pos[j] = pos0 + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = seq; b.logits[j] = true; }
    b.n_tokens = t.size();
    if (llama_decode(ctx, b)) { printf("decode failed\n"); exit(1); }
    std::vector<std::vector<float>> L;
    for (size_t j = 0; j < t.size(); ++j) { const float * l = llama_get_logits_ith(ctx, j); L.emplace_back(l, l + nv); }
    llama_batch_free(b);
    return L;
}

int main(int argc, char ** argv) {
    const int seq = getenv("SEQ") ? atoi(getenv("SEQ")) : 3;
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::stringstream ss; ss << std::ifstream(argv[2]).rdbuf();
    const std::vector<llama_token> pre = tok(vocab, ss.str(), true), tail = tok(vocab, argv[3], false);
    const int P = pre.size();
    auto piece = [&](int t) { char buf[64]; int n = llama_token_to_piece(vocab, t, buf, sizeof buf - 1, 0, true); buf[n > 0 ? n : 0] = 0; return std::string(buf); };
    auto argmax = [&](const std::vector<float> & l) { int a = 0; for (int k = 1; k < nv; ++k) if (l[k] > l[a]) a = k; return a; };

    // single: tail one token at a time
    std::vector<std::vector<float>> S;
    {
        llama_context * ctx = make_ctx(model, pre, seq);
        for (size_t i = 0; i < tail.size(); ++i) S.push_back(run(ctx, { tail[i] }, P + i, seq, nv)[0]);
        llama_free(ctx);
    }
    // verify replay
    llama_context * ctx = make_ctx(model, pre, seq);
    int pos = P;
    for (int a = 4; a + 1 < argc; a += 2) {
        const std::vector<llama_token> bt = tok(vocab, argv[a], false);
        const int keep = atoi(argv[a + 1]);
        auto L = run(ctx, bt, pos, seq, nv);
        printf("batch '%s' (%zu tokens) at pos %d, keep %d\n", argv[a], bt.size(), pos, keep);
        for (size_t j = 0; j < bt.size(); ++j) {
            const int p = pos + j - P;   // tail index whose logits this row predicts after
            if (p < 0 || p >= (int)S.size()) continue;
            double md = 0; for (int k = 0; k < nv; ++k) md = std::max(md, std::fabs((double)S[p][k] - L[j][k]));
            const int as = argmax(S[p]), av = argmax(L[j]);
            printf("  after %-10s max|d|=%-9.3g single %s %.3f   verify %s %.3f%s%s\n", ("'" + piece(bt[j]) + "'").c_str(), md,
                   piece(as).c_str(), S[p][as], piece(av).c_str(), L[j][av], bt[j] != tail[p] ? "  (row token differs from tail)" : "",
                   as != av ? "  <-- ARGMAX DIFFERS" : "");
        }
        pos += keep;
        llama_memory_seq_rm(llama_get_memory(ctx), seq, pos, -1);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
