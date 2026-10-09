// Same tokens decoded one at a time vs as one batch: first activation that differs
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>

struct cap {
    int pos0 = 0;                                   // position of column 0 in this ubatch
    int n_tok = 0;
    std::map<std::string, std::map<int, std::vector<float>>> rows;   // name -> pos -> row
    std::vector<std::string> order;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    cap * c = (cap *) ud;
    static const char * obs = getenv("OBS");
    if (obs && !strstr(t->name, obs)) return ask ? false : true;
    if (t->type != GGML_TYPE_F32 || t->ne[1] != c->n_tok || t->ne[2] != 1 || t->ne[3] != 1) return ask ? true : true;
    if (ask) return true;
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    std::string name = t->name;
    if (!c->rows.count(name)) c->order.push_back(name);
    for (int j = 0; j < c->n_tok; ++j) {
        c->rows[name][c->pos0 + j] = std::vector<float>(buf.begin() + (size_t)j * t->ne[0], buf.begin() + (size_t)(j + 1) * t->ne[0]);
    }
    return true;
}

int main(int argc, char ** argv) {
    const int N = getenv("NB") ? atoi(getenv("NB")) : 8;
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(256);
    const char * text = getenv("LONG") ? "Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre ." : "The quick brown fox jumps over the lazy dog because it wanted to";
    int nt = llama_tokenize(vocab, text, strlen(text), toks.data(), toks.size(), true, false);
    toks.resize(nt);
    cap c1, c2;
    for (int run = 0; run < 2; ++run) {
        cap & c = run == 0 ? c1 : c2;
        auto cp = llama_context_default_params();
        cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = run == 0 ? 1 : N;
        cp.n_threads = cp.n_threads_batch = 4;
        if (!getenv("NOCB")) { cp.cb_eval = cb; cp.cb_eval_user_data = &c; }
        llama_context * ctx = llama_init_from_model(model, cp);
        // shared prefix: tokens [0, nt-N) as one batch in both runs (ubatch 1 vs N differ anyway, so keep prefix ub-independent below)
        const int T = getenv("TAIL") ? atoi(getenv("TAIL")) : 0;
        const int n_pre = nt - N - T;
        for (int i = 0; i < nt; ) {
            int step = (i < n_pre || i >= nt - T) ? 1 : (run == 0 ? 1 : N);
            if (getenv("ALL") && run == 1) step = std::min(N, nt - i);
            if (getenv("REJ") && run == 1 && i >= 2) {
                // MTP-like verify: [real token, N-1 wrong drafts], keep 1, drop the drafts' KV
                const int nb = std::min(N, 256);
                llama_batch b = llama_batch_init(nb, 0, 1);
                for (int j = 0; j < nb; ++j) {
                    b.token[j] = j == 0 ? toks[i] : (llama_token)(1000 + 37 * j + i); b.pos[j] = i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = true;
                }
                b.n_tokens = nb;
                c.pos0 = i; c.n_tok = nb;
                if (llama_decode(ctx, b) != 0) { printf("decode failed\n"); return 1; }
                const int nv = llama_vocab_n_tokens(vocab);
                if (!c.rows.count("logits")) c.order.push_back("logits");
                const float * l = llama_get_logits_ith(ctx, 0); c.rows["logits"][i] = std::vector<float>(l, l + nv);
                llama_memory_seq_rm(llama_get_memory(ctx), 0, i + 1, -1);
                llama_batch_free(b);
                i += 1;
                continue;
            }
            llama_batch b = llama_batch_init(step, 0, 1);
            for (int j = 0; j < step; ++j) {
                b.token[j] = toks[i + j]; b.pos[j] = i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = true;
            }
            b.n_tokens = step;
            c.pos0 = i; c.n_tok = step;
            if (llama_decode(ctx, b) != 0) { printf("decode failed\n"); return 1; }
            if (getenv("NOCB")) {
                const int nv = llama_vocab_n_tokens(vocab);
                if (!c.rows.count("logits")) c.order.push_back("logits");
                for (int j = 0; j < step; ++j) { const float * l = llama_get_logits_ith(ctx, j); c.rows["logits"][i + j] = std::vector<float>(l, l + nv); }
            }
            llama_batch_free(b);
            i += step;
        }
        llama_free(ctx);
    }
    // compare rows of the last N positions, in graph order of the batched run
    int shown = 0;
    for (const auto & name : c2.order) {
        if (!c1.rows.count(name)) continue;
        double md = 0; int worst_pos = -1;
        const int T = getenv("TAIL") ? atoi(getenv("TAIL")) : 0;
        for (int p = (getenv("ALL") || getenv("REJ")) ? 0 : (T ? nt - T : nt - N); p < nt; ++p) {
            auto & a = c1.rows[name]; auto & b = c2.rows[name];
            if (!a.count(p) || !b.count(p) || a[p].size() != b[p].size()) continue;
            for (size_t k = 0; k < a[p].size(); ++k) {
                double d = std::fabs((double)a[p][k] - (double)b[p][k]);
                if (d > md) { md = d; worst_pos = p; }
            }
        }
        if (md > 0 && shown < 25) { printf("DIFF %-28s max|d|=%.3g (pos %d)\n", name.c_str(), md, worst_pos); ++shown; }
        if (md == 0 && shown == 0) printf("same %s\n", name.c_str());
    }
    llama_model_free(model);
    return 0;
}
