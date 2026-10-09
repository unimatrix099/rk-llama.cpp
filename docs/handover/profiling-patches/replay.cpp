// Replay the server's exact MTP verify batches on the target; report first greedy mismatch
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <map>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
static std::vector<int> nums(const std::string & s) { std::vector<int> v; std::stringstream ss(s); int x; while (ss >> x) v.push_back(x); return v; }
// capture rows of named per-token tensors at a target position
static int g_pos0 = 0, g_ntok = 0, g_want = -1; static bool g_on = false;
static std::map<std::string, std::vector<float>> g_rows; static std::vector<std::string> g_order;
static bool cbf(struct ggml_tensor * t, bool ask, void *) {
    static const char * obs = getenv("OBS");
    if (!g_on || !obs || !strstr(t->name, obs)) return ask ? false : true;
    if (t->type != GGML_TYPE_F32 || t->ne[1] != g_ntok || t->ne[2] != 1) return ask ? false : true;
    if (ask) return true;
    const int j = g_want - g_pos0; if (j < 0 || j >= g_ntok) return true;
    std::vector<float> row(t->ne[0]);
    ggml_backend_tensor_get(t, row.data(), (size_t)j * t->nb[1], t->ne[0] * sizeof(float));
    if (!g_rows.count(t->name)) g_order.push_back(t->name);
    g_rows[t->name] = row; return true;
}
static int argmax(const float * l, int n) { int b = 0; for (int i = 1; i < n; ++i) if (l[i] > l[b]) b = i; return b; }
int main(int argc, char ** argv) {
    // argv: model, prompt_tokens.txt, cycles.txt (lines: id_last pos n_acc sampled d0 d1 ...)
    const int seq = getenv("SEQ") ? atoi(getenv("SEQ")) : 3;
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_seq_max = getenv("NSEQ") ? atoi(getenv("NSEQ")) : 4;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.kv_unified = getenv("UNIFIED") != nullptr;
    if (getenv("FAOFF")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (getenv("OBS")) { cp.cb_eval = cbf; cp.cb_eval_user_data = nullptr; }
    const int cap_cycle = getenv("CAP_CYCLE") ? atoi(getenv("CAP_CYCLE")) : -1;
    g_want = getenv("CAP_POS") ? atoi(getenv("CAP_POS")) : -1;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::ifstream fp(argv[2]); std::string line; std::getline(fp, line); std::vector<int> prompt = nums(line);
    // prompt in the server's pieces
    std::vector<int> cuts = getenv("CUTS") ? nums(getenv("CUTS")) : std::vector<int>{(int)prompt.size()};
    int p0 = 0;
    for (int cut : cuts) {
        llama_batch b = llama_batch_init(cut - p0, 0, 1);
        for (int i = p0; i < cut; ++i) { int j = i - p0; b.token[j] = prompt[i]; b.pos[j] = i; b.n_seq_id[j] = 1; b.seq_id[j][0] = seq; b.logits[j] = i == cut - 1; }
        b.n_tokens = cut - p0; llama_decode(ctx, b); llama_batch_free(b); p0 = cut;
    }
    printf("first sampled (argmax of prompt end) = %d\n", argmax(llama_get_logits_ith(ctx, -1), nv));
    std::ifstream fc(argv[3]); int ci = 0, bad = 0;
    while (std::getline(fc, line)) {
        std::vector<int> v = nums(line); if (v.size() < 4) continue;
        const int id_last = v[0], pos = v[1], n_acc_srv = v[2], sampled_srv = v[3];
        std::vector<int> d(v.begin() + 4, v.end());
        const int seq_until = getenv("SEQ_UNTIL") ? atoi(getenv("SEQ_UNTIL")) : -1;
        const int seq_from  = getenv("SEQ_FROM")  ? atoi(getenv("SEQ_FROM"))  : -1;
        const bool seqdec = getenv("SEQDEC") || (seq_until >= 0 && ci < seq_until) || (seq_from >= 0 && ci >= seq_from);
        if (seqdec) {
            // same accepted path, one token at a time: id_last, then the accepted drafts
            int n_acc = 0, sampled = -1; bool done = false;
            for (int j = 0; j <= n_acc_srv; ++j) {
                llama_batch b1 = llama_batch_init(1, 0, 1);
                g_on = ci == cap_cycle; g_pos0 = pos + j; g_ntok = 1;
                b1.token[0] = j == 0 ? id_last : d[j - 1]; b1.pos[0] = pos + j; b1.n_seq_id[0] = 1; b1.seq_id[0][0] = seq; b1.logits[0] = true; b1.n_tokens = 1;
                llama_decode(ctx, b1); llama_batch_free(b1);
                const int a = argmax(llama_get_logits_ith(ctx, -1), nv);
                if (!done) { if (j < (int)d.size() && a == d[j]) ++n_acc; else { sampled = a; done = true; } }
            }
            if (getenv("DUMP")) printf("D %d %d %d\n", ci, n_acc, sampled);
            if ((n_acc != n_acc_srv || sampled != sampled_srv) && bad++ < 3)
                printf("cycle %d pos %d: seqdec n_acc=%d sampled=%d | server n_acc=%d sampled=%d\n", ci, pos, n_acc, sampled, n_acc_srv, sampled_srv);
            ++ci; continue;
        }
        const int nb = 1 + (int)d.size();
        llama_batch b = llama_batch_init(nb, 0, 1);
        const int pc = getenv("PERTURB_CYCLE") ? atoi(getenv("PERTURB_CYCLE")) : -1;
        const int ptok = getenv("PERTURB_TOK") ? atoi(getenv("PERTURB_TOK")) : 100;
        for (int j = 0; j < nb; ++j) { b.token[j] = j == 0 ? id_last : (ci == pc && j >= 2 ? ptok + j : d[j - 1]); b.pos[j] = pos + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = seq; b.logits[j] = true; }
        g_on = ci == cap_cycle; g_pos0 = pos; g_ntok = nb;
        b.n_tokens = nb; if (llama_decode(ctx, b)) { printf("decode failed\n"); return 1; } llama_batch_free(b);
        int n_acc = 0, sampled = -1;
        for (int j = 0; j < nb; ++j) { int a = argmax(llama_get_logits_ith(ctx, j), nv); if (j < (int)d.size() && a == d[j]) { ++n_acc; continue; } sampled = a; break; }
        if (ci == pc) { for (int j = 0; j < nb; ++j) printf("P row %d argmax %d\n", j, argmax(llama_get_logits_ith(ctx, j), nv)); }
        if (getenv("DUMP")) printf("D %d %d %d\n", ci, n_acc, sampled);
        if ((n_acc != n_acc_srv || sampled != sampled_srv) && bad++ < 3)
            printf("cycle %d pos %d: replay n_acc=%d sampled=%d | server n_acc=%d sampled=%d\n", ci, pos, n_acc, sampled, n_acc_srv, sampled_srv);
        // follow the server's decision
        if (getenv("REDECODE")) {
            // drop the whole verify batch, then decode the kept tokens one at a time
            llama_memory_seq_rm(llama_get_memory(ctx), seq, pos, -1);
            for (int j = 0; j <= n_acc_srv; ++j) {
                llama_batch b1 = llama_batch_init(1, 0, 1);
                b1.token[0] = j == 0 ? id_last : d[j - 1]; b1.pos[0] = pos + j; b1.n_seq_id[0] = 1; b1.seq_id[0][0] = seq; b1.logits[0] = true; b1.n_tokens = 1;
                g_on = false; llama_decode(ctx, b1); llama_batch_free(b1);
            }
        } else {
            llama_memory_seq_rm(llama_get_memory(ctx), seq, pos + n_acc_srv + 1, -1);
        }
        ++ci;
    }
    printf("cycles %d, mismatches %d\n", ci, bad);
    if (getenv("CAP_OUT")) { FILE * f = fopen(getenv("CAP_OUT"), "wb"); for (auto & n : g_order) { int len = n.size(), m = g_rows[n].size(); fwrite(&len, 4, 1, f); fwrite(n.data(), 1, len, f); fwrite(&m, 4, 1, f); fwrite(g_rows[n].data(), 4, m, f); } fclose(f); }
    return 0;
}
